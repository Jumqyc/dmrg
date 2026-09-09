import math
import warnings

import torch
import cvxpy as cp

from itertools import product

from base import Broomstick, MPO, cached_einsum
from ext_register import extends_Broomstick


# ---------------------------------------------------------------------------
# CGO helpers.
#
# One sample is a product operator
#
#     A_i = 1j * (R_1 (x) R_2 (x) ... (x) R_n),
#
# where each R_j is an independent random Hermitian matrix acting on a site
# of L0.  The product is Hermitian, so multiplying by 1j makes A_i
# anti-Hermitian.
# ---------------------------------------------------------------------------


def _interior_sites(hamiltonian: MPO, lend: int, rend: int) -> list[int]:
    '''
    Return the sites in [lend, rend) that are not coupled to the outside of
    the patch by any Hamiltonian coupling.  These form the region L0 on which
    the anti-Hermitian operators A_i are supported.
    '''
    if not (0 <= lend < rend <= hamiltonian.L):
        raise ValueError(
            f"Patch [{lend}, {rend}) is out of bounds for MPO of length "
            f"{hamiltonian.L}."
        )

    boundary = set()
    for _, _, locs in hamiltonian.couplings:
        inside = [loc for loc in locs if lend <= loc < rend]
        outside = [loc for loc in locs if not (lend <= loc < rend)]
        if inside and outside:
            boundary.update(inside)

    return [site for site in range(lend, rend) if site not in boundary]


def _random_hermitian(dim: int,
                      device: torch.device,
                      generator: torch.Generator) -> torch.Tensor:
    '''Draw a random Hermitian dim x dim matrix.'''
    t = torch.randn((dim, dim), dtype=torch.complex128,
                    device=device, generator=generator)
    t = t + 1j * torch.randn((dim, dim), dtype=torch.complex128,
                             device=device, generator=generator)
    return 0.5 * (t + t.conj().T)


def _sample_product_operators(l0: list[int],
                              num_sample: int,
                              dim: int,
                              device: torch.device,
                              seed: int | None):
    '''Sample R_1 (x) ... (x) R_n with independent Hermitian R_j.'''
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return [
        {site: _random_hermitian(dim, device, generator) for site in l0}
        for _ in range(num_sample)
    ]


def _patch_hamiltonian(hamiltonian: MPO,
                       lend: int,
                       rend: int) -> MPO:
    '''Build H_L from the Hamiltonian couplings fully inside the patch.'''
    H_L = MPO(
        rend - lend,
        hamiltonian.physical_dim,
        mapping=dict(hamiltonian.mapping),
        dtype=hamiltonian.dtype,
        device=hamiltonian.device,
    )
    for ops, coeff, locs in hamiltonian.couplings:
        if all(lend <= loc < rend for loc in locs):
            H_L.add_couplings(ops, coeff, tuple(loc - lend for loc in locs))
    H_L.build()
    return H_L


def _orthonormal_transform(gram: torch.Tensor) -> torch.Tensor:
    '''Return X such that X @ H @ X^dagger = I for the Gram matrix H.'''
    q = gram.shape[0] * gram.shape[1]
    G = gram.reshape(q, q).conj()
    return torch.linalg.inv(torch.linalg.cholesky(G))



def _build_cgo_dual(M_samples: list[torch.Tensor], B: torch.Tensor):
    '''Build the dual CGO SDP for the upper bound on <B>.

    M_samples[i] is the Hermitian projected commutator corresponding to
    -i P_V [H_L, A_i] P_V in CGO's bra/ket convention, and
    B = P_V B P_V.  Both must be expressed in an orthonormal basis of V_L.

    The dual problem is

        maximize   Tr(rho B)
        subject to rho >= 0, Tr(rho) = 1,
                   Tr(rho M_i) = 0 for all i

    which is the tighter upper bound on <B>.  For the lower bound on <B>,
    build the same problem with -B and negate the result.
    '''
    import numpy as np

    M = [m.detach().cpu().numpy() for m in M_samples]
    M = [0.5 * (m + m.conj().T) for m in M]
    B_np = B.detach().cpu().numpy()
    B_np = 0.5 * (B_np + B_np.conj().T)

    q = B_np.shape[0]
    m = len(M)

    rho = cp.Variable((q, q), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]
    constraints += [cp.real(cp.trace(rho @ M[i])) == 0 for i in range(m)]
    return cp.Problem(cp.Maximize(cp.real(cp.trace(rho @ B_np))), constraints)


def _solve_problem(problem,
                   solver: str,
                   solver_kwargs: dict | None = None) -> float | None:
    '''Solve one cvxpy problem and return its value, or None on failure.'''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            problem.solve(solver=solver, verbose=False, **(solver_kwargs or {}))
    except Exception:
        return None

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        return None
    if problem.value is None:
        return None

    value = float(problem.value)
    return value if math.isfinite(value) else None


def _solve_bound(dual,
                 solver_configs: tuple[tuple[str, dict], ...]) -> float:
    '''Solve one dual SDP, trying each solver configuration in order.'''
    for solver, solver_kwargs in solver_configs:
        value = _solve_problem(dual, solver, solver_kwargs)
        if value is not None:
            return value

    raise RuntimeError(
        'CGO SDP could not be solved with any of the configured solvers: '
        f'{[name for name, _ in solver_configs]}.'
    )


def _solve_cgo_bounds(M_samples: list[torch.Tensor],
                      B: torch.Tensor) -> tuple[float, float]:
    '''Solve the CGO dual SDPs and return (lower, upper).'''
    solver_configs = (
        ('CLARABEL', {}),
        ('SCS', {'eps': 1e-9, 'max_iters': 200000}),
    )

    upper = _solve_bound(_build_cgo_dual(M_samples, B), solver_configs)
    lower = -_solve_bound(_build_cgo_dual(M_samples, -B), solver_configs)
    return float(lower), float(upper)


def CGO_projections(self: Broomstick,
                    l: int,
                    operators: dict[int, torch.Tensor],
                    num_sample: int = 200,
                    seed: int | None = None):
    '''
    Build the projected CGO quantities for the observable `operators`.

    This is the tensor-network part of CGO: it computes the projected
    commutators and observable in an orthonormal basis of the patch
    subspace V_L.  Most users should call :func:`CGO` instead, which solves
    the SDPs and returns the upper/lower bounds.

    Args:
        l: radius of the patch L = [lend, rend) around the observable.
        operators: dict mapping global site indices to single-site operators;
                   this is the observable B.
        num_sample: number of random anti-Hermitian operators A_i.
        seed: optional RNG seed for the A_i sampling.

    Returns:
        M_samples: list of q x q projected commutators P_V [H_L,A_i] P_V in
                   the orthonormal basis of V_L.
        B: q x q projected observable P_V B P_V in the same basis.
        X: basis transformation, |tilde alpha> = sum_beta X[alpha,beta] |beta>.
    '''

    if not all(op.shape == (self.physical_dim, self.physical_dim)
               for op in operators.values()):
        raise ValueError(
            f"All operators must be of shape "
            f"({self.physical_dim},{self.physical_dim}), "
            f"but got {[op.shape for op in operators.values()]}"
        )

    # The optimization is performed on the patch L = [lend, rend).
    lend = max(0, min(operators.keys()) - l)
    rend = min(self.L, max(operators.keys()) + l + 1)

    D_l = self.state.bond_dim[lend]
    D_r = self.state.bond_dim[rend]
    opti_dim = D_l * D_r
    pairs = list(product(product(range(D_l), range(D_r)), repeat=2))

    # ------------------------------------------------------------------
    # step 1: Gram matrix of the boundary MPS states.
    # ------------------------------------------------------------------
    gram = torch.zeros(
        (D_l, D_r, D_l, D_r), dtype=torch.complex128, device=self.device
    )

    # CGO is used with l > 2, which guarantees rend - lend >= 4, so the
    # single-site patch case (rend - lend == 1) does not need special handling.
    def _compute_inner_product(il: int, ir: int, jl: int, jr: int) -> complex:
        lenv = cached_einsum(
            'ij,ik->jk',
            self.state[lend][il, ...],
            self.state[lend][jl, ...].conj()
        )
        renv = cached_einsum(
            'ji,ki->jk',
            self.state[rend - 1][..., ir],
            self.state[rend - 1][..., jr].conj()
        )
        for site in range(lend + 1, rend - 1):
            lenv = cached_einsum(
                'ab,acd,bce->de',
                lenv,
                self.state[site],
                self.state[site].conj()
            )
        return cached_einsum('ij,ij->', lenv, renv).item()

    for (il, ir), (jl, jr) in pairs:
        gram[il, ir, jl, jr] = _compute_inner_product(il, ir, jl, jr)

    # ------------------------------------------------------------------
    # step 2: sample anti-Hermitian operators A_i supported on L0.
    # ------------------------------------------------------------------
    l0 = _interior_sites(self.Hamiltonian, lend, rend)
    if not l0:
        raise ValueError(
            f"The CGO patch [{lend}, {rend}) has no interior sites. "
            "Increase l or use a larger observable support."
        )
    A_samples = _sample_product_operators(
        l0, num_sample, self.physical_dim, self.device, seed
    )

    # ------------------------------------------------------------------
    # step 3: build H_L, the part of H supported inside the patch.
    # ------------------------------------------------------------------
    H_L = _patch_hamiltonian(self.Hamiltonian, lend, rend)

    eye = torch.eye(self.physical_dim, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # step 4: projected commutators P_V [H_L, A_i] P_V.
    # ------------------------------------------------------------------
    def _one_sided_HL_R(bra_l: int, bra_r: int, ket_l: int, ket_r: int,
                        Ai: dict[int, torch.Tensor]) -> complex:
        r'''
        Compute <bra| H_L (R_1 \otimes ... \otimes R_n) |ket>, where the
        Hermitian R_j are given by Ai (keyed by global site index).
        '''
        lenv = cached_einsum(
            'sb,wst,tq,qc->bcw',
            self.state[lend].conj()[bra_l],
            H_L.tensors[0][0, ...],
            Ai.get(lend, eye),
            self.state[lend][ket_l]
        )
        for site in range(lend + 1, rend):
            lenv = cached_einsum(
                'aju,asb,uwst,tq,jqc->bcw',
                lenv,
                self.state[site].conj(),
                H_L.tensors[site - lend],
                Ai.get(site, eye),
                self.state[site]
            )
        # The last site's MPO index must be the end state D-1.
        return lenv[bra_r, ket_r, -1].item()

    def _one_sided_R_HL(bra_l: int, bra_r: int, ket_l: int, ket_r: int,
                        Ai: dict[int, torch.Tensor]) -> complex:
        r'''
        Compute <bra| (R_1 \otimes ... \otimes R_n) H_L |ket>, where the
        Hermitian R_j are given by Ai (keyed by global site index).
        '''
        lenv = cached_einsum(
            'sb,sp,wpt,tc->bcw',
            self.state[lend].conj()[bra_l],
            Ai.get(lend, eye),
            H_L.tensors[0][0, ...],
            self.state[lend][ket_l]
        )
        for site in range(lend + 1, rend):
            lenv = cached_einsum(
                'aju,asb,sp,uwpt,jtc->bcw',
                lenv,
                self.state[site].conj(),
                Ai.get(site, eye),
                H_L.tensors[site - lend],
                self.state[site]
            )
        # The last site's MPO index must be the end state D-1.
        return lenv[bra_r, ket_r, -1].item()

    def proj_commutator(il: int, ir: int, jl: int, jr: int,
                        Ai: dict[int, torch.Tensor]) -> complex:
        r'''
        Compute <jl,jr|[H_L, A_i]|il,ir>
            = i * <jl,jr|[H_L, (R_1 \otimes ... \otimes R_n)]|il,ir>,
        where A_i = i * (R_1 \otimes ... \otimes R_n) and Ai holds the R_j.

        The bra/ket convention matches _compute_inner_product, i.e. the
        (jl,jr) boundary state is conjugated and the (il,ir) state is not.

        Note: for the CGO identity [H, A_i] = [H_L, A_i] to hold, Ai must be
        supported on the interior L0 of the patch (sites not coupled to the
        outside of [lend, rend)).
        '''
        T1 = _one_sided_HL_R(jl, jr, il, ir, Ai)  # <jl,jr|H_L R|il,ir>
        T2 = _one_sided_R_HL(jl, jr, il, ir, Ai)  # <jl,jr|R H_L|il,ir>
        return 1j * (T1 - T2)

    M_samples = []
    for Ai in A_samples:
        M = torch.zeros(
            (D_l, D_r, D_l, D_r), dtype=torch.complex128, device=self.device
        )
        for (il, ir), (jl, jr) in pairs:
            M[il, ir, jl, jr] = proj_commutator(il, ir, jl, jr, Ai)
        M_samples.append(M.reshape(opti_dim, opti_dim))

    # ------------------------------------------------------------------
    # step 5: orthonormalize the boundary-state basis.
    #
    # gram[alpha, beta] = <beta|alpha> = H^T, where H[alpha, beta] =
    # <alpha|beta> is the usual Gram matrix.  Since H is Hermitian,
    # conj(gram) = H.  Do not take an extra transpose here.
    # ------------------------------------------------------------------
    X = _orthonormal_transform(gram)
    M_samples = [X @ M @ X.conj().T for M in M_samples]

    # ------------------------------------------------------------------
    # step 6: project the observable B onto V_L and transform it to the
    # orthonormal basis.  B[alpha,beta] = <beta|B|alpha>, so it transforms
    # with the same X as the commutators.
    # ------------------------------------------------------------------
    def _project_operator(bra_l: int, bra_r: int, ket_l: int, ket_r: int,
                          ops: dict[int, torch.Tensor]) -> complex:
        r'''
        Compute <bra| B |ket>, where B is the product operator given by ops
        (identity on sites not in ops).  The bra/ket convention matches
        _compute_inner_product.
        '''
        lenv = cached_einsum(
            'ib,ij,jc->bc',
            self.state[lend].conj()[bra_l],
            ops.get(lend, eye),
            self.state[lend][ket_l]
        )
        for site in range(lend + 1, rend - 1):
            lenv = cached_einsum(
                'ab,asc,st,btd->cd',
                lenv,
                self.state[site].conj(),
                ops.get(site, eye),
                self.state[site]
            )
        return cached_einsum(
            'ab,as,st,bt->',
            lenv,
            self.state[rend - 1].conj()[..., bra_r],
            ops.get(rend - 1, eye),
            self.state[rend - 1][..., ket_r]
        ).item()

    B = torch.zeros(
        (D_l, D_r, D_l, D_r), dtype=torch.complex128, device=self.device
    )
    for (il, ir), (jl, jr) in pairs:
        B[il, ir, jl, jr] = _project_operator(jl, jr, il, ir, operators)
    B = X @ B.reshape(opti_dim, opti_dim) @ X.conj().T

    return M_samples, B, X


@extends_Broomstick
def CGO(self: Broomstick,
        l: int,
        operators: dict[int, torch.Tensor],
        num_sample: int = 200,
        seed: int | None = None) -> tuple[float, float]:
    '''
    Compute CGO upper and lower bounds for <B> in a 1d DMRG state.

    Args:
        l: radius of the patch L = [lend, rend) around the observable.
        operators: dict mapping global site indices to single-site operators;
                   this is the observable B.
        num_sample: number of random anti-Hermitian operators A_i.
        seed: optional RNG seed for the A_i sampling.

    Returns:
        (lower, upper): bounds on <B>.
    '''
    M_samples, B, _ = CGO_projections(
        self, l=l, operators=operators, num_sample=num_sample, seed=seed
    )
    return _solve_cgo_bounds(M_samples, B)

