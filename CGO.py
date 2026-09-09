import math
import warnings

import cvxpy as cp
import numpy as np
import torch

from base import MPS, MPO, Broomstick, cached_einsum
from ext_register import extends_Broomstick


# ---------------------------------------------------------------------------
# CGO helpers.
#
# One sample is a product operator
#
#     A_i = 1j * (R_1 (x) R_2 (x) ... (x) R_n),
#
# where each R_j is an independent random Hermitian matrix acting on a site of
# L0.  The product is Hermitian, so multiplying by 1j makes A_i anti-Hermitian.
# Based on PhysRevB.94.195143.
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
                              seed: int | None) -> list[dict[int, torch.Tensor]]:
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


def _boundary_envs(state: MPS,
                   lend: int,
                   rend: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Gram matrices of the MPS blocks surrounding the patch.

    Returns (A, B) with A[kl, bl] = <L_bl|L_kl> for the block left of `lend`
    and B[kr, br] = <R_br|R_kr> for the block right of `rend`.  They are the
    identity exactly when the surrounding blocks are canonical, but keeping
    them makes the projection correct in any gauge.
    '''
    D_l, D_r = state.bond_dim[lend], state.bond_dim[rend]

    E = torch.ones(1, 1, 1, 1, dtype=state.dtype, device=state.device)
    for site in range(lend):
        E = cached_einsum('acbd,bsx,dsy->acxy',
                          E, state[site], state[site].conj())
    left = E.reshape(D_l, D_l)

    E = torch.ones(1, 1, 1, 1, dtype=state.dtype, device=state.device)
    for site in range(state.L - 1, rend - 1, -1):
        E = cached_einsum('asx,csy,xybd->acbd',
                          state[site], state[site].conj(), E)
    right = E.reshape(D_r, D_r)
    return left, right


def _orthonormal_transform(gram: torch.Tensor,
                           rtol: float = 1e-10) -> torch.Tensor:
    '''
    Return X such that X @ H @ X^dagger = I for the Gram matrix H.

    `gram` is the 4-index array gram[kl, kr, bl, br] = <bl,br|kl,kr>, so the
    Gram matrix is H = conj(gram).reshape(q, q).  The boundary-state basis is
    generally rank deficient (a patch of n sites offers only 2**n physical
    states for D_l * D_r boundary pairs), so the transform is built from the
    numerically non-zero eigenvectors of H rather than from a Cholesky factor,
    which would fail as soon as H has a zero eigenvalue.  The rows of the
    returned (rank, q) matrix are the orthonormal basis vectors expressed in
    the boundary basis.  For a positive-definite H this is the same transform
    as inv(cholesky(H)) up to a unitary rotation of the basis, which leaves the
    bounds unchanged.
    '''
    q = gram.shape[0] * gram.shape[1]
    H = gram.reshape(q, q).conj()
    H = 0.5 * (H + H.conj().T)
    evals, evecs = torch.linalg.eigh(H)
    keep = evals > rtol * evals.max().real.clamp_min(1e-300)
    if not bool(keep.any()):
        raise ValueError(
            'The patch Gram matrix has no non-zero eigenvalue; the CGO '
            'projection is empty.  Check the MPS and the patch [lend, rend).'
        )
    evals, evecs = evals[keep], evecs[:, keep]
    return (evecs / evals.sqrt().unsqueeze(0)).conj().T


def _product_matrix(state: MPS,
                    lend: int,
                    rend: int,
                    ops: dict[int, torch.Tensor],
                    eye: torch.Tensor) -> torch.Tensor:
    '''
    Matrix elements <bra|O|ket> of a product operator on the patch.

    Returns X[kl, kr, bl, br] = <bl,br| O |kl,kr>, where (kl, kr) are the ket
    boundary bond indices at `lend`/`rend` and (bl, br) the conjugated bra
    indices.  O acts as ops[site] on the sites present in `ops` and as the
    identity elsewhere.  One contraction per site produces all
    (D_l*D_r)**2 matrix elements at once, instead of one per boundary pair.
    '''
    T = state[lend]
    lenv = cached_einsum('kpc,qp,lqd->klcd', T, ops.get(lend, eye), T.conj())
    for site in range(lend + 1, rend):
        T = state[site]
        lenv = cached_einsum('klcd,cpe,qp,dqf->klef',
                             lenv, T, ops.get(site, eye), T.conj())
    return lenv.permute(0, 2, 1, 3)


def _mpo_commutator_matrix(state: MPS,
                           H_L: MPO,
                           lend: int,
                           rend: int,
                           R_ops: dict[int, torch.Tensor],
                           eye: torch.Tensor) -> torch.Tensor:
    '''
    Matrix elements of 1j * [H_L, R] on the patch.

    R = R_lend (x) ... (x) R_{rend-1} is the product operator held in `R_ops`
    (the identity where a site is missing), so the returned
    M[kl, kr, bl, br] = 1j <bl,br|[H_L, R]|kl,kr> is the projected commutator
    used by CGO.  The two orderings H_L R and R H_L are each contracted with
    one einsum per site, with the same boundary-index bookkeeping as
    _product_matrix.
    '''
    def local_operator(side: str, site: int) -> torch.Tensor:
        '''Local factor G[in, out, bra_phys, ket_phys] of H_L R or R H_L.'''
        W = H_L.tensors[site - lend]
        R = R_ops.get(site, eye)
        if side == 'right':                     # R acts on the ket leg
            return cached_einsum('iost,tu->iosu', W, R)
        # R acts on the bra leg: its first index stays open and pairs with the
        # bra state.
        return cached_einsum('sp,iopt->iost', R, W)

    def one_sided(side: str) -> torch.Tensor:
        T = state[lend]
        lenv = cached_einsum('kpc,oqp,lqd->klcdo', T,
                             local_operator(side, lend)[0], T.conj())
        for site in range(lend + 1, rend):
            T = state[site]
            # the last index of lenv is the incoming MPO index, i.e. the first
            # leg of the local operator
            lenv = cached_einsum('klcds,cpe,soqp,dqf->klefo',
                                 lenv, T, local_operator(side, site), T.conj())
        # the MPO right boundary is the end state D-1
        return lenv[..., -1].permute(0, 2, 1, 3)

    return 1j * (one_sided('right') - one_sided('left'))


def _build_cgo_dual(M_samples: list[torch.Tensor], B: torch.Tensor) -> cp.Problem:
    '''
    Build the dual CGO SDP for the upper bound on <B>.

    M_samples[i] is the Hermitian projected commutator corresponding to
    -i P_V [H_L, A_i] P_V in CGO's bra/ket convention, and B = P_V B P_V.  Both
    must be expressed in an orthonormal basis of V_L.

    The dual problem is

        maximize   Tr(rho B)
        subject to rho >= 0, Tr(rho) = 1,
                   Tr(rho M_i) = 0 for all i

    which is the tighter upper bound on <B>.  For the lower bound on <B>,
    build the same problem with -B and negate the result.
    '''
    M = [m.detach().cpu().numpy() for m in M_samples]
    M = [0.5 * (m + m.conj().T) for m in M]
    B_np = B.detach().cpu().numpy()
    B_np = 0.5 * (B_np + B_np.conj().T)

    q = B_np.shape[0]
    rho = cp.Variable((q, q), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]
    constraints += [cp.real(cp.trace(rho @ M[i])) == 0 for i in range(len(M))]
    return cp.Problem(cp.Maximize(cp.real(cp.trace(rho @ B_np))), constraints)


def _solve_problem(problem: cp.Problem,
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


def _solve_bound(dual: cp.Problem,
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
    '''
    Solve the CGO dual SDPs and return (lower, upper).

    CLARABEL is tried first: it is an interior-point solver that converges in
    ~10 iterations regardless of the number of constraints, which dominates the
    cost at the sample counts used here (m = 30).  SCS is kept as a fallback;
    it is first-order and wins for very small m (0.8 s vs 27 s at m = 4), but
    its iteration count grows quickly with m and with the accuracy target.
    '''
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
                    seed: int | None = None
                    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    '''
    Build the projected CGO quantities for the observable `operators`.

    This is the tensor-network part of CGO: it computes the projected
    commutators and the observable in an orthonormal basis of the patch
    subspace V_L.  Most users should call :func:`CGO` instead, which solves the
    SDPs and returns the upper/lower bounds.

    Args:
        l: radius of the patch L = [lend, rend) around the observable.
        operators: dict mapping global site indices to single-site operators;
                   this is the observable B.
        num_sample: number of random anti-Hermitian operators A_i.
        seed: optional RNG seed for the A_i sampling.

    Returns:
        M_samples: list of r x r projected commutators P_V [H_L,A_i] P_V in the
                   orthonormal basis of V_L, with r the rank of the patch Gram.
        B: r x r projected observable P_V B P_V in the same basis.
        X: basis transformation, |tilde alpha> = sum_beta X[alpha,beta] |beta>,
           of shape (r, D_l * D_r): the boundary basis is rank deficient, so
           only r of its directions are kept.
    '''
    if not all(op.shape == (self.physical_dim, self.physical_dim)
               for op in operators.values()):
        raise ValueError(
            f"All operators must be of shape "
            f"({self.physical_dim},{self.physical_dim}), "
            f"but got {[op.shape for op in operators.values()]}"
        )

    # the optimization is performed on the patch L = [lend, rend)
    lend = max(0, min(operators.keys()) - l)
    rend = min(self.L, max(operators.keys()) + l + 1)
    D_l = self.state.bond_dim[lend]
    D_r = self.state.bond_dim[rend]
    opti_dim = D_l * D_r
    eye = torch.eye(self.physical_dim, dtype=self.dtype, device=self.device)

    # ── sample anti-Hermitian operators A_i supported on the interior L0 ──
    l0 = _interior_sites(self.Hamiltonian, lend, rend)
    if not l0:
        raise ValueError(
            f"The CGO patch [{lend}, {rend}) has no interior sites. "
            "Increase l or use a larger observable support."
        )
    A_samples = _sample_product_operators(
        l0, num_sample, self.physical_dim, self.device, seed
    )
    H_L = _patch_hamiltonian(self.Hamiltonian, lend, rend)

    # ── matrix elements of the patch operators in the boundary basis ──
    #
    # Each quantity below is a (D_l, D_r, D_l, D_r) array
    #     X[kl, kr, bl, br] = <bl,br| O |kl,kr>,
    # where O is a product operator O_lend (x) ... (x) O_{rend-1}: the Gram
    # "operator" 1, the observable B, or 1j [H_L, R].  The boundary indices are
    # carried through the patch as batch dimensions, so each quantity costs one
    # contraction per site instead of one per boundary pair.
    gram_raw = _product_matrix(self.state, lend, rend, {}, eye)
    B_raw = _product_matrix(self.state, lend, rend, operators, eye)
    M_raw = [_mpo_commutator_matrix(self.state, H_L, lend, rend, Ai, eye)
             for Ai in A_samples]

    # ── weight by the Gram matrices of the surrounding blocks ──
    #
    # X[kl, kr, bl, br] above is a matrix element of the patch states; the
    # matrix element of the full state is
    #     A_env[kl, bl] * X[kl, kr, bl, br] * B_env[kr, br].
    # The weights are the identity when the surrounding blocks are canonical
    # (the usual case when the center sits inside the patch), but keeping them
    # makes the projection correct in any gauge.
    A_env, B_env = _boundary_envs(self.state, lend, rend)
    weight = A_env[:, None, :, None] * B_env[None, :, None, :]
    gram = weight * gram_raw
    B_full = weight * B_raw
    M_full = [weight * M for M in M_raw]

    # ── orthonormalize the boundary-state basis and transform the operators ──
    #
    # gram[alpha, beta] = <beta|alpha> = H^T, where H[alpha, beta] =
    # <alpha|beta> is the usual Gram matrix; since H is Hermitian, conj(gram) =
    # H.  The basis is rank deficient, so X has shape (r, D_l * D_r).
    X = _orthonormal_transform(gram)
    M_samples = [X @ M.reshape(opti_dim, opti_dim) @ X.conj().T
                 for M in M_full]
    B = X @ B_full.reshape(opti_dim, opti_dim) @ X.conj().T
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
