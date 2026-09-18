import math
import warnings

import cvxpy as cp
import torch

from base import MPS, MPO, Broomstick, cached_einsum
from ext_register import extends_Broomstick


# ---------------------------------------------------------------------------
# CGO helpers.
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


def _select_boundary_indices(weights: torch.Tensor,
                             chi: int | None,
                             discard_tol: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Choose which boundary indices to keep at the two cuts of the patch.

    `weights[il, ir]` is the weight <psi|P_{il,ir}|psi> of the boundary pair;
    with the orthogonality center inside the patch these weights sum to one and
    measure how much of the state each boundary configuration carries.  Indices
    are ranked by their marginal weight, largest first:

        chi          keep at most this many indices per cut;
        discard_tol  keep the smallest number whose discarded weight
                     1 - sum(weights over the kept pairs) is at most this.

    Keeping all indices (both arguments None) reproduces the untruncated
    projection.  Truncating shrinks the SDP - and, unlike a fixed `chi`, the
    discarded weight is the quantity that says whether the bounds are still
    certified: the CGO constraints are only valid if the state lies in the
    kept subspace.
    '''
    D_l, D_r = weights.shape
    if chi is None and discard_tol is None:
        return (torch.arange(D_l, device=weights.device),
                torch.arange(D_r, device=weights.device))

    order_l = torch.argsort(weights.sum(dim=1), descending=True)
    order_r = torch.argsort(weights.sum(dim=0), descending=True)
    k_max = min(D_l, D_r, chi) if chi is not None else min(D_l, D_r)
    k = k_max
    if discard_tol is not None:
        total = weights.sum().real
        for trial in range(1, k_max + 1):
            kept = weights[order_l[:trial]][:, order_r[:trial]].sum().real
            if kept >= total - discard_tol:
                k = trial
                break
    return (torch.sort(order_l[:k]).values, torch.sort(order_r[:k]).values)


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


def _build_cgo_dual(M_samples: list[torch.Tensor],
                    B: torch.Tensor,
                    constraint_tol: float = 0.0) -> cp.Problem:
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

    `constraint_tol` >= 0 replaces the equalities by |Tr(rho M_i)| <=
    constraint_tol.  The CGO constraints are exact only for an exact eigenstate
    that lies exactly in the patch subspace; a converged DMRG state still leaves
    the projected state violating them at a small but finite scale (the
    discarded weight, or sqrt(var(H)) times the size of the commutators).  The
    equalities are then numerically infeasible and the solvers fail or return a
    bogus interval.  Relaxing by that scale keeps the bounds valid - the
    projected state stays feasible and a larger feasible set can only widen the
    interval - and is what :func:`CGO` does when `constraint_tol=None`.
    '''
    M = [m.detach().cpu().numpy() for m in M_samples]
    M = [0.5 * (m + m.conj().T) for m in M]
    B_np = B.detach().cpu().numpy()
    B_np = 0.5 * (B_np + B_np.conj().T)

    q = B_np.shape[0]
    rho = cp.Variable((q, q), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]
    if constraint_tol > 0:
        constraints += [cp.abs(cp.real(cp.trace(rho @ M[i]))) <= constraint_tol
                        for i in range(len(M))]
    else:
        constraints += [cp.real(cp.trace(rho @ M[i])) == 0
                        for i in range(len(M))]
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
                      B: torch.Tensor,
                      constraint_tol: float = 0.0) -> tuple[float, float]:
    '''
    Solve the CGO dual SDPs and return (lower, upper).

    CLARABEL is tried first: it is an interior-point solver that converges in
    ~10 iterations regardless of the number of constraints, which dominates the
    cost at the sample counts used here (m = 30).  SCS is kept as a fallback;
    it is first-order and wins for very small m (0.8 s vs 27 s at m = 4), but
    its iteration count grows quickly with m and with the accuracy target.

    `constraint_tol` is forwarded to :func:`_build_cgo_dual`; see there for why
    the exact equalities are usually infeasible in practice.
    '''
    solver_configs = (
        ('CLARABEL', {}),
        ('SCS', {'eps': 1e-9, 'max_iters': 200000}),
    )

    upper = _solve_bound(_build_cgo_dual(M_samples, B, constraint_tol),
                         solver_configs)
    lower = -_solve_bound(_build_cgo_dual(M_samples, -B, constraint_tol),
                          solver_configs)
    return float(lower), float(upper)


def CGO_projections(self: Broomstick,
                    l: int,
                    operators: dict[int, torch.Tensor],
                    num_sample: int = 200,
                    seed: int | None = None,
                    chi: int | None = None,
                    discard_tol: float | None = None,
                    weight_tol: float = 1e-6,
                    return_info: bool = False
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
        chi: keep at most this many boundary indices per cut; None keeps all.
        discard_tol: keep the smallest number of boundary indices per cut whose
                     discarded state weight is at most this value.
        weight_tol: warn when the retained subspace holds less than
                     (1 - weight_tol) of the state weight.
        return_info: also return a dict with the projection diagnostics.

    Returns:
        M_samples: list of r x r projected commutators P_V [H_L,A_i] P_V in the
                   orthonormal basis of V_L, with r the rank of the patch Gram.
        B: r x r projected observable P_V B P_V in the same basis.
        X: basis transformation, |tilde alpha> = sum_beta X[alpha,beta] |beta>,
           of shape (r, q): the boundary basis is rank deficient, so only r of
           its q directions are kept.  If `return_info` is set, a dict with
           `q`, `rank`, `chi`, `weight_kept`, `gram_cond` and `x_cond` is
           returned as a fourth value.
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

    # ── keep only the most important boundary indices, when asked to ──
    #
    # weights[il, ir] = <psi|P_{il,ir}|psi> is how much of the state the
    # boundary pair (il, ir) carries; the sum over all pairs is <psi|psi>.  The
    # kept weight is the share of the state that still lies in the retained
    # subspace, i.e. the share for which the CGO constraints remain valid.
    ones = torch.ones(opti_dim, dtype=gram.dtype, device=gram.device)
    overlaps = (gram.reshape(opti_dim, opti_dim).conj() @ ones)
    weights = torch.diagonal(
        gram.reshape(opti_dim, opti_dim)).real.reshape(D_l, D_r)
    keep_l, keep_r = _select_boundary_indices(weights, chi, discard_tol)
    if len(keep_l) != D_l or len(keep_r) != D_r:
        sel = (keep_l[:, None] * D_r + keep_r[None, :]).reshape(-1)
        overlaps = overlaps[sel]

        def restrict(T: torch.Tensor) -> torch.Tensor:
            return T[keep_l][:, keep_r][:, :, keep_l][:, :, :, keep_r]

        gram = restrict(gram)
        B_full = restrict(B_full)
        M_full = [restrict(M) for M in M_full]

    # ── orthonormalize the boundary-state basis and transform the operators ──
    #
    # gram[alpha, beta] = <beta|alpha> = H^T, where H[alpha, beta] =
    # <alpha|beta> is the usual Gram matrix; since H is Hermitian, conj(gram) =
    # H.  The basis is rank deficient, so X has shape (r, q).
    X = _orthonormal_transform(gram)
    q = X.shape[1]
    M_samples = [X @ M.reshape(q, q) @ X.conj().T for M in M_full]
    B = X @ B_full.reshape(q, q) @ X.conj().T

    # ||P_V|psi>||^2: the share of the state inside the retained subspace.
    psi = X @ overlaps
    kept_weight = float(torch.linalg.norm(psi) ** 2)
    if kept_weight < 1 - weight_tol:
        warnings.warn(
            f'CGO: the retained patch subspace holds only {kept_weight:.12f} '
            f'of the state weight (discarded {1 - kept_weight:.3e} > '
            f'weight_tol = {weight_tol:.1e}); the CGO constraints are then not '
            f'satisfied by the state and the bounds are not certified. '
            f'Raise chi / discard_tol, or converge the state.'
        )
    if not return_info:
        return M_samples, B, X

    # How badly the projected state itself violates the CGO constraints.  It is
    # zero only for an exact eigenstate inside an untruncated patch subspace, so
    # this is the scale by which `CGO` relaxes the equalities when asked to
    # choose the relaxation automatically.
    violation = max(abs(complex(psi.conj() @ (M @ psi)) / kept_weight)
                    for M in M_samples) if M_samples else 0.0

    evals = torch.linalg.eigvalsh(0.5 * (gram.reshape(q, q)
                                         + gram.reshape(q, q).conj().T))
    evals = evals.double()
    positive = evals[evals > 1e-14 * evals.max()]
    svals = torch.linalg.svdvals(X)
    info = {
        'q': q,
        'rank': int(X.shape[0]),
        'chi': (int(len(keep_l)), int(len(keep_r))),
        'weight_kept': kept_weight,
        'constraint_violation': float(violation),
        'gram_cond': float(evals.max() / positive.min()),
        'x_cond': float(svals.max() / svals.min()),
    }
    return M_samples, B, X, info


@extends_Broomstick
def CGO(self: Broomstick,
        l: int,
        operators: dict[int, torch.Tensor],
        num_sample: int = 200,
        seed: int | None = None,
        chi: int | None = None,
        discard_tol: float | None = None,
        weight_tol: float = 1e-6,
        constraint_tol: float | None = 0.0,
        validate: bool = False) -> tuple[float, float]:
    '''
    Compute CGO upper and lower bounds for <B> in a 1d DMRG state.

    The bounds are certified only as far as the CGO constraints hold for the
    state: they require the state to lie in the patch subspace V_L, and they
    require the state to be an eigenstate of H.  `weight_tol` guards the first
    (the part of the state weight thrown away by `chi`/`discard_tol`, or by the
    rank truncation of a rank-deficient Gram), `validate` the second (through
    sqrt(var(H)), the norm of (H - E)|psi>).

    Args:
        l: radius of the patch L = [lend, rend) around the observable.
        operators: dict mapping global site indices to single-site operators;
                   this is the observable B.
        num_sample: number of random anti-Hermitian operators A_i.
        seed: optional RNG seed for the A_i sampling.
        chi: keep at most this many boundary indices per cut; None keeps all.
        discard_tol: keep the smallest number of boundary indices per cut whose
                     discarded state weight is at most this value.
        weight_tol: the kept state weight must reach 1 - weight_tol, otherwise
                    a warning says the bounds are not certified.
        constraint_tol: how far the projected state may miss the CGO
                    constraints Tr(rho [H_L, A_i]) = 0.  Pass a positive number
                    to enforce |Tr(rho M_i)| <= constraint_tol, or None to pick
                    the relaxation automatically from the violation the
                    projected state itself shows (see `CGO_projections`), which
                    keeps that state inside the feasible set.  0.0, the default,
                    restores the exact equalities; they are only satisfiable by
                    an exact eigenstate in an untruncated patch subspace and
                    otherwise make the SDP infeasible, so prefer None whenever
                    the exact form fails to solve.
        validate: also check sqrt(var(H)) against `weight_tol`.

    Returns:
        (lower, upper): bounds on <B>.
    '''
    if constraint_tol is None:
        M_samples, B, _, info = CGO_projections(
            self, l=l, operators=operators, num_sample=num_sample, seed=seed,
            chi=chi, discard_tol=discard_tol, weight_tol=weight_tol,
            return_info=True,
        )
        # A hair above the state's own violation, so that the projected state is
        # strictly inside the relaxed feasible set.
        constraint_tol = 1.01 * info['constraint_violation'] + 1e-12
    else:
        M_samples, B, _ = CGO_projections(
            self, l=l, operators=operators, num_sample=num_sample, seed=seed,
            chi=chi, discard_tol=discard_tol, weight_tol=weight_tol,
        )
    if validate:
        deviation = math.sqrt(max(self.compute_variance(), 0.0))
        if deviation > weight_tol:
            warnings.warn(
                f'CGO: sqrt(var(H)) = {deviation:.3e} > weight_tol = '
                f'{weight_tol:.1e}; the state is not an eigenstate of H, so '
                f'the constraints are violated at that scale and the bounds '
                f'are not certified.'
            )
    rank = B.shape[0]
    estimated = 16.0 * rank ** 4
    if estimated > 2e9:
        warnings.warn(
            f'CGO: the projected SDP has rank {rank}; the solver needs roughly '
            f'{estimated / 1e9:.1f} GB for its normal-equation matrix.  Pass '
            f'chi=<n> or discard_tol=<tol> to shrink the patch subspace '
            f'(the paper keeps 6 boundary indices per cut).'
        )
    return _solve_cgo_bounds(M_samples, B, constraint_tol)
