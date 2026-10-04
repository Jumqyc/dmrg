r'''
The semidefinite program shared by the two CGO implementations.

`extremal(observable, rows, solver_configs, constraint_tol)` returns the two ends of
min / max Tr(sigma observable) over the states sigma >= 0, Tr sigma = 1 with
Tr(sigma M_i) = 0, and None when no solver configuration solves both senses.
'''
import warnings

import cvxpy as cp
import numpy as np
from torch import Tensor


def _array(matrix: Tensor | np.ndarray) -> np.ndarray:
    '''
    A Hermitian matrix as a plain numpy array, for cvxpy.
    Args:
        matrix: a torch tensor or a numpy array, of shape (q, q).
    Returns:
        The array, symmetrised so that cvxpy sees an exactly Hermitian matrix.
    '''
    value = matrix.detach().cpu().numpy() if isinstance(matrix, Tensor) else np.asarray(matrix)
    return 0.5 * (value + value.conj().T)


def extremal(observable: Tensor | np.ndarray,
             rows: list[Tensor | np.ndarray],
             solver_configs: tuple[tuple[str, dict], ...],
             constraint_tol: float = 0.0) -> tuple[float, float, list[str], float] | None:
    r'''
    The two ends of the CGO program, trying each solver configuration in order: the
    smallest and the largest $\mathrm{Tr}(\sigma B)$ over the states $\sigma \succeq 0$,
    $\mathrm{Tr}\,\sigma = 1$ that also satisfy $\mathrm{Tr}(\sigma M_i) = 0$.  The two
    senses share one feasible set, so the program is built once and only the sign of the
    objective changes.
    Args:
        observable: the Hermitian matrix B of the objective, shape (q, q).
        rows: the constraint matrices M_i, Hermitian of shape (q, q); an empty list
            leaves the eigenvalue range of B.
        solver_configs: (solver name, solve options) pairs, tried in order until one
            solves both senses; a later one is the fallback for a solver that rejects an
            option name or fails to converge.
        constraint_tol: bound on $|\mathrm{Tr}(\sigma M_i)|$.  0.0 keeps the exact
            equalities; a positive value relaxes them, which a state that only nearly
            satisfies them needs for the program to stay feasible.
    Returns:
        (lower, upper, statuses, residual): the bounds, the two solver statuses, and the
        largest residual the solutions leave on the constraints and on the trace.  None
        when no configuration solves both senses.
    '''
    objective = _array(observable)
    matrices = [_array(row) for row in rows]
    sigma = cp.Variable(objective.shape, hermitian=True)
    constraints = [sigma >> 0, cp.real(cp.trace(sigma)) == 1]
    if constraint_tol > 0:
        constraints += [cp.abs(cp.real(cp.trace(m @ sigma))) <= constraint_tol
                        for m in matrices]
    else:
        constraints += [cp.real(cp.trace(m @ sigma)) == 0 for m in matrices]

    for solver, options in solver_configs:
        bounds, statuses = [], []
        for sign in (1.0, -1.0):
            problem = cp.Problem(cp.Minimize(sign * cp.real(cp.trace(objective @ sigma))),
                                 constraints)
            try:
                # a solver may not know an option name, and the fallback is the next
                # configuration, so the failure is reported by returning None
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    problem.solve(solver=solver, **(options or {}))
            except Exception:
                break
            if problem.status not in ('optimal', 'optimal_inaccurate') or problem.value is None:
                break
            bounds.append(float(sign * problem.value))
            statuses.append(problem.status)
        if len(bounds) != 2:
            continue
        solution = sigma.value
        residual = 1e30 if solution is None else max(
            [abs(float((m @ solution).trace().real)) for m in matrices]
            + [abs(float(solution.trace().real) - 1.0)])
        return bounds[0], bounds[1], statuses, residual
    return None
