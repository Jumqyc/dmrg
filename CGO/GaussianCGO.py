r'''
Commutator gauge optimization on the patch of a Gaussian state.

`GaussianCGO(covariance, internal)` takes the covariance of the whole state and the
Majoranas that are traced out; the patch is the complement, `V_L = Im rho_L` is the
support of its reduced state, and `basic_CGO` / `CGO` bound an observable on the patch.
'''
import itertools

import torch
from torch import Tensor

from base import CUDA,COMPLEX,REAL
from FreeFermion.linalg import hamiltonian, symplectic_form, wick, williamson

from sdp import extremal


class GaussianCGO:
    r'''
    CGO on the patch of a Gaussian state: the support of the patch block, the patch Hamiltonian, and the Majorana matrices the operators are expressed with.
    '''

    @staticmethod
    def majoranas(count: int,
                   device: torch.device = CUDA) -> list[Tensor]:
        r'''
        Dense representation of Majorana operators. 
        Args:
            count: the number of Majoranas, even.
            device: device of the matrices.
        Returns:
            The matrices, each of shape (2^m, 2^m), Hermitian, with square one and with
            $\{\gamma_a,\gamma_b\} = 2\delta_{ab}$.
        Raises:
            ValueError: if `count` is odd.
        '''
        if count % 2:
            raise ValueError(f'the number of Majoranas must be even, got {count}')
        x = torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX, device=device)
        y = torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX, device=device)
        z = torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX, device=device)
        modes = count // 2
        gammas = []
        for k in range(modes):
            prefix = torch.eye(1, dtype=COMPLEX, device=device)
            for _ in range(k):
                prefix = torch.kron(prefix, z)
            suffix = torch.eye(2 ** (modes - 1 - k), dtype=COMPLEX, device=device)
            gammas.append(torch.kron(torch.kron(prefix, x), suffix))
            gammas.append(torch.kron(torch.kron(prefix, y), suffix))
        return gammas


    @staticmethod
    def quadratic(gammas: list[Tensor],
                   matrix: Tensor) -> Tensor:
        r'''
        The operator $H = (i/4)\gamma^T M \gamma = (i/2)\sum_{a<b}M_{ab}\gamma_a\gamma_b$ on
        the Fock space.
        Args:
            gammas: the Majorana matrices of the modes the operator acts on.
            matrix: real antisymmetric Majorana matrix, shape (2m, 2m), of those modes.
        Returns:
            The Hermitian matrix of shape (2**m, 2**m).
        '''
        size = gammas[0].shape[0]
        operator = torch.zeros(size, size, dtype=COMPLEX, device=gammas[0].device)
        for a in range(len(gammas)):
            for b in range(a + 1, len(gammas)):
                operator = operator + 0.5j * matrix[a, b] * (gammas[a] @ gammas[b])
        return operator


    @classmethod
    def density(cls, gammas: list[Tensor],
                 covariance: Tensor) -> Tensor:
        r'''
        Fock-space density matrix of the Gaussian state of a covariance: the modular
        Hamiltonian is diagonal in the Williamson basis with the mode energies
        $2\,\mathrm{artanh}(\lambda_k)$, capped so that a pure mode sits in its ground state,
        and the state is $\rho \propto e^{-K}$.
        Args:
            gammas: the Majorana matrices of the modes of the covariance.
            covariance: real antisymmetric covariance of those modes, shape (2m, 2m).
        Returns:
            The density matrix of shape (2**m, 2**m), Hermitian, positive definite and of
            trace one, reproducing the covariance through
            $\Gamma_{ab} = \mathrm{Tr}(\rho\,(i/2)[\gamma_a,\gamma_b])$.
        '''
        modes = len(gammas) // 2
        pair = symplectic_form(2, dtype=covariance.dtype, device=covariance.device)
        rotation, lambdas = williamson(covariance)
        energies = -2.0 * torch.atanh(lambdas.clamp(max=1 - 1e-12)).clamp(max=20.0)
        modular = rotation.T @ torch.block_diag(
            *[energies[k] * pair for k in range(modes)]) @ rotation
        rho = torch.linalg.matrix_exp(-cls.quadratic(gammas, modular))
        return rho / rho.trace()


    def __init__(self,
                 covariance: Tensor,
                 internal: list[int],
                 tolerance: float = 1e-10,
                 device: torch.device = CUDA) -> None:
        r'''
        Build the patch data of one Gaussian state: the support of the reduced state on
        the patch, and the Majorana matrices every operator is expressed with.
        Args:
            covariance: real antisymmetric covariance of the Gaussian state, shape
                (2n, 2n), on CUDA.
            internal: the Majorana labels that are traced out, a non-empty even list; the
                patch is the complement, and every operator is expected on it.
            tolerance: relative eigenvalue below which a direction lies outside the
                support of the reduced state.
            device: device of the Fock-space matrices.
        Raises:
            ValueError: if `internal` is empty or odd, or if the reduced state has an
                empty support, which needs every patch mode to be maximally mixed.
        '''
        if not internal or len(internal) % 2:
            raise ValueError(f'internal legs must be a non-empty even list of Majoranas, '
                             f'got {len(internal)}')
        self.covariance = covariance
        self.internal = list(internal)
        self.external = sorted(set(range(covariance.shape[0])) - set(internal))
        self.gammas = self.majoranas(len(self.external), device)
        self.fock_dim = self.gammas[0].shape[0]
        self.position = {label: index for index, label in enumerate(self.external)}

        self.rho = self.density(self.gammas, covariance[self.external][:, self.external])
        values, vectors = torch.linalg.eigh(self.rho)
        self.basis = vectors[:, values.real > tolerance * float(values.real.max())]
        if not self.basis.shape[1]:
            raise ValueError('the reduced state has an empty support: the patch carries no state')
        self.size = int(self.basis.shape[1])

    def operator(self, string: list[tuple[complex, list[int]]]) -> Tensor:
        r'''
        The Fock-space matrix of a Majorana string on the patch,
        $\sum_A c_A \prod_{a\in A}\gamma_a$.
        Args:
            string: the Majorana string, a sequence of (coefficient, labels) terms, the
                labels in `external` in ascending order.
        Returns:
            The matrix of shape (fock, fock).
        Raises:
            ValueError: if a label does not belong to the patch.
        '''
        matrix = torch.zeros(self.fock_dim, self.fock_dim, dtype=COMPLEX,
                             device=self.gammas[0].device)
        for coefficient, labels in string:
            if any(label not in self.position for label in labels):
                raise ValueError(f'every label of {labels} must belong to the patch '
                                 f'{self.external}')
            product = torch.eye(self.fock_dim, dtype=COMPLEX, device=self.gammas[0].device)
            for label in labels:
                product = product @ self.gammas[self.position[label]]
            matrix = matrix + coefficient * product
        return matrix

    def patch_hamiltonian(self) -> list[tuple[complex, list[int]]]:
        r'''
        Patch Hamiltonian $H_L$ as a Majorana string: the flat Hamiltonian of the patch
        block, $M = \texttt{FreeFermion.linalg.hamiltonian}(\Gamma_L)$ with
        $\Gamma_L$ the block of the covariance on the patch.
        Returns:
            The (coefficient, labels) terms, purely imaginary, one per pair of patch
            Majoranas.
        '''
        matrix = hamiltonian(self.covariance[self.external][:, self.external]).cpu()
        return [(0.5j * float(matrix[a, b]), [self.external[a], self.external[b]])
                for a in range(len(self.external))
                for b in range(a + 1, len(self.external))]

    def enumerate_operators(self, degree: int) -> list[list[tuple[complex, list[int]]]]:
        r'''
        Every monomial of `degree` Majoranas on the patch, made anti-Hermitian so that
        its commutator with the quadratic $H_L$ is Hermitian.  A monomial has
        $\gamma_A^\dagger = (-1)^{|A|(|A|-1)/2}\gamma_A$, so the factor $i$ is needed
        exactly when that sign is $+1$.
        Args:
            degree: the number of Majoranas in every monomial.
        Returns:
            For every subset of the patch of that size, its (coefficient, labels).
        '''
        factor = 1j if (degree * (degree - 1) // 2) % 2 == 0 else 1.0
        return [[(factor, list(labels))]
                for labels in itertools.combinations(self.external, degree)]

    def sample_operators(self,
                         count: int,
                         seed: int | None = None) -> list[list[tuple[complex, list[int]]]]:
        r'''
        Random anti-Hermitian linear strings on the patch, $A_i = i\sum_a c_a\gamma_a$,
        the smallest family whose commutator with $H_L$ does not vanish.
        Args:
            count: the number of strings to draw.
            seed: optional RNG seed.
        Returns:
            For every string, its (coefficient, labels) terms.
        '''
        generator = torch.Generator(device=self.covariance.device)
        if seed is not None:
            generator.manual_seed(seed)
        coefficients = torch.randn((count, len(self.external)), dtype=torch.float64,
                                   device=self.covariance.device, generator=generator).cpu()
        return [[(1j * float(row[index]), [self.external[index]])
                 for index in range(len(self.external))] for row in coefficients]

    def by_wick(self, operator: list[tuple[complex, list[int]]]) -> float:
        r'''
        <O> of the Gaussian state, by Wick's theorem: every operator acts on the patch,
        so its block of the covariance is all it needs.
        Args:
            operator: the Majorana string, a sequence of (coefficient, labels) terms.
        Returns:
            sum_A c_A <gamma_A>, real.
        '''
        return float(sum(coefficient * wick(self.covariance, term)
                         for coefficient, term in operator).real)

    def basic_CGO(self, operator: list[tuple[complex, list[int]]]) -> tuple[float, float]:
        r'''
        Basic CGO bound: the eigenvalue range of $P_VBP_V$ in $V_L$, which brackets the
        expectation of the state because the state lies in $V_L$.
        Args:
            operator: the observable, a sequence of (coefficient, labels) terms.
        Returns:
            (lower, upper): bounds on <O>.
        '''
        projected = self.basis.conj().T @ self.operator(operator) @ self.basis
        values = torch.linalg.eigvalsh(projected)
        return float(values.min()), float(values.max())

    def CGO(self,
            operator: list[tuple[complex, list[int]]],
            operators: list[list[tuple[complex, list[int]]]] | None = None,
            degree: int = 2,
            solver: str = 'SCS',
            return_info: bool = False,
            constraint_tol: float = 0.0,
            **options) -> tuple[float, float] | tuple[float, float, dict]:
        r'''
        CGO bounds of a Majorana string: min/max Tr(sigma B_L) over the states of $V_L$
        with Tr(sigma [H_L, A_i]) = 0.
        Args:
            operator: the observable, a sequence of (coefficient, labels) terms.
            operators: the A_i family; None enumerates every monomial of `degree` Majoranas.
            degree: the family to enumerate when `operators` is None.
            solver: the cvxpy solver, SCS by default.
            return_info: also return the solver statuses, the family size, the dimension of
                V_L and the exact expectation.
            constraint_tol: bound on |Tr(sigma [H_L, A_i])|; 0.0 keeps the exact
                equalities, which a state that only nearly satisfies them needs relaxed.
            **options: passed to `cvxpy.Problem.solve`.
        Returns:
            (lower, upper), or (lower, upper, info) when `return_info` is set.
        Raises:
            ValueError: if some [H_L, A_i] is not Hermitian, which happens when an A_i is
                not anti-Hermitian and would make its constraint vacuous.
            RuntimeError: if the solver reaches no optimal status.
        '''
        family = operators if operators is not None else self.enumerate_operators(degree)
        h_l = self.operator(self.patch_hamiltonian())
        rows = []
        for string in family:
            a_i = self.operator(string)
            row = self.basis.conj().T @ (h_l @ a_i - a_i @ h_l) @ self.basis
            drift = float((row - row.conj().T).abs().max())
            if drift > 1e-9:
                raise ValueError(f'[H_L, A_i] is not Hermitian, max drift {drift:.1e}; '
                                 f'every A_i has to be anti-Hermitian')
            rows.append(row)
        observable = self.basis.conj().T @ self.operator(operator) @ self.basis
        solution = extremal(observable, rows, ((solver, options),), constraint_tol)
        if solution is None:
            raise RuntimeError(f'{solver} did not solve the CGO SDP, {len(family)} constraints')
        lower, upper, statuses, residual = solution
        if return_info:
            exact = self.by_wick(operator)
            return lower, upper, {'status': statuses, 'constraints': len(family),
                                  'dimension': self.size, 'residual': residual,
                                  'exact': exact,
                                  'inside': lower - 1e-7 <= exact <= upper + 1e-7}
        return lower, upper
