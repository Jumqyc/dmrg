r'''
Commutator gauge optimization on the patch of a Gaussian state.
'''
import itertools

import torch
from torch import Tensor
from base import COMPLEX, CUDA
from FreeFermion.linalg import hamiltonian, wick, williamson
from CGO.sdp import extremal
from CGO.cuda.gaussian import PatchProjector

type Term = tuple[complex, list[int]] # one (coefficient, labels) term of a Majorana string

class GaussianCGO:
    r'''
    CGO on the patch of a Gaussian state: the mixed modes of the patch block, the operators restricted to their span, and the SDP that bounds an observable.
    '''

    def __init__(self,
                 covariance: Tensor,
                 internal: list[int],
                 tolerance: float = 1e-8,
                 device: torch.device = CUDA) -> None:
        r'''
        Build the patch data of one Gaussian state: the block of the covariance on the
        patch and its pure and mixed modes in the Williamson basis.
        Args:
            covariance: real antisymmetric covariance of the Gaussian state, shape
                (2n, 2n), on CUDA.
            internal: the Majorana labels of the patch, a non-empty even list; the
                complement is traced out, and every operator is expected on the patch.
            tolerance: a mode counts as pure when its Williamson eigenvalue exceeds
                1 - tolerance, and a pure mode is fixed to its ground state.
            device: device of the amplitude arrays, of size `size`.
        Raises:
            ValueError: if `internal` is empty or odd, or if the patch has no mode.
        '''
        if not internal or len(internal) % 2:
            raise ValueError(f'internal, the patch, must be a non-empty even list of '
                             f'Majoranas, got {len(internal)}')
        self.covariance = covariance
        self.internal_index = sorted(internal)
        self.external_index = sorted(set(range(covariance.shape[0])) - set(internal))
        self.position = {label: index for index, label in enumerate(self.internal_index)}
        self.modes = len(self.internal_index) // 2
        self.device = device

        self.internal_block = covariance[self.internal_index][:, self.internal_index]
        self.rotation, self.occupations = williamson(self.internal_block)
        self.pure = [k for k in range(self.modes) if float(self.occupations[k]) > 1.0 - tolerance]
        self.mixed = [k for k in range(self.modes) if k not in self.pure]
        self.size = 2 ** len(self.mixed)

        # The register geometry the kernel is written against -- the mode masks, the
        # Jordan-Wigner signs and the coefficient columns -- is derived and owned by
        # the extension; the classification of the modes is its only input.
        self._projector = PatchProjector(
            self.rotation,
            torch.tensor([1 if k in self.mixed else 0 for k in range(self.modes)],
                         dtype=torch.int8, 
                         device=device))

    def project(self, *term: Term) -> Tensor:
        r'''
        Matrix of the Majorana string on V_L, the support of the reduced state.

        V_L is spanned by the mixed modes (both occupations) with every pure mode pinned to |0>, so a basis state splits into a mixed register (the row/column index, big-endian) and a pure register.  The amplitudes are carried as a batch over the pure register, of shape (2**len(pure), size, size), and one label expands over the normal modes at once: a normal-mode Majorana flips one occupation of either register and multiplies by its Jordan-Wigner sign, and amplitudes that end outside the pure vacuum are dropped, which is the projection onto V_L.  The expansion runs in ``CGO/cuda/gaussian.cu``, so the batch stays on the device.
        Args:
            term: the terms of the Majorana string, each (coefficient, labels), the labels in `internal_index` in ascending order.
        Returns:
            The matrix of shape (size, size).
        Raises:
            ValueError: if a label does not belong to the patch.
        '''
        matrix = torch.zeros((self.size, self.size), 
                             dtype=COMPLEX, 
                             device=self.device)
        for coefficient, labels in term:
            if any(label not in self.position for label in labels):
                raise ValueError(f'every label of {labels} must belong to the patch '
                                 f'{self.internal_index}')
            amps = self._projector.vacuum()   # the pure vacuum, shared by every string
            # The kernel left-multiplies, so walking the labels backwards is what
            # makes the product come out in the order they are written in; walking
            # them forwards would reverse the string, which flips the sign of every
            # monomial with d(d-1)/2 odd (d = 2, 3 mod 4).
            for label in reversed(labels):
                # gamma_a = sum_b R_ba gamma_b: the kernel expands the label over the
                # normal modes and applies the whole expansion to the batch.
                amps = self._projector.apply(amps, self.position[label])
            matrix = matrix + coefficient * amps[0]
        return matrix

    def sample_operators(self,
                         count: int,
                         seed: int | None = None) -> list[list[Term]]:
        r'''
        Random anti-Hermitian linear strings on the patch, A_i = i sum_a c_a gamma_a, the smallest family whose commutator with H_L does not vanish.
        Args:
            count: the number of strings to draw.
            seed: optional RNG seed.
        Returns:
            For every string, its (coefficient, labels) terms.
        '''
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        coefficients = torch.randn((count, len(self.internal_index)), 
                                   dtype=torch.float64,
                                   device=self.device, 
                                   generator=generator).cpu()
        return [[(1j * float(row[index]), [self.internal_index[index]])
                 for index in range(len(self.internal_index))] for row in coefficients]

    def by_wick(self, *term: Term) -> float:
        '''
        avg(O) of the Gaussian state, by Wick's theorem: every operator acts on the patch, so its block of the covariance is all it needs.
        Args:
            term: the terms of the Majorana string.
        Returns:
            sum_A c_A <gamma_A>, real.
        '''
        return float(sum(coefficient * wick(self.covariance, labels) for coefficient, labels in term).real)

    def basic_CGO(self, *term: Term) -> tuple[float, float]:
        r'''
        Basic CGO bound: the eigenvalue range of P_V B P_V in V_L, which brackets the
        expectation of the state because the state lies in V_L.
        Args:
            term: the terms of the observable.
        Returns:
            (lower, upper): bounds on <O>.
        '''
        values = torch.linalg.eigvalsh(self.project(*term))
        return float(values.min()), float(values.max())

    def CGO(self,
            *term: Term,
            operators: list[list[Term]] | None = None,
            degree: int = 2,
            solver: str = 'SCS',
            return_info: bool = False,
            constraint_tol: float = 0.0,
            energies: Tensor | None = None,
            **options) -> tuple[float, float] | tuple[float, float, dict]:
        r'''
        CGO bounds of a Majorana string: min/max Tr(sigma B_L) over the states of V_L
        with Tr(sigma [H_L, A_i]) = 0.
        Args:
            term: the terms of the observable.
            operators: the A_i family; None enumerates every monomial of `degree` Majoranas.
            degree: the family to enumerate when `operators` is None.
            solver: the cvxpy solver, SCS by default.
            return_info: also return the solver statuses, the family size, the dimension of
                V_L and the exact expectation.
            constraint_tol: bound on |Tr(sigma [H_L, A_i])|; 0.0 keeps the exact
                equalities, which a state that only nearly satisfies them needs relaxed.
            energies: energy of each normal mode of the state, shape (n,) for a
                covariance of shape (2n, 2n); None gives every mode the energy 1.
            **options: passed to cvxpy.Problem.solve.
        Returns:
            (lower, upper), or (lower, upper, info) when `return_info` is set.
        Raises:
            ValueError: if some [H_L, A_i] is not Hermitian, which happens when an A_i is
                not anti-Hermitian and would make its constraint vacuous.
            RuntimeError: if the solver reaches no optimal status.
        '''
        if operators is None:
            # a monomial made anti-Hermitian: gamma_A^dagger = (-1)^(|A|(|A|-1)/2) gamma_A, so
            # the factor i is needed exactly when that sign is +1
            factor = 1j if (degree * (degree - 1) // 2) % 2 == 0 else 1.0
            family = [[(factor, list(labels))]
                      for labels in itertools.combinations(self.internal_index, degree)]
        else:
            family = operators

        # H_L: the parent Hamiltonian of the whole state, keeping the terms whose support lies inside the patch
        full = hamiltonian(self.covariance, energies).cpu()
        labels = self.internal_index
        h_l = self.project(*[
            (0.5j * float(full[labels[a], labels[b]]), [labels[a], labels[b]])
            for a in range(len(labels))
            for b in range(a + 1, len(labels))])
        rows = []
        for string in family:
            a_i = self.project(*string)
            row = h_l @ a_i - a_i @ h_l
            drift = float((row - row.conj().T).abs().max())
            assert drift < 1e-9, ValueError(f'[H_L, A_i] is not Hermitian, max drift {drift:.1e}; every A_i has to be anti-Hermitian')
            rows.append(row)
        observable = self.project(*term)
        solution = extremal(observable, rows, ((solver, options),), constraint_tol)
        if solution is None:
            raise RuntimeError(f'{solver} did not solve the CGO SDP, {len(family)} constraints')
        lower, upper, statuses, residual = solution
        if return_info:
            exact = self.by_wick(*term)
            return lower, upper, {'status': statuses, 'constraints': len(family),
                                  'dimension': self.size, 'residual': residual,
                                  'exact': exact,
                                  'inside': lower - 1e-7 <= exact <= upper + 1e-7}
        return lower, upper
