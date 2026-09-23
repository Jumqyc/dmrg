"""Linear algebra helper of the real antisymmetric covariance matrix
``Gamma_ab = < (i/2) [gamma_a, gamma_b] >`` of a fermionic Gaussian state of
``n`` modes: the sampling of a pure covariance, and Williamson's theorem with
the derived quantities used by the Gaussian routes of this repository.

``A = i Gamma`` is Hermitian with eigenvalues coming in ``+-lambda_k`` pairs, so

    Gamma = R^T D R,   D = diag(lambda_1 J, ..., lambda_n J),   J = [[0, -1], [1, 0]],

with ``R`` real orthogonal and ``0 <= lambda_k <= 1``; the state is pure exactly
when every ``lambda_k = 1`` (Bravyi, quant-ph/0507182, eq. (18)-(19)).  The sign
of J is the convention under which the ``lambda_k`` are the positive eigenvalues
of ``i Gamma``, which is what the diagonalization below returns.

Williamson's theorem contributes the occupations ``p_k = (1 - lambda_k) / 2`` of
the normal modes and the modular ("thermal") matrix, which is the same similarity
transform with the block values ``artanh(lambda_k)``, i.e.
``W = -2i artanh(i Gamma)`` of ``rho = exp((i/4) gamma^T W gamma) / Z``.

The last three functions build a Gaussian matrix element from Wick's theorem in
the normal-mode basis: ``_mode_coefficients`` writes an operator (a mode d_k, a
mode d_k^dag or a physical Majorana) as a vector over the 2n mode operators,
``_vacuum_contraction`` assembles the matrix K_ij = <0|A_i A_j|0> of a list of
such operators, and ``pfaffian`` sums over its pairings, so that
<0|A_1 ... A_m|0> = Pf(K).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import Tensor

from base import CUDA, DTYPE

from FreeFermion.cuda import fermion


def vacuum_covariance(num_modes: int,
                      dtype: torch.dtype = torch.float64,
                      device: torch.device = CUDA) -> Tensor:
    '''
    covariance of the vacuum of num_modes Majorana modes, i.e. of the state
    whose normal modes are all empty and whose Williamson eigenvalues are all 1:
    the adjacent Majorana pairs (gamma_2k, gamma_2k+1) are paired with J.  This is
    the block diagonal form that randcov conjugates and that the bond blocks of a
    product state use.
    Args:
        num_modes: number of Majorana modes, even.
        dtype: dtype of the returned matrix.
        device: device of the returned matrix, CUDA by default.
    Returns:
        The real antisymmetric matrix J of shape (num_modes, num_modes), with
        J @ J = -1.
    Raises:
        ValueError: if num_modes is odd.
    '''
    if num_modes % 2 != 0:
        raise ValueError("The number of Majorana modes must be even.")
    j = torch.zeros((num_modes, num_modes), dtype=dtype, device=device)
    j[0::2, 1::2] = -torch.eye(num_modes // 2, dtype=dtype, device=device)
    j[1::2, 0::2] = torch.eye(num_modes // 2, dtype=dtype, device=device)
    return j


def randcov(dim: int,
            dtype: torch.dtype = torch.float64,
            device: torch.device = CUDA) -> Tensor:
    '''
    random pure Gaussian covariance of dim Majorana modes: an orthogonal q
    drawn from the Haar measure (the Q factor of a Gaussian matrix, with the signs
    of R fixed) conjugates the vacuum covariance J, whose 2x2 blocks [[0, -1],
    [1, 0]] pair the adjacent Majoranas (gamma_2k, gamma_2k+1) exactly as D of
    williamson_decomposition does.
    Args:
        dim: number of Majorana modes, even.
        dtype: dtype of the returned matrix.
        device: device of the returned matrix, CUDA by default.
    Returns:
        The real antisymmetric matrix q J q^T of shape (dim, dim), with
        (q J q^T)^2 = -1.
    Raises:
        ValueError: if dim is odd.
    '''
    if dim % 2 != 0:
        raise ValueError("Dimension must be even.")
    a = torch.randn(dim, dim, dtype=dtype, device=device)
    q,r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))

    return q @ vacuum_covariance(dim, dtype, device) @ q.T


def williamson_decomposition(gamma: Tensor) -> tuple[Tensor, Tensor]:
    '''
    Williamson normal form of the covariance gamma, from the Hermitian
    A = i gamma.  eigh returns the eigenvalues ascending in +-lambda_k pairs, so
    the last n columns are the eigenvectors w_k of the positive lambda_k.  The
    sqrt(2) in (sqrt(2) Re w_k, sqrt(2) Im w_k) is exactly the factor that makes
    the two rows orthonormal, so R is real orthogonal and gamma = R^T D R.
    The overall phase of every w_k is arbitrary; it cancels in D and in R^T D R.
    Args:
        gamma: real antisymmetric covariance, shape (2n, 2n).
    Returns:
        rotation: the real orthogonal matrix R, shape (2n, 2n).
        blocks: the real block diagonal matrix D, shape (2n, 2n), block k equal to
            lambda_k J, with the Williamson eigenvalues lambda_k = -D[2k, 2k + 1]
            ascending in [0, 1], shape (n,).
    '''
    num_modes = gamma.shape[0] // 2
    eigenvalues, eigenvectors = torch.linalg.eigh(1j * gamma.to(DTYPE))
    selected = eigenvectors[:, num_modes:]
    lambdas = eigenvalues[num_modes:].real
    real = math.sqrt(2.0) * selected.real
    imag = math.sqrt(2.0) * selected.imag
    rotation = torch.empty((2 * num_modes, 2 * num_modes),
                           dtype=real.dtype, device=real.device)
    rotation[0::2] = real.T
    rotation[1::2] = imag.T
    blocks = torch.zeros((2 * num_modes, 2 * num_modes),
                         dtype=lambdas.dtype, device=lambdas.device)
    blocks[0::2, 1::2] = -torch.diag(lambdas)
    blocks[1::2, 0::2] = torch.diag(lambdas)
    return rotation, blocks


def williamson_modes(gamma: Tensor,
                     tolerance: float = 1e-10) -> tuple[Tensor, Tensor]:
    '''
    normal modes d_k of the Gaussian state with covariance gamma.  The row
    pair (2k, 2k + 1) of R is (sqrt(2) Re w_k, sqrt(2) Im w_k), hence
    d_k = (1/2) sum_mu (R[2k] + i R[2k + 1])_mu gamma_mu, normalised so that
    {d_k, d_l^dag} = delta_kl and <d_k^dag d_l> = p_k delta_kl with
    p_k = (1 - lambda_k) / 2.  The overall phase of each d_k is arbitrary and
    drops out of the SDP.
    Args:
        gamma: real antisymmetric covariance, shape (2n, 2n).
        tolerance: smallest lambda_k accepted as positive.
    Returns:
        occupations: the occupation p_k, shape (n,).
        coeff: the complex coefficients of d_k over the 2n Majoranas, shape (n, 2n).
    Raises:
        ValueError: if the number of positive eigenvalues of i gamma is not n.
    '''
    rotation, blocks = williamson_decomposition(gamma)
    lambdas = -blocks[0::2, 1::2].diagonal()
    positive = int((lambdas > tolerance).sum())
    if positive != gamma.shape[0] // 2:
        raise ValueError(f'expected {gamma.shape[0] // 2} positive eigenvalues, '
                         f'got {positive}')
    return (1.0 - lambdas) / 2.0, (rotation[0::2] + 1j * rotation[1::2]) / 2


def williamson_modular(gamma: Tensor,
                       clip: float = 1e-12) -> tuple[Tensor, Tensor]:
    '''
    modular matrix W = -2i artanh(i gamma) of the thermal form
    rho = exp((i/4) gamma^T W gamma) / Z.  artanh is a matrix function, so it is
    the same similarity transform as the decomposition with the block values
    artanh(lambda_k): W = 2 R^T D' R.  Eigenvalues at the purity boundary are
    clipped, which only fixes the decoupled fully empty or occupied modes.
    Args:
        gamma: real antisymmetric covariance, shape (2n, 2n).
        clip: distance kept from |lambda_k| = 1 before artanh.
    Returns:
        modular: the real antisymmetric W, shape (2n, 2n).
        lambdas: the unclipped Williamson eigenvalues, shape (n,), ascending in
            [0, 1].
    '''
    rotation, blocks = williamson_decomposition(gamma)
    lambdas = -blocks[0::2, 1::2].diagonal()
    values = torch.atanh(lambdas.clamp(-1.0 + clip, 1.0 - clip))
    scaled = torch.zeros_like(blocks)
    scaled[0::2, 1::2] = -torch.diag(values)
    scaled[1::2, 0::2] = torch.diag(values)
    return 2.0 * rotation.T @ scaled @ rotation, lambdas


def pfaffian(matrix: Tensor) -> Tensor:
    '''
    Pfaffian of an antisymmetric matrix by skew-symmetric elimination with
    two-by-two pivots, Pf(A) = A[0,1] * Pf(Schur complement) with the sign of the
    pivoting permutation tracked; an odd-sized matrix has no pairing and gives 0.
    Kernel: already covered by CUDA in FreeFermion/cuda/fermion.cu (function pfaffian, one
    block per matrix, batched); this wrapper passes a single matrix.
    Args:
        matrix: real or complex antisymmetric matrix, shape (m, m).
    Returns:
        The scalar tensor sum over pairings of the signed products of the entries.
    '''
    return fermion.pfaffian(matrix.unsqueeze(0))[0]


def _mode_coefficients(coeff: Tensor, num_modes: int, operator: tuple) -> Tensor:
    '''
    a linear operator (a mode annihilation, a mode creation or a physical
    Majorana) as a vector over the 2*num_modes mode operators
    [d_0 ... d_{n-1}, d_0^dag ... d_{n-1}^dag]; a physical Majorana is
    gamma_mu = sum_k (2 conj(coeff[k, mu]) d_k + 2 coeff[k, mu] d_k^dag).
    Kernel: already covered by CUDA in FreeFermion/cuda/fermion.cu (function operator_vectors,
    one thread per operator, batched); this wrapper passes a single operator.
    Args:
        coeff: complex Williamson coefficients of d_k over the 2n Majoranas,
            shape (n, 2n).
        num_modes: number of modes n.
        operator: ('d', k) for d_k, ('c', k) for d_k^dag or ('gamma', mu) for
            the physical Majorana gamma_mu.
    Returns:
        The complex vector of length 2n.
    '''
    kinds = {'d': 0, 'c': 1}
    codes = torch.tensor([[kinds.get(operator[0], 2), operator[1]]],
                         dtype=torch.long, device=coeff.device)
    return fermion.operator_vectors(coeff, codes)[0]


def _vacuum_contraction(vectors: list[Tensor], num_modes: int) -> Tensor:
    '''
    the Wick contraction matrix with K[i, j] = <0|A_i A_j|0> for i < j and
    K[j, i] = -K[i, j].  In the mode vacuum the only non-zero contraction of the
    mode operators is <0|d_k d_l^dag|0> = delta_kl, so
    <0|A_i A_j|0> = sum_k a_i[d_k] a_j[d_k^dag]; the matrix is then
    antisymmetrised by hand, which is what the Pfaffian of the pairing expansion
    requires.
    Kernel: already covered by CUDA in FreeFermion/cuda/fermion.cu (function
    vacuum_contraction, one block per matrix, batched); this wrapper passes a
    single list of operators.
    Args:
        vectors: the m operators as mode-basis vectors, each of shape (2n,).
        num_modes: number of modes n.
    Returns:
        The complex antisymmetric contraction matrix, shape (m, m).
    '''
    return fermion.vacuum_contraction(torch.stack(vectors).unsqueeze(0), num_modes)[0]


def occupations(covariance: Tensor) -> Tensor:
    '''
    Williamson occupations p_k = (1 - lambda_k) / 2 of a covariance, with
    the lambda_k the positive eigenvalues of i Gamma.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
    Returns:
        The occupation of every normal mode, shape (n,), in [0, 1].
    '''
    eigenvalues = torch.linalg.eigvalsh(1j * covariance.to(torch.complex128)).real
    return (1.0 - eigenvalues[eigenvalues.shape[0] // 2:]) / 2.0


def purity(covariance: Tensor) -> float:
    '''
    purity Tr(rho^2) of a Gaussian state, the product of the binary purities
    of its normal modes: 1 for a pure state and 2**-n for the maximally mixed one.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
    Returns:
        The purity in (0, 1].
    '''
    p = occupations(covariance)
    return float((p * p + (1.0 - p) * (1.0 - p)).prod())


def entropy(covariance: Tensor) -> float:
    '''
    von Neumann entropy of a Gaussian state, the sum of the binary entropies
    of its normal modes.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
    Returns:
        The entropy in nats, 0 for a pure state.
    '''
    p = occupations(covariance).clamp(1e-15, 1.0 - 1e-15)
    return float(-(p * p.log() + (1.0 - p) * (1.0 - p).log()).sum())


def is_pure(covariance: Tensor, tolerance: float = 1e-9) -> bool:
    '''
    test the pure-state condition Gamma^2 = -1.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
        tolerance: largest max |Gamma^2 + 1| accepted as pure.
    Returns:
        True if the covariance is that of a pure state.
    '''
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    return float((covariance @ covariance + identity).abs().max()) < tolerance

