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

``wick`` evaluates an ordered product of Majorana operators in a Gaussian state,
which is the primitive every other contraction of this module is built from, and
``majorana_gram`` gives the Gram matrix of the Majorana products
``gamma_S |Gamma>`` over the Majoranas of a covariance, which the caller picks by
slicing the covariance down to them.
"""
import math

import torch

from torch import Tensor

from base import CUDA, DTYPE

from FreeFermion.cuda.fermion import pfaffian


def vacuum_covariance(num_modes: int,
                      dtype: torch.dtype = torch.float64,
                      device: torch.device = CUDA) -> Tensor:
    '''
    covariance of the vacuum of num_modes Majorana modes, i.e. of the state
    whose normal modes are all empty and whose Williamson eigenvalues are all 1:
    the adjacent Majorana pairs (gamma_2k, gamma_2k+1) are paired with J.  This is
    the block diagonal form that RandPureCov conjugates and that the bond blocks of a
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


def RandPureCov(dim: int,
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


def wick(covariance: Tensor, operators: Tensor | list[int]) -> Tensor:
    '''
    Gaussian expectation value of the ordered product of Majorana operators
    <gamma_{o_0} gamma_{o_1} ... gamma_{o_{m-1}}>, by Wick's theorem: the sum over
    pairings of the product of the contractions, which is the Pfaffian of the
    antisymmetrised contraction matrix

        <gamma_a gamma_b> = -i covariance[a, b] = K[a, b]   (a != b),
        <gamma_a gamma_a> = 1,

    so a repeated Majorana is its own square and contracts with itself, and no
    reduction of the product is needed before calling this.  The value vanishes
    for an odd number of operators.  This is the primitive every other fermionic
    contraction of this module is built from; it holds for any Gaussian state,
    mixed included, under the convention <gamma_a> = 0 that a covariance implies.
    Args:
        covariance: real antisymmetric covariance, shape (2n, 2n).
        operators: Majorana labels of the product, shape (..., m) with m <= 32,
            and a plain list of ints is read as one product.  The labels are
            0-based, and a negative one would wrap around in the index, so the
            range is checked whenever that costs nothing, i.e. for a list or a CPU
            tensor; a tensor already on the device is taken as given.
    Returns:
        The expectation values, shape (...); a single product gives a scalar
        tensor.
    Raises:
        ValueError: if a label of a list or CPU tensor input is outside the
            labels of the covariance.
    '''
    if isinstance(operators, torch.Tensor):
        labels, cheap = operators, operators.device.type == 'cpu'
    else:
        labels, cheap = torch.as_tensor(operators, dtype=torch.long), True
    if cheap and labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= covariance.shape[0]):
        raise ValueError(f'operators must be labels in [0, {covariance.shape[0]})')
    labels = labels.to(device=covariance.device, dtype=torch.long)
    single = labels.dim() == 1
    if single:
        labels = labels.unsqueeze(0)
    first, second = labels[:, :, None], labels[:, None, :]
    contractions = -1j * covariance[first, second].to(torch.complex128)
    contractions = torch.where(first == second, torch.ones_like(contractions), contractions)
    upper = torch.triu(contractions, diagonal=1)
    values = pfaffian(upper - upper.transpose(-1, -2))
    return values[0] if single else values


def majorana_gram(covariance: Tensor) -> Tensor:
    '''
    Gram matrix of the Majorana-excitation basis of a Gaussian state, over every Majorana of the given covariance.  The basis states are the products
    |S> = gamma_S |Gamma> for every subset S of the Majoranas, with gamma_S = prod_{s in S} gamma_s in ascending order, so 2r Majoranas give 2**(2r) states.
    Wick's theorem makes every overlap a Pfaffian of the contraction kernel
    K = -i covariance,

        <S|T> = (-1)**(|S| (|S| - 1) / 2 + N(S, T)) * table[S xor T],
        table[A] = Pf(K[A]),    N(S, T) = #{(s, t) in S x T : s > t},

    because gamma_S^dag gamma_T reduces to the single monomial gamma_{S xor T}
    with the sign (-1)**N(S, T), and only S xor T survives in the contraction.
    So 2**(2r) Pfaffians, one per subset, cover the whole matrix and every element
    is a table lookup afterwards.  The states are all normalised (the diagonal of
    the Gram is 1) but not orthogonal; the kernel of the Gram holds their linear
    dependence, which the caller removes by diagonalising the Gram.
    The state enters only through the two point functions of these Majoranas, so a
    sub-block of a larger covariance is a valid input: the caller chooses the
    Majoranas of the basis by slicing, which also keeps every label in range.
    Args:
        covariance: real antisymmetric covariance of the Gaussian state, of an even
            number 2r of Majoranas, shape (2r, 2r).  It may be the covariance of a
            mixed state; the Gram is then the matrix of correlators
            Tr(rho gamma_S^dag gamma_T).
    Returns:
        The Hermitian positive semi-definite Gram matrix, shape (2**2r, 2**2r),
        with the row and column of a subset given by its bitmask over the
        Majoranas.
    Raises:
        ValueError: if the covariance does not have an even number of Majoranas.
    '''
    num_majoranas = covariance.shape[0]
    if num_majoranas % 2:
        raise ValueError(f'the covariance must hold an even number of Majoranas, '
                         f'got {num_majoranas}')

    count = 2 ** num_majoranas
    device = covariance.device
    subsets = torch.arange(count, device=device)
    slots = torch.arange(num_majoranas, device=device)
    bits = ((subsets[:, None] >> slots[None, :]) & 1).bool()
    sizes = bits.sum(dim=1)

    # table[A] = Pf(K[A]) for every subset, one batch of Pfaffians per size
    table = torch.zeros(count, dtype=torch.complex128, device=device)
    table[0] = 1.0
    for size in range(1, num_majoranas + 1):
        selected = subsets[sizes == size]
        columns = slots[None, :].expand(len(selected), num_majoranas)
        pick = torch.where(bits[selected], columns, num_majoranas)
        table[selected] = wick(covariance, pick.sort(dim=1).values[:, :size])

    inversions = (bits.to(torch.float64)
                  @ (slots[:, None] > slots[None, :]).to(torch.float64)
                  @ bits.to(torch.float64).T).long()
    flip = sizes * (sizes - 1) // 2
    sign = torch.where(((flip[:, None] + inversions) % 2).bool(), -1.0, 1.0)
    return sign * table[subsets[:, None] ^ subsets[None, :]]

