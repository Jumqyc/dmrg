import math
import torch
from torch import Tensor
from base import CUDA, COMPLEX
from FreeFermion.cuda.fermion import pfaffian

def symplectic_form(num_modes: int,
                      dtype: torch.dtype = torch.float64,
                      device: torch.device = CUDA) -> Tensor:
    '''
    The real antisymmetric symplectic form J of 2n Majoranas, with J @ J = -1.
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
    random pure Gaussian covariance of dim Majorana modes: an orthogonal q drawn from the Haar measure conjugates the vacuum covariance J, whose 2x2 blocks [[0, -1], [1, 0]] pair the adjacent Majoranas (gamma_2k, gamma_2k+1) exactly as the D of williamson does.
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
    # q samples from the Haar measure of O(dim), and q J q^T is a random pure covariance.

    return q @ symplectic_form(dim, dtype, device) @ q.T


def williamson(gamma: Tensor) -> tuple[Tensor, Tensor]:
    r'''
    Williamson decomposition of the covariance gamma. 
    Args:
        gamma: real antisymmetric covariance, shape (2n, 2n).
    Returns:
        rotation: the real orthogonal matrix R, shape (2n, 2n).
        lambdas: the Williamson eigenvalues, shape (n,), ascending in [0, 1].
    '''
    num_modes = gamma.shape[0] // 2
    eigenvalues, eigenvectors = torch.linalg.eigh(1j * gamma.to(COMPLEX))
    selected = eigenvectors[:, num_modes:]
    lambdas = eigenvalues[num_modes:].real
    real = math.sqrt(2.0) * selected.real
    imag = math.sqrt(2.0) * selected.imag
    rotation = torch.empty((2 * num_modes, 2 * num_modes),
                           dtype=real.dtype,
                           device=real.device)
    rotation[0::2] = real.T
    rotation[1::2] = imag.T
    return rotation, lambdas


def hamiltonian(covariance: Tensor, spec: Tensor | None = None) -> Tensor:
    r'''
    Quadratic Hamiltonian H = (i/4) gamma^T M gamma of a Gaussian state, with one mode energy per normal mode: the block diagonal D = diag(lambda_k J) of the Williamson decomposition Gamma = R^T D R is replaced by diag(spec_k J), so

        M = -R^T diag(spec_k J) R,

    which inverts `GfPEPS.from_hamiltonian` for a pure covariance at the flat spec, where D = J already and $M = -covariance$ exactly, and the state is the ground state of H. A mixed covariance is flattened instead, so the ground state of M is the canonical purification $i\,\mathrm{sign}(i\,covariance)$ and not the state itself.  The ground state fixes only the sign of each energy, so every positive `spec` gives the same state and the magnitudes are free parameters of the Hamiltonian rather than of the state.
    Args:
        covariance: real antisymmetric covariance of the Gaussian state, shape
            (2n, 2n), on CUDA.
        spec: energy of each normal mode, shape (n,); None sets every energy to 1.
    Returns:
        The real antisymmetric Majorana matrix M, shape (2n, 2n), with $iM$ having
        eigenvalues $\pm\mathrm{spec}_k$.
    Raises:
        ValueError: if the Williamson rotation is not orthogonal, which happens when a
            mode is maximally mixed and the mode energies are not defined; or if `spec`
            does not hold one energy per mode.
    '''
    rotation, _ = williamson(covariance)
    identity = torch.eye(covariance.shape[0],
                         dtype=rotation.dtype,
                         device=rotation.device)
    assert float((rotation.T @ rotation - identity).abs().max()) < 1e-9, ValueError('the Williamson rotation is not orthogonal, so the covariance has a maximally mixed mode and the mode energies are not defined')
    if spec is None:
        return -rotation.T @ symplectic_form(covariance.shape[0],
                             covariance.dtype,
                             covariance.device) @ rotation
    modes = covariance.shape[0] // 2
    if spec.shape != (modes,):
        raise ValueError(f'spec holds one energy per mode, so shape ({modes},), '
                         f'got {tuple(spec.shape)}')
    form = symplectic_form(covariance.shape[0], covariance.dtype, covariance.device)
    form[0::2, 1::2] = -torch.diag(spec)
    form[1::2, 0::2] = torch.diag(spec)
    return -rotation.T @ form @ rotation



def occupations(covariance: Tensor) -> Tensor:
    '''
    Williamson occupations p_k = (1 - lambda_k) / 2 of a covariance, with the lambda_k the positive eigenvalues of i Gamma.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
    Returns:
        The occupation of every normal mode, shape (n,), in [0, 1].
    '''
    eigenvalues = torch.linalg.eigvalsh(1j * covariance.to(torch.complex128)).real
    return (1.0 - eigenvalues[eigenvalues.shape[0] // 2:]) / 2.0


def purity(covariance: Tensor) -> float:
    '''
    purity Tr(rho^2) of a Gaussian state, the product of the binary purities of its normal modes: 1 for a pure state and 2**-n for the maximally mixed one.
    Args:
        covariance: real antisymmetric covariance of 2n Majorana modes.
    Returns:
        The purity in (0, 1].
    '''
    p = occupations(covariance)
    return float((p * p + (1.0 - p) * (1.0 - p)).prod())


def entropy(covariance: Tensor) -> float:
    '''
    von Neumann entropy of a Gaussian state, the sum of the binary entropies of its normal modes.
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
    Gaussian expectation value of the ordered product of Majorana operators <gamma_{o_0} gamma_{o_1} ... gamma_{o_{m-1}}>, by Wick's theorem: the sum over pairings of the product of the contractions, which is the Pfaffian of the antisymmetrised contraction matrix

        <gamma_a gamma_b> = -i covariance[a, b] = K[a, b]   (a != b),
        <gamma_a gamma_a> = 1,

    so a repeated Majorana is its own square and contracts with itself, and no reduction of the product is needed before calling this.  The value vanishes for an odd number of operators.  This is the primitive every other fermionic contraction of this module is built from; it holds for any Gaussian state, mixed included, under the convention <gamma_a> = 0 that a covariance implies.
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

