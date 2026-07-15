import torch
import numpy as np
import opt_einsum as oe
import sys
import pickle as pkl


from math import sqrt
from typing import Literal
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Module-level cache for oe.contract_expression objects.
# Keys are (einsum_string, *shapes); values are the compiled expression.
# This avoids redundant path-optimization when the same contraction shape
# is encountered repeatedly (e.g. inside Lanczos or sweep environment updates).
# ---------------------------------------------------------------------------
_expr_cache: dict[tuple, oe.ContractExpression] = {}

def _get_expr(einsum_str: str, *shapes: tuple[int, ...]) -> oe.ContractExpression:
    """Return a cached ``oe.ContractExpression`` for *einsum_str* and *shapes*."""
    key = (einsum_str,) + shapes
    try:
        return _expr_cache[key]
    except KeyError:
        _expr_cache[key] = oe.contract_expression(
            einsum_str, *shapes, optimize='auto'
        )
        return _expr_cache[key]

def cached_einsum(einsum_str: str, *tensors: torch.Tensor) -> torch.Tensor:
    """Perform an einsum contraction using a cached ``oe.ContractExpression``."""
    shapes = tuple(t.shape for t in tensors)
    key = (einsum_str,) + shapes
    try:
        return _expr_cache[key](*tensors)
    except KeyError:
        _expr_cache[key] = oe.contract_expression(
            einsum_str, *shapes, optimize='auto'
        )
        return _expr_cache[key](*tensors)

class SpinOperator:
    '''
    A tensor representing spin operators (Sx, Sy, Sz) for a given physical dimension.
    '''
    def __init__(self, phys_dim:int, dtype:torch.dtype = torch.complex128, device:torch.device = torch.device('cuda')):
        self.data = torch.zeros((3, phys_dim, phys_dim),
                                 dtype=dtype,
                                 device=device,
                                 requires_grad=False)
        # Sz
        s = (phys_dim - 1) / 2
        for i in range(phys_dim):
            self.data[2, i, i] = s - i
            m = s - i
            if i == phys_dim - 1:
                break
            self.data[0, i   ,i+1] = sqrt(s * (s + 1) - m * (m - 1)) # S+
            self.data[1, i + 1, i] = sqrt(s * (s + 1) - m * (m - 1)) # S-

        self.data[0],self.data[1] = 0.5 * (self.data[0] + self.data[1]), -0.5j * (self.data[0] - self.data[1])# Sx, Sy

        self.phys_dim = phys_dim
        self.dtype = dtype
        self.device = device
    def to(self, device):
        '''
        Move the spin operator tensor to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        self.data = self.data.to(device)
    def __repr__(self):
        if self.phys_dim %2 == 0:
            return f"Spin {self.phys_dim//2} operator \n {self.data})"
        else:
            return f"Spin {self.phys_dim}/2 operator \n {self.data})"

class MPS:
    '''
    A class representing a Matrix Product State (MPS) in quantum physics and tensor networks.
    '''
    def __init__(self,
                 L: int,
                 phys_dim:int,
                 bond_dims:list[int]|None = None,
                 init_state:list[torch.Tensor]|None = None,
                 dtype:torch.dtype = torch.complex128,
                 device:torch.device = torch.device('cuda')):
        '''
        Args:
            L (int): Length of the MPS (number of sites).
            phys_dim (int): Physical dimension of each site.
            bond_dims (list[int] | None): List of bond dimensions for each bond in the MPS. Including the left and right virtual bonds, the length of this list should be L+1. If None, defaults to [1] + [phys_dim] * (L - 1) + [1].
            init_state (list[torch.Tensor] | None): Optional initial state for the MPS. If None, random tensors will be generated.
            dtype (torch.dtype): Data type for the MPS tensors (default: torch.complex128).
            device (torch.device): Device to store the MPS tensors (default: 'cuda').
        Raises:
            ValueError: If the length of bond_dims does not match L+1.
        '''
        if bond_dims is None:
            bond_dims = [1] + [phys_dim] * (L - 1) + [1]
        self.L = L
        self.phys_dim = phys_dim
        self.bond_dim = bond_dims
        if len(bond_dims) != L + 1:
            raise ValueError(f"Bond dimension list length {len(bond_dims)} does not match the expected length of {L + 1}.")
        self.dtype = dtype
        self.device = device
        self.center = 0
        # Center site index for canonical form, init as the middle of the chain

        # Initialize MPS tensors (left_bond_dim, phys_dim, right_bond_dim)
        if init_state is None:
            self.tensors = [
            torch.rand((bond_dims[i], phys_dim, bond_dims[i + 1]),
                       dtype=dtype,
                       device=device,
                       requires_grad=False) for i in range(L)
        ]
        else:
            self.tensors = init_state
            if any(t.shape != (bond_dims[i], phys_dim, bond_dims[i + 1]) for i, t in enumerate(init_state)):
                raise ValueError("Initial state tensors do not match the specified bond and physical dimensions.")

        self.canonicalize()  # Ensure the MPS is in canonical form upon initialization
        self.normalize()  # Normalize the MPS upon initialization

    def __getitem__(self, idx):
        return self.tensors[idx]
    def to(self, device):
        '''
        Move the MPS tensors to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        for i in range(self.L):
            self.tensors[i] = self.tensors[i].to(device)

    def move_center_to(self, new_center:int):
        '''
        Move the center site index to a new position in the MPS.
        Args:
            new_center (int): The new center site index (0 <= new_center < L).
        Raises:
            IndexError: If the new center index is out of bounds for the MPS length.
            '''
        if new_center == self.center:
            return  # No need to move if the new center is the same as the current center
        if new_center < 0 or new_center >= self.L:
            raise IndexError('New center index is out of bounds for the MPS length.')
        if new_center < self.center:
            for i in range(self.center, new_center, -1):  # move to the right
                self._right_canonicalize_site(i, update_center=True)
        elif new_center > self.center:
            for i in range(self.center, new_center):      # move to the left
                self._left_canonicalize_site(i, update_center=True)

    def expectation(self,operators, mode:Literal['direct','env'] = 'env')-> complex:
        '''
        Compute the expectation value of operators on the MPS.
        Accepts either a dict[int, torch.Tensor] of single-site operators or an MPO.

        To compute the norm, just pass {} as the operator list.
        Args:
            operators: dict[int, torch.Tensor] | MPO: A dictionary of operators to compute the expectation value,
                       or an MPO representing a full matrix product operator.
            mode: 'direct' or 'env' — contraction strategy (for dict operators only; MPO always uses direct).
        Returns:
            complex: The expectation value of the operators on the MPS.
        Raises:
            ValueError: If the operator shapes do not match the physical dimension of the MPS tensors.
        '''
        # Check if operators is an MPO instance (imported locally to avoid circular refs)
        if type(operators).__name__ == 'MPO':
            return self._mpo_expectation(operators)
        # Otherwise treat as dict[int, torch.Tensor]
        if any(op.shape != (self.phys_dim, self.phys_dim) for op in operators.values()):
            raise ValueError(f"Operator shapes do not match the physical dimension {self.phys_dim}.")
        elif any(idx < 0 or idx >= self.L for idx in operators.keys()):
            raise ValueError(f"Operator indices must be in the range [0, {self.L - 1}].")
        if mode == 'direct':
            return self._full_dir_contract(operators) # uses brute-force contraction of the MPS with the given operators to compute the expectation value.
        elif mode == 'env':
            return self._full_env_contract(operators) # uses canonical condition.

    def norm(self,mode:Literal['direct','env'] = 'env')->complex:
        if mode == 'env':
            return torch.einsum('abc,abc->', self.tensors[self.center].conj(), self.tensors[self.center]).item()
        elif mode == 'direct':
            return self.expectation({})

    def normalize(self):
        '''
        Normalize the MPS by dividing each tensor by the square root of the norm.
        '''
        norm:complex = self.norm(mode='env')
        self.tensors[self.center] /= sqrt(norm.real)

    @torch.no_grad()
    def _mpo_expectation(self, mpo) -> complex:
        '''
        Compute <psi|O|psi> by contracting MPS* @ MPO @ MPS from left to right.
        The MPO tensors follow the convention: tensors[0] = left boundary (carries site-0 ops),
        tensors[1..L-1] = site 1..L-1 operators, tensors[L] = right boundary.
        '''
        # Site 0: use MPO tensors[0] (left boundary + site-0 operators)
        lenv = torch.einsum('bpa,dqc,uwpq->acw',
                            self.tensors[0].conj(), self.tensors[0], mpo.tensors[0])
        for i in range(1, self.L):
            lenv = torch.einsum('bdw,bpa,dqc,wxpq->acx',
                                lenv, self.tensors[i].conj(), self.tensors[i], mpo.tensors[i])
        # lenv shape: (1, 1, D); pick the MPO end state (index -1)
        return lenv[0, 0, -1].item()

    @torch.no_grad()
    def _full_dir_contract(self,operators:dict[int,torch.Tensor]):
        '''
        Brute-force contraction of the MPS with the given operators to compute the expectation value.
        '''
        try:
            op = operators[0]
            lenv = torch.einsum('aib,ajc,ij->bc', self.tensors[0].conj(), self.tensors[0], op)
        except KeyError:
            lenv = torch.einsum('aib,aic->bc', self.tensors[0].conj(), self.tensors[0])
        for i in range(1,self.L):
            try:
                op = operators[i]
                lenv = torch.einsum('ab,aic,bjd,ij->cd', lenv, self.tensors[i].conj(), self.tensors[i], op)
            except KeyError:
                lenv = torch.einsum('ab,aic,bid->cd', lenv, self.tensors[i].conj(), self.tensors[i])
        return torch.trace(lenv).item()


    @torch.no_grad()
    def _full_env_contract(self, operators: dict[int, torch.Tensor]):
        '''
        Compute the expectation value of a list of operators on the MPS using the canonical form.
        '''
        if not operators: # If the operator list is empty, return the norm of the MPS.
            return self.norm(mode='env')

        leftmost = min(operators.keys())
        rightmost = max(operators.keys())

        self.move_center_to(leftmost)
        lenv = torch.einsum('aic,ajd,ij->cd',
                                    self.tensors[leftmost].conj(),
                                    self.tensors[leftmost],
                                    operators[leftmost])
        for i in range(leftmost + 1, rightmost + 1):
            try:
                op = operators[i]
                lenv = torch.einsum('ab,aic,bjd,ij->cd', lenv,
                                    self.tensors[i].conj(), self.tensors[i], op)
            except KeyError:
                lenv = torch.einsum('ab,aic,bid->cd', lenv,
                                    self.tensors[i].conj(), self.tensors[i])
        return torch.trace(lenv).item()

    @torch.no_grad()
    def entanglement_entropy(self,bond:int,alpha:float|None = None, svd_tol:float = 1e-14)-> float:
        '''
        Compute the entanglement entropy of ith and i+1th site.
        Args:
            bond (int): The bond index between sites i and i+1 (0 <= bond < L-1).
            alpha (float | None): The Renyi entropy order. If None, the von Neumann entropy is computed. (default: None)
            svd_tol (float): Tolerance for singular value truncation. (default: 1e-14)
        '''
        if bond < 0 or bond >= self.L - 1:
            raise IndexError(f"Bond index {bond} is out of bounds for MPS of length {self.L}.")
        self.move_center_to(bond)
        s = torch.linalg.svdvals(self.tensors[bond].reshape(-1, self.bond_dim[bond + 1]))
        s = s[s > svd_tol]
        if alpha is None:
            return -torch.sum(s**2 * torch.log(s**2)).item()
        else:
            return 1/(1-alpha) * torch.log(torch.sum(s**(2*alpha))).item()


    @torch.no_grad()
    def _left_canonicalize_site(self, i, update_center=True):
        '''
        Canonicalize the MPS site at index i to the left canonical form using QR decomposition. Optionally update the center site index.
                |          |           |
        --i-1--A[i-1]--i-- A[i]-i+1- A[i+1]-i+2-
        =>
                |        |           |
        -i-1--A[i-1]--i--Q---R-i+1- A[i+1]-i+2-
        Args:
            i (int): Index of the site to canonicalize (0 <= i < L-1).
            update_center (bool): Whether to update the center site index after canonicalization (default: True).
        Raises:
            IndexError: If the index i is out of bounds for the MPS length.
        '''
        if i < 0 or i >= self.L-1:
            raise IndexError(f"Index {i} is out of bounds for MPS of length {self.L}.")
        self.tensors[i] = self.tensors[i].reshape(-1, self.bond_dim[i + 1])
        q,r = torch.linalg.qr(self.tensors[i])
        new_bond = q.shape[-1]
        self.bond_dim[i+1] = new_bond
        self.tensors[i] = q.reshape(self.bond_dim[i],self.phys_dim,new_bond) # (bond_dim[i], phys_dim, new_bond)
        self.tensors[i+1] = torch.einsum('ab,bcd->acd', r, self.tensors[i+1])
        if update_center:
            self.center = i + 1
        del q,r

    @torch.no_grad()
    def _right_canonicalize_site(self, i, update_center=True):
        '''
        Canonicalize the MPS site at index i to the right canonical form using RQ decomposition. Optionally update the center site index.
                |          |           |
        --i-1--A[i-1]--i-- A[i]-i+1- A[i+1]-i+2-
        =>
                |              |          |
        -i-1--A[i-1]--i--R.T---Q.T-i+1- A[i+1]-i+2-
        Args:
            i (int): Index of the site to canonicalize (1 <= i < L).
            update_center (bool): Whether to update the center site index after canonicalization (default: True).
        Raises:
            IndexError: If the index i is out of bounds for the MPS length.
        '''
        if i <= 0 or i >= self.L:
            raise IndexError(f"Index {i} is out of bounds for MPS of length {self.L}.")
        self.tensors[i] = self.tensors[i].reshape(self.bond_dim[i], -1)
        q,r = torch.linalg.qr(self.tensors[i].H)
        q,r = q.H, r.H
        self.bond_dim[i] = q.shape[0]
        self.tensors[i] = q.reshape(self.bond_dim[i], self.phys_dim, self.bond_dim[i+1])
        self.tensors[i-1] = torch.einsum('lpb,br->lpr', self.tensors[i-1], r)
        if update_center:
            self.center = i - 1
        del r, q

    def canonicalize(self):
        '''
        Canonicalize the MPS in the specified mode.
        '''
        for i in range(self.center): # to center -1
            self._left_canonicalize_site(i, update_center=True)

        for i in range(self.L - 1, self.center, -1):   # i = L-1, L-2,...,self.center+1
            self._right_canonicalize_site(i, update_center=True)
    def copy(self):
        '''
        Create a deep copy of the MPS instance.
        '''
        return MPS(self.L,
                      self.phys_dim,
                      self.bond_dim.copy(),
                      [t.clone().detach() for t in self.tensors],
                      self.dtype,
                      self.device)
    def __repr__(self):
        return f"MPS(L={self.L}, phys_dim={self.phys_dim}, bond_dim={self.bond_dim}, center={self.center})"
    def __sizeof__(self):
        return sum(t.__sizeof__() for t in self.tensors) * self.dtype.itemsize

    @property
    def shape(self)-> list[int]:
        '''
        Return the dimension of virtual bond of the MPS and the MPO.
        '''
        return self.bond_dim

class MPO:
    '''
    A class representing a Matrix Product Operator (MPO) in quantum physics and tensor networks.
    '''
    @torch.no_grad()
    def __init__(self,
                 L:int,
                 physical_dim:int,
                 mapping:dict[str,torch.Tensor]|None = None,
                 dtype:torch.dtype = torch.complex128,
                 device:torch.device = torch.device('cuda')):
        '''
        Args:
            L (int): Length of the MPO (number of sites).
            physical_dim (int): Physical dimension of the MPO.
            mapping (dict[str, torch.Tensor]): A dictionary mapping operator names to their corresponding torch.Tensor representations. If None, default mappings will be used.
        '''
        self.dtype = dtype
        self.device = device
        self.L = L
        self.couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]] = []
        self.physical_dim = physical_dim
        self.tensors: list[torch.Tensor] = []
        self.bond_dim = 0

        if mapping is None:
            spin_op = SpinOperator(phys_dim=physical_dim, dtype=dtype, device=device).data
            self.mapping = {
                'X': spin_op[0],
                'Y': spin_op[1],
                'Z': spin_op[2],
                'I': torch.eye(physical_dim, dtype=dtype, device=device)
            }
        else:
            self.mapping = mapping

    def to(self, device):
        '''
        Move the MPO tensors to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device)
        for k in self.mapping:
            self.mapping[k] = self.mapping[k].to(device)

    def copy(self):
        '''
        Create a deep copy of the MPO instance.
        '''
        new = MPO(self.L,
                  self.physical_dim,
                  mapping={k: v.clone().detach() for k, v in self.mapping.items()},
                  dtype=self.dtype,
                  device=self.device)
        new.couplings = [(ops, coeff, locs) for ops, coeff, locs in self.couplings]
        if self.tensors:
            new.tensors = [t.clone().detach() for t in self.tensors]
        new.bond_dim = self.bond_dim
        if hasattr(self, 'state_idx'):
            new.state_idx = self.state_idx.copy()
        return new

    def __repr__(self):
        return f"MPO(L={self.L}, physical_dim={self.physical_dim}, bond_dim={self.bond_dim})"
    def __size__(self):
        return sum(t.numel() for t in self.tensors) * self.dtype.itemsize

    # ── Arithmetic dunders ──────────────────────────────────────────────

    def __add__(self, other):
        '''
        MPO + MPO: merge couplings.   MPO + dict: add single-site operators.
        '''
        if isinstance(other, MPO):
            new = self.copy()
            # Copy any missing operator matrices from other into new.mapping
            for op_name in set(op for ops, _, _ in other.couplings for op in ops):
                if op_name not in new.mapping:
                    base = op_name.split('@')[0]
                    mat = new.mapping[base[0]]
                    for ch in base[1:]:
                        mat = mat @ new.mapping[ch]
                    new.mapping[op_name] = mat.to(dtype=new.dtype, device=new.device)
            # Directly merge couplings (they are already merged by add_couplings)
            new.couplings.extend(other.couplings)
            new.build()
            return new
        elif isinstance(other, dict):
            new = self.copy()
            for site, op_mat in other.items():
                name = f"__dict_{len(new.mapping)}"
                new.mapping[name] = op_mat.to(dtype=new.dtype, device=new.device)
                new.add_couplings((name,), 1.0, (site,))
            new.build()
            return new
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, dict):
            return self.__add__(other)
        return NotImplemented

    def __mul__(self, other):
        '''
        MPO * scalar: scale couplings.
        MPO * MPO:   operator composition (contract physical indices).
        '''
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            if new.tensors:
                # scale the tensors directly (simpler for already-built MPOs)
                for i in range(len(new.tensors)):
                    new.tensors[i] = new.tensors[i] * other
            else:
                new.couplings = [(ops, coeff * other, locs) for ops, coeff, locs in new.couplings]
                new.build()
            return new
        elif isinstance(other, MPO):
            # MPO * MPO: contract physical indices, outer-product the bond dims
            new = MPO(self.L, self.physical_dim,
                      mapping={k: v.clone().detach() for k, v in self.mapping.items()},
                      dtype=self.dtype, device=self.device)
            new.tensors = []
            for i in range(len(self.tensors)):
                t1 = self.tensors[i]
                t2 = other.tensors[i]
                # t1: (wL1, wR1, p, q), t2: (wL2, wR2, q, r)  →  (wL1*wL2, wR1*wR2, p, r)
                wL1, wR1 = t1.shape[0], t1.shape[1]
                wL2, wR2 = t2.shape[0], t2.shape[1]
                new_t = torch.einsum('abpq,cdqr->acbdpr', t1, t2).reshape(
                    wL1 * wL2, wR1 * wR2, self.physical_dim, self.physical_dim)
                new.tensors.append(new_t)
            new.bond_dim = self.bond_dim * other.bond_dim
            return new
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1.0)

    def __sub__(self, other):
        return self.__add__((-1.0) * other)

    # ── Coupling management ─────────────────────────────────────────────

    def add_couplings(self,operators:tuple[str,...],coeff:complex,locations:NDArray[np.int64]|tuple[int,...]):
        '''
        Add new couplings to the MPO.
        Args:
            operators (tuple[str]): A tuple of operator names (e.g., ('X', 'Y')).
            coeffs (complex): A complex coefficient for the coupling.
            locations (tuple[int]): A tuple of site indices where the operators act.
        Raises:
            ValueError: If the length of operators does not match the number of locations,
                        or if any site index is out of bounds for the MPO length,
                        or if any operator is not defined in the mapping.
        '''
        if len(operators) != len(locations):
            raise ValueError(f"Length of operators {len(operators)} does not match the number of locations {len(locations)}.")
        if any(loc < 0 or loc >= self.L for loc in locations):
            raise ValueError(f"Site indices {locations} are out of bounds for MPO of length {self.L}.")
        if not all(op in self.mapping for op in operators):
            raise ValueError(f"Operators {operators} are not defined in the mapping. Available operators: {list(self.mapping.keys())}.")

        # Merge consecutive operators on the same site
        merged_ops = []
        merged_locs = []
        current_op = operators[0]
        current_loc = locations[0]
        for op, loc in zip(operators[1:], locations[1:]):
            if loc == current_loc:
                current_op = current_op + op
            else:
                merged_ops.append(self._register_merged_op(current_op))
                merged_locs.append(current_loc)
                current_op = op
                current_loc = loc
        merged_ops.append(self._register_merged_op(current_op))
        merged_locs.append(current_loc)
        self.couplings.append((tuple(merged_ops), coeff, tuple(merged_locs)))

    def _register_merged_op(self, op_string: str) -> str:
        '''
        Register a merged operator matrix if not already present, return its name.
        '''
        if op_string not in self.mapping:
            base = op_string.split('@')[0]  # e.g. "XY@0" → "XY"
            mat = self.mapping[base[0]]
            for ch in base[1:]:
                mat = mat @ self.mapping[ch]
            self.mapping[op_string] = mat.to(dtype=self.dtype, device=self.device)
        return op_string

    @staticmethod
    def _state_key(c_idx: int, ops_prefix: tuple[str, ...], gap: int) -> tuple:
        '''Build a state key.

        gap = 1 states are shared across couplings (safe because NN
        couplings have no identity pass-through that could create cross-terms).
        gap > 1 states are unique per coupling to prevent spurious paths.
        '''
        if gap == 1:
            return ops_prefix + ('__g1',)
        return (c_idx,) + ops_prefix + (f'__g{gap}',)

    def build(self):
        '''
        Build the MPO tensors from the current couplings and mapping.
        '''
        # ── enumerate intermediate states ──
        state_gaps: dict[tuple, int] = {}  # key → gap
        for c_idx, (ops, _, locs) in enumerate(self.couplings):
            for n in range(len(ops) - 1):
                gap = locs[n + 1] - locs[n]
                key = self._state_key(c_idx, ops[:n + 1], gap)
                state_gaps[key] = max(state_gaps.get(key, 0), gap)

        sorted_keys = sorted(state_gaps, key=lambda x: (len(x), x))
        self.state_idx = {k: i + 1 for i, k in enumerate(sorted_keys)}
        self.bond_dim = len(self.state_idx) + 2  # start (0) + end (D-1)

        # ── allocate tensors ──
        D = self.bond_dim
        p = self.physical_dim
        self.tensors = [torch.zeros((1, D, p, p), dtype=self.dtype, device=self.device)]
        self.tensors += [torch.zeros((D, D, p, p), dtype=self.dtype, device=self.device)
                         for _ in range(1, self.L)]
        self.tensors += [torch.zeros((D, 1, p, p), dtype=self.dtype, device=self.device)]

        # ── identity diagonals ──
        eye = torch.eye(p, dtype=self.dtype, device=self.device)
        self.tensors[0][0, 0] = eye                       # site-0 start
        self.tensors[self.L][D - 1, 0] = eye               # right boundary
        for i in range(1, self.L):
            self.tensors[i][0, 0] = eye                    # bulk start
            self.tensors[i][D - 1, D - 1] = eye            # bulk end

        # identity on skipped sites for long-range couplings
        for key, gap in state_gaps.items():
            if gap <= 1:
                continue
            s = self.state_idx[key]
            c_idx = key[0]                                 # unique key → c_idx
            _, _, locs = self.couplings[c_idx]
            for n in range(len(locs) - 1):
                if locs[n + 1] - locs[n] != gap:
                    continue
                for k in range(locs[n] + 1, locs[n + 1]):
                    if 0 < k < self.L:
                        self.tensors[k][s, s] = eye

        # ── place operator matrices ──
        for c_idx, (ops, coeff, locs) in enumerate(self.couplings):
            for n, (op_str, site) in enumerate(zip(ops, locs)):
                mat = self.mapping[op_str]
                if n == 0:
                    mat = coeff * mat

                state_in = 0 if n == 0 else \
                    self.state_idx[self._state_key(c_idx, ops[:n], locs[n] - locs[n - 1])]
                state_out = D - 1 if n == len(ops) - 1 else \
                    self.state_idx[self._state_key(c_idx, ops[:n + 1], locs[n + 1] - locs[n])]

                if site == 0:
                    self.tensors[0][0, state_out] += mat
                elif site == self.L:
                    self.tensors[self.L][state_in, 0] += mat
                else:
                    self.tensors[site][state_in, state_out] += mat
    def __sizeof__(self):
        return sum(t.__sizeof__() for t in self.tensors) * self.dtype.itemsize


class Broomstick:
    '''
    A class that optimize an MPS given an MPO Hamiltonian.

    Performs 2-site DMRG sweeping on the MPS to minimize the energy with respect to the MPO Hamiltonian. The MPS and MPO must be initialized and built externally before being passed in.

    '''
    def __init__(self,
                 mps: MPS,
                 mpo: MPO,
                 max_bond_dim: int = 50,
                 svd_tol: float = 1e-14,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        self.state = mps
        self.Hamiltonian = mpo

        if self.state.L != self.Hamiltonian.L:
            raise ValueError(f"MPS length {self.state.L} does not match MPO length {self.Hamiltonian.L}.")
        self.L = mps.L
        
        self.physical_dim = mps.phys_dim
        if self.physical_dim != self.Hamiltonian.physical_dim:
            raise ValueError(f"MPS physical dimension {self.physical_dim} does not match MPO physical dimension {self.Hamiltonian.physical_dim}.")
        
        self.max_bond_dim = max_bond_dim
        self.svd_tol = svd_tol
        self.device = device if device is not None else mps.device
        self.dtype = dtype if dtype is not None else mps.dtype

        self.center = 0
        self.renv = []
        self.lenv = []

    def to(self, device):
        '''
        Move the DMRG engine state and Hamiltonian to the specified device.
        '''
        self.device = device
        self.Hamiltonian.to(device)
        self.state.to(device)

    def copy(self):
        '''
        Create a deep copy of the DMRG_engine instance.
        '''
        new = Broomstick(self.state.copy(),
                          self.Hamiltonian.copy(),
                          max_bond_dim=self.max_bond_dim,
                          svd_tol=self.svd_tol,
                          device=self.device,
                          dtype=self.dtype)
        return new

    def cache_envs(self):
        L = self.L
        self.renv:list[torch.Tensor] = [None] * (L + 1)
        self.lenv:list[torch.Tensor] = [None] * (L + 1)

        # Right boundary environment: MPO bond dim = Hamiltonian.bond_dim, final state (D-1) set to 1
        self.renv[L] = torch.zeros(1, self.Hamiltonian.bond_dim, 1,
                                   dtype=self.dtype, device=self.device)
        self.renv[L][0, -1, 0] = 1.0
        for i in range(L - 1, 1, -1): # from L-1 to 2
            # renv[i] = A[i]^* ⊗ W[i] ⊗ A[i] ⊗ renv[i+1]
            A_conj = self.state.tensors[i].conj()
            A_i = self.state.tensors[i]
            self.renv[i] = cached_einsum('apb,cqd,bjd,ijpq->aic', 
                                         self.state.tensors[i].conj(), 
                                         self.state.tensors[i],
                                         self.renv[i + 1],
                                         self.Hamiltonian.tensors[i])
        self.lenv[0] = torch.ones(1, 1, 1, dtype=self.dtype, device=self.device)

    def move_center_to(self, new_center:int):
        '''
        Move the center site index to a new position in the MPS and update the left and right environment tensors accordingly.
        Args:
            new_center (int): The new center site index (0 <= new_center < L).
        Raises:
            IndexError: If the new center index is out of bounds for the MPS length.
        '''
        if new_center == self.center:
            return  # No need to move if the new center is the same as the current center
        if new_center < 0 or new_center >= self.L-1:
            raise IndexError('New center index is out of bounds for the MPS length.')

        if new_center < self.center:
            while self.center != new_center:  # move to the left
                self.state._right_canonicalize_site(self.center+1, update_center=True)
                self.state.move_center_to(self.center - 1)                
                expr = _get_expr('apb,cqd,bjd,ijpq->aic',
                                 self.state.tensors[self.center + 1].shape, 
                                 self.state.tensors[self.center + 1].shape,
                                 self.renv[self.center + 2].shape,
                                 self.Hamiltonian.tensors[self.center + 1].shape)
                self.renv[self.center + 1] = cached_einsum('apb,cqd,bjd,ijpq->aic',
                    self.state.tensors[self.center + 1].conj(), 
                    self.state.tensors[self.center + 1],
                    self.renv[self.center + 2],
                    self.Hamiltonian.tensors[self.center + 1])
                self.center -= 1

        elif new_center > self.center:
            while self.center != new_center: # move to the right
                self.state.move_center_to(self.center + 1)
                self.lenv[self.center + 1] = cached_einsum('aic,apb,cqd,ijpq->bjd',
                    self.lenv[self.center], 
                    self.state.tensors[self.center].conj(), 
                    self.state.tensors[self.center],
                    self.Hamiltonian.tensors[self.center])
                self.center += 1
        self.center = new_center

    def sweep(self, num_sweeps: int = 5, compute_variance: bool = False):
        '''
        Perform the Density Matrix Renormalization Group (DMRG) algorithm to find
        the ground state of the spin chain Hamiltonian.

        Each sweep renders a multi-line progress display (flush area) that
        overwrites itself in-place in the terminal.  The final summary of
        each sweep covers the flush area before the next sweep begins.

        Args:
            num_sweeps: Number of full DMRG sweeps (right + left passes).
            compute_variance: If True, compute <H²> - <H>² at the end
        '''
        import sys

        self.cache_envs()
        for sweep in range(num_sweeps):
            sys.stderr.write(f'── Sweep {sweep + 1}/{num_sweeps} ──\n')
            sys.stderr.flush()
            E, trunc = self.singlesweep()
            print(f'Sweep {sweep + 1}/{num_sweeps}: E = {E / self.L:.12f}, max_trunc_err = {trunc:.3e}')
        if compute_variance:
            print(f'DMRG stops at sweep {sweep + 1}/{num_sweeps}: energy = {E / self.L:.12f}, variance of energy = {self.compute_variance() / self.L:.3e}')
        else:
            print(f'DMRG stops at sweep {sweep + 1}/{num_sweeps}: energy = {E / self.L:.12f}')

    @torch.no_grad()
    def singlesweep(self):
        '''
        Perform a single sweep of the DMRG algorithm, optimizing the MPS tensors
        at each site.  Returns (energy, max_truncation_error).

        Renders a multi-line progress display that overwrites itself in-place
        (flush area).  The final summary line covers the flush area on exit.

        Environment tensors are updated incrementally without QR
        canonicalization: the SVD inside update() already guarantees that the
        MPS tensors are left- / right-isometric as the sweep progresses.

        The sweep order is right (0 → L-1) then left (L-2 → 0).  The center site index is updated accordingly, and the MPS is left in a canonical form at the end of the sweep.
        '''
        if not self.lenv or not self.renv:
            self.cache_envs()
        max_trunc = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        max_trunc_float = 0.0          # cached CPU value for progress display
        E_float = 0.0                   # cached energy for progress display

        total_steps = (self.L - 1) + (self.L - 2)  # right + left passes
        step = 0

        SYNC_INTERVAL = 8 # avoid GPU→CPU sync on every step; sync every 8 steps for progress display

        # ── right sweep ──
        for i in range(self.L - 1):
            E, trunc_err = self.update(direction='right')
            max_trunc = torch.maximum(max_trunc, trunc_err)
            step += 1
            if step % SYNC_INTERVAL == 0:
                E_float = E.item()
            if i < self.L - 2:
                self.lenv[i + 1] = cached_einsum('aic,apb,cqd,ijpq->bjd',
                                                 self.lenv[i], 
                                        self.state.tensors[i].conj(), 
                                        self.state.tensors[i],
                                        self.Hamiltonian.tensors[i])
            self.center = i + 1

            bar = self._progress_bar(step, total_steps, E_float / self.L)
            self._flush_render(
                f"    {bar}",
                f"    bond_dim @ center = {self.state.bond_dim[self.center]:4d}",
                f"    max_trunc_err     = {max_trunc_float:.3e}",
            )

        max_trunc_float = max_trunc.item()
        E_float = E.item()

        # special handling for the last site: renormalize the last tensor to be right-isometric
        s_norm = torch.linalg.norm(self.state.tensors[self.L - 1], dim=(1, 2))
        s_norm = torch.clamp(s_norm, min=1e-30)
        self.state.tensors[self.L - 1] = self.state.tensors[self.L - 1] / s_norm[:, None, None]
        self.state.tensors[self.L - 2] = (
            self.state.tensors[self.L - 2] * s_norm[None, None, :]
        )
        self.center = self.L - 2
        self.state.center = self.L - 2

        # ── recompute renv[L-1] using the now-right-isometric site L-1 ──
        
        self.renv[self.L - 1] = cached_einsum('apb,cqd,bjd,ijpq->aic',
                                        self.state.tensors[self.L - 1].conj(), 
                                     self.state.tensors[self.L - 1],
                                     self.renv[self.L],
                                     self.Hamiltonian.tensors[self.L - 1])

        # ── left sweep ──
        for i in range(self.L - 3, -1, -1):
            self.center = i
            self.state.center = i
            E, trunc_err = self.update(direction='left')
            max_trunc = torch.maximum(max_trunc, trunc_err)
            step += 1
            if step % SYNC_INTERVAL == 0:
                E_float = E.item()
            if i > 0:
                self.renv[i + 1] = cached_einsum('apb,cqd,bjd,ijpq->aic',
                                                 self.state.tensors[i + 1].conj(), 
                                        self.state.tensors[i + 1],
                                        self.renv[i + 2],
                                        self.Hamiltonian.tensors[i + 1])

            bar = self._progress_bar(step, total_steps, E_float / self.L)
            self._flush_render(
                f"  {bar}",
                f"    bond_dim @ center = {self.state.bond_dim[self.center + 1]:4d}",
                f"    max_trunc_err      = {max_trunc_float:.3e}",
            )

        self._flush_finish()
        # final sync at end of sweep
        max_trunc_float = max_trunc.item()
        E_float = E.item()
        return E_float, max_trunc_float

    @torch.no_grad()
    def _lanczos(self,
                 i: int,
                 v0: torch.Tensor,
                 n_iter: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Lanczos diagonalisation of the effective Hamiltonian at bond *i*.

        Args:
            i: center site index.
            v0: 4‑leg tensor ``(bond_dim[i], phys_dim, phys_dim, bond_dim[i+2])``.
            n_iter: maximum number of Lanczos iterations.
        Returns:
            ``(E0, v_ground)`` where v_ground has the same 4‑leg shape as v0.
        """

        v0 = v0 / torch.linalg.norm(v0)

        # store Krylov vectors as flat arrays for efficient linear algebra
        vecs = torch.zeros((n_iter + 1, *v0.shape), dtype=v0.dtype, device=v0.device)
        alphas = torch.zeros(n_iter, dtype=torch.float64, device=v0.device)
        betas = torch.zeros(n_iter, dtype=torch.float64, device=v0.device)

        # pre computed contraction
        # usually D > phisycal_dim^2, so we can save some time by precomputing the left and right environments
        lenv = torch.einsum('apb,pqij->abqij', 
                            self.lenv[i],
                            self.Hamiltonian.tensors[i])
        renv = torch.einsum('crd,qrkl->cdqkl',
                            self.renv[i + 2],
                            self.Hamiltonian.tensors[i + 1])
        def helper(v: torch.Tensor) -> torch.Tensor:
            return cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, v)
    
        vecs[0] = v0

        for k in range(n_iter):
            v_k = vecs[k]
            w = helper(v_k)

            alpha = torch.sum(v_k.conj() * w).real
            alphas[k] = alpha

            if k == 0:
                w = w - alpha * v_k
            else:
                w = w - alpha * v_k - betas[k - 1] * vecs[k - 1]

            beta = torch.linalg.norm(w).real
            betas[k] = beta

            vecs[k + 1] = w / beta
            vecs[k + 1][beta == 0] = 0
        # for performance reasons, we will NOT check for convergence and run
        # all n_iter iterations to avoid GPU→CPU syncs.  The caller can check
        # the eigenvalue convergence if desired.

        T = torch.diag(alphas) + torch.diag(betas[:n_iter - 1], diagonal=1) \
                               + torch.diag(betas[:n_iter - 1], diagonal=-1)

        eigvals, eigvecs_T = torch.linalg.eigh(T)

        E0 = eigvals[0]
        v_ground = torch.einsum('a,aijkl->ijkl', 
                                eigvecs_T[:n_iter, 0].to(dtype=vecs.dtype), 
                                vecs[:n_iter,...])

        return E0, v_ground

    def update(self, direction: Literal['left', 'right'], n_iter: int = 8) -> tuple['torch.Tensor', 'torch.Tensor']:
        '''
        Perform 2-site optimization of MPS with effective Hamiltonian.
        Returns (energy_tensor, truncation_error_tensor) — both are 0-d GPU
        tensors to avoid GPU→CPU syncs.  The caller is responsible for
        syncing only when needed (e.g. for progress display).
        Args:
            direction: 'left' or 'right', indicating the sweep direction.
            n_iter: Maximum number of Lanczos iterations for the effective Hamiltonian diagonalization.
        '''
        i = self.center

        assert self.center == self.state.center, "Center site index mismatch between DMRG_engine and MPS."

        # step 1, form the current 2-site state (warm start for Lanczos)
        state = cached_einsum('bjx,xld->bjld',self.state.tensors[i],
                     self.state.tensors[i + 1])

        # step 2, matrix-free Lanczos — Heff is never materialised
        E, v = self._lanczos(i, state, n_iter=n_iter)
        # v is 4‑leg, same shape as state

        # step 3, svd and update the MPS tensors
        p = self.state.phys_dim
        u, s, vh = torch.linalg.svd(
            v.reshape(self.state.bond_dim[i] * p, p * self.state.bond_dim[i + 2]), full_matrices=False)

        s2_total = torch.sum(s * s)
        mask = (s > self.svd_tol)
        s = s[mask]
        u = u[:, mask]
        vh = vh[mask, :]

        new_bond = min(len(s), self.max_bond_dim)
        s = s[:new_bond]
        u = u[:, :new_bond]
        Vh = vh[:new_bond, :]

        s2_kept = torch.sum(s * s)
        trunc_err = torch.where(s2_total > 0, 1.0 - s2_kept / s2_total,
                                torch.tensor(0.0, device=s.device, dtype=torch.float64))

        s /= torch.sqrt(s2_kept)
        if direction == 'right':
            self.state.tensors[i] = u.reshape(self.state.bond_dim[i], self.state.phys_dim, new_bond)
            self.state.tensors[i + 1] = (s.to(Vh.dtype).unsqueeze(1) * Vh).reshape(new_bond, self.state.phys_dim, self.state.bond_dim[i + 2])
            self.state.center = i + 1  # singular values absorbed into site i+1

        elif direction == 'left':
            self.state.tensors[i] = (u * s.to(u.dtype).unsqueeze(0)).reshape(
                self.state.bond_dim[i], 
                self.state.phys_dim, 
                new_bond)
            self.state.tensors[i + 1] = Vh.reshape(
                new_bond, 
                self.state.phys_dim, 
                self.state.bond_dim[i + 2])
            self.state.center = i  # singular values stay at site i
        self.state.bond_dim[i + 1] = new_bond
        return E.real, trunc_err

    @property
    @torch.no_grad()
    def energy(self)->complex:
        '''
        Compute the energy of the current MPS state with respect to the Hamiltonian MPO.
        '''
        lenv = cached_einsum('apc,pqij,aib,cjd->bqd',
                             self.lenv[self.center],
                         self.Hamiltonian.tensors[self.center],
                         self.state.tensors[self.center].conj(), 
                         self.state.tensors[self.center])
        renv = cached_einsum('apc,qpij,bia,djc->bqd',self.renv[self.center + 2],
                         self.Hamiltonian.tensors[self.center + 1],
                         self.state.tensors[self.center + 1].conj(), 
                         self.state.tensors[self.center + 1])

        return torch.einsum('bqd,bqd->', lenv, renv).item()
    
    def __sizeof__(self):
        return self.state.__sizeof__() + \
                self.Hamiltonian.__sizeof__() + \
                (sum(t.__sizeof__() for t in self.lenv) +
                sum(t.__sizeof__() for t in self.renv)) * self.dtype.itemsize

    def _flush_render(self, *lines: str):
        '''Render multiple flush lines, overwriting the previous flush block in-place.
        Uses ANSI escape codes to move the cursor up and overwrite previously
        rendered lines. Call _flush_finish() to move past the flush area when done.
        Args:
            *lines: One string per line of output. Must always be called with the
                    same number of lines within a flush session for correct behaviour.
        '''
        n = len(lines)
        prev = getattr(self, '_flush_count', 0)
        if prev > 0:
            sys.stderr.write(f'\033[{prev}F')
        for line in lines:
            sys.stderr.write(f'\033[K{line}\n')
        sys.stderr.flush()
        self._flush_count = n

    def _flush_finish(self):
        '''Clear the flush area and move the cursor past it.'''
        import sys
        n = getattr(self, '_flush_count', 0)
        if n > 0:
            sys.stderr.write(f'\033[{n}F')
            sys.stderr.write('\033[0J')
            sys.stderr.flush()
            self._flush_count = 0

    @staticmethod
    def _progress_bar(step: int, total: int,
                      energy: float, bar_width: int = 30):
        '''Return a compact progress-bar string (single line).  Used as a building
        block by the multi-line flush display; does NOT write to stderr itself.'''
        frac = step / total
        filled = int(bar_width * frac)
        bar = '█' * filled + '░' * (bar_width - filled)
        return (f"Step {step}/{total} |{bar}| {frac * 100:3.0f}%  "
                f"E={energy:.8f}")
    def compute_variance(self) -> float:
        '''
        Compute var(H) = <H²> - <H>² for the current state.
        Contracts <ψ|H²|ψ> directly without building the H² MPO explicitly,
        keeping memory O(B²·D²) instead of O(D⁴).
        '''
        W = self.Hamiltonian.tensors
        A = self.state.tensors
        lenv = cached_einsum('bpa,uwpr,vzrq,dqc->acwz',A[0].conj(), W[0], W[0], A[0])
        for i in range(1, self.L):
            lenv = cached_einsum('bdxy,bpa,xwpr,yzrq,dqc->acwz',
                                 lenv, A[i].conj(), W[i], W[i], A[i])
        # lenv shape: (1, 1, D, D) → pick the final MPO states
        E2 = lenv[0, 0, -1, -1].real.item()

        E = self.state.expectation(self.Hamiltonian)
        var = (E2 - (E.real) ** 2)
        return max(var, 0.0)  # clamp numerical noise below zero
    
    def save_as_pkl(self,filename: str | None = None):
        '''
        Save the current state of the DMRG engine, including the MPS and MPO, to a pickle file.
        The filename is generated based on the current timestamp and the number of sweeps completed.
        '''
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"dmrg_state_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pkl.dump(self, f)
        
        print(f"DMRG engine state saved to {filename}")
    
    @staticmethod
    def load_from_pkl(filename: str) -> 'Broomstick':
        '''
        Load a DMRG engine state from a pickle file.
        Args:
            filename: The path to the pickle file containing the saved DMRG engine state.
        Returns:
            An instance of the Broomstick class with the loaded state.
        '''
        with open(filename, 'rb') as f:
            stick = pkl.load(f)
        return stick
