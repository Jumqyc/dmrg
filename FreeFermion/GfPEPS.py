import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import Tensor
from dataclasses import dataclass

from base import CUDA
from FreeFermion.linalg import randcov

'''
Based on arxiv:2012.04666: C.-M. Jian, B. Bauer, A. Keselman and A. W. W. Ludwig,
"Criticality and entanglement in non-unitary quantum circuits and tensor networks
of non-interacting fermions", Phys. Rev. B 106, 054309 (2022), Sec. III:
fermionic Gaussian tensor networks.

A node carries ``dim_of_node(node)`` Majorana modes and is described by its
covariance matrix ``Gamma_ij = < i/2 [gamma_i, gamma_j] >`` (Eq. (2) of the
reference), which is real, antisymmetric and satisfies ``Gamma^2 = -1`` for a
pure Gaussian state. The modes of a node are ordered as

    [ external modes | bond block 1 | bond block 2 | ... ],

where the bond blocks follow the order of ``graph.edges``. A bond pairs the
k-th mode of the block at its first end with the k-th mode of the block at its
second end and is contracted with the projector
``prod_k (1 + i gamma_k upsilon_k)/2`` (Sec. III.2 of the reference, Eq. (9)).
'''

@dataclass(frozen=True, order=True)
class Bond:
    '''
    Represents a bond between two nodes in the graph.

    Attributes:
        first (int): The smaller node.
        second (int): The larger node.
        bond_dim (int): The number of Majorana modes carried by each end of the
            bond (the ``chi`` of the reference, so the physical bond dimension
            of the leg is ``2**(bond_dim / 2)``).
    '''
    first: int
    second: int
    bond_dim: int

    def __post_init__(self):
        assert self.first < self.second, "first must be less than second."
        assert self.bond_dim > 0, "bond_dim must be a positive integer."

    def other(self, node: int) -> int:
        '''
        Returns the node at the other end of the bond.
        '''
        return self.second if self.first == node else self.first


class Graph:
    def __init__(self,
                 num_nodes: int,
                 edges: tuple[Bond, ...]):
        self.num_nodes = num_nodes
        self.edges = edges

        assert all(0 <= bond.first
                   and bond.second < num_nodes 
                   for bond in edges), \
            "Bond nodes must be within the range of the number of nodes."

    def neighbours(self, node: int) -> list[int]:
        '''
        Returns the nodes that are directly connected to the given node.
        '''
        return sorted({bond.other(node) for bond in self.edges
                       if node in (bond.first, bond.second)})

class GfPEPS:
    def __init__(self, 
                 graph: Graph,
                 ext_dim: list[int],
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = CUDA):
        '''
        graph: An instance of the Graph class representing the structure of the system.
        ext_dim: A list of integers representing the external dimensions for each node.
        dtype: Data type of the covariance matrices.
        device: Device the covariance matrices live on (``cuda`` by default).
        '''
        self.graph = graph
        self.ext_dim = ext_dim
        self.dtype = dtype
        self.device = device
        self.tensors = [
            randcov(self.dim_of_node(node), dtype=dtype, device=device)
            for node in range(self.graph.num_nodes)
        ]

    @classmethod
    def _from_data(cls,
                   graph: Graph,
                   ext_dim: list[int],
                   tensors: list[Tensor]) -> 'GfPEPS':
        '''
        Build a GfPEPS from known covariance matrices, bypassing the random initialization.
        '''
        state = cls.__new__(cls)
        state.graph = graph
        state.ext_dim = list(ext_dim)
        state.dtype = tensors[0].dtype
        state.device = tensors[0].device
        state.tensors = list(tensors)
        return state

    def dim_of_node(self, node: int) -> int:
        return self.ext_dim[node] + sum(
            bond.bond_dim for bond in self.connected_nodes(node)
        )

    def connected_nodes(self, node: int) -> list[Bond]:
        '''
        Returns a list of Bonds that are directly connected to the given node, in
        the order in which they appear in ``graph.edges``. The contractions keep
        that order, so relabelling a node never moves its modes.
        '''
        return [
            bond for bond in self.graph.edges 
            if bond.first == node or bond.second == node
        ]

    def layout(self, node: int) -> list[tuple[Bond, slice]]:
        '''
        Returns the slice of every bond block of a node inside its covariance
        matrix, in the order of ``graph.edges``. The external modes come first.
        '''
        blocks, start = [], self.ext_dim[node]
        for bond in self.connected_nodes(node):
            blocks.append((bond, slice(start, start + bond.bond_dim)))
            start += bond.bond_dim
        return blocks

    def contract_sites(self, i: int, j: int) -> 'GfPEPS':
        '''
        Contract the sites i and j into a single site and fuse its bonds.

        Parallel bonds are fused first, so i and j are joined by a single bond
        with ``chi`` Majorana modes, which is contracted with Eq. (9) of the
        reference (Jian-Bauer-Keselman-Ludwig, Phys. Rev. B 106, 054309 (2022),
        arXiv:2012.04666). The sites are then merged into one, keeping the external
        modes of both:

          1. contract the bond between i and j into ``psi``, which carries the
             open modes of i followed by the open modes of j. The equation is
             evaluated in the block form of the reference,
             ``gamma = [[G_LL, G_LR], [-G_LR^T, G_RR]]`` with the open modes of i
             first and ``upsilon = [[U_LL, U_LR], [-U_LR^T, U_RR]]`` with the
             contracted modes of j first. A bond is contracted with the projector
             ``prod_k (1 + i gamma_k upsilon_k)/2`` of its first and its second
             end, so when i is its second end the sign of the contracted modes of
             i is flipped (they only couple to j);
          2. build the bonds of the new graph: the old ones in the same order,
             with the bond between i and j dropped and the other bonds of i and j
             pointing at the merged site, which takes the label of the smaller
             site. If that exchanges the two ends of such a bond, the sign of its
             block is flipped so that the projector with the site at the other end
             stays the same;
          3. rearrange ``psi`` into the layout of the merged site,
             ``[ ext of i | ext of j | its bond blocks in the order of the new
             bonds ]``;
          4. fuse the parallel bonds that the merge may have created.

        Only the two contracted tensors and the labels change, every other site
        keeps its modes. ``self`` is left unchanged.
        '''
        if not (0 <= i < self.graph.num_nodes and 0 <= j < self.graph.num_nodes):
            raise ValueError(f"Sites must be node indices of the graph, got {i} and {j}.")
        if i == j:
            raise ValueError("A site cannot be contracted with itself.")

        state = self.fuse_parallel_bonds()
        blocks_i, blocks_j = dict(state.layout(i)), dict(state.layout(j))
        bond = next((bond for bond in blocks_i if bond.other(i) == j), None)
        if bond is None:
            raise ValueError(f"Sites {i} and {j} are not connected by a bond.")
        gamma_block, upsilon_block = blocks_i[bond], blocks_j[bond]

        open_i = list(range(gamma_block.start)) + list(range(gamma_block.stop, state.dim_of_node(i)))
        open_j = list(range(upsilon_block.start)) + list(range(upsilon_block.stop, state.dim_of_node(j)))
        n_i, chi = len(open_i), bond.bond_dim

        gamma = state.tensors[i]
        if bond.first != i:
            gamma = gamma.clone()
            gamma[gamma_block, :] *= -1
            gamma[:, gamma_block] *= -1
        upsilon = state.tensors[j]

        dtype, device = gamma.dtype, gamma.device
        n_open = n_i + len(open_j)
        eye = torch.eye(chi, dtype=dtype, device=device)

        kernel = torch.zeros((2 * chi, 2 * chi), dtype=dtype, device=device)
        kernel[:chi, :chi] = gamma[gamma_block][:, gamma_block]
        kernel[:chi, chi:] = eye
        kernel[chi:, :chi] = -eye
        kernel[chi:, chi:] = upsilon[upsilon_block][:, upsilon_block]

        coupling = torch.zeros((n_open, 2 * chi), dtype=dtype, device=device)
        coupling[:n_i, :chi] = gamma[open_i][:, gamma_block]
        coupling[n_i:, chi:] = -upsilon[upsilon_block][:, open_j].T

        psi = torch.zeros((n_open, n_open), dtype=dtype, device=device)
        psi[:n_i, :n_i] = gamma[open_i][:, open_i]
        psi[n_i:, n_i:] = upsilon[open_j][:, open_j]
        psi += coupling @ torch.linalg.solve(kernel, coupling.T)

        in_psi = {i: {mode: p for p, mode in enumerate(open_i)},
                  j: {mode: len(open_i) + p for p, mode in enumerate(open_j)}}

        low, high = min(i, j), max(i, j)

        def relabel(node: int) -> int:
            return node if node < high else node - 1

        edges, parts = [], []
        for old in state.graph.edges:
            if old == bond:
                continue
            if old in blocks_i:
                source, block = i, blocks_i[old]
            elif old in blocks_j:
                source, block = j, blocks_j[old]
            else:
                edges.append(Bond(relabel(old.first), relabel(old.second), old.bond_dim))
                continue
            other = relabel(old.other(source))
            new = Bond(min(low, other), max(low, other), old.bond_dim)
            parts.append((source, block, (old.first == source) != (new.first == low)))
            edges.append(new)

        order, negate = [], []
        order += [in_psi[i][mode] for mode in range(state.ext_dim[i])]
        order += [in_psi[j][mode] for mode in range(state.ext_dim[j])]
        for source, block, flip in parts:
            start = len(order)
            order += [in_psi[source][mode] for mode in range(block.start, block.stop)]
            if flip:
                negate += range(start, len(order))
        index = torch.tensor(order, dtype=torch.long, device=psi.device)
        merged = psi[index][:, index]
        if negate:
            flipped = torch.tensor(negate, dtype=torch.long, device=psi.device)
            merged[flipped, :] *= -1
            merged[:, flipped] *= -1

        ext_dim = [0] * (state.graph.num_nodes - 1)
        tensors = [None] * (state.graph.num_nodes - 1)
        ext_dim[low] = state.ext_dim[i] + state.ext_dim[j]
        tensors[low] = merged
        for node in range(state.graph.num_nodes):
            if node not in (i, j):
                ext_dim[relabel(node)] = state.ext_dim[node]
                tensors[relabel(node)] = state.tensors[node]

        return GfPEPS._from_data(Graph(len(ext_dim), 
                                tuple(edges)),
                                 ext_dim, tensors).fuse_parallel_bonds()

    def fuse_parallel_bonds(self) -> 'GfPEPS':
        '''
        Fuse every group of bonds that connects the same pair of sites into a
        single bond.

        The modes of the fused bond are the blocks of its original bonds
        concatenated in the order of ``graph.edges``, which is the same order at
        either end, so the pairing and the state are unchanged and only the layout
        of the covariance matrices moves. ``self`` is left unchanged.
        '''
        pairs = {}
        for bond in self.graph.edges:
            pairs.setdefault((bond.first, bond.second), []).append(bond)
        edges = tuple(Bond(first, second, sum(bond.bond_dim for bond in group))
                      for (first, second), group in pairs.items())

        tensors = []
        for node in range(self.graph.num_nodes):
            blocks = {}
            for bond, block in self.layout(node):
                blocks.setdefault(bond.other(node), []).append(block)
            order = list(range(self.ext_dim[node]))
            for group in blocks.values():
                for block in group:
                    order += range(block.start, block.stop)
            tensor = self.tensors[node]
            if order != list(range(tensor.shape[0])):
                index = torch.tensor(order, dtype=torch.long, device=tensor.device)
                tensor = tensor[index][:, index]
            tensors.append(tensor)

        return GfPEPS._from_data(Graph(self.graph.num_nodes, edges), self.ext_dim, tensors)


class GfPEPO:
    def __init__(self, 
                 graph: Graph,
                 ext_dim_in: list[int],
                 ext_dim_out: list[int],
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = CUDA):
        '''
        graph: An instance of the Graph class representing the structure of the system.
        ext_dim_in: A list of integers representing the external dimensions for each node.
        ext_dim_out: A list of integers representing the external dimensions for each node.
        dtype: Data type of the covariance matrices.
        device: Device the covariance matrices live on (``cuda`` by default).
        '''
        self.graph = graph
        self.ext_dim_in = ext_dim_in
        self.ext_dim_out = ext_dim_out
        self.dtype = dtype
        self.device = device
        self.tensors = [
            randcov(self.dim_of_node(node), dtype=dtype, device=device)
            for node in range(self.graph.num_nodes)
        ]

    def dim_of_node(self, node: int) -> int:
        return self.ext_dim_in[node] + self.ext_dim_out[node] + sum(
            bond.bond_dim for bond in self.connected_nodes(node)
        )

    def connected_nodes(self, node: int) -> list[Bond]:
        '''
        Returns a list of Bonds that are directly connected to the given node, in
        the order in which they appear in ``graph.edges``.
        '''
        return [
            bond for bond in self.graph.edges 
            if bond.first == node or bond.second == node
        ]

    @classmethod
    def _from_data(cls,
                   graph: Graph,
                   ext_dim_in: list[int],
                   ext_dim_out: list[int],
                   tensors: list[Tensor]) -> 'GfPEPO':
        '''
        Build a GfPEPO from known covariance matrices, bypassing the random initialization.
        '''
        operator = cls.__new__(cls)
        operator.graph = graph
        operator.ext_dim_in = list(ext_dim_in)
        operator.ext_dim_out = list(ext_dim_out)
        operator.dtype = tensors[0].dtype
        operator.device = tensors[0].device
        operator.tensors = list(tensors)
        return operator

    @property
    def ext_dim(self) -> list[int]:
        '''
        Number of external modes of every node, ordered as
        ``[ in modes | out modes | bond blocks ]``. A node that carries the
        external modes of several original nodes keeps those blocks one after
        the other.
        '''
        return [ext_in + ext_out
                for ext_in, ext_out in zip(self.ext_dim_in, self.ext_dim_out)]


def _contract_all(network) -> tuple[Tensor, dict[int, list[int]]]:
    '''
    Contract every bond of a state or operator network and return its covariance
    matrix on the remaining external modes together with the positions of the
    external modes of every original node. The external modes of the resulting
    nodes are ordered as the bonds of the original network keep them, so the
    modes of a node are contiguous and keep their original order.
    '''
    net = GfPEPS._from_data(network.graph, network.ext_dim, network.tensors)
    source = {node: [node] for node in range(network.graph.num_nodes)}
    while net.graph.edges:
        bond = net.graph.edges[0]
        i, j = bond.first, bond.second
        low, high = min(i, j), max(i, j)
        net = net.contract_sites(i, j)
        merged = source[i] + source[j]
        source = {node if node < high else node - 1: nodes
                  for node, nodes in source.items() if node not in (i, j)}
        source[low] = merged

    covariance = torch.block_diag(*[net.tensors[node] for node in sorted(source)])
    positions, start = {}, 0
    for node in sorted(source):
        for original in source[node]:
            positions[original] = list(range(start, start + network.ext_dim[original]))
            start += network.ext_dim[original]
    return covariance, positions


class Sandwich:
    def __init__(self, 
                 state: GfPEPS,
                 operator: GfPEPO,
                 connecting_pts: list[tuple[int, int]]):
        '''
        state: An instance of GfPEPS representing the state.
        operator: An instance of GfPEPO representing the operator.
        connecting_pts: A list of tuples where each tuple contains two integers representing the nodes in the state and operator that are connected.
        '''
        self.state = state
        self.operator = operator
        self.connecting_pts = connecting_pts

        if not all(    0 <= p[0] < self.state.graph.num_nodes
                   and 0 <= p[1] < self.operator.graph.num_nodes 
                   for p in self.connecting_pts):
            raise ValueError("Connecting points must be within the range of the number of nodes in the state and operator.")

    def contract(self)-> float:
        '''
        Contract the state with the operator along the connecting points and
        return the normalized value of ``<psi|O|psi>``.

        Both networks are contracted internally first, so every node contributes a
        covariance matrix on its external modes. A connecting point ``(s, o)``
        pairs the external modes of the state node ``s`` with those of the
        operator node ``o`` in their layout order, so every state mode and every
        operator mode has to appear in exactly one connecting point and the paired
        dimensions have to agree; otherwise the contraction would leave open modes.

        A covariance matrix fixes its Gaussian tensor only up to the normalization
        and phase of that tensor, so the value that the networks determine is the
        normalized magnitude ``|<psi|O|psi>| = det((1 - G_state G_operator)/2)**(1/4)``
        of the two contracted pure Gaussian tensors.
        '''
        state, state_modes = _contract_all(self.state)
        operator, operator_modes = _contract_all(self.operator)

        state_order, operator_order = [], []
        for s, o in self.connecting_pts:
            if len(state_modes[s]) != len(operator_modes[o]):
                raise ValueError(f"The external dimensions of the connected nodes {s} and {o} do not match.")
            state_order += state_modes[s]
            operator_order += operator_modes[o]
        if sorted(state_order) != list(range(state.shape[0])) \
                or sorted(operator_order) != list(range(operator.shape[0])):
            raise ValueError("Every external mode has to appear in exactly one connecting point.")

        order = [None] * state.shape[0]
        for s, o in zip(state_order, operator_order):
            order[s] = o
        operator = operator[order][:, order]

        identity = torch.eye(state.shape[0], dtype=state.dtype, device=state.device)
        value = torch.linalg.det((identity - state @ operator) / 2).clamp(min=0.0)
        return float(value ** 0.25)

    def state_reduction(self)->GfPEPS:
        '''
        Reduce the state to the sites that connect to the operator.

        Repeatedly a site that is not a connecting point but still has a neighbour
        is contracted into that neighbour (preferring a connecting point), and the
        connecting points are relabelled as the labels change. This leaves exactly
        the connecting points, and shrinks a component without a connecting point
        to a single site. ``self.state`` is left unchanged.
        '''
        state = self.state
        keep = {point[0] for point in self.connecting_pts}

        while True:
            victim = next((node for node in range(state.graph.num_nodes)
                           if node not in keep and state.graph.neighbours(node)), None)
            if victim is None:
                return state

            neighbours = state.graph.neighbours(victim)
            partner = next((node for node in neighbours if node in keep), neighbours[0])
            state = state.contract_sites(victim, partner)
            low, high = min(victim, partner), max(victim, partner)
            keep = {low if node in (victim, partner) else node if node < high else node - 1
                    for node in keep}
