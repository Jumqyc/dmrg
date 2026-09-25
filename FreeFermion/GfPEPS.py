import torch

from torch import Tensor
from dataclasses import dataclass

from base import CUDA
from FreeFermion.linalg import RandPureCov, vacuum_covariance

from typing import Optional,NewType

'''
Based on arxiv:2012.04666: C.-M. Jian, B. Bauer, A. Keselman and A. W. W. Ludwig,
"Criticality and entanglement in non-unitary quantum circuits and tensor networks
of non-interacting fermions", Phys. Rev. B 106, 054309 (2022), Sec. III:
fermionic Gaussian tensor networks.

A node carries ``dim_of_node(node)`` Majorana modes and is described by its
covariance matrix ``Gamma_ij = < i/2 [gamma_i, gamma_j] >`` that are grouped into blocks by the legs of the graph. 
'''

@dataclass(frozen=True, order=True)
class Bond:
    '''
    Represents a bond between two nodes in the graph.
    Attributes:
        first (int): The node at the gamma end.
        second (int): The node at the upsilon end.
        bond_dim (int): The number of Majorana modes carried by each end of the
            bond (the ``chi`` of the reference, so the physical bond dimension
            of the leg is ``2**(bond_dim / 2)``).
        index (int): Which of the parallel bonds between the same pair of nodes
            this one is; two parallel bonds are otherwise equal, so the index is
            what makes their legs tellable apart.
    '''
    first: int
    second: int
    bond_dim: int
    index: int = 0

    def __post_init__(self):
        assert self.first != self.second, "the two ends must be different nodes."
        assert self.bond_dim > 0, "bond_dim must be a positive integer."
        assert self.index >= 0, "index must not be negative."

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
        seen:dict[tuple[int, int], int] = {}
        numbered:list[Bond] = []
        for bond in edges:
            pair:tuple[int, int] = (bond.first, bond.second) if bond.first < bond.second else (bond.second, bond.first)
            index:int = seen.get(pair, 0)
            seen[pair] = index + 1 
            numbered.append(Bond(bond.first, bond.second, bond.bond_dim, index))
        self.edges = tuple(numbered)

        assert all(0 <= bond.first < num_nodes and 0 <= bond.second < num_nodes for bond in edges), "Bond nodes must be within the range of the number of nodes."

    def neighbors(self, node: int) -> list[int]:
        '''
        Returns the nodes that are directly connected to the given node.
        '''
        return sorted({bond.other(node) for bond in self.edges
                       if node in (bond.first, bond.second)})

    def bonds(self, node: int) -> list[Bond]:
        '''
        the bonds that meet the given node, which are the legs of that node apart
        from its external modes.
        Args:
            node: the node label.
        Returns:
            The bonds, in the order in which they appear in ``self.edges``.
        '''
        return [bond for bond in self.edges
                if bond.first == node or bond.second == node]

Leg = Optional[Bond]

class Node:
    '''
    One Gaussian tensor of a network: its covariance, stored as blocks that are keyed by legs.  A leg is the external modes (None) or one bond of the graph, and ``blocks[(a, b)]`` holds the (a,b) block, both orders being stored, so the whole set of blocks is exactly the covariance while a block is addressed by the legs it connects instead of by offsets into a matrix.
    '''
    def __init__(self, blocks: dict[tuple[Leg, Leg], Tensor], sizes: dict[Leg, int]):
        self.blocks = blocks
        self.sizes = sizes

    def legs(self) -> list[Leg]:
        '''
        the legs of this node.
        Returns:
            The legs, the external modes first and then the bonds in order. (first external, then the bonds in the order of the graph edges, so that relabelling a node never moves its modes)
        '''
        return sorted(self.sizes, key=lambda leg: (0, None) if leg is None else (1, leg))

    def __getitem__(self, legs: tuple[Leg, Leg]) -> Tensor:
        '''
        one block of the covariance, addressed by the legs it connects.
        Args:
            legs: the two legs, in either order.
        Returns:
            The block, of shape (sizes[first], sizes[second]).
        '''
        return self.blocks[legs]

    @classmethod
    def from_dense(cls, covariance: Tensor, sizes: dict[Leg, int]) -> 'Node':
        '''
        build a node from a dense covariance whose blocks follow the legs of sizes,
        in the order in which sizes lists them; the only place where the layout of a
        dense matrix is read.
        Args:
            covariance: the real antisymmetric covariance, of size sum(sizes.values()).
            sizes: the number of modes of every leg, in the order of the matrix.
        Returns:
            The node holding the blocks.
        '''
        offsets:dict[Leg, slice[int,int,int]] = {}
        position = 0
        for leg in sizes:
            offsets[leg] = slice(position, position + sizes[leg])
            position += sizes[leg]
        legs = list(sizes)
        blocks = {}
        for index, first in enumerate(legs):
            blocks[(first, first)] = covariance[offsets[first], offsets[first]]
            for second in legs[index + 1:]:
                blocks[(first, second)] = covariance[offsets[first], offsets[second]]
                blocks[(second, first)] = covariance[offsets[second], offsets[first]]
        return cls(blocks, sizes)

    def assemble(self, order: list[Leg] | None = None) -> Tensor:
        '''
        the dense covariance of this node with its blocks placed in the given leg
        order, which is what the local linear algebra of a contraction needs.
        Args:
            order: the legs in the order of the matrix, legs() by default.
        Returns:
            The real antisymmetric covariance, of size sum(sizes[leg] for leg in order).
        '''
        legs = self.legs() if order is None else order
        offsets:dict[Leg, slice[int,int,int]] = {}
        position = 0
        for leg in legs:
            offsets[leg] = slice(position, position + self.sizes[leg])
            position += self.sizes[leg]
        block = self.blocks[(legs[0], legs[0])]
        matrix = torch.zeros((position, position), dtype=block.dtype, device=block.device)
        for index, first in enumerate(legs):
            matrix[offsets[first], offsets[first]] = self.blocks[(first, first)]
            for second in legs[index + 1:]:
                matrix[offsets[first], offsets[second]] = self.blocks[(first, second)]
                matrix[offsets[second], offsets[first]] = self.blocks[(second, first)]
        return matrix

    def coupling(self, legs: list[Leg], bond: Bond) -> Tensor:
        '''
        the blocks that couple a set of legs to one bond, stacked in the order of the legs, which is the C of the contraction formula.
        Args:
            legs: the legs to stack.
            bond: the bond they couple to.
        Returns:
            The matrix of shape (sum(sizes[leg] for leg in legs), sizes[bond]).
        '''
        return torch.cat([self[leg, bond] for leg in legs], dim=0)

    def flipped(self, bonds: list[Bond]) -> 'Node':
        '''
        this node with the sign of the basis of the given bonds flipped, which is what turns the basis of the second end of a bond into the one of its first end.  A block of two flipped bonds is quadratic in their modes, so only the blocks that couple exactly one flipped bond to another leg change sign.
        Args:
            bonds: the bonds whose basis is flipped.
        Returns:
            The node in the flipped basis.
        '''
        flipped = set(bonds)
        blocks = {legs: (block if (legs[0] in flipped) == (legs[1] in flipped) else -block)
                  for legs, block in self.blocks.items()}
        return Node(blocks, self.sizes)

    def renamed(self, mapping: dict[Bond, Bond]) -> 'Node':
        '''
        the same blocks under the new names of their legs, which is how a node
        follows a relabelling of the graph.  Every leg keeps its position and its
        blocks, only the keys change.
        Args:
            mapping: old bond -> new bond.
        Returns:
            The node keyed by the new bonds.
        '''
        if not mapping:
            return self
        blocks = {(mapping.get(first, first), mapping.get(second, second)): block
                  for (first, second), block in self.blocks.items()}
        sizes = {mapping.get(leg, leg): size for leg, size in self.sizes.items()}
        return Node(blocks, sizes)

    def glue(self, other: 'Node', bonds: list[Bond], flip: list[Bond]) -> 'Node':
        '''
        glue this node to another along the bonds between them, Eq. (9) of
        Jian-Bauer-Keselman-Ludwig (arXiv:2012.04666):

            psi = blockdiag(G_oo, U_oo) + C K^-1 C^T,   K = [[G_bb, I], [-I, U_bb]],

        with C the coupling of the open legs to the contracted modes and K the kernel
        of the projector prod_k (1 + i gamma_k upsilon_k)/2; the equation holds for
        any set of modes, so all the bonds between the two nodes are contracted at
        once, cross blocks between two of them included, which is what makes a second
        bond between the same two nodes unnecessary to glue on its own.  The projector
        pairs the first end of a bond as gamma with its second end as upsilon, so this
        node, which takes the gamma slot, has its bond basis flipped for the bonds of
        which it is the second end.  A singular K, which two uncorrelated nodes have,
        is handled with the pseudo-inverse: the covariance is well defined there, only
        the solve is not.
        Args:
            other: the node at the other end of the bonds.
            bonds: the bonds to contract, all of them between the two nodes.
            flip: the bonds of which this node is the second end, and which therefore
                have to be flipped to take the gamma slot.
        Returns:
            The glued node, keyed by the external leg (the modes of both nodes) and
            by the surviving bonds of both.
        '''
        self = self.flipped(flip) if flip else self
        contracted = set(bonds)
        open_i = [leg for leg in self.legs() if leg not in contracted]
        open_j = [leg for leg in other.legs() if leg not in contracted]
        n_i = sum(self.sizes[leg] for leg in open_i)
        n_open = n_i + sum(other.sizes[leg] for leg in open_j)
        chi = sum(self.sizes[bond] for bond in bonds)

        block = self.assemble(bonds)
        dtype, device = block.dtype, block.device
        eye = torch.eye(chi, dtype=dtype, device=device)
        k = torch.cat([torch.cat([block, eye], dim=1),
                       torch.cat([-eye, other.assemble(bonds)], dim=1)], dim=0)
        c = torch.zeros((n_open, 2 * chi), dtype=dtype, device=device)
        c[:n_i, :chi] = torch.cat([self.coupling(open_i, bond) for bond in bonds],dim=1)
        # the coupling of the second end enters transposed in the reference formula,
        # and the antisymmetry of its blocks turns that into the same open x bond block
        c[n_i:, chi:] = torch.cat([other.coupling(open_j, bond) for bond in bonds],
                                         dim=1)

        psi = torch.zeros((n_open, n_open), dtype=dtype, device=device)
        psi[:n_i, :n_i] = self.assemble(open_i)
        psi[n_i:, n_i:] = other.assemble(open_j)
        try:
            solved = torch.linalg.solve(k, c.T)
        except torch.linalg.LinAlgError:
            solved = torch.linalg.pinv(k) @ c.T
        psi = psi + c @ solved

        # the merged node keeps the external modes of both nodes first and then the
        # surviving bonds; psi holds them as [gamma open | upsilon open], so the two
        # external blocks simply have to move next to each other
        external = self.sizes[None] + other.sizes[None]
        order = list(range(self.sizes[None])) + list(range(n_i, n_i + other.sizes[None])) \
            + list(range(self.sizes[None], n_i)) + list(range(n_i + other.sizes[None], n_open))
        sizes:dict[Leg, int] = {None: external,
                 **{leg: (self.sizes[leg] if leg in self.sizes else other.sizes[leg])
                    for leg in open_i + open_j if leg is not None}}
        return Node.from_dense(psi[order][:, order], sizes)

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
        self.ext_dim = list(ext_dim)
        self.dtype = dtype
        self.device = device
        self._tensors = None
        dense = [RandPureCov(self.dim_of_node(node), dtype=dtype, device=device)
                 for node in range(graph.num_nodes)]
        self._tensors = dense
        self.nodes = [Node.from_dense(dense[node], self.node_sizes(node))
                      for node in range(self.graph.num_nodes)]

    def node_legs(self, node: int) -> list[Leg]:
        '''
        the legs of a node in the order of its dense covariance.
        Args:
            node: the node label.
        Returns:
            The external leg first, then one leg per bond in the order of graph.edges.
        '''
        return [None] + self.graph.bonds(node)

    def node_sizes(self, node: int) -> dict[Leg, int]:
        '''
        the number of modes of every leg of a node.
        Args:
            node: the node label.
        Returns:
            The sizes, keyed by leg.
        '''
        return {None: self.ext_dim[node],
                **{bond: bond.bond_dim for bond in self.graph.bonds(node)}}

    @classmethod
    def _from_data(cls,
                   graph: Graph,
                   ext_dim: list[int],
                   tensors: list[Tensor]) -> 'GfPEPS':
        '''
        build a GfPEPS from dense node covariances, whose blocks follow the graph and
        ext_dim, bypassing the random initialization.
        '''
        state = cls.__new__(cls)
        state.dtype = tensors[0].dtype
        state.device = tensors[0].device
        state.graph = graph
        state.ext_dim = list(ext_dim)
        state._tensors = list(tensors)
        state.nodes = [Node.from_dense(tensor, state.node_sizes(node))
                       for node, tensor in enumerate(state._tensors)]
        return state

    @property
    def tensors(self) -> list[Tensor]:
        '''
        the dense covariance of every node, assembled from the blocks of the nodes on
        first use.
        Returns:
            The list of node covariances, in node order.
        '''
        if self._tensors is None:
            self._tensors = [node.assemble(self.node_legs(index))
                             for index, node in enumerate(self.nodes)]
        return self._tensors

    def __getitem__(self, node: int) -> Node:
        '''
        the Gaussian tensor of one node, whose blocks are addressed by leg.
        Args:
            node: the node label.
        Returns:
            The node.
        '''
        return self.nodes[node]

    def covariance(self, node: int) -> Tensor:
        '''
        the dense covariance of one node, in the order of node_legs.
        Args:
            node: the node label.
        Returns:
            The real antisymmetric covariance of the node.
        '''
        return self.tensors[node]

    @classmethod
    def from_covariances(cls,
                         graph: Graph,
                         ext_dim: list[int],
                         tensors: list[Tensor]) -> 'GfPEPS':
        '''
        Build a network from known node covariances, checking that every one
        is a real antisymmetric matrix of the size the graph and ext_dim ask for.
        Args:
            graph: the structure of the system.
            ext_dim: external Majorana modes of every node.
            tensors: the node covariances, one per node, in node order.
        Returns:
            The GfPEPS holding those covariances.
        Raises:
            ValueError: if the node counts disagree, or a covariance has the wrong
                size or is not antisymmetric.
        '''
        if len(ext_dim) != graph.num_nodes or len(tensors) != graph.num_nodes:
            raise ValueError(f'expected {graph.num_nodes} nodes, got ext_dim of length '
                             f'{len(ext_dim)} and {len(tensors)} tensors')
        state = cls._from_data(graph, ext_dim, tensors)
        for node, tensor in enumerate(state.tensors):
            size = state.dim_of_node(node)
            if tuple(tensor.shape) != (size, size):
                raise ValueError(f'node {node}: covariance has shape {tuple(tensor.shape)}, '
                                 f'expected {(size, size)}')
            scale = max(1.0, float(tensor.abs().max()))
            if float((tensor + tensor.T).abs().max()) > 1e-9 * scale:
                raise ValueError(f'node {node}: covariance is not antisymmetric')
        return state

    @classmethod
    def from_global_covariance(cls, covariance: Tensor) -> 'GfPEPS':
        '''
        Wrap the covariance of a whole system as a single node without bonds,
        so that a state built elsewhere can be used with the methods of this class.
        Args:
            covariance: real antisymmetric covariance of 2n Majorana modes.
        Returns:
            The single-node GfPEPS on those modes.
        '''
        return cls.from_covariances(Graph(1, ()), [covariance.shape[0]], [covariance])

    @classmethod
    def from_hamiltonian(cls, matrix: Tensor) -> 'GfPEPS':
        '''
        Ground state of the quadratic Hamiltonian H = (i/4) gamma^T M gamma
        as a single node: Gamma = i sign(iM).
        Args:
            matrix: the real antisymmetric Majorana matrix M, shape (2n, 2n).
        Returns:
            The single-node GfPEPS of that ground state.
        Raises:
            ValueError: if the ground state is degenerate, so that Gamma is not
                defined by the sign function.
        '''
        eigenvalues, eigenvectors = torch.linalg.eigh(1j * matrix.to(torch.complex128))
        if float(eigenvalues.abs().min()) < 1e-9 * float(eigenvalues.abs().max()):
            raise ValueError('the Hamiltonian has zero modes, so its ground state is '
                             'degenerate; pass an explicit covariance to '
                             'from_global_covariance')
        covariance = 1j * (eigenvectors * torch.sign(eigenvalues)) @ eigenvectors.conj().T
        if float(covariance.imag.abs().max()) > 1e-9:
            raise ValueError('the ground state is degenerate and Gamma is not real; '
                             'pass an explicit covariance to from_global_covariance')
        return cls.from_global_covariance(covariance.real)

    @classmethod
    def product_state(cls,
                      graph: Graph,
                      ext_dim: list[int],
                      site_covariance: Tensor | None = None,
                      dtype: torch.dtype = torch.float64,
                      device: torch.device = CUDA) -> 'GfPEPS':
        '''
        Network of uncorrelated nodes: every node covariance is block
        diagonal, with the given site state on its external modes (the vacuum by
        default) and the vacuum covariance on every bond block, so the whole
        network is pure and no site is correlated with another.
        Args:
            graph: the structure of the system.
            ext_dim: external Majorana modes of every node.
            site_covariance: covariance put on the external modes of every node, so
                every node needs the same ext_dim; None uses the vacuum.
            dtype: dtype of the node covariances.
            device: device of the node covariances.
        Returns:
            The product-state GfPEPS.
        '''
        tensors = []
        for node in range(graph.num_nodes):
            blocks = [site_covariance if site_covariance is not None
                      else vacuum_covariance(ext_dim[node], dtype, device)]
            blocks += [vacuum_covariance(bond.bond_dim, dtype, device)
                       for bond in graph.edges if node in (bond.first, bond.second)]
            tensors.append(torch.block_diag(*blocks))
        return cls.from_covariances(graph, ext_dim, tensors)

    def dim_of_node(self, node: int) -> int:
        return self.ext_dim[node] + sum(
            bond.bond_dim for bond in self.graph.bonds(node)
        )

    def contract_sites(self, i: int, j: int) -> 'GfPEPS':
        '''
        Contract the sites i and j into a single site.
        '''
        if not (0 <= i < self.graph.num_nodes and 0 <= j < self.graph.num_nodes):
            raise ValueError(f"Sites must be node indices of the graph, got {i} and {j}.")
        if i == j:
            raise ValueError("A site cannot be contracted with itself.")

        bonds = [bond for bond in self.graph.edges
                 if i in (bond.first, bond.second) and j in (bond.first, bond.second)]
        if not bonds:
            raise ValueError(f"Sites {i} and {j} are not connected by a bond.")
        low, high = min(i, j), max(i, j)
        merged = self.nodes[i].glue(self.nodes[j], bonds,
                                    [bond for bond in bonds if bond.first != i])

        def relabel(node: int) -> int:
            return low if node in (i, j) else (node if node < high else node - 1)

        # a surviving bond keeps its modes, its bond dimension and its direction, so
        # the merged site can only ever be the end of it that its old node was
        surviving = [bond for bond in self.graph.edges if bond not in bonds]
        graph = Graph(self.graph.num_nodes - 1,
                      tuple(Bond(relabel(bond.first), relabel(bond.second), bond.bond_dim)
                            for bond in surviving))
        mapping = dict(zip(surviving, graph.edges))

        ext_dim = [self.ext_dim[node] for node in range(self.graph.num_nodes) if node != high]
        ext_dim[low] = merged.sizes[None]
        nodes = [None] * graph.num_nodes
        for node in range(self.graph.num_nodes):
            if node != i and node != j:
                nodes[relabel(node)] = self.nodes[node].renamed(mapping)
        nodes[low] = merged.renamed(mapping)

        network = GfPEPS.__new__(GfPEPS)
        network.graph = graph
        network.ext_dim = ext_dim
        network.nodes = nodes
        block = next(iter(nodes[0].blocks.values()))
        network.dtype, network.device = block.dtype, block.device
        network._tensors = None
        return network

    def contract_all(self, from_end: bool = False) -> tuple['GfPEPS', dict[tuple[int, int], int]]:
        '''
        Contract every bond of the network into a single node and report where the external modes of every original node ended up, so that a sub-block of the result can be addressed by node instead of by raw mode index.  The bonds are taken from the front of the graph or, with from_end, from the back; the result is the same state with the mode order recorded        in the labels.
        Args:
            from_end: contract the bonds in the opposite order.
        Returns:
            (state, labels) with the one-node state and
            labels[(original node, its mode)] = the mode index in state.tensors[0].
        '''
        covariance, positions = _contract_all(self, from_end=from_end)
        labels = {(node, mode): index
                  for node, indices in positions.items()
                  for mode, index in enumerate(indices)}
        return GfPEPS.from_global_covariance(covariance), labels

    def partial_trace(self, sites: list[int], from_end: bool = False) -> Tensor:
        '''
        Exact reduced state of a set of nodes: contract every bond of the network first, which keeps the state pure, and then trace out the rest, which for a Gaussian state is the sub-block of the covariance on the kept modes.
        Args:
            sites: the original node labels whose external modes are kept.
            from_end: contract the bonds in the opposite order (same result).
        Returns:
            The real antisymmetric covariance of the kept modes, of size
            sum(ext_dim[node] for node in sites); it is generally mixed.
        '''
        state, labels = self.contract_all(from_end=from_end)
        modes = [labels[(node, mode)] for node in sorted(sites)
                 for mode in range(self.ext_dim[node])]
        covariance = state.tensors[0]
        return covariance[modes][:, modes]


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
            RandPureCov(self.dim_of_node(node), dtype=dtype, device=device)
            for node in range(self.graph.num_nodes)
        ]

    def dim_of_node(self, node: int) -> int:
        return self.ext_dim_in[node] + self.ext_dim_out[node] + sum(
            bond.bond_dim for bond in self.graph.bonds(node)
        )

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


def _contract_all(network,
                  from_end: bool = False) -> tuple[Tensor, dict[int, list[int]]]:
    '''
    Contract every bond of a state or operator network and return its covariance matrix on the remaining external modes together with the positions of the external modes of every original node. The external modes of the resulting nodes are ordered as the bonds of the original network keep them, so the modes of a node are contiguous and keep their original order. The bonds are contracted from the front of the graph, or from the back with ``from_end``; the result is the same covariance up to the mode order it reports.
    '''
    net = GfPEPS._from_data(network.graph, network.ext_dim, network.tensors)
    source = {node: [node] for node in range(network.graph.num_nodes)}
    while net.graph.edges:
        bond = net.graph.edges[-1] if from_end else net.graph.edges[0]
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
                           if node not in keep and state.graph.neighbors(node)), None)
            if victim is None:
                return state

            neighbours = state.graph.neighbors(victim)
            partner = next((node for node in neighbours if node in keep), neighbours[0])
            state = state.contract_sites(victim, partner)
            low, high = min(victim, partner), max(victim, partner)
            keep = {low if node in (victim, partner) else node if node < high else node - 1
                    for node in keep}
