from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor

from base import REAL

SQRT3 = 3.0 ** 0.5


class Boundary(Enum):
    '''Whether a direction of the lattice ends or closes on itself.'''
    OPEN = 'open'
    PERIODIC = 'periodic'


@dataclass(frozen=True)
class Lattice:
    '''
    A lattice on the plane: the primitive vectors, the sites of a unit cell and the
    bonds as (basis, basis, cell offset), every bond of the infinite lattice once.
    '''
    name: str
    vectors: Tensor
    basis: Tensor
    bonds: tuple[tuple[int, int, tuple[int, int]], ...]

    @property
    def per_cell(self) -> int:
        '''Sites in a unit cell.'''
        return self.basis.shape[0]

    def rectangle(self, n1: int, n2: int) -> 'Geometry':
        '''A finite piece of the lattice, open in both directions.'''
        return Geometry.build(self, n1, n2, (Boundary.OPEN, Boundary.OPEN))

    def cylinder(self, n1: int, n2: int) -> 'Geometry':
        '''A finite piece of the lattice, periodic along a1 and open along a2.'''
        return Geometry.build(self, n1, n2, (Boundary.PERIODIC, Boundary.OPEN))

    def torus(self, n1: int, n2: int) -> 'Geometry':
        '''A finite piece of the lattice, periodic in both directions.'''
        return Geometry.build(self, n1, n2, (Boundary.PERIODIC, Boundary.PERIODIC))


SQUARE = Lattice(name='square',
                 vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=REAL),
                 basis=torch.zeros((1, 2), dtype=REAL),
                 bonds=((0, 0, (1, 0)), (0, 0, (0, 1))))
HONEYCOMB = Lattice(name='honeycomb',
                    vectors=torch.tensor([[-SQRT3 / 2.0, 3.0 / 2.0],
                                          [SQRT3 / 2.0, 3.0 / 2.0]], dtype=REAL),
                    basis=torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=REAL),
                    bonds=((0, 1, (0, 0)), (0, 1, (-1, 0)), (0, 1, (0, -1))))
TRIANGULAR = Lattice(name='triangular',
                     vectors=torch.tensor([[1.0, 0.0], [0.5, SQRT3 / 2.0]], dtype=REAL),
                     basis=torch.zeros((1, 2), dtype=REAL),
                     bonds=((0, 0, (1, 0)), (0, 0, (0, 1)), (0, 0, (1, -1))))
KAGOME = Lattice(name='kagome',
                 vectors=torch.tensor([[1.0, 0.0], [0.5, SQRT3 / 2.0]], dtype=REAL),
                 basis=torch.tensor([[0.5, 0.0], [0.25, SQRT3 / 4.0],
                                     [0.75, SQRT3 / 4.0]], dtype=REAL),
                 bonds=((0, 1, (0, 0)), (1, 2, (0, 0)), (2, 0, (0, 0)),
                        (2, 0, (0, 1)), (0, 1, (1, -1)), (1, 2, (-1, 0))))
LIEB = Lattice(name='lieb',
               vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=REAL),
               basis=torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=REAL),
               bonds=((1, 0, (0, 0)), (1, 0, (1, 0)), (2, 0, (0, 0)), (2, 0, (0, 1))))
CHECKERBOARD = Lattice(name='checkerboard',
                       vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=REAL),
                       basis=torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=REAL),
                       bonds=((0, 1, (0, 0)), (0, 1, (-1, 0)), (0, 1, (0, -1)),
                              (0, 1, (-1, -1))))


@dataclass(frozen=True)
class Geometry:
    '''
    One finite instance of a lattice: n1 x n2 cells with one boundary condition per
    direction, the sites, the bonds that survive the boundaries, and the site bookkeeping
    ``cell`` and ``basis``.
    '''
    lattice: Lattice
    n1: int
    n2: int
    boundary: tuple[Boundary, Boundary]
    positions: Tensor
    bonds: tuple[tuple[int, int], ...]
    cell: Tensor
    basis: Tensor

    @classmethod
    def build(cls, lattice: Lattice, n1: int, n2: int,
              boundary: tuple[Boundary, Boundary]) -> 'Geometry':
        '''
        Instantiate a lattice on n1 x n2 cells.
        Args:
            lattice: the lattice on the plane.
            n1: number of cells along a1, at least 1.
            n2: number of cells along a2, at least 1.
            boundary: the boundary condition of direction a1 and of direction a2.
        Returns:
            The geometry, its bonds deduplicated and without self loops.
        Raises:
            ValueError: if n1 or n2 is below 1, or a bond of the infinite lattice becomes
                a self loop, which happens when a cell is its own neighbour.
        '''
        if n1 < 1 or n2 < 1:
            raise ValueError(f'n1 and n2 must be at least 1, got {n1} and {n2}')
        cells = [(i, j) for j in range(n2) for i in range(n1)]
        sites = lattice.per_cell
        positions = torch.stack([i * lattice.vectors[0] + j * lattice.vectors[1]
                                 + lattice.basis[site]
                                 for i, j in cells for site in range(sites)])
        cell = torch.tensor([(i, j) for i, j in cells for _ in range(sites)],
                            dtype=torch.long)
        basis = torch.tensor([site for _ in cells for site in range(sites)],
                             dtype=torch.long)

        def label(i: int, j: int, site: int) -> int:
            return sites * (i + n1 * j) + site

        bonds: set[tuple[int, int]] = set()
        for i, j in cells:
            for first, second, (di, dj) in lattice.bonds:
                ii = (i + di) % n1 if boundary[0] is Boundary.PERIODIC else i + di
                jj = (j + dj) % n2 if boundary[1] is Boundary.PERIODIC else j + dj
                if not (0 <= ii < n1 and 0 <= jj < n2):
                    continue
                r, rp = label(i, j, first), label(ii, jj, second)
                if r == rp:
                    raise ValueError(f'{lattice.name} with n1 = {n1} and n2 = {n2} makes '
                                     f'site {r} its own neighbour through the bond '
                                     f'{(first, second, (di, dj))}; use a larger cell')
                bonds.add((min(r, rp), max(r, rp)))
        return cls(lattice, n1, n2, boundary, positions, tuple(sorted(bonds)), cell, basis)

    @property
    def sites(self) -> int:
        '''Number of sites of the instance.'''
        return self.positions.shape[0]

    def patch(self, a: int, b: int, i0: int = 0, j0: int = 0) -> list[int]:
        '''
        Site labels of the a x b parallelogram of cells that starts at cell (i0, j0);
        a periodic direction wraps, an open one has to stay inside.
        Args:
            a: cells along a1.
            b: cells along a2.
            i0: first cell along a1.
            j0: first cell along a2.
        Returns:
            The sorted site labels of the patch.
        Raises:
            ValueError: if an open direction would leave the lattice.
        '''
        sites = self.lattice.per_cell
        labels = []
        for j in range(j0, j0 + b):
            for i in range(i0, i0 + a):
                ii = i % self.n1 if self.boundary[0] is Boundary.PERIODIC else i
                jj = j % self.n2 if self.boundary[1] is Boundary.PERIODIC else j
                if not (0 <= ii < self.n1 and 0 <= jj < self.n2):
                    raise ValueError(f'the patch leaves the open lattice at cell '
                                     f'({i}, {j}) of a {self.n1} x {self.n2} instance')
                first = sites * (ii + self.n1 * jj)
                labels += list(range(first, first + sites))
        return sorted(labels)

    def snake_order(self) -> list[int]:
        '''
        The sites row by row along a2, alternating the direction along a1, so that
        consecutive sites are neighbours wherever the rows are open.
        Returns:
            The site labels in the order the snake visits them.
        '''
        sites = self.lattice.per_cell
        order = []
        for j in range(self.n2):
            row = range(self.n1) if j % 2 == 0 else range(self.n1 - 1, -1, -1)
            for i in row:
                first = sites * (i + self.n1 * j)
                order += list(range(first, first + sites))
        return order

    def cut_bonds(self, patch: Iterable[int]) -> int:
        '''
        Number of bonds with one end in the patch and one outside it: the lattice
        measure of the boundary, zero when the patch is the whole instance.
        Args:
            patch: site labels of the patch.
        Returns:
            The count of cut bonds.
        '''
        inside = set(patch)
        return sum(1 for r, rp in self.bonds if (r in inside) != (rp in inside))

    def cut_length(self, patch: Iterable[int]) -> float:
        '''
        Length of the cut bonds, the boundary length of the patch in units of the
        nearest-neighbour distance, defined for any shape and boundary condition.
        Args:
            patch: site labels of the patch.
        Returns:
            The summed length of the bonds that leave the patch.
        '''
        inside = set(patch)
        return float(sum(float((self.positions[r] - self.positions[rp]).norm())
                         for r, rp in self.bonds if (r in inside) != (rp in inside)))

    def boundary_sites(self, patch: Iterable[int]) -> int:
        '''
        Number of patch sites with a bond that leaves the patch.
        Args:
            patch: site labels of the patch.
        Returns:
            The count of boundary sites.
        '''
        inside = set(patch)
        boundary = set()
        for r, rp in self.bonds:
            if (r in inside) != (rp in inside):
                boundary.add(r if r in inside else rp)
        return len(boundary)

    def colors(self, kind: str) -> Tensor:
        '''
        An integer colour per site of one of the standard patterns of these lattices:
        the checkerboard (i + j) mod 2, the Semenoff colouring by basis site, or the
        three-colouring (i - j) mod 3.
        Args:
            kind: 'checkerboard', 'semenoff' or 'three-colouring'.
        Returns:
            The colour of every site, shape (sites,).
        Raises:
            ValueError: if the pattern does not close on a periodic direction, or the
                kind is unknown.
        '''
        if kind == 'checkerboard':
            self._require_periodic(2, 'checkerboard')
            return (self.cell[:, 0] + self.cell[:, 1]) % 2
        if kind == 'semenoff':
            return self.basis.clone()
        if kind == 'three-colouring':
            self._require_periodic(3, 'three-colouring')
            return (self.cell[:, 0] - self.cell[:, 1]) % 3
        if kind == 'bipartite':
            return self._bipartition()
        raise ValueError(f"unknown colouring {kind!r}; expected 'checkerboard', "
                         f"'semenoff', 'three-colouring' or 'bipartite'")

    def _bipartition(self) -> Tensor:
        '''
        The two-colouring of a bipartite lattice, found by breadth-first search.
        Returns:
            The colour of every site, shape (sites,), 0 or 1.
        Raises:
            ValueError: if the lattice has an odd cycle, so no two-colouring exists.
        '''
        neighbours: list[list[int]] = [[] for _ in range(self.sites)]
        for r, rp in self.bonds:
            neighbours[r].append(rp)
            neighbours[rp].append(r)
        colour = torch.full((self.sites,), -1, dtype=torch.long)
        for start in range(self.sites):
            if colour[start] >= 0:
                continue
            colour[start] = 0
            queue = [start]
            while queue:
                site = queue.pop()
                for other in neighbours[site]:
                    if colour[other] < 0:
                        colour[other] = 1 - colour[site]
                        queue.append(other)
                    elif colour[other] == colour[site]:
                        raise ValueError(f'the {self.lattice.name} lattice has an odd cycle '
                                         f'through sites {site} and {other}, so it is not '
                                         f'bipartite')
        return colour

    def _require_periodic(self, factor: int, kind: str) -> None:
        '''
        Check that a pattern of the given period closes on every periodic direction.
        Args:
            factor: the period of the pattern in cells.
            kind: the name of the pattern, for the error message.
        Raises:
            ValueError: if a periodic direction is not a multiple of the period, which
                would leave a seam where the lattice closes on itself.
        '''
        for axis, size in ((0, self.n1), (1, self.n2)):
            if self.boundary[axis] is Boundary.PERIODIC and size % factor:
                raise ValueError(f'the {kind} pattern has period {factor} cells, but '
                                 f'direction {axis + 1} closes after {size} cells, so the '
                                 f'pattern would have a seam on the closed lattice')

    def potential(self, rule: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        '''
        An on-site value per site from a rule of the cell coordinates and the basis site.
        Args:
            rule: rule(cell, basis) -> values, with cell of shape (sites, 2) and basis of
                shape (sites,); it has to be periodic with the size of every periodic
                direction, otherwise it has a seam on the closed lattice.
        Returns:
            The value of every site in the repository's real precision, shape (sites,).
        Raises:
            ValueError: if the rule does not return a floating tensor, or does not close
                on a periodic direction.
        '''
        values = rule(self.cell, self.basis)
        if not values.dtype.is_floating_point:
            raise ValueError(f'the rule returned {values.dtype}, expected a floating value '
                             f'per site')
        values = values.to(REAL)
        for axis, size in ((0, self.n1), (1, self.n2)):
            if self.boundary[axis] is not Boundary.PERIODIC:
                continue
            shifted = self.cell.clone()
            shifted[:, axis] = self.cell[:, axis] + size
            closing = rule(shifted, self.basis)
            if not closing.dtype.is_floating_point or not torch.allclose(closing.to(REAL),
                                                                        values):
                raise ValueError(f'the rule changes when direction {axis + 1} closes '
                                 f'after {size} cells, so it has a seam on the closed '
                                 f'lattice; use a rule periodic with {size} cells')
        return values

    def graph(self, edges: Sequence[tuple[int, int]], bond_dims: Sequence[int]):
        '''
        A GfPEPS graph on the sites of this instance: the topology is the caller's
        choice, so the same geometry serves a lattice graph, a snake or a tree.
        Args:
            edges: the node pairs that carry a bond.
            bond_dims: the Majorana modes of every bond, one per edge.
        Returns:
            The Graph of the tensor network.
        Raises:
            ValueError: if the edge and bond_dim counts disagree.
        '''
        from FreeFermion.GfPEPS import Bond, Graph

        if len(edges) != len(bond_dims):
            raise ValueError(f'{len(edges)} edges but {len(bond_dims)} bond dimensions')
        return Graph(self.sites, tuple(Bond(first, second, dim)
                                       for (first, second), dim in zip(edges, bond_dims)))

    def attach(self, network, node_of_site: Sequence[int] | None = None,
               mode_of_site: Sequence[int] | None = None) -> 'Bound':
        '''
        Attach a GfPEPS to the sites of this instance, on demand: the network keeps its
        own graph, and this is the single place where its nodes and their external modes
        are mapped to sites.
        Args:
            network: a GfPEPS whose external modes carry the state.
            node_of_site: the node of every site; None means node = site.
            mode_of_site: the first external mode of every site inside its node; None means
                mode 0, which is one node of two external modes per site.
        Returns:
            The Bound, which addresses patches by site.
        Raises:
            ValueError: if the map does not cover the sites, names a node the network does
                not have, or gives a site no room for its two Majoranas.
        '''
        node = tuple(range(self.sites)) if node_of_site is None else tuple(node_of_site)
        mode = ((0,) * self.sites) if mode_of_site is None else tuple(mode_of_site)
        if len(node) != self.sites or len(mode) != self.sites:
            raise ValueError(f'{len(node)} nodes and {len(mode)} mode offsets for '
                             f'{self.sites} sites')
        claimed: set[tuple[int, int]] = set()
        for site, (label, first) in enumerate(zip(node, mode)):
            if not 0 <= label < network.graph.num_nodes:
                raise ValueError(f'site {site} names node {label} of a network with '
                                 f'{network.graph.num_nodes} nodes')
            if first < 0 or first + 2 > network.ext_dim[label]:
                raise ValueError(f'site {site} claims the modes {first} and {first + 1} of '
                                 f'node {label}, which has {network.ext_dim[label]} external '
                                 f'modes')
            for offset in (first, first + 1):
                if (label, offset) in claimed:
                    raise ValueError(f'two sites claim mode {offset} of node {label}')
                claimed.add((label, offset))
        return Bound(network, self, node, mode)


@dataclass(frozen=True)
class Bound:
    '''
    A GfPEPS attached to the sites of a geometry, so that a patch is a list of sites:
    the covariance is returned in the physical Majorana layout, site r owning 2r and 2r+1.
    '''
    network: object
    geometry: Geometry
    node: tuple[int, ...]
    mode: tuple[int, ...]

    def covariance(self) -> Tensor:
        '''
        The covariance of the whole network in the physical layout.
        Returns:
            The real antisymmetric covariance, shape (2 * sites, 2 * sites).
        '''
        return self._modes(range(self.geometry.sites))

    def reduced(self, patch: Iterable[int]) -> Tensor:
        '''
        The exact reduced covariance of a patch through the network's partial_trace,
        reordered into the physical layout of the patch.
        Args:
            patch: site labels of the patch.
        Returns:
            The real antisymmetric covariance of the patch.
        '''
        kept = sorted(set(patch))
        nodes = sorted({self.node[site] for site in kept})
        covariance = self.network.partial_trace(nodes)
        offset, position = {}, 0
        for node in nodes:
            offset[node] = position
            position += self.network.ext_dim[node]
        order = [offset[self.node[site]] + self.mode[site] + step for site in kept
                 for step in range(2)]
        return covariance[order][:, order]

    def _modes(self, sites: Iterable[int]) -> Tensor:
        '''
        The covariance of the given sites, in the physical layout of the network.
        Args:
            sites: site labels, taken in the order given.
        Returns:
            The sub-block of the contracted covariance.
        '''
        state, labels = self.network.contract_all()
        modes = [labels[(self.node[site], self.mode[site] + step)] for site in sites
                 for step in range(2)]
        return state.tensors[0][modes][:, modes]


def square(n1: int, n2: int,
           boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
           ) -> Geometry:
    '''Square lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(SQUARE, n1, n2, boundary)


def honeycomb(n1: int, n2: int,
              boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
              ) -> Geometry:
    '''Honeycomb lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(HONEYCOMB, n1, n2, boundary)


def triangular(n1: int, n2: int,
               boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
               ) -> Geometry:
    '''Triangular lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(TRIANGULAR, n1, n2, boundary)


def kagome(n1: int, n2: int,
           boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
           ) -> Geometry:
    '''Kagome lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(KAGOME, n1, n2, boundary)


def lieb(n1: int, n2: int,
         boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
         ) -> Geometry:
    '''Lieb lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(LIEB, n1, n2, boundary)


def checkerboard(n1: int, n2: int,
                 boundary: tuple[Boundary, Boundary] = (Boundary.PERIODIC, Boundary.PERIODIC)
                 ) -> Geometry:
    '''Checkerboard lattice with the given boundary conditions, a torus by default.'''
    return Geometry.build(CHECKERBOARD, n1, n2, boundary)
