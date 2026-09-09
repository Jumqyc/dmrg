'''Minimal CGO bound demo.

Run with:

    ../.venv/bin/python playground/cgo_demo.py

It builds a small Heisenberg chain, finds the ground state with DMRG, and
prints the CGO lower/upper bounds for a two-site observable.
'''

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from base import MPS, MPO, Broomstick
import CGO  # registers Broomstick.CGO


CUDA = torch.device('cuda')
DTYPE = torch.complex128


def main() -> None:
    L = 20
    max_bond_dim = 8

    # Heisenberg chain: H = sum_i S_i . S_{i+1}
    torch.manual_seed(0)
    mps = MPS(L, 2, device=CUDA, dtype=DTYPE)
    mpo = MPO(L, 2, device=CUDA, dtype=DTYPE)
    for i in range(L - 1):
        for name in ('X', 'Y', 'Z'):
            mpo.add_couplings((name, name), 1.0, (i, i + 1))
    mpo.build()

    stick = Broomstick(
        mps, mpo,
        max_bond_dim=max_bond_dim,
        svd_tol=1e-14,
        device=CUDA,
        dtype=DTYPE,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        stick.sweep(num_sweeps=4)

    # Observable B = Sz_2 Sz_3
    sz = torch.diag(torch.tensor([0.5, -0.5],device = CUDA, dtype=DTYPE))
    operators = {10: sz, 11: sz}

    exact = float(stick.state.expectation(operators).real)
    lower, upper = stick.CGO(
        l=2,
        operators=operators,
        num_sample=30,
    )

    print(f'ground-state energy: {stick.energy:.12f}')
    print(f'exact <Sz_2 Sz_3>:    {exact:+.12f}')
    print(f'CGO bounds:           [{lower:+.12f}, {upper:+.12f}]')
    print(f'bound width:          {upper - lower:.3e}')
    print(f'exact in bounds:      {lower <= exact <= upper}')

    assert lower <= exact <= upper


if __name__ == '__main__':
    main()
