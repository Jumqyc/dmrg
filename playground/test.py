import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import Broomstick, MPS, MPO,SpinOperator
import numpy as np
import torch


L = 30
MAXBONDDIM = 256

def heisenberg():

    # ── Heisenberg model ────────────────────────────────────────────────────
    mps = MPS(L, phys_dim=2)
    heisenberg_mpo = MPO(L, phys_dim=2)

    nn = np.array([(i, i+1) for i in range(L-1)]).T
    for i,j in zip(nn[0], nn[1]):
        heisenberg_mpo.add_couplings(('X', 'X'), 1.0, (i,j))
        heisenberg_mpo.add_couplings(('Y', 'Y'), 1.0, (i,j))
        heisenberg_mpo.add_couplings(('Z', 'Z'), 1.0, (i,j))
    heisenberg_mpo.build()

    heisenberg = Broomstick(mps, heisenberg_mpo, max_bond_dim=MAXBONDDIM, svd_tol=1e-16)
    heisenberg.sweep(num_sweeps=5)
    print(f"Heisenberg model done. Expected energy density: {-0.4431471805599453}")
    print(sys.getsizeof(heisenberg) / (1024 * 1024), "MB")
    del heisenberg


def ising():
    # ── Ising model ─────────────────────────────────────────────────────────
    ising_mps = MPS(L, phys_dim=2)
    ising_mpo = MPO(L, phys_dim=2)

    for i in range(L-1):
        ising_mpo.add_couplings(('Z','Z'), -1.0, (i, i+1))
    h = 0
    for i in range(L):
        ising_mpo.add_couplings(('X',), -h, (i,))
    ising_mpo.build()

    ising = Broomstick(ising_mps, ising_mpo, max_bond_dim=MAXBONDDIM)
    ising.sweep(num_sweeps=3)

    print(f"Ising model done. Expected energy density: {-0.25 * (1 - 1 / L)}")
    print(sys.getsizeof(ising) / (1024 * 1024), "MB")
    del ising

def vbs():
    # ── VBS model ───────────────────────────────────────────────────────────
    vbs_mps = MPS(L, phys_dim=2)
    vbs_mpo = MPO(L, phys_dim=2)

    for i in range(L-1):
        vbs_mpo.add_couplings(('X','X'), 1.0, (i, i+1))
        vbs_mpo.add_couplings(('Y','Y'), 1.0, (i, i+1))
        vbs_mpo.add_couplings(('Z','Z'), 1.0, (i, i+1))
    for i in range(L-2):
        vbs_mpo.add_couplings(('X','X'), 0.5, (i, i+2))
        vbs_mpo.add_couplings(('Y','Y'), 0.5, (i, i+2))
        vbs_mpo.add_couplings(('Z','Z'), 0.5, (i, i+2))
    vbs_mpo.build()

    vbs = Broomstick(vbs_mps, vbs_mpo, max_bond_dim=MAXBONDDIM)
    vbs.sweep(num_sweeps=3)

    print(sys.getsizeof(vbs) / (1024 * 1024), "MB")
    del vbs
    print("VBS model done. Expected energy density: −0.375")

def aklt():
    # ── AKLT model (spin-1, exact VBS ground state, E=0) ────────────────────


    phys_dim = 3

    # Spin-1 operators
    S1 = SpinOperator(phys_dim).data  # [Sx, Sy, Sz]
    Sx, Sy, Sz = S1[0], S1[1], S1[2]
    I3 = torch.eye(phys_dim, dtype=torch.complex128, device='cuda')

    # Build mapping with all product operators needed for (S·S)²
    mapping = {'X': Sx, 'Y': Sy, 'Z': Sz, 'I': I3}
    op_list = [Sx, Sy, Sz]
    op_names = ['X', 'Y', 'Z']
    for i, a_name in enumerate(op_names):
        for j, b_name in enumerate(op_names):
            mapping[f'{a_name}{b_name}'] = op_list[i] @ op_list[j]

    aklt_mpo = MPO(L, phys_dim=phys_dim, mapping=mapping)

    # P⁽²⁾ = 1/6 (S·S)² + 1/2 (S·S) + 1/3
    for site in range(L - 1):
        # Linear: 1/2 * S·S
        aklt_mpo.add_couplings(('X', 'X'), 0.5, (site, site + 1))
        aklt_mpo.add_couplings(('Y', 'Y'), 0.5, (site, site + 1))
        aklt_mpo.add_couplings(('Z', 'Z'), 0.5, (site, site + 1))
        # Quadratic: 1/6 * (S·S)²  → 9 terms ab⊗ab for a,b∈{X,Y,Z}
        for a in op_names:
            for b in op_names:
                aklt_mpo.add_couplings((f'{a}{b}', f'{a}{b}'), 1.0 / 6.0, (site, site + 1))
        # Constant: 1/3
        aklt_mpo.add_couplings(('I', 'I'), 1.0 / 3.0, (site, site + 1))

    aklt_mpo.build()

    dmrg = Broomstick(MPS(L, phys_dim), aklt_mpo, max_bond_dim=MAXBONDDIM)
    dmrg.sweep(num_sweeps=2)
    print("AKLT model done. Expected energy density: 0.0")
    print('---------------')

def uls():
    # similar to AKLT but with different coefficients
    phys_dim = 3
    S1 = SpinOperator(phys_dim).data  # [Sx, Sy, Sz]
    Sx, Sy, Sz = S1[0], S1[1], S1[2]
    I3 = torch.eye(phys_dim, dtype=torch.complex128, device='cuda')

    # Build mapping with all product operators needed for (S·S)²
    mapping = {'X': Sx, 'Y': Sy, 'Z': Sz, 'I': I3}
    op_list = [Sx, Sy, Sz]
    op_names = ['X', 'Y', 'Z']
    for i, a_name in enumerate(op_names):
        for j, b_name in enumerate(op_names):
            mapping[f'{a_name}{b_name}'] = op_list[i] @ op_list[j]

    uls_h = MPO(L, phys_dim=phys_dim, mapping=mapping)

    # P⁽²⁾ = 1/2 (S·S)² + 1/2 (S·S) + 1/3
    for site in range(L - 1):
        # Linear: 1/2 * S·S
        uls_h.add_couplings(('X', 'X'), 0.5, (site, site + 1))
        uls_h.add_couplings(('Y', 'Y'), 0.5, (site, site + 1))
        uls_h.add_couplings(('Z', 'Z'), 0.5, (site, site + 1))
        # Quadratic: 1/2 * (S·S)²  → 9 terms ab⊗ab for a,b∈{X,Y,Z}
        for a in op_names:
            for b in op_names:
                uls_h.add_couplings((f'{a}{b}', f'{a}{b}'), 1.0 / 2.0, (site, site + 1))
        uls_h.add_couplings(('I', 'I'), 1.0 / 3.0, (site, site + 1))

    uls_h.build()

    dmrg = Broomstick(MPS(L, phys_dim), uls_h, max_bond_dim=MAXBONDDIM)
    dmrg.sweep(num_sweeps=4)



# heisenberg()
# ising()
# vbs()
# aklt()
uls()