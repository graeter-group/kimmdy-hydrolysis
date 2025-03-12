from MDAnalysis.core.universe import Atom
import MDAnalysis as mda
import numpy as np
from math import sqrt

from kimmdy_hydrolysis.utils import get_aproach_penalty


def normalize(v):
    return v / np.linalg.norm(v)


def test_normalize():
    v = np.array([1, 0, 0])
    assert normalize(v).tolist() == [1, 0, 0]

    v = np.array([1, 1, 0])
    assert normalize(v).tolist() == [1 / sqrt(2), 1 / sqrt(2), 0]

    v = np.array([1, 1, 1])
    assert normalize(v).tolist() == [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]


def test_find_qm_waters_and_oh_by_example():
    pass


def test_angle_penalties():
    gro = "tests/test_files/gly-npt.gro"
    u = mda.Universe(str(gro))
    ix_cc = 11  # C
    ix_n = 13  # N
    ix_oc = ix_cc + 1  # O carbonyl
    ix_n = ix_cc + 2  # N
    assert isinstance(u.atoms, mda.AtomGroup)
    assert u.atoms[ix_cc].name == "C", f"Expected C, got {u.atoms[ix_cc].name}"
    assert u.atoms[ix_n].name == "N", f"Expected N, got {u.atoms[ix_n].name}"
    assert u.atoms[ix_oc].name == "O", f"Expected O, got {u.atoms[ix_oc].name}"
    c_alpha = u.select_atoms(f"name CA and same residue as index {ix_cc}")[0]
    ix_ca = int(c_alpha.index)
    assert u.atoms[ix_ca].name == "CA", f"Expected CA, got {u.atoms[ix_ca].name}"
    c_carbonyl = u.atoms[ix_cc]
    o_carbonyl = u.atoms[ix_oc]
    n_peptide = u.atoms[ix_n]
    water_os = u.select_atoms(f"name OW and resname SOL and around 5 index {ix_cc}")

    angle_penalties = []
    penalties = []
    distances = []
    for o in water_os:
        angle_penalty, distance, penalty = get_aproach_penalty(
            o_water=o,
            c_carbonyl=c_carbonyl,
            o_carbonyl=o_carbonyl,
            n_peptide=n_peptide,
            c_alpha=c_alpha,
        )
        angle_penalties.append(angle_penalty)
        distances.append(distance)
        penalties.append(penalty)

    assert len(angle_penalties) == 11
    assert round(angle_penalties[0], 4) == 51.6687
    assert round(angle_penalties[1], 4) == 176.9778

    assert round(distances[0], 4) == 4.3143
    assert round(distances[1], 4) == 3.884
    assert round(distances[2], 4) == 3.3191
    assert round(distances[3], 4) == 3.5102

    assert round(penalties[0], 4) == 0.4227
    assert round(penalties[1], 4) == 0.6821
