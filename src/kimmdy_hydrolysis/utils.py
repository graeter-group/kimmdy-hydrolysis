import logging
from math import degrees, sqrt

from MDAnalysis.core.universe import Atom
from kimmdy.parsing import read_plumed
from kimmdy.topology.atomic import Bond
from kimmdy.topology.topology import Topology
import numpy as np
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import calc_angles
from kimmdy_hydrolysis.constants import nN_per_kJ_per_mol_nm
from pathlib import Path

logger = logging.getLogger("kimmdy.hydrolysis.utils")

BONDSTATS_COLUMNS = "ai,aj,mean_d,mean_f,delta_d,b0"

def bondstats_to_csv(stats: dict, path: str|Path):
    ls = []
    ls.append(BONDSTATS_COLUMNS)
    for k, s in stats.items():
        ls.append(f"{k[0]},{k[1]},{s['mean_d']:.6f},{s['mean_f']:.6f},{s['delta_d']:.6f},{s['b0']:.6f}")

    with open(path, "w") as f:
        f.write("\n".join(ls))

def bondstats_from_csv(path: str | Path) -> dict[tuple[str, str], dict]:
    stats: dict[tuple[str, str], dict] = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            ai, aj, mean_d, mean_f, delta_d, b0 = line.split(",")
            stats[(ai, aj)] = {
                "mean_d": float(mean_d),
                "mean_f": float(mean_f),
                "delta_d": float(delta_d),
                "b0": float(b0),
            }

    return stats

def get_bondstats(top: Topology, distances: dict[float, dict[str, float]], peptide_bonds: dict[str, Bond], bond_to_plumed_id: dict[tuple[str, str], str]) -> dict[tuple[str, str], dict]:
    stats: dict[tuple[str, str], dict] = {}
    for bond in peptide_bonds.values():
        ai = top.atoms[bond.ai]
        aj = top.atoms[bond.aj]
        k = (bond.ai, bond.aj)
        bondtype = top.ff.bondtypes.get((ai.type, aj.type))
        plumed_id = bond_to_plumed_id.get((bond.ai, bond.aj))
        if not bondtype or bondtype.c0 is None or bondtype.c1 is None:
            raise ValueError("Could not find bondtype")
        assert plumed_id is not None, f"bond {bond} not found in plumed input"
        b0 = float(bondtype.c0)
        kb = float(bondtype.c1)
        dissociation_energy = 500
        ds = np.asarray([values[plumed_id] for values in distances.values()])
        beta = np.sqrt(kb / (2 * dissociation_energy))
        d_inflection = (beta * b0 + np.log(2)) / beta
        # if the bond is stretched beyond the inflection point,
        # take the inflection point force because this force must have acted on the bond at some point
        ds_mask = ds > d_inflection
        ds[ds_mask] = d_inflection
        dds = ds - b0
        forces = (
            2 * beta * dissociation_energy * np.exp(-beta * dds) * (1 - np.exp(-beta * dds))
        ) * nN_per_kJ_per_mol_nm

        mean_d = np.mean(ds)
        mean_f = np.mean(forces)

        stats[k] = {
            "mean_d": mean_d,
            "mean_f": mean_f,
            "delta_d": mean_d - b0,
            "b0": b0,
        }
    return stats


def read_plumed_input(path: str | Path) -> dict[tuple[str, str], str]:
    if not isinstance(path, Path):
        path = Path(path)
    plumed = read_plumed(path)
    d = {}
    for k, v in plumed["labeled_action"].items():
        if v["keyword"] != "DISTANCE":
            continue
        atoms = v["atoms"]
        bondkey = tuple(sorted(atoms, key=int))
        d[bondkey] = k
    return d


def get_peptide_bonds_from_top(top: Topology) -> dict[str, Bond]:
    bs = {}
    for bond in top.bonds.values():
        a = top.atoms[bond.ai]
        b = top.atoms[bond.aj]
        if a.residue in ["NME", "ACE"] or b.residue in ["NME", "ACE"]:
            continue
        if a.atom == "C" and b.atom == "N":
            bs[a.nr] = bond

    return bs


def normalize(v):
    return v / np.linalg.norm(v)


def get_aproach_penalty(
    o_water: Atom, c_carbonyl: Atom, o_carbonyl: Atom, n_peptide: Atom, c_alpha: Atom
) -> tuple[float, float, float]:

    c = c_carbonyl.position
    o = o_carbonyl.position
    n = n_peptide.position
    ca = c_alpha.position
    ow = o_water.position
    c_n = n - c
    n_c = c - n
    c_ca = ca - c
    c_o = o - c
    o_c = c - o
    c_ow = ow - c

    distance = float(np.linalg.norm(c_ow))

    plane_normal = np.cross(n_c, c_ca)
    plane_normal = normalize(plane_normal)

    c_ow_projected = c_ow - np.dot(c_ow, plane_normal) * plane_normal
    c_ow_projected = normalize(c_ow_projected)

    # Bürgi-Dunitz angle
    # O-C-O angle close to angle of 107 deg
    # The BD is the angle between the approach vector of O_nucl
    # and the electrophilic C and the C=O bond
    bd = degrees(calc_angles(*AtomGroup([o_water, c_carbonyl, o_carbonyl]).positions))
    bd_penalty = abs(bd - 107)

    # Flippin-Lodge angle
    # The FL is an angle that estimates the displacement of the nucleophile,
    # at its elevation, toward or away from the particular R and R' substituents
    # attached to the electrophilic atom
    dot = np.dot(c_ow_projected, o_c)
    oc_norm = np.linalg.norm(o_c)
    fl = degrees(np.arccos(dot / (1 * oc_norm)))
    fl_penalty = abs(fl - 0)
    angle_penalty = sqrt(bd_penalty**2 + fl_penalty**2)
    # weigh all penalties equally
    max_bd_penalty = 180
    max_fl_penalty = 180
    max_distance_penalty = 5
    penalty = (
        (bd_penalty / max_bd_penalty)
        + (fl_penalty / max_fl_penalty)
        + (min(distance, max_distance_penalty) / max_distance_penalty)
    ) / 3
    return angle_penalty, distance, penalty
