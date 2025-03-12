import logging
from math import degrees, sqrt

from MDAnalysis.core.universe import Atom
from kimmdy.topology.atomic import Bond
import numpy as np
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import calc_angles
from kimmdy_hydrolysis.constants import nN_per_kJ_per_mol_nm
from pathlib import Path

logger = logging.getLogger("kimmdy.hydrolysis.utils")


def read_bond_lengths(path: str | Path) -> dict[tuple[str, str], float]:
    lengths = {}
    with open(path, "r") as f:
        next(f)
        for l in f:
            i, j, d = l.strip().split(",")
            lengths[(i, j)] = float(d)
    return lengths


def ds_to_forces(
    ds: np.ndarray, dissociation_energy: float, b0: float, kb: float
) -> np.ndarray:
    beta = np.sqrt(kb / (2 * dissociation_energy))
    d_inflection = (beta * b0 + np.log(2)) / beta
    # if the bond is stretched beyond the inflection point,
    # take the inflection point force because this force must have acted on the bond at some point
    ds_mask = ds > d_inflection
    ds[ds_mask] = d_inflection
    dds = ds - b0

    # kJ/mol/nm -> nN
    forces = (
        2 * beta * dissociation_energy * np.exp(-beta * dds) * (1 - np.exp(-beta * dds))
    ) * nN_per_kJ_per_mol_nm
    return forces


def get_peptide_bonds_from_top(top) -> list[Bond]:
    bs = []
    for b in top.bonds.values():
        ai = top.atoms[b.ai]
        aj = top.atoms[b.aj]
        if ai.residue == "ACE" or aj.residue == "NME":
            continue
        if ai.atom == "C" and aj.atom == "N":
            bs.append(b)

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
    # logger.info(f"Water O ix: {o_water.ix}")
    # logger.info(f"bd penalty: {bd_penalty}")
    # logger.info(f"fl penalty: {fl_penalty}")
    # logger.info(f"total angle penalty: {angle_penalty}")
    # logger.info(f"distance: {distance}")
    return angle_penalty, distance, penalty
