import logging
from math import degrees, sqrt

from MDAnalysis.core.universe import Atom
from kimmdy.topology.atomic import Bond
from kimmdy.topology.topology import Topology
import numpy as np
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import calc_angles

logger = logging.getLogger("kimmdy.hydrolysis.utils")


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
