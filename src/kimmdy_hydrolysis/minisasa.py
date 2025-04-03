"""
Based on SASAAnalysis: <https://github.com/pegerto/mdakit_sasa>

to get multiple SASA's (for different atoms) in one go and not have to
rebuild the strucuture every time.
"""

import MDAnalysis as mda
from MDAnalysis.core.groups import Atom
from MDAnalysis.exceptions import NoDataError
import freesasa
import logging


logger = logging.getLogger("kimmdy.hydrolysis")

freesasa.setVerbosity(freesasa.silent)

class MiniSasa:
    def __init__(self, u: mda.Universe, mda_selection: str = "not resname SOL and not resname CL and not resname NA"):
        self.u = u
        self.mda_selection = mda_selection
        self.structure = self.update_structure()
        self.params = freesasa.Parameters()

        # WARNING:
        # Looks like we have to use just one thread for now
        # See <https://github.com/freesasa/freesasa-python/blob/7ead59e34ebe456b7ed27682455c6bf5bd0e7de7/src/freesasa.pyx#L222-L225>
        # self.params.setNThreads(1)

    def update_structure(self):
        """
        FreeSasa structure accepts PDBS if not available requires to reconstruct the structure using `addAtom`
        """
        structure = freesasa.Structure()
        # NOTE: the order is important here later
        # when we want the SASA per atom
        # NOTE: from mda docs: AtomGroups originating from a selection are sorted and duplicate elements are removed
        for a in self.u.select_atoms(self.mda_selection):
            a: Atom
            x, y, z = a.position
            prev_n = structure.nAtoms()
            try:
                resname = a.resname
            except NoDataError:
                resname = 'ANY' # Default classifier value
            try:
                structure.addAtom(a.type.rjust(2), resname, a.resnum.item(), a.segid, x, y, z)
            except Exception as e:
                print(e)
                print(a)

            next_n = structure.nAtoms()
            if next_n - prev_n != 1:
                m = f"Atom {a} not added to structure, nAtoms: {prev_n} -> {next_n}"
                logger.error(m)
                logger.error(a.__dict__)
                raise ValueError(m)

        self.structure = structure

    def calc(self):
        self.result = freesasa.calc(self.structure, self.params)
        self.n_atoms: int = self.result.nAtoms()
        return self.result

    def per_atom(self, i):
        return self.result.atomArea(i)


def get_baseline_sasa():
    """Calculate the SASA of a C in the peptide bond of a capped GLY-GLY didpeptide"""
    u = mda.Universe("./assets/two-gly.pdb")
    sasa = MiniSasa(u)
    ix_cc = 8
    res = sasa.calc()
    res.atomArea(ix_cc)
