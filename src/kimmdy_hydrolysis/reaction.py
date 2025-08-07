import logging
from math import inf
from pathlib import Path
from typing import Any

import MDAnalysis as mda
from kimmdy.plugins import ReactionPlugin
from kimmdy.constants import nN_per_kJ_per_mol_nm
from kimmdy.recipe import (
    Bind,
    Break,
    CustomTopMod,
    DeferredRecipeSteps,
    Recipe,
    RecipeCollection,
    RecipeStep,
    Relax,
)
from kimmdy.tasks import TaskFiles
from kimmdy.topology.atomic import Bond
from kimmdy.topology.topology import Topology
from kimmdy.topology.utils import get_residue_by_bonding

from kimmdy_hydrolysis.minisasa import MiniSasa
from kimmdy_hydrolysis.utils import get_aproach_penalty, get_peptide_bonds_from_top
from kimmdy.plugin_utils import (
    bondstats_to_csv,
    bondstats_from_csv,
    calculate_bondstats,
)
from kimmdy_hydrolysis.rates import (
    experimental_reaction_rate_per_s,
    theoretical_reaction_rate_per_s,
)

logger = logging.getLogger("kimmdy.hydrolysis")


class HydrolysisReaction(ReactionPlugin):
    """Hydrolyses peptide bonds of the backbone."""

    def get_recipe_collection(self, files: TaskFiles) -> RecipeCollection:
        logger = files.logger

        # settings from the config
        self.max_sasa = self.config.max_sasa
        self.ph_value = self.config.ph_value
        self.external_force = self.config.external_force
        self.bondstats_at_0 = None
        if self.config.bondstats_at_0 != "":
            self.bondstats_at_0 = bondstats_from_csv(self.config.bondstats_at_0)

        self.theoretical = self.config.theoretical_rates.use
        if self.theoretical:
            logger.info(f"Using theoretical rates")
            self.A = (
                self.config.theoretical_rates.empirical_attempt_frequency * 1e12
            )  # A from 1/ps to 1/s
            self.ts1 = self.config.theoretical_rates.ts1
            self.ts2 = self.config.theoretical_rates.ts2
            self.ts1_force_scaling = self.config.theoretical_rates.ts1_force_scaling
            self.ts2_force_scaling = self.config.theoretical_rates.ts2_force_scaling
            self.critical_force = self.config.theoretical_rates.critical_force

        self.temperature = self.config.temperature
        self.step = self.config.step
        self.recipes = []
        self.sasa_per_bond: dict[str, list[float]] = {}
        # times are shared between all bonds
        self.times: list[float] = []
        self.timespans: list[tuple[float, float]]
        self.peptide_bonds = get_peptide_bonds_from_top(self.runmng.top)

        self.init_timings(files)
        self.init_universe()

        did_read_sasa = self.use_cached_sasa()
        if not did_read_sasa:
            self.calculate_sasa()
            self.cache_sasa()

        if self.config.external_force == -1:
            did_read_bondstats = self.use_cached_bondstats()
            if not did_read_bondstats:
                plumed_out = files.input["plumed_out"]
                plumed_in = files.input["plumed"]
                if plumed_out is None or plumed_in is None:
                    m = f"External force not specified but no plumed file found"
                    logger.error(m)
                    raise ValueError(m)
                self.bondstats = calculate_bondstats(
                    top=self.runmng.top,
                    plumed_in=plumed_in,
                    plumed_out=plumed_out,
                    dt=self.config.dt_distances,
                    edissoc_dat=files.input["edissoc.dat"],
                )
                self.cache_bondstats()

        logger.info(f"Got {len(self.times)} times for SASA calculation")
        for id_c, b in self.peptide_bonds.items():
            r = Recipe(
                recipe_steps=DeferredRecipeSteps(
                    key=id_c, callback=self.get_steps_for_id_c_at_t
                ),
                rates=self.sasas_to_rates(sasas=self.sasa_per_bond[id_c], bond=b),
                timespans=self.timespans,
            )
            self.recipes.append(r)

        return RecipeCollection(self.recipes)

    def use_cached_bondstats(self) -> bool:
        if self.config.recompute_bondstats:
            return False
        if not Path(self.bondstatsfile).exists():
            m = f"bondstatsfile does not exist. Not using cached bondstats."
            logger.info(m)
            return False

        self.bondstats = bondstats_from_csv(self.bondstatsfile)
        return True

    def cache_bondstats(self) -> None:
        bondstats_to_csv(self.bondstats, self.bondstatsfile)

    def sasas_to_rates(self, sasas: list[float], bond: Bond) -> list[float]:
        if self.external_force != -1:
            force = self.external_force
        else:
            # get force on bond
            bondkey = (bond.ai, bond.aj)
            force = self.bondstats[bondkey]["mean_f"]
            if self.bondstats_at_0 is not None:
                # subtract the force at 0
                force_at_0 = self.bondstats_at_0[bondkey]["mean_f"]
                force = force - force_at_0

            # force from bondstats is in gromacs units
            force = force * nN_per_kJ_per_mol_nm

            # set negative average forces to 0
            force = max(force, 0)

        logger.debug(
            f"Calculating rates for bond {bond.ai} {bond.aj} with force {force}"
        )
        if self.theoretical:
            k_hyd_per_s = theoretical_reaction_rate_per_s(
                force=force,
                A=self.A,
                temperature=self.temperature,
                ph_value=self.ph_value,
                ts1=self.ts1,
                ts2=self.ts2,
                ts1_force_scaling=self.ts1_force_scaling,
                ts2_force_scaling=self.ts2_force_scaling,
            )
            k_hyd = k_hyd_per_s * 1e-12
        else:
            k_hyd_per_s = experimental_reaction_rate_per_s(force, self.temperature)

            k_hyd = k_hyd_per_s * 1e-12  # rates in 1/ps
            # scale by pH value
            # concentration of OH-
            c_oh_experiment = 10 ** (-(14 - 7.4))
            c_oh = 10 ** (-(14 - self.ph_value))
            ph_scaling = c_oh / c_oh_experiment
            k_hyd = k_hyd * ph_scaling

        # scale by SASA
        rates = []
        for sasa in sasas:
            sasa_scaling = sasa / self.max_sasa
            rates.append(sasa_scaling * k_hyd)
        return rates

    def get_steps_for_id_c_at_t(self, key: str, ttime: float) -> list[RecipeStep]:
        """Get the steps for a given bond at a given time.

        The bond is identified by the atom id of the C atom.
        """
        b = self.peptide_bonds[key]
        ix_cc = int(b.ai) - 1  # C
        ix_n = int(b.aj) - 1  # N
        ix_oc = ix_cc + 1  # O carbonyl
        ix_n = ix_cc + 2  # N
        assert isinstance(self.u.atoms, mda.AtomGroup)
        assert (
            self.u.atoms[ix_cc].name == "C"
        ), f"Expected C, got {self.u.atoms[ix_cc].name}"
        assert (
            self.u.atoms[ix_n].name == "N"
        ), f"Expected N, got {self.u.atoms[ix_n].name}"
        assert (
            self.u.atoms[ix_oc].name == "O"
        ), f"Expected O, got {self.u.atoms[ix_oc].name}"
        c_alpha = self.u.select_atoms(f"name CA and same residue as index {ix_cc}")[0]
        ix_ca = int(c_alpha.index)
        assert (
            self.u.atoms[ix_ca].name == "CA"
        ), f"Expected CA, got {self.u.atoms[ix_ca].name}"
        c_carbonyl = self.u.atoms[ix_cc]
        o_carbonyl = self.u.atoms[ix_oc]
        n_peptide = self.u.atoms[ix_n]

        logger.info(
            f"Hydrolyzing bond between C with ix {ix_cc} and N with ix {ix_n} at time {ttime} ps"
        )

        frame = round(ttime / self.ps_per_frame)
        snapshot = self.u.trajectory[frame]

        logger.info(
            f"Trajectory time: {self.u.trajectory.time:.3f} ps at frame {frame}"
        )
        logger.info(f"Time from runmanager: {ttime} ps")
        logger.info(f"Time of u.trajectory: {snapshot.time:.3f} ps")

        if round(self.u.trajectory.time, 3) != round(ttime, 3):
            m = f"Mismatch between time chosen by the runmanager and time received"
            logger.error(m)
            raise ValueError(m)

        water_os = self.u.select_atoms(
            f"name OW and resname SOL and around {self.config.cutoff} index {ix_cc}"
        )
        if len(water_os) == 0:
            m = f"No water molecules found around C{c_carbonyl.resid} at index {ix_cc} with cutoff {self.config.cutoff} returning empty list of steps."
            logger.warning(m)
            return []

        chosen_water = None
        lowest_penalty = inf
        for o in water_os:
            angle_penalty, distance, penalty = get_aproach_penalty(
                o_water=o,
                c_carbonyl=c_carbonyl,
                o_carbonyl=o_carbonyl,
                n_peptide=n_peptide,
                c_alpha=c_alpha,
            )
            if penalty < lowest_penalty:
                chosen_water = o
                lowest_penalty = penalty
                logger.info(
                    f"Found better water {chosen_water} with penalty {penalty} and angle_penalty {angle_penalty} and distance {distance}"
                )

        if chosen_water is None:
            raise ValueError("No water O was chosen")
        ix_o = int(chosen_water.index)
        logger.info(f"Chose water O ix: {ix_o}")

        ix_h1 = ix_o + 1
        ix_h2 = ix_o + 2

        steps = []
        # break peptide bond
        steps.append(Break(atom_ix_1=ix_cc, atom_ix_2=ix_n))

        # # break O-H bonds in water
        # Those bonds don't exist with settles instead of flexible tip3p water
        # (though kimmdy adds them temporarily to the topology to provide bonds for interpolating with slow growth)
        if self.runmng.top.bonds.get((str(ix_o + 1), str(ix_h1 + 1))) is not None:
            steps.append(Break(atom_ix_1=ix_o, atom_ix_2=ix_h1))
        if self.runmng.top.bonds.get((str(ix_o + 1), str(ix_h2 + 1))) is not None:
            steps.append(Break(atom_ix_1=ix_o, atom_ix_2=ix_h2))

        def f(top: Topology) -> Topology:
            id_cc = str(ix_cc + 1)
            id_n = str(ix_n + 1)
            id_o = str(ix_o + 1)
            id_h1 = str(ix_h1 + 1)
            id_h2 = str(ix_h2 + 1)

            c_carbonyl = top.atoms[id_cc]
            n_peptide = top.atoms[id_n]
            o_water = top.atoms[id_o]
            h1_water = top.atoms[id_h1]
            h2_water = top.atoms[id_h2]

            # rename atoms and atomtypes
            c_side = get_residue_by_bonding(c_carbonyl, top.atoms)
            n_side = get_residue_by_bonding(n_peptide, top.atoms)

            def fix_charge(a):
                residuetype = top.ff.residuetypes.get(a.residue)
                if residuetype is None:
                    m = f"Could not find residuetype for {a.residue}"
                    logger.error(m)
                    return
                rtype_atom = residuetype.atoms.get(a.atom)
                if rtype_atom is None:
                    m = f"Could not find residuetype atom for {a.atom}"
                    logger.error(m)
                    return
                a.charge = rtype_atom.charge

            for a in c_side.values():
                a.residue = "C" + a.residue
                if a.atom == "O":
                    a.atom = "OC1"
                    a.type = "O2"
                fix_charge(a)
            for a in n_side.values():
                a.residue = "N" + a.residue
                if a.atom == "H":
                    a.atom = "H1"
                elif a.atom == "N":
                    a.type = "N3"
                elif a.atom == "H":
                    a.type = "H1"
                elif a.atom == "HA1" or a.atom == "HA2":
                    a.type = "HP"
                fix_charge(a)

            o_water.resnr = c_carbonyl.resnr
            o_water.residue = c_carbonyl.residue
            o_water.type = "O2"
            o_water.atom = "OC2"
            o_water.cgnr = c_carbonyl.cgnr
            fix_charge(o_water)

            h1_water.resnr = n_peptide.resnr
            h1_water.residue = n_peptide.residue
            h1_water.type = "H"
            h1_water.atom = "H2"
            h1_water.cgnr = n_peptide.cgnr
            fix_charge(h1_water)

            h2_water.resnr = n_peptide.resnr
            h2_water.residue = n_peptide.residue
            h2_water.type = "H"
            h2_water.atom = "H3"
            h2_water.cgnr = n_peptide.cgnr
            fix_charge(h2_water)

            return top

        # use custom topology modification to rename atoms and atomtypes
        steps.append(CustomTopMod(f))

        # re-assemble into terminal amino acids (with deprotonated C- and protonated N-terminus)
        steps.append(Bind(atom_ix_1=ix_o, atom_ix_2=ix_cc))
        steps.append(Bind(atom_ix_1=ix_h1, atom_ix_2=ix_n))
        steps.append(Bind(atom_ix_1=ix_h2, atom_ix_2=ix_n))

        # use slow growth to relax into new parameters
        steps.append(Relax())

        return steps

    def times_to_timespans(self, times: list[float]) -> list[tuple[float, float]]:
        """Transforms an array of times into a list of time intervals
        times are an array of times at which the SASA was calculated
        timespans are the time intervals between those times
        as tuples of (start, end)
        """
        timespans = []
        for i in range(len(times) - 1):
            timespans.append((times[i], times[i + 1]))
        return timespans

    def init_timings(self, files: TaskFiles):
        self.gro = files.input["gro"]
        self.xtc = files.input["xtc"]
        logger.debug(f"Using gro: {self.gro}")
        logger.debug(f"Using xtc: {self.xtc}")

        if self.xtc is None or self.gro is None:
            m = "No xtc file found"
            logger.error(m)
            raise ValueError(m)
        logger.info(
            f"Using xtc {self.xtc.name} in {self.xtc.parent.name} for trajectory"
        )
        self.sasafile = self.xtc.with_name(".kimmdy.sasa")
        self.bondstatsfile = self.xtc.with_name(".kimmdy.bondstats")

        md_instance = self.xtc.stem
        timings = self.runmng.timeinfos.get(md_instance)
        if timings is None:
            m = f"No timings from mdp file found for {md_instance}"
            logger.error(m)
            raise ValueError(m)

        logger.info(f"timings: {timings}")
        self.timings = timings
        self.xtc_trr_ratio = timings.trr_nst / timings.xtc_nst

        self.dt = timings.dt
        self.nframes_stepsize = self.xtc_trr_ratio * self.step
        # make sure the stepsize is an integer
        if not self.nframes_stepsize.is_integer():
            m = f"Stepsize {self.nframes_stepsize} is not an integer. Derived from xtc_nst {timings.xtc_nst} and trr_nst {timings.trr_nst}"
            logger.error(m)
            raise ValueError(m)

        self.nframes_stepsize = int(self.nframes_stepsize)
        self.ps_per_frame = round(self.dt * timings.xtc_nst, 3)

    def init_universe(self):
        self.u = mda.Universe(str(self.gro), str(self.xtc))
        # reset to first frame just in case
        frame = self.u.trajectory[0]

        logger.info(f"First frame: {frame.frame} with time {frame.time} ps")
        # validate that the next frame is dt * xtc_nst ps away
        frame = self.u.trajectory[1]
        if round(frame.time, 3) != round(self.dt, 3) * self.timings.xtc_nst:
            m = f"Expected the next frame to be {self.dt * self.timings.xtc_nst} ps away, got {frame.time} ps"
            logger.error(m)
            raise ValueError(m)

        # reset to first frame
        frame = self.u.trajectory[0]

    def calculate_sasa(self):
        logger.info(
            f"Calculating SASA for {len(self.peptide_bonds)} bonds. Step={self.step}"
        )
        logger.info(f"Universe has {len(self.u.trajectory)} frames")
        self.times = []
        self.sasa_per_bond = {k: [] for k in self.peptide_bonds.keys()}
        sasa = MiniSasa(self.u)
        # skip the first frame
        for frame in self.u.trajectory[1 :: self.nframes_stepsize]:
            time = round(frame.time, 3)
            logger.debug(
                f"Calculating SASA for frame {frame.frame} with time rounded {time}"
            )
            self.times.append(time)
            sasa.update_structure()
            sasa.calc()
            for k in self.peptide_bonds.keys():
                ix_cc = int(k) - 1
                s = sasa.per_atom(ix_cc)
                try:
                    self.sasa_per_bond[k].append(s)
                except IndexError:
                    m = f"Indices of atoms for mdanalysis selection for sasa calculation do not match with the peptide bonds."
                    logger.error(m)
                    raise ValueError(m)

        # but retain the first time [0.0] because
        # because it will combine with the next time
        # into the first timespan
        # the times are in ps. We round to fs.
        self.timespans = self.times_to_timespans([0.0] + self.times)

    def cache_sasa(self):
        write_sasa_free(
            times=self.times,
            sasa_per_bond=self.sasa_per_bond,
            metadata={
                "dt": self.dt,
                "xtc_trr_ratio": self.xtc_trr_ratio,
                "step": self.step,
                "ps_per_frame": self.ps_per_frame,
                "nframes_stepsize": self.nframes_stepsize,
            },
            path=self.sasafile,
        )

    def use_cached_sasa(self) -> bool:
        if self.config.recompute_sasa:
            return False
        if not Path(self.sasafile).exists():
            m = f"sasafile {self.sasafile} does not exist. Not using cached SASA."
            logger.info(m)
            return False

        result = read_sasa_free(self.sasafile)
        if result is None:
            m = f"Could not read sasafile {self.sasafile}. Not using cached SASA."
            logger.warning(m)
            return False
        metadata, times, sasa_per_bond = result

        if float(metadata["dt"]) != self.dt:
            m = f"dt from sasafile {metadata['dt']} does not match {self.dt} from mdp. Not using cached SASA."
            logger.warning(m)
            return False
        if float(metadata["xtc_trr_ratio"]) != self.xtc_trr_ratio:
            m = f"xtc_trr_ratio from sasafile {metadata['xtc_trr_ratio']} does not match {self.xtc_trr_ratio} from mdp. Not using cached SASA."
            logger.warning(m)
            return False
        if int(metadata["step"]) != self.step:
            m = f"step from sasafile {metadata['step']} does not match {self.step} from config. Not using cached SASA."
            logger.warning(m)
            return False
        if float(metadata["ps_per_frame"]) != self.ps_per_frame:
            m = f"ps_per_frame from sasafile {metadata['ps_per_frame']} does not match {self.ps_per_frame} from mdp. Not using cached SASA."
            logger.warning(m)
            return False
        if int(metadata["nframes_stepsize"]) != self.nframes_stepsize:
            m = f"nframes_stepsize from sasafile {metadata['nframes_stepsize']} does not match {self.nframes_stepsize} from mdp. Not using cached SASA."
            logger.warning(m)
            return False

        self.times = times
        self.sasa_per_bond = sasa_per_bond
        self.timespans = self.times_to_timespans([0.0] + self.times)

        m = f"Using cached SASA"
        logger.info(m)

        return True


def write_sasa_free(
    times: list[float],
    sasa_per_bond: dict[str, list[float]],
    metadata: dict[str, Any],
    path: str | Path,
) -> None:
    with open(path, "w") as f:
        # header with metadata
        # df, xtc_trr_ratio, dt, step, ps_per_frame, nframes_stepsize,
        f.write("---meta\n")
        for key, value in metadata.items():
            f.write(f"{key} = {value}\n")

        # times
        f.write(f"---times\n")
        for t in times:
            f.write(f"{t:.3f}\n")

        # sasa
        f.write(f"---sasa\n")
        for k, sasas in sasa_per_bond.items():
            f.write(f"#{k}\n")
            for s in sasas:
                f.write(f"{s}\n")


def read_sasa_free(
    path: str | Path,
) -> None | tuple[dict, list[float], dict[str, list[float]]]:
    with open(path, "r") as f:
        l = f.readline()
        if not l.startswith("---meta"):
            m = f"File {path} does not start with ---meta. Not using cached SASA."
            logger.warning(m)
            return None

        metadata = {}
        l = f.readline()
        while not l.startswith("---"):
            key, value = l.split("=")
            metadata[key.strip()] = value.strip()
            l = f.readline()

        times = []
        if not l.startswith("---times"):
            m = f"File {path} does not contain ---times. Not using cached SASA."
            logger.warning(m)
            return None

        l = f.readline()
        while not l.startswith("---"):
            times.append(float(l.strip()))
            l = f.readline()

        sasa_per_bond = {}
        if not l.startswith("---sasa"):
            m = f"File {path} does not contain ---sasa. Not using cached SASA."
            logger.warning(m)
            return None

        k = None
        l = f.readline()
        while l:
            if l.startswith("#"):
                k = l.strip()[1:]
                sasa_per_bond[k] = []
            else:
                sasa_per_bond[k].append(float(l.strip()))
            l = f.readline()

    return metadata, times, sasa_per_bond
