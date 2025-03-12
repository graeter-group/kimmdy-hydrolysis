import logging
from math import inf
from kimmdy.parsing import read_distances_dat, read_plumed
from kimmdy.topology.atomic import Bond
from kimmdy.topology.topology import Topology
from kimmdy.topology.utils import get_residue_by_bonding
import numpy as np
from kimmdy_hydrolysis.minisasa import MiniSasa
from kimmdy_hydrolysis.constants import K_b, T_EXPERIMENT

import MDAnalysis as mda
from kimmdy.plugins import ReactionPlugin
from kimmdy.recipe import (
    Bind,
    Break,
    CustomTopMod,
    Recipe,
    RecipeCollection,
    RecipeStep,
    Relax,
    DeferredRecipeSteps,
)
from kimmdy.tasks import TaskFiles
from kimmdy_hydrolysis.utils import (
    ds_to_forces,
    get_aproach_penalty,
    get_peptide_bonds_from_top,
    read_bond_lengths,
)

logger = logging.getLogger("kimmdy.hydrolysis")


def e_ts1(
    force: float = 0,
    ts1: float = 80,
    ts1_force_scaling: float = 1.67,
) -> float:
    return ts1 - ts1_force_scaling * force


def e_ts2(
    force: float = 0,
    ts2: float = 92,
    ts2_force_scaling: float = 25.83,
) -> float:
    return ts2 - ts2_force_scaling * force


def low_force_log_rate(force):
    log_slope = 26.26
    log_offset = -19.77
    return log_slope * force + log_offset


def high_force_log_rate(force, temperature: float = T_EXPERIMENT):
    """
    (Intercept) -20.342988   1.946146  -10.45 5.36e-10 ***
    t             0.070648   0.006458   10.94 2.30e-10 ***
    f             1.605233   0.100667   15.95 1.43e-13 ***
    """
    log_slope_t = 0.070648
    log_slope_f = 1.605233
    log_offset = -20.342988
    log_k = log_offset + log_slope_t * temperature + log_slope_f * force

    return log_k


def experimental_reaction_rate_per_s(
    force: float, temperature: float = T_EXPERIMENT
) -> float:
    critical_force = 0.7
    interpolation_width = 0.05

    if force <= (critical_force - interpolation_width):
        log_k = low_force_log_rate(force)
    elif force > (critical_force + interpolation_width):
        log_k = high_force_log_rate(force, temperature)
    else:
        # linear interpolation between the two linear regimes
        low = low_force_log_rate(force)
        high = high_force_log_rate(force)
        high_percentage = (force - (critical_force - interpolation_width)) / (
            2 * interpolation_width
        )
        low_percentage = 1 - high_percentage
        log_k = low_percentage * low + high_percentage * high

    k = np.exp(log_k)
    return k


def theoretical_reaction_rate_per_s(
    force: float = 0,
    ts1: float = 80,
    ts2: float = 92,
    ts1_force_scaling: float = 1.67,
    ts2_force_scaling: float = 25.83,
    A: float = 1e11,  # 1/s
    temperature: float = 300,
    ph_value: float = 7.4,
) -> float:
    """Calculate reaction rate in 1/s

    see SI of pill et al. 2019
    <http://dx.doi.org/10.1002/anie.201902752>
    """
    # energy barriers in kJ/mol
    # high force regime, TS1 is rate-determining
    E_ts1 = e_ts1(force, ts1, ts1_force_scaling)
    # low force regime, TS2 is rate-determining
    E_ts2 = e_ts2(force, ts2, ts2_force_scaling)

    # concentration of OH-
    c_oh = 10 ** (-(14 - ph_value))
    c_oh_experiment = 10 ** (-(14 - 7.4))

    A1 = A / c_oh_experiment / 10
    k1 = A1 * np.exp(-E_ts1 / (K_b * temperature))
    k2 = A1 * np.exp(-E_ts2 / (K_b * temperature))  # k2' in the paper
    # reaction rate in 1/s (depending on how A is chosen)
    k_hyd = (k1 * k2 * c_oh) / (k1 + k2)

    logger.debug(f"TS1: {E_ts1} TS2: {E_ts2} Force: {force} k_hyd: {k_hyd}")

    return k_hyd


class HydrolysisReaction(ReactionPlugin):
    """A custom reaction plugin."""

    def get_recipe_collection(self, files: TaskFiles) -> RecipeCollection:
        logger = files.logger
        logger.debug(f"Calling hydrolysis reaction with config: {self.config}")

        # settings from the config
        self.max_sasa = self.config.max_sasa
        self.ph_value = self.config.ph_value
        self.external_force = self.config.external_force
        if self.config.eq_bond_lengths != "":
            self.eq_lengths = read_bond_lengths(self.config.eq_bond_lengths)
        else:
            self.eq_lengths = None

        self.theoretical = self.config.theoretical_rates.use
        if self.theoretical:
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
        self.sasa_per_bond: list[list[float]] = []
        # times are shared between all bonds
        self.times: list[float] = []
        self.timespans: list[tuple[float, float]]
        self.bonds = get_peptide_bonds_from_top(self.runmng.top)
        plumed = None
        if self.config.external_force == -1:
            plumed_out = files.input["plumed_out"]
            plumed_in = files.input["plumed"]
            if plumed_out is None or plumed_in is None:
                m = f"External force not specified but no plumed file found"
                logger.error(m)
                raise ValueError(m)
            self.distances = read_distances_dat(plumed_out)
            plumed = read_plumed(plumed_in)

            self.bond_to_plumed_id = {}
            for k, v in plumed["labeled_action"].items():
                if v["keyword"] != "DISTANCE":
                    continue
                atoms = v["atoms"]
                bondkey = tuple(sorted(atoms, key=int))
                self.bond_to_plumed_id[bondkey] = k

        self.init_universe(files)

        self.calculate_sasa(bonds=self.bonds, step=self.step)

        logger.debug(f"Got {len(self.times)} times for SASA calculation")
        logger.debug(
            f"Latest SASA values for each bond: {[ss[-1] for ss in self.sasa_per_bond]}"
        )

        for i, b in enumerate(self.bonds):
            r = Recipe(
                recipe_steps=DeferredRecipeSteps(
                    key=i, callback=self.get_steps_for_bond_at_i
                ),
                rates=self.sasas_to_rates(sasas=self.sasa_per_bond[i], bond=b),
                timespans=self.timespans,
            )
            self.recipes.append(r)

        return RecipeCollection(self.recipes)

    def sasas_to_rates(self, sasas: list[float], bond: Bond) -> list[float]:
        if self.external_force != -1:
            force = self.external_force
        else:
            # calculate force on bond
            ai = self.runmng.top.atoms[bond.ai]
            aj = self.runmng.top.atoms[bond.aj]
            id = self.bond_to_plumed_id.get((bond.ai, bond.aj))
            if id is None:
                raise ValueError(f"Could not find plumed id for bond {ai} {aj}")

            bondtype = self.runmng.top.ff.bondtypes.get((ai.type, aj.type))
            if bondtype is None:
                bondtype = self.runmng.top.ff.bondtypes.get((aj.type, ai.type))
            if bondtype is None:
                raise ValueError("Could not find bondtype")
            if bondtype.c0 is None or bondtype.c1 is None:
                raise ValueError("Could not find bondtype")
            b0 = float(bondtype.c0)
            kb = float(bondtype.c1)

            ds = np.asarray(self.distances[id])

            if self.eq_lengths is not None:
                observed_b0 = self.eq_lengths.get((bond.ai, bond.aj))
                if observed_b0 is None:
                    m = f"Could not find observed bond length in {self.config.eq_bond_lengths} using default b0"
                    logger.warning(m)
                    observed_b0 = b0
            else:
                observed_b0 = b0

            d = np.mean(ds)
            forces = ds_to_forces(
                ds=np.array(d), dissociation_energy=500, b0=observed_b0, kb=kb
            )
            force = float(np.mean(forces))
            # set negative average forces to 0
            force = max(force, 0)

        logger.debug(
            f"Calculating rates for bond {bond.ai} {bond.aj} with force {force}"
        )
        if self.theoretical:
            logger.info(f"Using theoretical rates")
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
            logger.info(f"Using experimental rates")
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

    def get_steps_for_bond_at_i(
        self, key: int, time_index: int, ttime: float
    ) -> list[RecipeStep]:
        b = self.bonds[key]
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

        # add 1 because the first frame is the initial frame
        # for which we don't report a rate
        frame = (time_index + 1) * self.step
        # FIXME: is this wrong?
        max_frames = len(self.u.trajectory)
        logger.info(
            f"Max frames: {max_frames}, frame: {frame}, time_index: {time_index}"
        )
        logger.info(f"with steps: {self.step}")

        frame = min(frame, max_frames - 1)
        logger.info(f"New frame: {frame}")

        logger.info(
            f"Hydrolyzing bond between C with ix {ix_cc} and N with ix {ix_n} at time_index {time_index}. With step {self.step} results in frame index={frame}"
        )

        self.u.trajectory[frame]
        logger.info(f"Time: {self.u.trajectory.time:3} ps")
        logger.info(f"Time from runmanager: {ttime} ps")

        # FIXME: ignore for now
        if round(self.u.trajectory.time, 3) != round(ttime, 3):
            m = f"Mismatch between time chosen by the runmanager and index received"
            logger.error(m)
            # raise ValueError(m)

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
                    # FIXME: handle this
                    logger.error(m)
                    return
                rtype_atom = residuetype.atoms.get(a.atom)
                if rtype_atom is None:
                    m = f"Could not find residuetype atom for {a.atom}"
                    # FIXME: handle this
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

        # # use slow growth to relax into new parameters
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

    def init_universe(self, files: TaskFiles):
        gro = files.input["gro"]
        xtc = files.input["xtc"]
        logger.debug(f"Using gro: {gro}")
        logger.debug(f"Using xtc: {xtc}")

        if xtc is None or gro is None:
            m = "No xtc file found"
            logger.error(m)
            raise ValueError(m)
        logger.info(f"Using xtc {xtc.name} in {xtc.parent.name} for trajectory")
        self.u = mda.Universe(str(gro), str(xtc))

        md_instance = xtc.stem
        timings = self.runmng.timeinfos.get(md_instance)
        if timings is None:
            m = f"No timings from mdp file found for {md_instance}"
            logger.error(m)
            raise ValueError(m)

        # reset to first frame just in case
        frame = self.u.trajectory[0]
        logger.info(f"First frame: {frame.frame} with time {frame.time} ps")
        self.xtc_trr_ratio = timings.xtc_nst / timings.trr_nst
        self.dt = timings.dt
        logger.info(f"timings: {timings}")

        # validate that the next frame is dt * xtc_nst ps away
        frame = self.u.trajectory[1]
        if round(frame.time, 3) != round(self.dt, 3) * timings.xtc_nst:
            m = f"Expected the next frame to be {self.dt * timings.xtc_nst} ps away, got {frame.time} ps"
            logger.error(m)
            raise ValueError(m)

        # reset to first frame
        frame = self.u.trajectory[0]

    def calculate_sasa(self, bonds, step: int = 1):
        logger.info(f"Calculating SASA for {len(bonds)} bonds. Step={step}")
        logger.info(f"Universe has {len(self.u.trajectory)} frames")
        self.times = []
        self.sasa_per_bond = [[] for _ in range(len(bonds))]
        sasa = MiniSasa(self.u)
        # skip the first frame
        for frame in self.u.trajectory[1:]:
            time = round(frame.time, 3)
            logger.debug(
                f"Calculating SASA for frame {frame.frame} with time rounded {time}"
            )
            self.times.append(time)
            sasa.update_structure()
            sasa.calc()
            for i, b in enumerate(bonds):
                # NOTE:this breaks if the protein is not the first
                # molecule in the universe
                ix_cc = int(b.ai) - 1  # C
                s = sasa.per_atom(ix_cc)
                self.sasa_per_bond[i].append(s)

        # but retain the first time [0.0] because
        # because it will combine with the next time
        # into the first timespan
        # the times are in ps. We round to fs.
        self.timespans = self.times_to_timespans([0.0] + self.times)
