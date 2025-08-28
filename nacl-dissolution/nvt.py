# /// script
# requires-python = ">=3.11, <3.12"
# dependencies = [
#     "ase",
#     "nequix",
#     "tqdm",
# ]
#
# ///

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.bussi import Bussi
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from nequix.calculator import NequixCalculator
from tqdm import tqdm

atoms = read("nacl_h2o_opt.xyz")
atoms.calc = NequixCalculator(capacity_multiplier=1.05)
atoms.cell = [27.8, 27.8, 27.8]
atoms.pbc = True
timestep_fs = 1.0
steps = 200_000
temperature_K = 400.0
thermostat_tau_fs = 200.0


MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)

dyn = Bussi(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    taut=thermostat_tau_fs * units.fs,
)


traj = Trajectory(
    "/data/NFS/radish/tekoker/nequix-examples/nacl-dissolution/nvt.traj", "w", atoms
)
dyn.attach(traj.write, interval=10)
logger = MDLogger(
    dyn, atoms, "nvt.log", header=True, stress=False, peratom=False, mode="w"
)
dyn.attach(logger, interval=10)

update_interval = 10
pbar = tqdm(total=steps, desc="NVT", unit="step")


def _update_pbar():
    pbar.update(update_interval)


dyn.attach(_update_pbar, interval=update_interval)

# Run dynamics
dyn.run(steps)
pbar.update(steps - pbar.n)
pbar.close()
