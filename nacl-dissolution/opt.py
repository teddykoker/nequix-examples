# /// script
# requires-python = ">=3.11, <3.12"
# dependencies = [
#     "ase",
#     "nequix",
# ]
#
# ///

from ase.io import read, write
from ase.optimize import FIRE
from nequix.calculator import NequixCalculator

atoms = read("nacl_h2o.xyz")
atoms.calc = NequixCalculator(capacity_multiplier=1.05)
atoms.cell = [27.8, 27.8, 27.8]
atoms.pbc = True

opt = FIRE(atoms, trajectory="opt.traj", logfile="opt.log")
opt.run(fmax=0.5, steps=200)
write("nacl_h2o_opt.xyz", atoms)