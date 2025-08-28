# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ase",
# ]
# ///

from ase.build import bulk, molecule
from ase.io import write

a = 5.72
nacl_bulk = bulk('NaCl', 'rocksalt', a=a, cubic=True)
nacl_crystal = nacl_bulk.repeat((2, 2, 2))

write("nacl.xyz", nacl_crystal)

h2o = molecule("H2O")
write("h2o.xyz", h2o)
