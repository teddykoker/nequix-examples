# /// script
# requires-python = ">=3.10"
# dependencies = ["ase", "phonopy", "nequix", "matplotlib", "seekpath"]
# ///

import matplotlib.pyplot as plt
import numpy as np
import phonopy
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from mpl_toolkits.axes_grid1 import ImageGrid
from phonopy.phonon.band_structure import BandPlot
import equinox as eqx
import jax
import jax.numpy as jnp
from nequix.calculator import NequixCalculator
from nequix.data import dict_to_graphstuple, atomic_numbers_to_indices, preprocess_graph
from nequix.pft.hessian import hessian_linearized

distance = 0.01
filename = "mp-149.yaml"
model_names = ["nequix-mp-1", "nequix-mp-1-pft", "nequix-mp-1-pft"]
autodiffs = [False, False, True]

def relax(atoms):
    atoms = FrechetCellFilter(atoms)
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.01)


ph_ref = phonopy.load(filename)
bs_mlips = []

for model_name, autodiff in zip(model_names, autodiffs):
    ase_cell = Atoms(
        cell=ph_ref.unitcell.cell,
        symbols=ph_ref.unitcell.symbols,
        scaled_positions=ph_ref.unitcell.scaled_positions,
        pbc=True,
    )
    ase_cell.calc = NequixCalculator(model_name=model_name)
    relax(ase_cell)

    ph_atoms = phonopy.structure.atoms.PhonopyAtoms(
        cell=ase_cell.get_cell(),
        scaled_positions=ase_cell.get_scaled_positions(),
        symbols=ase_cell.get_chemical_symbols(),
    )
    ph_mlip = phonopy.Phonopy(
        ph_atoms,
        supercell_matrix=ph_ref.supercell_matrix,
        primitive_matrix=ph_ref.primitive_matrix,
    )

    if autodiff:
        # calculate force constants using linearized Hessian
        ase_supercell = Atoms(
            cell=ph_mlip.supercell.cell,
            symbols=ph_mlip.supercell.symbols,
            scaled_positions=ph_mlip.supercell.scaled_positions,
            pbc=True,
        )
        atom_indices = atomic_numbers_to_indices(ase_cell.calc.config["atomic_numbers"])
        graph = dict_to_graphstuple(
            preprocess_graph(
                ase_supercell, atom_indices, ase_cell.calc.config["cutoff"], False
            )
        )
        hessian = hessian_linearized(ase_cell.calc.model, graph)
        ph_mlip.force_constants = np.array(hessian, copy=True)

    else:
        # calculate force constants using finite displacements
        forcesets = []
        ph_mlip.generate_displacements(distance=distance)
        for supercell in ph_mlip.supercells_with_displacements:
            scell = Atoms(
                cell=supercell.cell,
                symbols=supercell.symbols,
                scaled_positions=supercell.scaled_positions,
                pbc=True,
            )
            scell.calc = NequixCalculator(model_name=model_name)
            forces = scell.get_forces()
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]

            forcesets.append(forces)
        ph_mlip.forces = forcesets
        ph_mlip.produce_force_constants()

    ph_mlip.symmetrize_force_constants()
    ph_mlip.auto_band_structure()
    bs_mlips.append(ph_mlip._band_structure)

ph_ref.auto_band_structure()
bs_ref = ph_ref._band_structure


# plot band structures
n = len([x for x in bs_ref.path_connections if not x])
fig = plt.figure(figsize=(5, 3))
axs = ImageGrid(
    fig,
    111,
    nrows_ncols=(1, n),
    axes_pad=0.11,
    label_mode="L",
)

bp = BandPlot(axs)
bp.decorate(
    bs_ref._labels, bs_ref._path_connections, bs_ref._frequencies, bs_ref._distances
)
bp.plot(
    bs_ref._distances,
    bs_ref._frequencies,
    bs_ref._path_connections,
    fmt="k-",
    label="DFT/PBE",
)
colors = ["r", "b", "g"]
for model_name, bs_mlip, color, autodiff in zip(
    model_names, bs_mlips, colors, autodiffs
):
    bp.plot(
        bs_mlip._distances,
        bs_mlip._frequencies,
        bs_mlip._path_connections,
        fmt=f"{color}" + ("--" if autodiff else "-"),
        label=model_name + (" (autodiff)" if autodiff else ""),
    )

axs[0].legend_.remove()
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
plt.subplots_adjust(top=0.8)
plt.savefig(f"mp-149_bs.png", bbox_inches="tight", dpi=300)
plt.close()
