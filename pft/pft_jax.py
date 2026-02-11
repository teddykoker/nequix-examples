# /// script
# requires-python = ">=3.10"
# dependencies = ["jax[cuda12]", "equinox", "optax", "phonopy", "numpy", "nequix"]
# ///

import time

import ase
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import phonopy
from nequix.data import atomic_numbers_to_indices, preprocess_graph
from nequix.model import Nequix
from nequix.calculator import NequixCalculator
from tqdm import tqdm


def train_hessian(model, graph, ref_hessian, n_epochs=200, lr=0.003):
    def energy_fn(model, pos_flat):
        pos = pos_flat.reshape(graph["positions"].shape)
        offset = graph["shifts"] @ graph["cell"]
        disp = pos[graph["senders"]] - pos[graph["receivers"]] + offset
        return model.node_energies(
            disp, graph["species"], graph["senders"], graph["receivers"]
        ).sum()

    grad_fn = jax.grad(energy_fn, argnums=1)

    def hessian_fn(model, x):
        return jax.jacfwd(lambda pos: grad_fn(model, pos))(x)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, pos_flat):
        def loss_fn(model):
            hessian = hessian_fn(model, pos_flat)
            return jnp.abs(hessian - ref_hessian).mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, loss

    pos_flat = graph["positions"].flatten()
    losses = []
    step_times = []  # profile
    for epoch in tqdm(range(n_epochs)):
        jax.block_until_ready(model)  # profile
        step_start = time.perf_counter()  # profile
        model, opt_state, loss = train_step(model, opt_state, pos_flat)
        jax.block_until_ready(model)  # profile
        step_time = time.perf_counter() - step_start  # profile
        losses.append(float(loss))
        if epoch >= 5:  # profile
            step_times.append(step_time)  # profile

    avg_step_time = sum(step_times) / len(step_times)  # profile
    device = jax.devices()[0]  # profile
    mem_stats = device.memory_stats()  # profile
    peak_mem = mem_stats.get("peak_bytes_in_use", 0) / 1024**3  # profile
    return losses, avg_step_time, peak_mem


def train_hvp(model, graph, ref_hessian, n_epochs=200, lr=0.003):
    def energy_fn(model, pos_flat):
        pos = pos_flat.reshape(graph["positions"].shape)
        offset = graph["shifts"] @ graph["cell"]
        disp = pos[graph["senders"]] - pos[graph["receivers"]] + offset
        return model.node_energies(
            disp, graph["species"], graph["senders"], graph["receivers"]
        ).sum()

    grad_fn = jax.grad(energy_fn, argnums=1)

    def hvp_fn(model, x, v):
        return jax.jvp(lambda pos: grad_fn(model, pos), (x,), (v,))[1]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, pos_flat, idx):
        def loss_fn(model):
            v = jnp.zeros_like(pos_flat).at[idx].set(1.0)
            hvp = hvp_fn(model, pos_flat, v)
            return jnp.abs(hvp - ref_hessian[:, idx]).mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, loss

    pos_flat = graph["positions"].flatten()
    losses = []
    step_times = []  # profile
    rng_key = jax.random.key(0)

    for epoch in tqdm(range(n_epochs)):
        jax.block_until_ready(model)  # profile
        step_start = time.perf_counter()  # profile
        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, (), 0, pos_flat.shape[0])
        model, opt_state, loss = train_step(model, opt_state, pos_flat, idx)
        jax.block_until_ready(model)  # profile
        step_time = time.perf_counter() - step_start  # profile
        losses.append(float(loss))
        if epoch >= 5:  # profile
            step_times.append(step_time)  # profile

    avg_step_time = sum(step_times) / len(step_times)  # profile
    device = jax.devices()[0]  # profile
    mem_stats = device.memory_stats()  # profile
    peak_mem = mem_stats.get("peak_bytes_in_use", 0) / 1024**3  # profile
    return losses, avg_step_time, peak_mem


def main():
    n_epochs = 200
    cutoff = 5.0

    ph_ref = phonopy.load("../phonon/mp-149.yaml")
    ph_ref.produce_force_constants()

    plt.figure(figsize=(6, 4))
    plt.imshow(np.log(np.abs(ph_ref.force_constants[:, :, 0, 0])), cmap="plasma")
    plt.colorbar(label=r"$\log(|\Phi|)$")
    plt.savefig("force_constants.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    atoms = ase.Atoms(
        symbols=ph_ref.supercell.symbols,
        positions=ph_ref.supercell.positions,
        cell=ph_ref.supercell.cell,
        pbc=True,
    )
    atom_indices = atomic_numbers_to_indices(set(atoms.get_atomic_numbers()))
    graph = preprocess_graph(atoms, atom_indices, cutoff, targets=False)

    # (n, n, 3, 3) -> (3n, 3n)
    ref_hessian = (
        jnp.array(ph_ref.force_constants, dtype=jnp.float32)
        .swapaxes(1, 2)
        .reshape(graph["n_node"][0] * 3, graph["n_node"][0] * 3)
    )

    key = jax.random.key(0)
    model = Nequix(
        key=key,
        n_species=1,
        cutoff=cutoff,
        hidden_irreps="32x0e + 32x1o + 32x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
    )

    loss_hvp, avg_step_hvp, mem_hvp = train_hvp(model, graph, ref_hessian, n_epochs=n_epochs)

    calc = NequixCalculator("nequix-mp-1")
    model = calc.model
    graph = preprocess_graph(atoms, calc.atom_indices, calc.cutoff, targets=False)
    loss_pre, avg_step_pre, mem_pre = train_hvp(
        model, graph, ref_hessian, n_epochs=n_epochs, lr=0.0001
    )
    print(f"Pretrained: {avg_step_pre * 1000:.1f}ms/step, {mem_pre:.2f}GB")
    key = jax.random.key(0)
    model = Nequix(
        key=key,
        n_species=1,
        cutoff=cutoff,
        hidden_irreps="32x0e + 32x1o + 32x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
    )
    graph = preprocess_graph(atoms, atom_indices, cutoff, targets=False)
    loss_full, avg_step_full, mem_full = train_hessian(model, graph, ref_hessian, n_epochs=n_epochs)

    print(f"Full: {avg_step_full * 1000:.1f}ms/step, {mem_full:.2f}GB")
    print(f"HVP: {avg_step_hvp * 1000:.1f}ms/step, {mem_hvp:.2f}GB")
    print(f"Speedup: {avg_step_full / avg_step_hvp:.1f}x, Memory: {mem_full / mem_hvp:.1f}x")

    steps = np.arange(len(loss_full))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(
        steps,
        loss_full,
        "b-",
        lw=2,
        label=f"Nequix 80K Hessian ({avg_step_full * 1000:.1f}ms/step, {mem_full:.2f}GB)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set(
        xlabel="Step",
        ylabel=r"Hessian MAE [meV/Å$^2$/atom]",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/pft_loss_jax_full.png", dpi=300, bbox_inches="tight")
    ax.plot(
        steps,
        loss_hvp,
        "r--",
        lw=2,
        label=f"Nequix 80K HVP ({avg_step_hvp * 1000:.1f}ms/step, {mem_hvp:.2f}GB)",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/pft_loss_hvp.png", dpi=300, bbox_inches="tight")
    ax.plot(
        steps,
        loss_pre,
        "k-",
        lw=2,
        label=f"Nequix MP 700K HVP ({avg_step_pre * 1000:.1f}ms/step, {mem_pre:.2f}GB)",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/pft_loss_pretrained.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
