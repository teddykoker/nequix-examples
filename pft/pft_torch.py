# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "phonopy", "seekpath", "numpy", "nequix[torch]", "ase"]
# ///

import time

import ase
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import torch
from tqdm import tqdm

from nequix.data import atomic_numbers_to_indices, preprocess_graph
from nequix.torch.model import NequixTorch
from nequix.calculator import NequixCalculator


def train_hessian(model, graph, ref_hessian, n_epochs=200, lr=0.003):
    def energy_fn(pos_flat):
        pos = pos_flat.view(*graph["positions"].shape)
        offset = graph["shifts"] @ graph["cell"]
        disp = pos[graph["senders"]] - pos[graph["receivers"]] + offset
        return model.node_energies(
            disp, graph["species"], graph["senders"], graph["receivers"]
        ).sum()

    grad_fn = torch.func.grad(energy_fn)
    hessian_fn = torch.compile(torch.func.jacfwd(grad_fn))

    pos_flat = graph["positions"].flatten()
    losses = []
    step_times = []  # profile
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = graph["positions"].device  # profile
    if device.type == "cuda":  # profile
        torch.cuda.reset_peak_memory_stats(device)  # profile

    for epoch in tqdm(range(n_epochs)):
        if device.type == "cuda":  # profile
            torch.cuda.synchronize()  # profile
        step_start = time.perf_counter()  # profile
        optimizer.zero_grad()
        hessian = hessian_fn(pos_flat)
        loss = (hessian - ref_hessian).abs().mean()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":  # profile
            torch.cuda.synchronize()  # profile
        step_time = time.perf_counter() - step_start  # profile
        if epoch >= 5:  # profile
            step_times.append(step_time)  # profile
        losses.append(loss.item())

    avg_step_time = sum(step_times) / len(step_times)  # profile
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3  # profile
    return losses, avg_step_time, peak_mem


def train_hvp(model, graph, ref_hessian, n_epochs=200, lr=0.003):
    def energy_fn(pos_flat):
        pos = pos_flat.view(*graph["positions"].shape)
        offset = graph["shifts"] @ graph["cell"]
        disp = pos[graph["senders"]] - pos[graph["receivers"]] + offset
        return model.node_energies(
            disp, graph["species"], graph["senders"], graph["receivers"]
        ).sum()

    grad_fn = torch.func.grad(energy_fn)
    hvp_fn = torch.compile(lambda x, v: torch.func.jvp(grad_fn, (x,), (v,))[1])

    pos_flat = graph["positions"].flatten()
    losses = []
    step_times = []  # profile
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = graph["positions"].device  # profile
    if device.type == "cuda":  # profile
        torch.cuda.reset_peak_memory_stats(device)  # profile

    for epoch in tqdm(range(n_epochs)):
        if device.type == "cuda":  # profile
            torch.cuda.synchronize()  # profile
        step_start = time.perf_counter()  # profile
        optimizer.zero_grad()
        idx = torch.randint(pos_flat.shape[0], (1,), device=device).item()
        v = torch.zeros_like(pos_flat)
        v[idx] = 1.0
        hvp = hvp_fn(pos_flat, v)
        loss = (hvp - ref_hessian[:, idx]).abs().mean()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":  # profile
            torch.cuda.synchronize()  # profile
        step_time = time.perf_counter() - step_start  # profile
        if epoch >= 5:  # profile
            step_times.append(step_time)  # profile
        losses.append(loss.item())

    avg_step_time = sum(step_times) / len(step_times)  # profile
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3  # profile
    return losses, avg_step_time, peak_mem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 200
    cutoff = 5.0

    ph_ref = phonopy.load("../phonon/mp-149.yaml")
    ph_ref.produce_force_constants()

    atoms = ase.Atoms(
        symbols=ph_ref.supercell.symbols,
        positions=ph_ref.supercell.positions,
        cell=ph_ref.supercell.cell,
        pbc=True,
    )
    atom_indices = atomic_numbers_to_indices(set(atoms.get_atomic_numbers()))
    g = preprocess_graph(atoms, atom_indices, cutoff, targets=False)
    graph = {
        k: torch.as_tensor(
            v, device=device, dtype=torch.float32 if v.dtype == np.float32 else torch.long
        )
        for k, v in g.items()
    }

    # (n, n, 3, 3) -> (3n, 3n)
    ref_hessian = (
        torch.tensor(ph_ref.force_constants, dtype=torch.float32, device=device)
        .swapaxes(1, 2)
        .reshape(g["n_node"][0] * 3, g["n_node"][0] * 3)
    )

    torch.manual_seed(0)
    model = NequixTorch(
        n_species=1,
        cutoff=cutoff,
        hidden_irreps="32x0e + 32x1o + 32x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
    ).to(device)
    loss_full, avg_step_full, mem_full = train_hessian(model, graph, ref_hessian, n_epochs=n_epochs)

    torch.manual_seed(0)
    model = NequixTorch(
        n_species=1,
        cutoff=cutoff,
        hidden_irreps="32x0e + 32x1o + 32x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
    ).to(device)
    loss_hvp, avg_step_hvp, mem_hvp = train_hvp(model, graph, ref_hessian, n_epochs=n_epochs)

    print(f"Full: {avg_step_full * 1000:.1f}ms/step, {mem_full:.2f}GB")
    print(f"HVP: {avg_step_hvp * 1000:.1f}ms/step, {mem_hvp:.2f}GB")
    print(f"Speedup: {avg_step_full / avg_step_hvp:.1f}x, Memory: {mem_full / mem_hvp:.1f}x")

    calc = NequixCalculator("nequix-mp-1", backend="torch", use_kernel=False)
    model = calc.model.to(device)
    model.train()
    g = preprocess_graph(atoms, calc.atom_indices, calc.cutoff, targets=False)
    graph = {
        k: torch.as_tensor(
            v, device=device, dtype=torch.float32 if v.dtype == np.float32 else torch.long
        )
        for k, v in g.items()
    }
    torch.manual_seed(0)
    loss_pre, avg_step_pre, mem_pre = train_hvp(
        model, graph, ref_hessian, n_epochs=n_epochs, lr=0.0001
    )
    print(f"Pretrained: {avg_step_pre * 1000:.1f}ms/step, {mem_pre:.2f}GB")

    steps = np.arange(len(loss_full))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        steps,
        loss_full,
        "b-",
        lw=2,
        label=f"Nequix 80K Hessian ({avg_step_full * 1000:.1f}ms/step, {mem_full:.2f}GB)",
    )
    ax.plot(
        steps,
        loss_hvp,
        "r--",
        lw=2,
        label=f"Nequix 80K HVP ({avg_step_hvp * 1000:.1f}ms/step, {mem_hvp:.2f}GB)",
    )
    ax.plot(
        steps,
        loss_pre,
        "k--",
        lw=2,
        label=f"Nequix MP 700K HVP ({avg_step_pre * 1000:.1f}ms/step, {mem_pre:.2f}GB)",
    )
    ax.set(
        xlabel="Step",
        ylabel=r"Hessian MAE [meV/Å$^2$/atom]",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/pft_loss_torch.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
