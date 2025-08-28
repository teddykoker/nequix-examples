# NaCl Dissolution

<img src="nacl_h2o_opt.png" alt="NaCl in water" width="400" />

Based on experiments from *Crumbling crystals: on the dissolution mechanism of
NaCl in water* [(O'Neill et al.)](https://doi.org/10.1039/D4CP03115F), repeated
in *A foundation model for atomistic materials chemistry* [(Batatia et
al.)](https://arxiv.org/abs/2401.00096). 625 water molecules are placed around a
4 x 4 x 4 NaCl structure in a 27.8 Å cubic simulation cell, a NaCl
concentrations in water of 2.84 mol/kg. We then run NVT molecular dynamics at
400K for 0.2 ns with a 1 fs timestep.

Run NVT molecular dynamics on `nacl_h2o_opt.xyz`

```bash
uv run nvt.py
```

Trajectory will be saved to `nvt.traj`.

## Creating starting configuration

This has already been done to create `nacl_h2o_opt.xyz`, but I include the steps
to create it for reproducability. Create reference structures for 4 x 4 x 4 NaCl
nano crystal and H2O:

```bash
uv run nacl_h2o.py
```

Pack water molecules around crystal (requires
[Julia](https://julialang.org/install/), as it uses
[Packmol.jl](https://github.com/m3g/Packmol.jl)):


```bash
julia packmol.jl nacl_h2o.inp
```

Optimize structure with Nequix:

```bash
uv run opt.py
```