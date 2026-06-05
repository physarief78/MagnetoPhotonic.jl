# MagnetoPhotonic

`MagnetoPhotonic` packages the lab's FDTD + magneto-optics code into a reusable,
tested Julia library. It leaves the original standalone scripts untouched and
provides a **coupled** simulation stack:

- **Geometry / obstacles** — `Vec2`, polygon/waveguide/tapered/cylinder/letter shapes,
  point-in-polygon rasterization with sub-pixel fill fractions, a `Scene` builder, and a
  named device library (`not_gate_60um`, `passive_waveguide`, `hm_test_pattern`).
- **Grids** — uniform, graded, and non-uniform propagation Yee axes with dual spacings + CFL.
- **FDTD core** — non-uniform split-field Yee `update_H!`/`update_E!` with **CFS-CPML**
  absorbing boundaries (κ-stretch + ψ convolution).
- **Material models** — Drude–Lorentz **ADE dispersion** and **magneto-optic gyration**
  (Ey↔Ez) updates, with pole fitting verified by `discrete_pole_chi`.
- **Customizable physics model** — a tunable two-sublattice GdFeCo model: the
  **four-temperature (3TM/4TM)** electron/lattice/spin baths coupled to the
  **Landau–Lifshitz–Bloch (LLB)** magnetization with the validation-matched TM-first /
  RE-delayed branch-selection channel → deterministic all-optical switching.
- **Drivers** — `run_pump_probe_sim` (pump → relax → switching readout) and
  `run_convergence_study` (Cauchy L2-vs-Δx, plus a transmitted spectrum).
- **Visualization / IO** — device OBJ + SVG export, field-slice extraction, an optional
  Makie heatmap-video / scene-plan renderer, HDF5/serialized state, and a DFT spectrum.

The original CUDA scripts remain the validation reference:

- `Main_Research/pump_probe_switching_empirical_params.jl`
- `Convergence_study/3D/3D_dielectric.jl`, `Convergence_study/3D/3D_dispersive.jl`
- `Device_Design/*.jl`

## Quick Start

From the `Lab` directory:

```julia
julia --project=MagnetoPhotonic.jl -e "using Pkg; Pkg.test()"        # 12 testsets, all CPU
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/examples/pump_probe_run.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/examples/convergence_dielectric.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/examples/convergence_dispersive.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/examples/device_render_3d.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/examples/device_cross_section.jl
```

## General EM-FDTD API

The package also provides an additive MEEP-like EM layer for 1-D, 2-D, and 3-D
free-space/material simulations. The coupled magneto-optic `FDTDState` path remains
available; `Simulation` is the cleaner general EM entry point.

```julia
using MagnetoPhotonic

p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=8e-15, t0=32e-15,
                      omega=2pi * p.c0 / 800e-9)

sim = Simulation(; cell=(2e-6,), dx=5e-9, dimension=1,
                 sources=[PointSource(pulse, :Ez, 0.4e-6)],
                 boundary=PML(20), params=p)
mon = PointMonitor(:Ez, 1.4e-6)
run!(sim; until=80e-15, monitors=[mon])
```

Use `dimension=2, mode=:TM` for `(Ez,Hx,Hy)`, `dimension=2, mode=:TE` for
`(Hz,Ex,Ey)`, and `dimension=3` for the pure-EM 3-D Yee path. The public
`Simulation(; ...)` constructor returns a concretely typed driver, so the
time-stepping path specializes on dimension, mode, grid, field, CPML, and ADE
state types.

`PML(n)` uses the same CFS-CPML psi-convolution form in 1-D, 2-D, and 3-D.
`FluxMonitor(axis, x)` returns signed, plane-integrated Poynting flux through the
requested normal axis. `Periodic()` is implemented as a post-update boundary
copy; that is useful for simple wraparound tests, but true Bloch/bandstructure
work should use wrapped derivative stencils instead.

Materials can be specified by refractive index:

```julia
scene = Scene()
add_shape!(scene, Box(1e-6, 2e-6, -1, 1, -1, 1), Material("n=2 slab"; index=2.0))
```

Drude-Lorentz ADE poles are supported through `Material(...; poles=...)` in all
dimensions. The diagonal permittivity used by the grid is the nondispersive
background `epsr` plus the ADE pole response. Use `epsr=1.0` when the fitted
poles already represent the full target dielectric response relative to vacuum;
otherwise set `epsr` to the intended background only.

Convergence replication scripts live under `studies/`:

```julia
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/studies/conv_wave_in_a_box.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/studies/conv_wave_with_pml.jl
julia --project=MagnetoPhotonic.jl MagnetoPhotonic.jl/studies/conv_3d.jl
```

`pump_probe_run.jl` prints a full switch on a small CPU-scale grid:

```
mean m_TM_x : 1.0   -> -0.99      # FeCo reversed
mean m_RE_x : -0.998 -> 0.968     # Gd reversed
switched fraction : 1.0
```

## Tuning the model

Every GdFeCo parameter is a keyword with the empirical default; override any to
build a custom model and drop it straight into a simulation:

```julia
model = MagnetoOpticModel(; T_Curie=560.0, Q_voigt_TM=0.025, alpha0_TM=0.04)
run_pump_probe_sim(config; model=model, thermal_kick=1150.0, relax_steps=8000)
```

## Coupled solver API

```julia
using MagnetoPhotonic

grid  = uniform_grid((0.0, 1.5e-6), (-0.4e-6, 0.4e-6), (-0.4e-6, 0.4e-6), 0.1e-6)
scene = Scene()
add_shape!(scene, Box(0.0,1.5e-6, -0.15e-6,0.15e-6, -0.15e-6,0.15e-6), Material("Si3N4"; epsr=4.0))
add_shape!(scene, Box(0.7e-6,0.8e-6, -0.15e-6,0.15e-6, -0.15e-6,0.15e-6),
           Material("GdFeCo"; epsr=1.0, model=MagnetoOpticModel()))   # marks magneto-optic cells
geo = rasterize(scene, grid; subpixel=1)

p     = FDTDParams(800e-9)
pulse = GaussianPulse(; amplitude=1.0, tau=15e-15, t0=45e-15, omega=2pi*p.c0/800e-9)
state = FDTDState(grid, geo; dt=cfl_dt(grid, p; courant=0.3), params=p,
                  source=(pulse, :Ez, (3, 4, 4)), T=Float64,
                  model=MagnetoOpticModel(), n_pml=6,
                  enable_magneto_optic=true, multiphysics_every=4)

run!(state, 300)            # pump: Yee + CPML + ADE + MO + 4TM + LLB
for _ in 1:6000; relax_step!(state, 1e-15); end   # cool-down: 4TM + LLB only
```

## Backend status (honest scope)

The compute kernels (`update_H!`/`update_E!`, ADE, gyration, 4TM, LLB) are **CPU**
implementations and are what the test suite validates. CUDA, HDF5, and Makie are
optional weak-dependency **extensions**: loading `CUDA` enables device-array allocation,
`HDF5` enables `save_hdf5_state`, and `CairoMakie` enables `plot_scene` /
`render_field_video`. Porting the inner loops to KernelAbstractions for true GPU
execution, and running the full quantitative validation gate against the original CUDA
scripts at production scale, remain the next step.
