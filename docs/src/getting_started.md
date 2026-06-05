# Getting Started

## Installation

`MagnetoPhotonic` targets **Julia ≥ 1.12**. From the repository root:

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'      # 88 tests, CPU only
```

The compute path is pure Julia/CPU. `CUDA`, `HDF5` and `CairoMakie` are *optional* weak
dependencies loaded as package extensions (device arrays, HDF5 I/O, plotting/video); none
are required to run or test the package.

## Anatomy of a `Simulation`

The general EM entry point is the `Simulation` constructor. You provide:

| Keyword | Meaning |
|---|---|
| `cell` | physical extent — `(Lx,)`, `(Lx,Ly)` or `(Lx,Ly,Lz)` |
| `dx` (or `resolution`) | uniform cell size |
| `dimension` | `1`, `2`, or `3` |
| `mode` | `:TM` (`Ez,Hx,Hy`) or `:TE` (`Hz,Ex,Ey`) — 2-D only |
| `geometry` | a `Scene` of shapes + materials (default: empty/vacuum) |
| `sources` | a vector of `PointSource` / `PlaneSource` |
| `boundary` | `PML(n)`, `PEC()`, or `Periodic()` |
| `courant` | CFL fraction (default `0.5`) |

`dimension` defaults to `length(cell)`, and `dt` is set automatically from the CFL condition.

## Your first simulation

```julia
using MagnetoPhotonic

p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=4e-15, t0=16e-15, omega=2pi*p.c0/800e-9)

sim = Simulation(; cell=(4e-6,), dx=10e-9, dimension=1,
                 sources=[PointSource(pulse, :Ez, 2e-6)],
                 boundary=PML(20), courant=0.5)

mon = PointMonitor(:Ez, 3e-6)        # probe the field at x = 3 µm
run!(sim; until=120e-15, monitors=[mon])

using Printf
@printf("peak |Ez| = %.3f\n", maximum(abs, mon.values))
```

```text
peak |Ez| = 1.942
```

## Stepping and running

- `step!(sim)` advances one time step.
- `run!(sim, nsteps; monitors=…, callback=…)` runs a fixed number of steps.
- `run!(sim; until=T, monitors=…)` runs until physical time `T`.

A `callback` receives the `sim` each step — handy for custom diagnostics:

```julia
peak = Ref(0.0)
run!(sim, 2000; callback = s -> (peak[] = max(peak[], field_energy(s.fields, s.grid, p))))
```

## Where next

- [EM-FDTD Tutorial](@ref) — 1-D/2-D/3-D, materials, boundaries, monitors, validation.
- [Magneto-Optic Switching](@ref) — the coupled 4TM + LLB all-optical switching model.
- [API Reference](@ref) — the full public surface.
