# Getting Started

## Installation

`MagnetoPhotonic` targets **Julia ≥ 1.12**. It is **not yet in the General registry**, so install
it directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/physarief78/MagnetoPhotonic.jl")
```

or, equivalently, from the Pkg REPL (press `]`):

```
pkg> add https://github.com/physarief78/MagnetoPhotonic.jl
```

Then load it:

```julia
using MagnetoPhotonic
```

### Optional extensions

The CPU path needs nothing else. GPU execution, HDF5 I/O, and plotting are **optional weak
dependencies** loaded as package extensions — add the one you want and it activates automatically:

```julia
Pkg.add("CUDA")        # CUDA GPU backend
Pkg.add("HDF5")        # HDF5 schema I/O
Pkg.add("CairoMakie")  # figures / field video
```

See [Capabilities](@ref) for the backend and precision details.

### Developing the package

To hack on the package or run its test suite, clone it and use its own project:

```julia
# git clone https://github.com/physarief78/MagnetoPhotonic.jl
julia --project=MagnetoPhotonic.jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

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

- [Fundamentals](@ref) — the physics & numerics behind the solver.
- [EM-FDTD Tutorial](@ref) — 1-D/2-D/3-D, materials, boundaries, monitors, validation.
- [Magneto-Optic Switching](@ref) — the coupled 4TM + LLB all-optical switching model.
- [Capabilities](@ref) — what the package can compute, backends/GPU, performance, validation.
- [API Reference](@ref) — the full public surface.
