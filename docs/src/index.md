# MagnetoPhotonic.jl

A Julia package for **magneto-photonic time-domain electromagnetics**: a general,
MEEP-like **1-D / 2-D / 3-D FDTD** engine coupled to a tunable **magneto-optic** material
model (four-temperature electron–lattice–spin dynamics + Landau–Lifshitz–Bloch
magnetization + Kerr/Faraday gyrotropy) for simulating **all-optical magnetization
switching** in integrated photonic devices.

!!! note "Umbrella package"
    `MagnetoPhotonic` collects magneto-photonic solvers under one roof. It ships an **FDTD**
    solver today; a **Discontinuous Galerkin Time-Domain (DGTD)** solver is planned.

## Highlights

- **General EM-FDTD** in 1-D, 2-D (TM/TE) and 3-D on a non-uniform Yee grid.
- **CFS-CPML** absorbing boundaries (ψ-convolution) in every dimension; also `PEC`, `Periodic`.
- **Materials by refractive index** and **Drude–Lorentz ADE dispersion** in all dimensions.
- **Point / plane-wave** sources, and **point / field / flux / DFT** monitors.
- **Magneto-optic** stack: 4-temperature thermal bath + LLB magnetization + magneto-optic
  gyration, driving deterministic all-optical switching.
- **88 passing tests** on Julia 1.12 (CPU).

## Contents

```@contents
Pages = ["getting_started.md", "em_fdtd.md", "magneto_optics.md", "api.md"]
Depth = 2
```

## Installation

The package targets **Julia ≥ 1.12**.

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

From your own code:

```julia
using Pkg; Pkg.develop(path="path/to/MagnetoPhotonic.jl")
using MagnetoPhotonic
```

## A 30-second example

```julia
using MagnetoPhotonic
p = FDTDParams()
pulse = GaussianPulse(; amplitude=1.0, tau=4e-15, t0=16e-15, omega=2pi*p.c0/800e-9)

sim = Simulation(; cell=(4e-6,), dx=10e-9, dimension=1,
                 sources=[PointSource(pulse, :Ez, 2e-6)], boundary=PML(20))
mon = PointMonitor(:Ez, 3e-6)
run!(sim; until=120e-15, monitors=[mon])
```

```text
steps=14390   peak |Ez| @ probe = 1.942   energy final/peak = 3.58e-22
```

## License

[MIT](https://opensource.org/licenses/MIT) © 2026 Muhammad Arief Mulyana.
