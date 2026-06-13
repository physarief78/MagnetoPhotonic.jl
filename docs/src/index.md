# MagnetoPhotonic.jl

A Julia package for **magneto-photonic time-domain electromagnetics**: a general,
Meep-like **1-D / 2-D / 3-D FDTD** engine *coupled* to a tunable, **dynamical magneto-optic**
material model (four-temperature electron–lattice–spin dynamics + two-sublattice
Landau–Lifshitz–Bloch magnetization + Kerr/Faraday gyrotropy) for simulating **all-optical
magnetization switching** in integrated photonic devices.

!!! note "Umbrella package"
    `MagnetoPhotonic` collects magneto-photonic solvers under one roof. It ships an **FDTD**
    solver today; a **Discontinuous Galerkin Time-Domain (DGTD)** solver is planned.

!!! tip "Origin & philosophy"
    This package generalizes the from-scratch, GPU-accelerated multiphysics solver built for an
    UNPAD physics thesis on an all-optical magneto-optical NOT gate. It is **Meep-like** in its
    electromagnetic core but goes beyond it: the magnetization is a *dynamical* field that the
    optical pump can reverse, not a static input. See the [Introduction](@ref) for the full
    rationale and the Meep comparison, and [Capabilities](@ref) for the validation against the
    production reference.

## Highlights

- **General EM-FDTD** in 1-D, 2-D (TM/TE) and 3-D on a non-uniform Yee grid.
- **CFS-CPML** absorbing boundaries (ψ-convolution) in every dimension; also `PEC`, `Periodic`.
- **Materials by refractive index** and **Drude–Lorentz ADE dispersion** in all dimensions.
- **Point / plane-wave / guided-mode** sources, and **point / field / flux / DFT / polarimetric**
  monitors.
- **Coupled magneto-optic** stack: 4-temperature thermal bath + LLB magnetization + magnetization-
  dependent gyration, driving deterministic all-optical switching.
- **GPU-native**: every kernel runs on CPU or CUDA via KernelAbstractions, with a hand-tuned
  native `@cuda` Maxwell path validated against a production reference to ~0.1 %.
- **CPU test suite passing** on Julia 1.12.

## Contents

```@contents
Pages = ["introduction.md", "fundamentals.md", "getting_started.md", "em_fdtd.md", "magneto_optics.md", "capabilities.md", "api.md"]
Depth = 2
```

## Installation

The package targets **Julia ≥ 1.12** and is **not yet registered**, so install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/physarief78/MagnetoPhotonic.jl")
using MagnetoPhotonic
```

GPU, HDF5 I/O, and plotting are optional extensions — `Pkg.add("CUDA")` / `Pkg.add("HDF5")` /
`Pkg.add("CairoMakie")` and they load automatically. See [Getting Started](@ref) for details.

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

Licensed under the [GNU General Public License v3.0 only](https://www.gnu.org/licenses/gpl-3.0.html)
(`GPL-3.0-only`). Copyright © 2026 Muhammad Arief Mulyana.
