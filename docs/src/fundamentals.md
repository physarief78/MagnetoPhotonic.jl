# Fundamentals

This page explains the physics and numerics the package is built on — enough to read the
[tutorials](@ref "EM-FDTD Tutorial") and interpret results. It is conceptual; for the exact
function signatures see the [API Reference](@ref).

## Units and conventions

All quantities are **SI**: lengths in metres, time in seconds, frequencies in rad/s. A
`FDTDParams` bundle carries the physical constants (`c0`, `eps0`, `mu0`) and default
material indices. The time step `dt` is chosen automatically from the Courant–Friedrichs–Lewy
(CFL) limit for the grid (`cfl_dt`); you set the Courant *fraction* (`courant`, default 0.5).

## The Yee FDTD scheme

FDTD solves Maxwell's curl equations by leap-frogging the electric and magnetic fields in time on
a **staggered (Yee) grid**: E-field components live on cell edges, H-field components on cell
faces, each offset by half a cell and half a time step. Updating one from the curl of the other,

```math
\mu_0 \,\partial_t \mathbf{H} = -\nabla\times\mathbf{E}, \qquad
\partial_t \mathbf{D} = \nabla\times\mathbf{H}, \qquad \mathbf{D} = \varepsilon_0\varepsilon_r\,\mathbf{E},
```

gives a second-order-accurate, explicit scheme. The package implements this in 1-D, 2-D
(`:TM` = `Ez,Hx,Hy` or `:TE` = `Hz,Ex,Ey`) and 3-D, on a **non-uniform** grid (so you can refine
locally — e.g. around a thin film — without paying for fine cells everywhere). Second-order
spatial accuracy is confirmed by a convergence study (`L2 ∝ Δx²`).

## Absorbing boundaries (CFS-CPML)

To simulate open space on a finite grid, outgoing waves must leave without reflecting. The package
uses a **complex-frequency-shifted convolutional perfectly matched layer (CFS-CPML)**: inside a
boundary layer each spatial derivative is augmented by a convolution-memory term `ψ` that is
updated recursively,

```math
\psi^{n} = b\,\psi^{n-1} + a\,(\partial_x f), \qquad \partial_x f \;\rightarrow\; \tfrac{1}{\kappa}\partial_x f + \psi,
```

with per-axis coefficients `(inv_kappa, a, b)` that ramp through the layer. In the bulk the
coefficients reduce to `κ=1, a=b=0`, so the update is plain Yee. `PML(n)` requests an `n`-cell
layer; `PEC()` (reflecting walls) and `Periodic()` are also available. The `ψ` slabs are stored
only in the boundary region, so a thick PML costs little memory.

## Dispersive materials (Drude–Lorentz ADE)

Frequency-dependent permittivity is handled by the **auxiliary differential equation (ADE)**
method. Each Drude–Lorentz pole contributes a polarization `P` and current `J` integrated
alongside the fields via a semi-implicit recursion (`J = 2ΔP/dt − J`), so a single time-domain run
captures the material's full dispersion. Materials can be specified either by a real refractive
index or by a set of poles (`DLPole`); helpers fit poles to a target permittivity at a wavelength.
ADE updates touch only the **active cells** (those inside dispersive material), kept as compact
per-component index lists.

## The coupled magneto-optic model

This is what makes the package more than an FDTD engine. The GdFeCo film is described by a tunable
`MagnetoOpticModel` with three interacting layers, all advanced inside the time loop.

### Optical layer — gyrotropic permittivity

Magneto-optic activity is an **antisymmetric, off-diagonal** contribution to the permittivity
tensor that couples the transverse field components (`Ey`, `Ez`). Its strength is set by
Voigt-like parameters weighted by each sublattice's magnetization,

```math
Q_\text{eff} = Q_\text{TM}\, m_\text{TM} + Q_\text{RE}\, m_\text{RE},
```

implemented as an off-diagonal ADE coupling. Because `Q_eff` depends on the **current**
magnetization, a probe reads the *present* magnetic state — and a pump that changes the
magnetization changes the optical response in the same run.

### Thermal layer — four-temperature bath

The absorbed optical power (computed per step as local E·J work) heats a **four-temperature
model**: electron `Te`, lattice `Tl`, and two spin reservoirs `Ts_TM`, `Ts_RE`. They exchange
energy through coupling constants and heat capacities, with the electron bath taking the optical
load and driving the spins. This is the path by which light raises the film above its Curie point.

### Magnetic layer — two-sublattice LLB

Each sublattice magnetization evolves by a **Landau–Lifshitz–Bloch (LLB)** equation, which —
unlike Landau–Lifshitz–Gilbert — allows the *magnitude* to shrink and reverse near and above the
Curie temperature, exactly what thermal switching requires. The instantaneous equilibrium
magnetization `m_eq(T)` comes from a coupled mean-field **Brillouin** function (precomputed into a
lookup table for speed, with a few fixed-point iterations for the mutual molecular field). The two
sublattices are **antiferromagnetically coupled**, so they demagnetize and recover on different
timescales — the origin of the transient ferromagnetic state during switching.

### One coupled step

```
H update  →  source  →  E update  →  diagonal ADE dispersion
          →  magneto-optic gyration ADE  →  accumulate absorbed power
          →  (every k steps) 4TM + LLB advance  →  m feeds back into Q_eff
```

A pump phase runs this with the film heating; a relax phase advances only the 4TM+LLB as the film
cools and settles; a probe phase freezes the magnetization and reads it out polarimetrically.

## Precision strategy

The production configuration uses **Float64 arithmetic** with **Float32 field storage** (halving
field memory with no measurable accuracy cost), and **Float32 storage for the CPML ψ** with Float64
arithmetic. The last choice is reference-faithful and frees ~0.5 GiB on a 6 GiB card, which keeps
it off the memory cliff with no change to the physics (validated bit-for-bit on the pump). The
stiff 4TM+LLB ODE always integrates in Float64 for stability. See [Capabilities](@ref) for the
backend/precision details.

## Validation philosophy

With no external ground truth for the full coupled problem, correctness rests on three legs:

1. **Convergence** — field and energy observables converge as `O(Δx²)` under grid refinement.
2. **A CPU test suite** — unit and acceptance tests for CPML absorption, ADE dispersion, flux,
   the LLB switching, backend parity, and the production-replication monitors.
3. **Cross-check against the production CUDA reference** — the full pump→relax→probe pipeline
   reproduces the thesis solver to ~0.1 % (see [Capabilities](@ref)).
