# Introduction

## What MagnetoPhotonic.jl is

MagnetoPhotonic.jl is a Julia package for **magneto-photonic time-domain electromagnetics**. At
its core is a general, Meep-like **finite-difference time-domain (FDTD)** solver in 1-D, 2-D and
3-D. What sets it apart is that the optical solve is **coupled to a dynamical magneto-optic
material model** — a four-temperature electron–lattice–spin bath, a two-sublattice
Landau–Lifshitz–Bloch (LLB) magnetization, and a magnetization-dependent gyrotropic permittivity
tensor — so that a guided laser pulse can *change the magnetic state of the material as the
simulation runs*.

The package is built to study **all-optical magnetization switching** and magneto-optic logic in
integrated photonic devices, but the electromagnetic engine underneath is a general-purpose FDTD
you can use for ordinary nanophotonics.

## Motivation: light that both computes and routes

The slowing of electronic transistor scaling motivates **all-photon logic**, in which light both
performs and routes computation. Magneto-optical (MO) materials offer an attractive write–read
primitive: a **pump** pulse *writes* a magnetic state, and a **probe** pulse *reads* it through
polarization rotation (the Faraday/Kerr effect). A logic gate built on this primitive needs only
optical inputs and outputs.

Simulating such a gate from first principles is a genuinely multiphysics problem — and that is the
gap this package fills.

## Origin: an UNPAD physics thesis

The package generalizes the solver built for an undergraduate physics thesis at **Universitas
Padjadjaran (UNPAD)**, which assessed the feasibility of a compact **all-photon NOT gate**:

- A single linearly-polarized **800 nm, 40 fs pump** guided onto an **8 nm GdFeCo film** drives the
  electron and spin temperatures above the Curie point and **deterministically reverses the net
  magnetization** within a few picoseconds (thermally-driven, helicity-independent all-optical
  switching — AO-HIS).
- A weak **532 nm probe**, evaluated through frozen *initial* and *switched* magnetic states
  against a bare-waveguide reference, leaves the transmission/reflection/absorption budget
  essentially unchanged (`T + R + A ≈ 1`, reflectivity ≈ 27 %) yet shows a **sign-reversing Faraday
  rotation** with a switching contrast of ≈ 1.7° — the polarization-encoded logic output.
- The platform is fabrication-light **silicon nitride (Si₃N₄)**.

The thesis built the entire coupled solver from scratch in Julia, on a consumer GPU, “to ensure
full transparency and control over all numerical assumptions.” MagnetoPhotonic.jl is that solver,
refactored into a reusable, tested, documented library that reproduces the thesis’s production
results to ~0.1 % (see [Capabilities](@ref)).

## Why not just use Meep?

Established FDTD packages — [Meep](https://meep.readthedocs.io/), commercial tools — model
magneto-optics as a **static** input: you fix a magnetization (or bias field) and they compute the
Faraday/Kerr response of that frozen state. That answers the *read* question.

All-optical switching is the *inverse, write* problem: the magnetization is a **dynamical unknown**
that reverses *because* the pulse heats the film through its Curie temperature. Resolving it
requires three systems to advance together in one time loop —

1. the **electromagnetic field** (Maxwell/Yee),
2. the **absorbed-power → temperature** path (the four-temperature bath), and
3. the **magnetization** (two-sublattice LLB),

with the gyrotropic permittivity recomputed from the *current* magnetization every step. Meep can
read a fixed MO state; it cannot switch one. That coupling is the reason this package exists.

### Same core, different scope

The electromagnetic layer is deliberately Meep-like, so the mental model transfers:

| | Meep | MagnetoPhotonic.jl |
|---|---|---|
| Yee FDTD (1/2/3-D), PML/CPML, dispersive media, sources, flux/DFT monitors | ✅ | ✅ |
| Magnetization | static input | **dynamical** (4TM + LLB) |
| Thermal model | none | 4-temperature bath |
| All-optical switching | ✗ | **headline use case** |
| Stack | C++ / Python·Scheme | pure **Julia**, GPU-native |

Meep remains far more mature and broadly featured (adjoint design, eigenmode decomposition,
near-to-far, cylindrical coordinates, MPI, a large material library and community). Reach for Meep
for general nanophotonics; reach for MagnetoPhotonic.jl when the magnetization must *evolve*. The
full comparison and capability matrix is in [Capabilities](@ref).

## Design principles

| Principle | In the code |
|---|---|
| **Transparency over black-box** | Every kernel is readable Julia you can audit and modify. |
| **Coupled multiphysics is first-class** | The magnetization-dependent gyrotropy is evaluated inside the step loop, not bolted on after. |
| **Runs on accessible hardware** | Reference results came from a 6 GB consumer RTX 3050; the GPU path is tuned to fit there. |
| **Internally validated** | Convergence study + CPU test suite + bit-level comparison to the production CUDA reference. |
| **One language** | Numerics, orchestration, geometry and I/O are all Julia, composed through multiple dispatch. |

## Where to go next

- New to the physics or the numerics? → [Fundamentals](@ref)
- Want to run something now? → [Getting Started](@ref)
- Learn by example → [EM-FDTD Tutorial](@ref) and [Magneto-Optic Switching](@ref)
- What can it actually compute, and how fast? → [Capabilities](@ref)
- Function-by-function → [API Reference](@ref)
