# Capabilities

What the package can compute today, on what hardware, how fast, and how well it is validated —
plus an honest list of what it does *not* do yet.

## Capability matrix

| Area | What you get |
|---|---|
| **Dimensions** | 1-D, 2-D (`:TM` = `Ez,Hx,Hy` / `:TE` = `Hz,Ex,Ey`), 3-D |
| **Grid** | non-uniform Yee grid (uniform, graded, or propagation axes); automatic CFL `dt` |
| **Boundaries** | CFS-CPML (`PML(n)`, ψ-convolution), `PEC`, `Periodic` |
| **Materials** | real refractive index; Drude–Lorentz **ADE dispersion** (all dimensions); subpixel averaging |
| **Sources** | `GaussianPulse` / `ContinuousSource`; `PointSource`, `PlaneSource`, guided `ModeSource` |
| **Mode solver** | scalar waveguide eigenmode (`solve_waveguide_mode`) for mode sources/overlap |
| **Monitors** | `PointMonitor`, `FieldMonitor` (slices), `FluxMonitor` (signed Poynting), `DFTMonitor`, and the polarimetric `ProbeReadout` |
| **Magneto-optics** | tunable two-sublattice GdFeCo: 4-temperature bath + two-sublattice **LLB** + magnetization-dependent gyrotropy |
| **Read-out** | pump–probe **T / R / A**, **Faraday & Kerr** rotation + ellipticity, group delay, switched fraction |
| **Geometry** | `Box`, `PolygonShape`, `Waveguide`, `TaperedWaveguide`, `Cylinder`, `Letter`; `Scene` builder; OBJ/SVG export; device library (`not_gate_60um`, …) |
| **I/O & viz** | HDF5 schema I/O; field video; device cross-sections / 3-D renders (optional extensions) |
| **Backends** | CPU and CUDA via KernelAbstractions, with a native `@cuda` Maxwell fast path |

## What it can *compute* — the physics outputs

- **Electromagnetic propagation**: time-domain fields, energy, transmitted/reflected spectra,
  signed power flux, near-field DFT amplitudes.
- **All-optical switching**: the full coupled pump dynamics — absorbed power → 4-temperature
  evolution → LLB sublattice reversal — yielding per-cell temperatures, magnetizations, switched
  fraction, and the transient ferromagnetic window.
- **Magneto-optic read-out**: launch a probe through frozen *initial* and *switched* magnetic
  states and recover the polarization rotation contrast (the logic output), with a bare-waveguide
  reference normalization for T/R/A.

## Backends and GPU

A backend abstraction (`CPUBackend`, `CUDABackend`) isolates all device specifics:

- **Every kernel is written once in [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)** and runs on CPU or CUDA. `using CUDA` loads the GPU
  hooks from a weak-dependency package extension; nothing GPU-related is needed for the CPU path.
- **A native `@cuda` fast path** for the two Maxwell (Yee H/E) kernels — ~98 % of production step
  time — engages for the validated `CUDABackend` + `Float64`-compute case, recovering codegen
  headroom that the generic KernelAbstractions path leaves on the table.
- **AMD / other GPUs**: because the kernels are backend-agnostic, adding an `AMDGPUBackend` is a
  small extension (a backend struct + a ~40-line AMDGPU extension mirroring the CUDA one); the
  KernelAbstractions kernels then run unchanged.

### Precision

Compute is **Float64**; field **storage is Float32** (half the memory, no measurable accuracy
cost); CPML ψ is **Float32 storage with Float64 arithmetic**. The last choice is reference-faithful
and frees ~0.5 GiB, keeping a 6 GiB card off its memory cliff with **no change to the physics**
(validated bit-for-bit on the pump). The stiff 4TM+LLB ODE always integrates in Float64.

## Performance

On the reference 3-D production device (a 4099×190×100 non-uniform mesh, ≈ 78 M cells) on a
consumer **GeForce RTX 3050 (6 GB)**, the coupled pump runs at roughly **90–96 s per 1000 steps**,
at parity with the hand-written CUDA reference (~92.7 s/1000). The Maxwell kernels dominate
(~98 % of step time); the 4-temperature/LLB and read-out add ~1–2 %.

!!! note "Measuring fairly on a laptop GPU"
    A consumer card boost-clocks when cool and throttles under sustained load, so **absolute**
    s/1000-step numbers are only comparable **back-to-back from the same thermal state**. Use the
    per-kernel split and clock-independent ratios when benchmarking. `examples/perf_test_pump.jl`
    and `examples/perf_test_probe.jl` provide bounded, profiled, no-output runs for this.

## Validation against the reference

On the exact production NOT-gate mesh, the package reproduces the thesis's hand-written CUDA
production solver:

| Quantity | Package | Reference | Agreement |
|---|---|---|---|
| Switched fraction | 0.7823 | 0.7831 | 1 cell of 1323 |
| Absorbed fluence `F_abs` | 0.813 mJ/cm² | 0.814 mJ/cm² | 0.12 % |
| Per-cell `U_abs` | — | — | correlation 1.0000 |
| Peak electron temperature | 2186.7 K | 2188.1 K | 0.06 % |
| Faraday contrast `Δθ_F` | 1.684° | 1.690° | within 0.4 % |
| Probe `T / R / A` budget | matches | matches | `T+R+A ≈ 1` |
| HDF5 schema (shapes + dtypes) | — | — | exact |

The lone per-cell disagreement is a single boundary cell sitting exactly at the switching-fluence
threshold — a floating-point-order effect, not a numerical instability. Second-order spatial
accuracy is independently confirmed by the convergence study (`L2 ∝ Δx²`).

## What it does *not* do (yet)

Being explicit about scope:

- **No adjoint / inverse design, eigenmode decomposition, or near-to-far-field transforms.**
- **No cylindrical coordinates**; Cartesian 1-D/2-D/3-D only.
- **Single node.** No MPI domain decomposition or multi-GPU; sized to fit one consumer card.
- **The coupled magneto-optic switching path is 3-D-oriented** (the GdFeCo film + waveguide
  device), though the general EM-FDTD layer is fully 1-D/2-D/3-D.
- **One magnetic material family**: a tunable two-sublattice ferrimagnet (GdFeCo-style). Every
  parameter is a keyword, but other magnetic orderings are not modeled.
- **A DGTD solver is planned** under the same umbrella but not yet implemented.

For general-purpose nanophotonics needing the features above, [Meep](https://meep.readthedocs.io/)
is the mature choice; see [Introduction](@ref) for the full positioning. MagnetoPhotonic.jl's niche
is the **coupled, dynamical magneto-optic switch**.
