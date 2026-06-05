# API Reference

A grouped overview of the public surface. All names are exported by `using MagnetoPhotonic`.

## Parameters and constants

| Symbol | Description |
|---|---|
| `FDTDParams()` / `FDTDParams(λ0)` | physical constants (`c0`, `mu0`, `eps0`) and material indices; the `λ0` form fills Sellmeier indices for Si₃N₄ / SiO₂ |
| `get_n_si3n4(λ)`, `get_n_sio2(λ)` | Sellmeier refractive indices |
| `cfl_dt(grid, p; courant)` | stable time step from the CFL condition |
| `EM_FIELD_STORAGE_TYPE` | default field element type (`Float32`) |

## Grids

| Symbol | Description |
|---|---|
| `uniform_grid(xlim, dx)` / `(xlim, ylim, dx)` / `(xlim, ylim, zlim, dx)` | uniform 1-D/2-D/3-D grid |
| `Grid1D`, `Grid2D`, `Grid3D`, `Axis1D` | grid / axis types |
| `graded_axis(...)`, `propagation_axis(...)` | non-uniform (graded / film-refined) axes |
| `dim(grid)`, `min_spacing(grid)`, `size(grid)` | grid queries |

## Simulation and stepping

| Symbol | Description |
|---|---|
| `Simulation(; cell, dx, dimension, mode, geometry, sources, boundary, courant, params)` | build a simulation |
| `step!(sim)` | advance one time step |
| `run!(sim, nsteps; monitors, callback)` | run a fixed number of steps |
| `run!(sim; until=T, monitors, callback)` | run until physical time `T` |

## Sources

| Symbol | Description |
|---|---|
| `GaussianPulse(; amplitude, t0, tau, omega, phase)` | Gaussian-modulated sinusoid |
| `ContinuousSource(; amplitude, omega, phase)` | continuous-wave source |
| `PointSource(pulse, component, position)` | soft point source (interpolated to the grid) |
| `PlaneSource(pulse, component; axis, position)` | transverse current sheet → plane wave |
| `source_value(src, t)` | the source's scalar value at time `t` |

## Boundaries

| Symbol | Description |
|---|---|
| `PML(n)` | `n`-cell CFS-CPML (ψ-convolution), 1-D/2-D/3-D |
| `PEC()` | reflecting perfect-conductor walls |
| `Periodic()` | post-update edge copy (simple wraparound) |

## Monitors

| Symbol | Description |
|---|---|
| `PointMonitor(component, pos)` | interpolated field time-trace at a physical point |
| `FieldMonitor(component; every)` | field slices for animation |
| `FluxMonitor(axis, pos)` | signed, plane-integrated Poynting flux |
| `DFTMonitor(component, pos)` | trace + on-demand spectrum |
| `record!(monitor, sim)`, `monitor_data(monitor)` | record / read out a monitor |

## Geometry and materials

| Symbol | Description |
|---|---|
| `Scene()`, `add_shape!(scene, shape, material)` | scene builder |
| `Material(name; index= / epsr= / poles= / model=)`, `Medium(; index=)` | materials |
| `Box`, `PolygonShape`, `Waveguide`, `TaperedWaveguide`, `Cylinder`, `Letter` | shape primitives |
| `rasterize`, `rasterize_1d`, `rasterize_2d` | scene → permittivity + active-cell lists |
| `not_gate_60um`, `passive_waveguide`, `hm_test_pattern` | device library |
| `write_device_obj`, `write_plan_svg`, `plot_scene` | export / render geometry |

## Fields, energy, dispersion

| Symbol | Description |
|---|---|
| `allocate_fields(grid; …)`, `field_energy(fields, grid, p)` | field allocation / energy |
| `field_slice(fields; plane, component)`, `capture_frames(...)` | slices for plotting/video |
| `DLPole`, `create_pole(ω0, γ, Δε·ω0², dt, eps0)`, `discrete_pole_chi(...)` | Drude–Lorentz ADE poles |
| `compute_spectrum(t, signal)` | DFT magnitude spectrum |

## Magneto-optic model

| Symbol | Description |
|---|---|
| `MagnetoOpticModel(; T_Curie, Q_voigt_TM, alpha0_TM, …)` | tunable GdFeCo model |
| `GdFeCoParameters` | the underlying parameter struct |
| `NullModel` | pure-EM (no multiphysics) |
| `run_pump_probe_sim(config; model, thermal_kick, relax_steps, …)` | high-level switching driver |
| `FDTDState`, `step!`, `relax_step!`, `run!` | low-level coupled solver |
| `ThermalState`, `MagnetizationState`, `llb_step`, `build_m_eq_lut` | 4TM / LLB building blocks |
| `probe_jones_angles_deg`, `energy_balance` | Kerr/Faraday + T/R/A helpers |

## Drivers and configuration

| Symbol | Description |
|---|---|
| `SimConfig`, `GridConfig`, `SourceConfig`, `DeviceConfig`, `PMLConfig`, `ModelConfig`, `RenderConfig` | configuration structs |
| `convergence_study(; dimension, mode, dispersive, pml, dxs, L, T_max)` | Cauchy convergence + spectrum |
| `run_convergence_study`, `run_convergence_study_3D`, `run_convergence_study_3D_dispersive` | convenience wrappers |

## I/O and extensions

| Symbol | Description |
|---|---|
| `save_state`, `load_state` | serialized state I/O |
| `save_hdf5_state`, `load_hdf5_state` | HDF5 I/O (requires the `HDF5` extension) |
| `render_field_video` | heatmap video (requires the `CairoMakie` extension) |

!!! note
    Loading `CUDA`, `HDF5` or `CairoMakie` activates the corresponding package extension
    (device arrays, HDF5 I/O, plotting/video). The core solver runs without any of them.
