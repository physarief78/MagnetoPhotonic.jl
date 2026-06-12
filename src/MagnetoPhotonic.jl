module MagnetoPhotonic

using Adapt
using KernelAbstractions
using LinearAlgebra
using Printf
using SparseArrays

include("core/Constants.jl")
include("core/Config.jl")
include("core/Backend.jl")
include("core/Precision.jl")
include("core/Materials.jl")
include("physics/Models.jl")

include("geometry/VecMath.jl")
include("geometry/Shapes.jl")
include("geometry/Raster.jl")
include("geometry/Scene.jl")
include("geometry/DeviceBuilder.jl")
include("geometry/Devices.jl")

include("grid/Grid.jl")
include("geometry/RasterDims.jl")
include("geometry/ReferenceRaster.jl")

include("fdtd/Fields.jl")
include("fdtd/CPML.jl")
include("fdtd/Boundary.jl")
include("fdtd/ModeSolver.jl")
include("fdtd/Source.jl")
include("fdtd/Dispersion.jl")
include("fdtd/MagnetoOptic.jl")
include("fdtd/Maxwell.jl")

include("physics/Thermal.jl")
include("physics/Magnetization.jl")
include("physics/Coupling.jl")
include("physics/Polarimetry.jl")

include("fdtd/Kernels.jl")

# Solver depends on both the fdtd kernels and the physics state types.
include("fdtd/Solver.jl")

include("sim/Monitors.jl")
include("sim/ProbeReadout.jl")
include("sim/Simulation.jl")
include("sim/Run.jl")
include("sim/Experiment.jl")

include("viz/Theme.jl")
include("viz/DeviceView.jl")
include("viz/FieldVideo.jl")
include("viz/Diagnostics.jl")

include("io/HDF5IO.jl")

include("drivers/PumpProbe.jl")
include("drivers/Convergence.jl")
include("drivers/Presets.jl")

export EM_FIELD_STORAGE_TYPE, FDTDParams, get_n_sio2, get_n_si3n4, cfl_dt
export Medium, medium_epsr, medium_material
export GridConfig, SourceConfig, DeviceConfig, PMLConfig, ModelConfig, ProbeConfig, OutputConfig, BackendConfig, SimConfig, RenderConfig
export AbstractBackend, CPUBackend, CUDABackend, backend, ka_device, synchronize, array_type, has_gpu
export zeros_backend, fill_backend, adapt_backend, to_host, default_compute_type, resolve_compute_type, compute_type
export Vec2, vx, vy, dot2, norm2, normalize2, as_tuple
export AbstractShape, Box, PolygonShape, Waveguide, TaperedWaveguide, Cylinder, Letter
export generate_waveguide_polygon, generate_tapered_polygon, generate_H_geometry, generate_M_geometry
export polygon_area, BBox, get_bbox, is_inside_polygon, is_inside_any, fill_fraction
export Material, Scene, SceneEntry, add_shape!, rasterize, rasterize_1d, rasterize_2d
export straight, taper, cosine_bend, ybranch, film_region, waveguide_device
export not_gate_60um, passive_waveguide, hm_test_pattern
export Axis1D, Grid1D, Grid2D, Grid3D, uniform_axis, graded_axis, propagation_axis, uniform_grid, grid_from_config, min_spacing, dim
export not_gate_reference_grid, ref_graded_axis, ref_propagation_axis
export not_gate_reference_geometry, not_gate_reference_polygons, active_axis_position_map
export Fields1D, Fields2D, FieldState, allocate_fields, field_energy
export CPMLProfiles, SlabCPML, build_cpml_profiles, build_cpml_profiles_nonuniform
export CPMLAxis, CPMLState, build_cpml
export AbstractBoundary, PEC, Periodic, PML
export GaussianPulse, ContinuousSource, CarrierGaussianPulse, gaussian_pulse_value, source_value
export AbstractEMSource, PointSource, PlaneSource, ModeSource, inject!, inject_soft!
export solve_waveguide_mode, gaussian_mode_profile
export DLPole, create_pole, discrete_pole_chi, build_pump_poles, build_probe_poles, build_probe_mo_poles
export ADEState, allocate_ade_state, patch_E_dispersive!
export MagnetoOpticADEState, allocate_mo_ade_state, patch_E_mo_gyration!
export update_H!, update_E!, update_H_1d!, update_E_1d!, update_H_2d!, update_E_2d!
export FDTDState, step!, relax_step!, run!, flush_multiphysics!
export freeze_multiphysics!, enable_multiphysics!, reset_ade_states!, configure_probe_mode!
export magnetization_snapshot, apply_magnetization!
export Simulation, PointMonitor, FieldMonitor, FluxMonitor, DFTMonitor
export Transmission, Reflection, Absorption, Polarimetry, SwitchedFraction, FilmAverage, AbsorbedPower, Progress, NaNGuard
export HotCellTrace, CallbackMonitor, switching_metrics
export ProbeReadout, probe_shot, probe_contrast, film_active_snapshot
export record!, monitor_data
export Phase, Pump, Relax, Probe, Result, init_state, run_phase!, run_experiment
export AbstractPhysicsModel, NullModel, FerrimagnetParameters, GdFeCoParameters, MagnetoOpticModel
export ferrimagnet, gdfeco_parameters
export optical_coupling, absorbed_power_density
export ThermalState, AbsorptionState, thermal_step!, update_4tm, update_Te_energy
export MagnetizationState, llb_step, lookup_m_eq_lut, hot_weight
export brillouin, order_cap, build_m_eq_lut, cayley_rotate
export multiphysics_step!, accumulate_absorption!
export probe_jones_angles_deg, energy_balance
export PUB_THEME, MATERIAL_COLORS
export extrude_waveguide_mesh, write_device_obj, write_plan_svg, plot_scene
export field_slice, capture_frames, render_field_video, compute_spectrum, diagnostic_summary
export save_state, load_state, save_hdf5_state, load_hdf5_state, save_run, load_run
export save_magnetization, load_magnetization, load_magnetization!, load_production_magnetization
export write_production_h5, write_goldstd_h5, h5_schema_signature, assert_h5_schema_compatible
export production_frame_writer, record_production_frame!, close_frame_writer!
export probe_frame_writer, record_probe_frame!
export run_pump_probe_sim, convergence_study, run_convergence_study, run_convergence_study_3D, run_convergence_study_3D_dispersive
export gdfeco_pump_probe

end
