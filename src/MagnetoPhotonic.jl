module MagnetoPhotonic

include("core/Constants.jl")
include("core/Config.jl")
include("core/Backend.jl")
include("core/Materials.jl")

include("geometry/VecMath.jl")
include("geometry/Shapes.jl")
include("geometry/Raster.jl")
include("geometry/Scene.jl")
include("geometry/Devices.jl")

include("grid/Grid.jl")
include("geometry/RasterDims.jl")

include("fdtd/Fields.jl")
include("fdtd/CPML.jl")
include("fdtd/Boundary.jl")
include("fdtd/Source.jl")
include("fdtd/Dispersion.jl")
include("fdtd/MagnetoOptic.jl")
include("fdtd/Maxwell.jl")

include("physics/Models.jl")
include("physics/Thermal.jl")
include("physics/Magnetization.jl")
include("physics/Coupling.jl")
include("physics/Polarimetry.jl")

# Solver depends on both the fdtd kernels and the physics state types.
include("fdtd/Solver.jl")

include("sim/Monitors.jl")
include("sim/Simulation.jl")
include("sim/Run.jl")

include("viz/Theme.jl")
include("viz/DeviceView.jl")
include("viz/FieldVideo.jl")
include("viz/Diagnostics.jl")

include("io/HDF5IO.jl")

include("drivers/PumpProbe.jl")
include("drivers/Convergence.jl")

export EM_FIELD_STORAGE_TYPE, FDTDParams, get_n_sio2, get_n_si3n4, cfl_dt
export Medium, medium_epsr, medium_material
export GridConfig, SourceConfig, DeviceConfig, PMLConfig, ModelConfig, SimConfig, RenderConfig
export AbstractBackend, CPUBackend, CUDABackend, backend, zeros_backend, adapt_backend
export Vec2, vx, vy, dot2, norm2, normalize2, as_tuple
export AbstractShape, Box, PolygonShape, Waveguide, TaperedWaveguide, Cylinder, Letter
export generate_waveguide_polygon, generate_tapered_polygon, generate_H_geometry, generate_M_geometry
export polygon_area, BBox, get_bbox, is_inside_polygon, is_inside_any, fill_fraction
export Material, Scene, SceneEntry, add_shape!, rasterize, rasterize_1d, rasterize_2d
export not_gate_60um, passive_waveguide, hm_test_pattern
export Axis1D, Grid1D, Grid2D, Grid3D, uniform_axis, graded_axis, propagation_axis, uniform_grid, min_spacing, dim
export Fields1D, Fields2D, FieldState, allocate_fields, field_energy
export CPMLProfiles, SlabCPML, build_cpml_profiles, build_cpml_profiles_nonuniform
export CPMLAxis, CPMLState, build_cpml
export AbstractBoundary, PEC, Periodic, PML
export GaussianPulse, ContinuousSource, gaussian_pulse_value, source_value
export AbstractEMSource, PointSource, PlaneSource, inject!, inject_soft!
export DLPole, create_pole, discrete_pole_chi, build_pump_poles, build_probe_poles, build_probe_mo_poles
export ADEState, allocate_ade_state, patch_E_dispersive!
export MagnetoOpticADEState, allocate_mo_ade_state, patch_E_mo_gyration!
export update_H!, update_E!, update_H_1d!, update_E_1d!, update_H_2d!, update_E_2d!
export FDTDState, step!, relax_step!, run!
export Simulation, PointMonitor, FieldMonitor, FluxMonitor, DFTMonitor, record!, monitor_data
export AbstractPhysicsModel, NullModel, GdFeCoParameters, MagnetoOpticModel
export optical_coupling, absorbed_power_density
export ThermalState, thermal_step!, update_4tm, update_Te_energy
export MagnetizationState, llb_step, lookup_m_eq_lut, hot_weight
export brillouin, order_cap, build_m_eq_lut, cayley_rotate
export multiphysics_step!
export probe_jones_angles_deg, energy_balance
export PUB_THEME, MATERIAL_COLORS
export extrude_waveguide_mesh, write_device_obj, write_plan_svg, plot_scene
export field_slice, capture_frames, render_field_video, compute_spectrum, diagnostic_summary
export save_state, load_state, save_hdf5_state, load_hdf5_state
export run_pump_probe_sim, convergence_study, run_convergence_study, run_convergence_study_3D, run_convergence_study_3D_dispersive

end
