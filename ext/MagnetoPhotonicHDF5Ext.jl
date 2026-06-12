module MagnetoPhotonicHDF5Ext

using MagnetoPhotonic
using Dates
import HDF5

_scalar(x) = x isa Number || x isa AbstractString || x isa Symbol || x isa Bool ||
             x isa DataType || x === nothing
_h5_value(x) = x
_h5_value(x::Symbol) = String(x)
_h5_value(x::DataType) = string(x)
_h5_value(::Nothing) = "nothing"
_h5_value(x::Tuple) = all(v -> v isa Number || v isa Bool, x) ? collect(x) : string(x)

function _write_any!(h5, path::AbstractString, x)
    xh = MagnetoPhotonic.to_host(x)
    if _scalar(xh) || xh isa Tuple
        h5[path] = _h5_value(xh)
    elseif xh isa AbstractVector && !isbitstype(eltype(xh))
        for (i, item) in enumerate(xh)
            _write_any!(h5, "$path/$i", item)
        end
    elseif xh isa AbstractArray
        h5[path] = xh
    elseif xh isa NamedTuple
        for k in keys(xh)
            _write_any!(h5, "$path/$(String(k))", getproperty(xh, k))
        end
    elseif xh isa AbstractDict
        for (k, v) in xh
            _write_any!(h5, "$path/$(String(k))", v)
        end
    elseif isstructtype(typeof(xh))
        for f in fieldnames(typeof(xh))
            _write_any!(h5, "$path/$(String(f))", getfield(xh, f))
        end
    else
        h5[path] = string(xh)
    end
    return h5
end

function _write_state!(h5, state)
    _write_any!(h5, "state/n", state.n)
    _write_any!(h5, "state/dt", state.dt)
    _write_any!(h5, "state/backend", typeof(state.backend))
    _write_any!(h5, "state/compute_T", state.compute_T)
    for component in (:Ex, :Ey, :Ez, :Hx, :Hy, :Hz, :Dx, :Dy, :Dz)
        _write_any!(h5, "state/fields/$(String(component))", getfield(state.fields, component))
    end
    state.thermal === nothing || _write_any!(h5, "state/thermal", state.thermal)
    state.mag === nothing || _write_any!(h5, "state/magnetization", state.mag)
    _write_any!(h5, "state/region", state.region)
    return h5
end

function MagnetoPhotonic.save_hdf5_state(filename::AbstractString, result; overwrite::Bool=true)
    mode = overwrite ? "w" : "cw"
    HDF5.h5open(filename, mode) do h5
        _write_any!(h5, "provenance/package", "MagnetoPhotonic")
        _write_any!(h5, "provenance/version", string(pkgversion(MagnetoPhotonic)))
        _write_any!(h5, "provenance/saved_at", string(now()))
        if result isa MagnetoPhotonic.Result
            _write_any!(h5, "config", result.cfg)
            _write_any!(h5, "summary", result.summary)
            _write_any!(h5, "phase_results", result.phase_results)
            _write_any!(h5, "monitors", result.monitor_data)
            _write_state!(h5, result.state)
        elseif result isa MagnetoPhotonic.FDTDState
            _write_state!(h5, result)
        else
            _write_any!(h5, "result", result)
            hasproperty(result, :state) && _write_state!(h5, getproperty(result, :state))
        end
    end
    return filename
end

_read_node(obj::HDF5.Dataset) = HDF5.read(obj)
function _read_node(obj)
    out = Dict{Symbol,Any}()
    for k in keys(obj)
        out[Symbol(k)] = _read_node(obj[k])
    end
    return out
end

function MagnetoPhotonic.load_hdf5_state(filename::AbstractString)
    HDF5.h5open(filename, "r") do h5
        return _read_node(h5)
    end
end

function _ensure_group(parent, path::AbstractString)
    isempty(path) && return parent
    g = parent
    for part in split(path, '/')
        isempty(part) && continue
        if !(part in keys(g))
            HDF5.create_group(g, part)
        end
        g = g[part]
    end
    return g
end

function _put!(h5, path::AbstractString, value)
    parts = split(path, '/')
    name = String(parts[end])
    group_path = join(parts[1:end-1], '/')
    g = _ensure_group(h5, group_path)
    g[name] = _h5_value(MagnetoPhotonic.to_host(value))
    return h5
end

_get(nt, key::Symbol, default=nothing) = nt === nothing ? default :
    nt isa AbstractDict ? get(nt, key, default) :
    hasproperty(nt, key) ? getproperty(nt, key) : default

_vec(nt, key::Symbol, n::Integer; fill_value=NaN) = begin
    v = _get(nt, key, nothing)
    v === nothing ? fill(Float64(fill_value), n) : Float64.(MagnetoPhotonic.to_host(v))
end
_ivec(nt, key::Symbol, n::Integer; fill_value=0) = begin
    v = _get(nt, key, nothing)
    v === nothing ? fill(Int(fill_value), n) : Int.(MagnetoPhotonic.to_host(v))
end
_f32movie(nt, key::Symbol, dims::NTuple{3,Int}) = begin
    v = _get(nt, key, nothing)
    v === nothing ? zeros(Float32, dims) : Float32.(MagnetoPhotonic.to_host(v))
end

function _put_empty_dataset!(h5, path::AbstractString, ::Type{T}, dims::NTuple{N,Int};
                             chunk=nothing) where {T,N}
    parts = split(path, '/')
    name = String(parts[end])
    group_path = join(parts[1:end-1], '/')
    g = _ensure_group(h5, group_path)
    if chunk === nothing
        HDF5.create_dataset(g, name, T, dims)
    else
        HDF5.create_dataset(g, name, T, dims; chunk=chunk)
    end
    return h5
end

function _h5_dataset_exists(h5, path::AbstractString)
    g = h5
    for part in split(path, '/')
        isempty(part) && continue
        part in keys(g) || return false
        g = g[part]
    end
    return true
end

# Movie dims follow the reference layout: spatial-slice-major, frame LAST —
# (Nx, Ny, nf) / (Nx, Nz, nf) / (Ny, Nz, nf), chunked one full slice per frame.
function _put_movie!(h5, path::AbstractString, source, key::Symbol, dims::NTuple{3,Int})
    # The pump movie datasets may already exist (streamed during the run by the
    # production frame writer) — leave them untouched.
    _h5_dataset_exists(h5, path) && return h5
    chunk = (max(1, dims[1]), max(1, dims[2]), 1)
    v = _get(source, key, nothing)
    if v === nothing
        any(==(0), dims) && return _put!(h5, path, zeros(Float32, dims))
        return _put_empty_dataset!(h5, path, Float32, dims; chunk=chunk)
    end
    vh = MagnetoPhotonic.to_host(v)
    if vh isa AbstractArray && any(==(0), size(vh)) && all(>(0), dims)
        return _put_empty_dataset!(h5, path, Float32, dims; chunk=chunk)
    end
    return _put!(h5, path, Float32.(vh))
end

function _dims_from_state(state)
    grid = state.grid
    return (length(grid.x.centers), length(grid.y.centers), length(grid.z.centers),
            getproperty(state.region, :n_material))
end

# Permittivity slices in the reference's native orientation — eps_z_xy is
# (Nx, Ny) at the waveguide mid-height plane (slice_z_pos), eps_z_xz is (Nx, Nz)
# at y = slice_y_pos, the yz slices are (Ny, Nz). No transposes.
function _eps_slices(state; x_um=(10.0, 20.0, 30.0, 40.0), slice_z_pos=nothing, slice_y_pos=0.0)
    eps = Float64.(MagnetoPhotonic.to_host(state.epsr))
    grid = state.grid
    kz = slice_z_pos === nothing ? cld(length(grid.z.centers), 2) :
         MagnetoPhotonic._source_index(grid.z, slice_z_pos)
    jy = MagnetoPhotonic._source_index(grid.y, slice_y_pos)
    xy = eps[:, :, kz]
    xz = eps[:, jy, :]
    yz = Dict{Int,Any}()
    for x in x_um
        idx = MagnetoPhotonic._source_index(grid.x, x * 1e-6)
        yz[Int(round(x))] = eps[idx, :, :]
    end
    return xy, xz, yz
end

function _write_metadata!(h5, state; metadata=NamedTuple(), frame_count::Integer=0)
    Nx, Ny, Nz, _ = _dims_from_state(state)
    xy, xz, yz = _eps_slices(state; slice_z_pos=_get(metadata, :eps_slice_z, nothing),
                             slice_y_pos=_get(metadata, :eps_slice_y, 0.0))
    p = state.params
    gd = state.model isa MagnetoPhotonic.MagnetoOpticModel ? state.model.params : MagnetoPhotonic.FerrimagnetParameters()
    scalars = Dict{String,Any}(
        "Nx"=>Nx, "Ny"=>Ny, "Nz"=>Nz, "dt"=>state.dt,
        "eps_r_vac"=>p.epsr_vac, "eps_r_sio2"=>p.epsr_sio2,
        "Q_voigt_TM"=>gd.Q_voigt_TM, "Q_voigt_RE"=>gd.Q_voigt_RE,
        "Hsw0_T"=>gd.Hsw0, "pump_E_scale"=>_get(metadata, :pump_E_scale, 1.0),
        "target_F_abs_mJcm2"=>_get(metadata, :target_F_abs_mJcm2, NaN),
        "E_amplitude_base"=>_get(metadata, :E_amplitude_base, NaN),
        "E_amplitude_scale_to_target_F_abs"=>_get(metadata, :E_amplitude_scale_to_target_F_abs, 1.0),
        "E_amplitude"=>_get(metadata, :E_amplitude, NaN),
        "absorbed_fluence_avg_mJcm2"=>_get(metadata, :absorbed_fluence_avg_mJcm2, NaN),
        "absorbed_fluence_local_max_mJcm2"=>_get(metadata, :absorbed_fluence_local_max_mJcm2, NaN),
        "fluence_scale_from_base"=>_get(metadata, :fluence_scale_from_base, _get(metadata, :E_amplitude_scale_to_target_F_abs, 1.0)),
        "U_abs_local_max_Jm3"=>_get(metadata, :U_abs_local_max_Jm3, NaN),
        "absorption_model_id"=>state.absorption_model === :ade_work ? 2 : 1,
        "brillouin_iters"=>state.brillouin_iters,
        "debug_max_probe_steps"=>_get(metadata, :debug_max_probe_steps, 0),
        "debug_max_pump_steps"=>_get(metadata, :debug_max_pump_steps, 0),
        "debug_nan_trace_dense_steps"=>_get(metadata, :debug_nan_trace_dense_steps, 0),
        "debug_nan_trace_interval"=>_get(metadata, :debug_nan_trace_interval, 0),
        "debug_nan_trace_steps"=>_get(metadata, :debug_nan_trace_steps, 0),
        "debug_progress_interval"=>_get(metadata, :debug_progress_interval, 0),
        "frame_skip"=>_get(metadata, :frame_skip, 0),
        "initial_m_RE_x_reduced"=>_get(metadata, :initial_m_RE_x_reduced, -1.0),
        "initial_m_TM_x_reduced"=>_get(metadata, :initial_m_TM_x_reduced, 1.0),
        "n_relaxation_trajectories"=>_get(metadata, :n_relaxation_trajectories, 1),
        "probe_amplitude_V_m"=>_get(metadata, :probe_amplitude_V_m, 1.0e6),
        "probe_duration_s"=>_get(metadata, :probe_duration_s, 0.6e-12),
        "probe_lambda_nm"=>_get(metadata, :probe_lambda_nm, 532.0),
        "probe_post_x_um_requested"=>_get(metadata, :probe_post_x_um_requested, 48.0),
        "probe_pre_x_um_requested"=>_get(metadata, :probe_pre_x_um_requested, 33.0),
        "probe_spectrum_bins"=>_get(metadata, :probe_spectrum_bins, 5),
        "probe_trace_stride"=>_get(metadata, :probe_trace_stride, 10),
        "pump_MLUPS"=>_get(metadata, :pump_MLUPS, NaN),
        "pump_elapsed_s"=>_get(metadata, :pump_elapsed_s, NaN),
        "pump_ns_per_cell_step"=>_get(metadata, :pump_ns_per_cell_step, NaN),
        "steps_pump_nominal"=>_get(metadata, :steps_pump_nominal, state.n),
        "steps_pump_run"=>_get(metadata, :steps_pump_run, state.n),
        # Boolean run-flags are stored as Int8 in the reference HDF5 (see
        # pump_probe_switching_empirical_params.jl `g_meta[...] = Int8(...)`); match
        # the dtype exactly so a strict (require_types=true) schema check passes.
        "debug_disable_ade"=>Int8(0), "debug_disable_multiphysics"=>Int8(0), "debug_disable_source"=>Int8(0),
        "debug_nan_trace"=>Int8(0), "debug_skip_equilibration"=>Int8(0), "debug_stop_after_pump"=>Int8(0),
        "debug_stop_after_pump_effective"=>Int8(0), "deterministic_relaxation"=>Int8(1),
        "async_frame_io"=>Int8(0), "run_probe"=>Int8(1), "save_probe_frames"=>Int8(1),
        "save_pump_frames"=>Int8(frame_count > 0 ? 1 : 0),
        "use_split_maxwell_pml_effective"=>Int8(0),
    )
    strings = Dict{String,Any}(
        "Hsw_role"=>"Hsw0 is a parameter record; no applied switching field during optical run",
        "Q_spin_formula"=>"Q_spin reservoirs follow the coupled four-temperature spin-bath energy exchange",
        "Q_spin_units"=>"J/m^3",
        "absorption_ade_work_formula"=>"positive local E dot J_ADE work accumulated in active magneto-optic cells",
        "absorption_ade_work_note"=>"matches the production :ade_work fluence path; nonpositive local work is clipped for heating",
        "absorption_cycle_average_formula"=>"0.5*omega*eps0*eps_imag*|E|^2",
        "absorption_model"=>String(state.absorption_model),
        "ade_precision"=>string(state.compute_T), "ade_update_model"=>"diagonal ADE polarization current",
        "cpml_precision"=>string(state.compute_T), "em_field_storage_precision"=>string(eltype(state.fields.Ez)),
        "field_precision"=>string(eltype(state.fields.Ez)), "magnetization_internal_units"=>"reduced sublattice moments",
        "magnetization_model"=>"two-sublattice ferrimagnet LLB with 4TM reservoirs",
        "magnetization_model_source"=>"MagnetoPhotonic generalized ferrimagnet model",
        "magnetization_physical_units"=>"A/m", "magnetization_precision"=>string(state.compute_T),
        "maxwell_coef_precision"=>string(state.compute_T), "maxwell_update_model"=>"Yee + CPML + ADE + MO",
        "pml_launch_model"=>"KernelAbstractions",
        "precision"=>"storage Float32 / selectable compute precision",
        "production_curve_cleanup"=>"package production replication writer",
        "pump_coupling"=>"mode source into NOT-gate arm",
        "pump_frame_hdf5_chunking"=>"frame-major float32 movies",
        "pump_frame_storage_precision"=>"Float32",
        "thermal_precision"=>string(state.compute_T),
        "use_split_maxwell_pml_requested"=>"false",
        "initial_magnetization_source"=>"ferrimagnet equilibrium at T0",
    )
    for (k, v) in scalars
        _put!(h5, "metadata/$k", v)
    end
    for (k, v) in strings
        _put!(h5, "metadata/$k", v)
    end
    _put!(h5, "metadata/bulk_grid_shape", Int[Nx, Ny, Nz])
    _put!(h5, "metadata/pml_cells_xyz", Int[_get(metadata, :pml_x, 0), _get(metadata, :pml_y, 0), _get(metadata, :pml_z, 0)])
    _put!(h5, "metadata/pml_thickness_nm_xyz", Float64[_get(metadata, :pml_x_nm, NaN), _get(metadata, :pml_y_nm, NaN), _get(metadata, :pml_z_nm, NaN)])
    _put!(h5, "metadata/threads_3d", Int[8, 8, 4])
    # The reference stores cell-centre coordinates in micrometres (the package grid is in
    # SI metres); scale by 1e6 so this metadata is bit-identical to the reference file and
    # the existing archive/ analysis reads package output unchanged.
    _put!(h5, "metadata/x_coords", state.grid.x.centers .* 1e6)
    _put!(h5, "metadata/y_coords", state.grid.y.centers .* 1e6)
    _put!(h5, "metadata/z_coords", state.grid.z.centers .* 1e6)
    _put!(h5, "metadata/eps_z_xy", xy)
    _put!(h5, "metadata/eps_z_xz", xz)
    for key in (10, 20, 30, 40)
        _put!(h5, "metadata/eps_z_yz_$key", yz[key])
    end
end

const _FILM_TS_KEYS = (
    :mag_time, :mx_TM, :mx_RE,
    :m_TM_x_reduced, :m_TM_y_reduced, :m_TM_z_reduced,
    :m_RE_x_reduced, :m_RE_y_reduced, :m_RE_z_reduced,
    :M_TM_x_Apm, :M_TM_y_Apm, :M_TM_z_Apm,
    :M_RE_x_Apm, :M_RE_y_Apm, :M_RE_z_Apm,
    :M_net_x_Apm, :M_net_y_Apm, :M_net_z_Apm,
    :Te_K, :Tl_K, :Ts_TM_K, :Ts_RE_K, :Te_avg, :Tl_avg, :Ts_avg,
    :mag_TM_norm_avg, :mag_RE_norm_avg,
)

function _write_film_timeseries!(h5, group::AbstractString, data; n::Integer=0)
    n = n > 0 ? n : length(_vec(data, :mag_time, 0))
    for key in _FILM_TS_KEYS
        _put!(h5, "$group/$(String(key))", _vec(data, key, n))
    end
end

function _write_pump_group!(h5, state; data=NamedTuple(), active=NamedTuple(), movies=NamedTuple())
    Nx, Ny, Nz, Na = _dims_from_state(state)
    n = length(_vec(data, :mag_time, 0))
    _write_film_timeseries!(h5, "pump", data; n=n)
    for key in (:m_TM_x_reduced_active_cells, :m_TM_y_reduced_active_cells, :m_TM_z_reduced_active_cells,
                :m_RE_x_reduced_active_cells, :m_RE_y_reduced_active_cells, :m_RE_z_reduced_active_cells,
                :M_TM_x_Apm_active_cells, :M_TM_y_Apm_active_cells, :M_TM_z_Apm_active_cells,
                :M_RE_x_Apm_active_cells, :M_RE_y_Apm_active_cells, :M_RE_z_Apm_active_cells,
                :M_net_x_Apm_active_cells, :M_net_y_Apm_active_cells, :M_net_z_Apm_active_cells,
                :mag_TM_norm_active_cells, :mag_RE_norm_active_cells,
                :Te_active_cells, :Tl_active_cells, :Ts_TM_active_cells, :Ts_RE_active_cells)
        _put!(h5, "pump/$(String(key))", _vec(active, key, Na))
    end
    for key in (:U_abs_J_m3, :U_abs_active_cells, :active_cell_volume_m3, :pabs_peak_W_m3, :pabs_peak_active_cells)
        _put!(h5, "pump/$(String(key))", _vec(active, key, Na; fill_value=0.0))
    end
    _put!(h5, "pump/active_linear_index", _ivec(active, :active_linear_index, Na))
    _put!(h5, "pump/hot_cell_index", Int(_get(active, :hot_cell_index, 1)))
    nf = Int(_get(movies, :frame_count, 0))
    _put_movie!(h5, "pump/Ez_xy", movies, :Ez_xy, (Nx, Ny, nf))
    _put_movie!(h5, "pump/Ez_xz", movies, :Ez_xz, (Nx, Nz, nf))
    for key in (:Ez_yz_10, :Ez_yz_20, :Ez_yz_30, :Ez_yz_40)
        _put_movie!(h5, "pump/$(String(key))", movies, key, (Ny, Nz, nf))
    end
end

function _write_relax_group!(h5, group::AbstractString, data; elapsed_s=NaN)
    _write_film_timeseries!(h5, group, data)
    _put!(h5, "$group/elapsed_s", elapsed_s)
end

function _write_shot_group!(h5, state; data=NamedTuple(), active=NamedTuple(), elapsed_s=NaN)
    Nx, Ny, Nz, Na = _dims_from_state(state)
    _write_relax_group!(h5, "shot_1", data; elapsed_s=elapsed_s)
    for key in (:final_Te_K_active, :final_Tl_K_active, :final_Ts_RE_K_active, :final_Ts_TM_K_active,
                :final_U_abs_J_m3_active, :final_m_RE_x_reduced_active, :final_m_RE_y_reduced_active,
                :final_m_RE_z_reduced_active, :final_m_TM_x_reduced_active, :final_m_TM_y_reduced_active,
                :final_m_TM_z_reduced_active)
        fallback = startswith(String(key), "final_m_RE_x") ? _vec(active, :m_RE_x_reduced_active_cells, Na) :
                   startswith(String(key), "final_m_RE_y") ? _vec(active, :m_RE_y_reduced_active_cells, Na) :
                   startswith(String(key), "final_m_RE_z") ? _vec(active, :m_RE_z_reduced_active_cells, Na) :
                   startswith(String(key), "final_m_TM_x") ? _vec(active, :m_TM_x_reduced_active_cells, Na) :
                   startswith(String(key), "final_m_TM_y") ? _vec(active, :m_TM_y_reduced_active_cells, Na) :
                   startswith(String(key), "final_m_TM_z") ? _vec(active, :m_TM_z_reduced_active_cells, Na) :
                   zeros(Float64, Na)
        _put!(h5, "shot_1/$(String(key))", _get(active, key, fallback))
    end
    for key in (:active_cell_MLUPS, :core_Uabs_fraction_of_max, :final_core_switch_fraction,
                :final_switch_fraction, :final_core_mean_m_RE_x, :final_core_mean_m_TM_x,
                :final_mixture_check_m_RE_x, :final_mixture_check_m_TM_x,
                :final_re_reversed_fraction, :final_switched_cell_mean_m_RE_x,
                :final_switched_cell_mean_m_TM_x, :final_tm_reversed_fraction,
                :final_unswitched_cell_mean_m_RE_x, :final_unswitched_cell_mean_m_TM_x,
                :ns_per_active_cell_step)
        _put!(h5, "shot_1/$(String(key))", Float64(_get(active, key, NaN)))
    end
    _put!(h5, "shot_1/final_core_cell_count", Int(_get(active, :final_core_cell_count, Na)))
    _put!(h5, "shot_1/hot_cell_index", Int(_get(active, :hot_cell_index, 1)))
    n = length(_vec(data, :mag_time, 0))
    _put!(h5, "shot_1/hot_cell_Te_K", _vec(active, :hot_cell_Te_K, n))
    _put!(h5, "shot_1/hot_cell_mx_RE", _vec(active, :hot_cell_mx_RE, n))
    _put!(h5, "shot_1/hot_cell_mx_TM", _vec(active, :hot_cell_mx_TM, n))
    _put!(h5, "shot_1/switched_fraction_TM_time", _vec(active, :switched_fraction_TM_time, n; fill_value=NaN))
end

function _write_probe_metadata_only!(h5, state; metadata=NamedTuple())
    gd = state.model isa MagnetoPhotonic.MagnetoOpticModel ? state.model.params : MagnetoPhotonic.FerrimagnetParameters()
    p = MagnetoPhotonic.FDTDParams(532e-9)
    dp = MagnetoPhotonic.build_probe_poles(state.dt, p.eps0, gd)
    mp = MagnetoPhotonic.build_probe_mo_poles(state.dt, p.eps0, gd)
    _put!(h5, "probe/Hsw0_T_for_parameter_record", gd.Hsw0)
    _put!(h5, "probe/Q_voigt_RE", gd.Q_voigt_RE)
    _put!(h5, "probe/Q_voigt_TM", gd.Q_voigt_TM)
    _put!(h5, "probe/absorption_model", String(state.absorption_model))
    _put!(h5, "probe/absorption_model_id", state.absorption_model === :ade_work ? 2 : 1)
    _put!(h5, "probe/eps_probe_target_re_im", [gd.eps_real_probe, gd.eps_imag_probe])
    _put!(h5, "probe/final_core_switch_fraction_from_shot_1", _get(metadata, :final_core_switch_fraction, NaN))
    _put!(h5, "probe/final_switch_fraction_from_shot_1", _get(metadata, :final_switch_fraction, NaN))
    _put!(h5, "probe/geometry_note", "60um NOT-gate Y-branch waveguide with active GdFeCo film")
    _put!(h5, "probe/mo_coupling", "off-diagonal magneto-optic ADE coupling from frozen sublattice magnetization")
    _put!(h5, "probe/output_h5", _get(metadata, :output_h5, ""))
    _put!(h5, "probe/source_h5", _get(metadata, :source_h5, ""))
    _put!(h5, "probe/probe_run_mode", "gold_standard_probe_only")
    _put!(h5, "probe/probe_tau_note", "probe_tau is the Gaussian envelope tau used by the reference probe")
    _put!(h5, "probe/probe_tau_s", _get(metadata, :probe_tau_s, 24e-15))
    _put!(h5, "probe/readout_scope", "mode-overlap DFT, net flux energies, Jones Faraday/Kerr readout")
    _put!(h5, "probe/tra_method", "bare-waveguide reference netflux normalization with initial/switched contrast")
    # Reference layout: one ROW per pole, columns (C1, C2, C3) — shape (n_poles, 3).
    _put!(h5, "probe/probe_diag_poles_C1_C2_C3", reduce(vcat, ([p.C1 p.C2 p.C3] for p in dp)))
    _put!(h5, "probe/probe_mo_poles_C1_C2_C3", reduce(vcat, ([p.C1 p.C2 p.C3] for p in mp)))
    _put!(h5, "probe/probe_pole_positions_omega0_gamma", Float64[0.0 3.0e14; 4.0e15 8.0e14])
end

# `pump_active` is the post-pump (pre-relax) active-cell snapshot (the reference's
# pump group) and `shot_active` the post-relax one (shot_1); both default to the
# legacy single `active` payload. With `overwrite=false` an existing file is
# opened read-write so movie datasets streamed during the run are preserved.
function MagnetoPhotonic.write_production_h5(path::AbstractString, state;
        pump=NamedTuple(), relaxation=NamedTuple(), shot_1=relaxation, active=NamedTuple(),
        pump_active=active, shot_active=active,
        movies=NamedTuple(), metadata=NamedTuple(), overwrite::Bool=true)
    mode = overwrite ? "w" : "cw"
    HDF5.h5open(path, mode) do h5
        _write_metadata!(h5, state; metadata=metadata, frame_count=Int(_get(movies, :frame_count, 0)))
        _write_pump_group!(h5, state; data=pump, active=pump_active, movies=movies)
        _write_relax_group!(h5, "relaxation", relaxation; elapsed_s=_get(metadata, :relaxation_elapsed_s, NaN))
        _write_shot_group!(h5, state; data=shot_1, active=shot_active, elapsed_s=_get(metadata, :shot_elapsed_s, NaN))
        _write_probe_metadata_only!(h5, state; metadata=metadata)
        _put!(h5, "probe/reference/Q_voigt_RE", state.model.params.Q_voigt_RE)
        _put!(h5, "probe/reference/Q_voigt_TM", state.model.params.Q_voigt_TM)
        _put!(h5, "probe/reference/dt", state.dt)
        _put!(h5, "probe/reference/lambda_nm", 532.0)
        _put!(h5, "probe/reference/omega_bins_rad_s", (2pi * MagnetoPhotonic.FDTDParams().c0 / 532e-9) .* (1 .+ range(-0.04, 0.04; length=5)))
        _put!(h5, "probe/reference/pole_fit_note", "probe poles fitted at lambda_probe")
        _put!(h5, "probe/reference/probe_amplitude_V_m", _get(metadata, :probe_amplitude_V_m, 1.0e6))
        _put!(h5, "probe/reference/probe_frame_count", 0)
        _put!(h5, "probe/reference/probe_frame_skip", 0)
        _put!(h5, "probe/reference/probe_frame_target", _get(metadata, :probe_frame_target, 903))
        _put!(h5, "probe/reference/probe_tau_fs", 24.0)
        _put!(h5, "probe/reference/state_label", "reference")
        _put!(h5, "probe/reference/steps_probe", _get(metadata, :steps_probe, 0))
        _put!(h5, "probe/reference/x_post_um", 48.0)
        _put!(h5, "probe/reference/x_pre_um", 33.0)
        _put!(h5, "phase2_resume_finalization/Hsw0_T", state.model.params.Hsw0)
        _put!(h5, "phase2_resume_finalization/dt_relax_s", _get(metadata, :dt_relax_s, NaN))
        _put!(h5, "phase2_resume_finalization/elapsed_s", _get(metadata, :phase2_elapsed_s, NaN))
        _put!(h5, "phase2_resume_finalization/final_core_switch_fraction", _get(shot_active, :final_core_switch_fraction, NaN))
        _put!(h5, "phase2_resume_finalization/final_switch_fraction", _get(shot_active, :final_switch_fraction, NaN))
        _put!(h5, "phase2_resume_finalization/source_h5", _get(metadata, :source_h5, ""))
        _put!(h5, "phase2_resume_finalization/status", "complete")
        _put!(h5, "phase2_resume_finalization/steps_relax", _get(metadata, :steps_relax, 0))
    end
    return path
end

function _write_probe_shot!(h5, group::AbstractString, shot; state=nothing, write_movies::Bool=false,
                            eps_slice_z=nothing)
    for key in (:A, :R, :T, :T_plus_R_plus_A, :absorbed_energy_J, :dt,
                :ellipticity_faraday_deg, :ellipticity_kerr_deg, :group_delay_fs,
                :incident_energy_J, :input_pulse_rms_fs, :lambda_nm,
                :net_energy_post_J, :net_energy_pre_J, :probe_amplitude_V_m,
                :probe_frame_count, :probe_frame_skip, :probe_frame_target,
                :probe_tau_fs, :probe_trace_stride, :pulse_broadening_fs,
                :reflected_energy_J, :steps_probe, :theta_faraday_deg, :theta_kerr_deg,
                :throughput_loss, :transmitted_energy_J, :transmitted_pulse_rms_fs,
                :x_post_um, :x_pre_um)
        _put!(h5, "$group/$(String(key))", _get(shot, key, NaN))
    end
    _put!(h5, "$group/is_reference", Int8(_get(shot, :is_reference, 0)))
    _put!(h5, "$group/Q_voigt_RE", state === nothing ? NaN : state.model.params.Q_voigt_RE)
    _put!(h5, "$group/Q_voigt_TM", state === nothing ? NaN : state.model.params.Q_voigt_TM)
    for key in (:T_omega, :R_omega, :Sx_incident_im, :Sx_incident_re,
                :Sx_post_net_im, :Sx_post_net_re, :Sx_pre_net_im, :Sx_pre_net_re,
                :jones_faraday_Ey_re_im, :jones_faraday_Ez_re_im,
                :jones_kerr_Ey_re_im, :jones_kerr_Ez_re_im,
                :omega_bins_rad_s, :Ey_trace_post, :Ey_trace_pre,
                :Ez_trace_post, :Ez_trace_pre, :trace_time_s)
        _put!(h5, "$group/$(String(key))", _get(shot, key, Float64[]))
    end
    _put!(h5, "$group/normalization_method", _get(shot, :normalization_method, "gold_standard_bare_waveguide_reference_netflux"))
    _put!(h5, "$group/pole_fit_note", _get(shot, :pole_fit_note, "probe ADE poles fitted at lambda_probe"))
    _put!(h5, "$group/state_label", _get(shot, :state_label, group))
    if write_movies && hasproperty(shot, :probe_frame_indices)
        _put!(h5, "$group/probe_frame_indices", shot.probe_frame_indices)
    end
    if write_movies && hasproperty(shot, :probe_frame_times_s)
        _put!(h5, "$group/probe_frame_times_s", shot.probe_frame_times_s)
    end
    if write_movies
        frames = _get(shot, :frames, NamedTuple())
        frame_count = Int(_get(shot, :probe_frame_count, 0))
        if state === nothing
            for key in (:Ey_xy, :Ez_xy, :Ey_yz_post, :Ey_yz_pre, :Ez_yz_post, :Ez_yz_pre)
                _put_movie!(h5, "$group/$(String(key))", frames, key, (frame_count, 0, 0))
            end
        else
            Nx, Ny, Nz, _ = _dims_from_state(state)
            _put_movie!(h5, "$group/Ey_xy", frames, :Ey_xy, (Nx, Ny, frame_count))
            _put_movie!(h5, "$group/Ez_xy", frames, :Ez_xy, (Nx, Ny, frame_count))
            _put_movie!(h5, "$group/Ey_yz_post", frames, :Ey_yz_post, (Ny, Nz, frame_count))
            _put_movie!(h5, "$group/Ey_yz_pre", frames, :Ey_yz_pre, (Ny, Nz, frame_count))
            _put_movie!(h5, "$group/Ez_yz_post", frames, :Ez_yz_post, (Ny, Nz, frame_count))
            _put_movie!(h5, "$group/Ez_yz_pre", frames, :Ez_yz_pre, (Ny, Nz, frame_count))
        end
        if state !== nothing
            xpre = _get(shot, :x_pre_um, 33.0)
            xpost = _get(shot, :x_post_um, 48.0)
            xy, _, yz = _eps_slices(state; x_um=(xpre, xpost), slice_z_pos=eps_slice_z)
            _put!(h5, "$group/eps_xy", xy)
            _put!(h5, "$group/eps_yz_post", yz[Int(round(xpost))])
            _put!(h5, "$group/eps_yz_pre", yz[Int(round(xpre))])
        end
    end
end

function MagnetoPhotonic.write_goldstd_h5(path::AbstractString, shots; state=nothing,
        reference=getproperty(shots, :reference), initial=getproperty(shots, :initial),
        switched=getproperty(shots, :switched), contrast=MagnetoPhotonic.probe_contrast(initial, switched),
        metadata=NamedTuple(), overwrite::Bool=true)
    mode = overwrite ? "w" : "cw"
    HDF5.h5open(path, mode) do h5
        if state !== nothing
            _write_probe_metadata_only!(h5, state; metadata=merge((output_h5=path,), metadata))
        end
        slice_z = _get(metadata, :eps_slice_z, nothing)
        _write_probe_shot!(h5, "probe/reference", reference; state=state, write_movies=false)
        _write_probe_shot!(h5, "probe/initial", initial; state=state, write_movies=true, eps_slice_z=slice_z)
        _write_probe_shot!(h5, "probe/switched", switched; state=state, write_movies=true, eps_slice_z=slice_z)
        for key in (:delta_A, :delta_R, :delta_T, :delta_ellipticity_faraday_deg,
                    :delta_ellipticity_kerr_deg, :delta_group_delay_fs,
                    :delta_pulse_broadening_fs, :delta_theta_faraday_deg,
                    :delta_theta_kerr_deg, :initial_T_R_A, :initial_T_plus_R_plus_A,
                    :switched_T_R_A, :switched_T_plus_R_plus_A)
            _put!(h5, "probe/contrast/$(String(key))", _get(contrast, key, NaN))
        end
    end
    return path
end

# Streaming pump-movie writer. Creates (or opens) the production HDF5 with the
# six |Ez| movie datasets in the reference layout — Ez_xy (Nx,Ny,nf),
# Ez_xz (Nx,Nz,nf), Ez_yz_10..40 (Ny,Nz,nf), chunked one slice per frame — and
# fills one frame per call during the pump loop, like the reference does in-loop
# (buffering ~5 GB of frames in RAM is not an option). Typical use:
#   w = production_frame_writer(path, state; frame_count=903, slice_z_pos=0.2e-6)
#   Pump(monitors=[..., CallbackMonitor(sim -> record_production_frame!(w, sim.state); every=79)])
#   ...run pump...; close_frame_writer!(w)
# then write_production_h5(path, ...; overwrite=false) fills in everything else.
mutable struct ProductionFrameWriter
    h5::HDF5.File
    dsets::Dict{Symbol,HDF5.Dataset}
    kz::Int
    jy::Int
    ix::NTuple{4,Int}
    frame_idx::Int
    frame_count::Int
    # Preallocated |Ez| staging buffers (reference's Ez_*_abs_gpu / cpu_Ez_* pattern):
    # abs runs device-side into dev_*, one copyto! lands it in host_*, the HDF5 write
    # reads host_*. Allocating these fresh per frame (the old Array(Float32.(abs.(view))))
    # churned ~230 MB/1000 steps of host garbage → 4-6 s of GC pauses per 1000 steps.
    dev_xy::Any
    dev_xz::Any
    dev_yz::Any
    host_xy::Matrix{Float32}
    host_xz::Matrix{Float32}
    host_yz::Matrix{Float32}
end

function MagnetoPhotonic.production_frame_writer(path::AbstractString, state;
        frame_count::Integer, slice_z_pos::Real=NaN, slice_y_pos::Real=0.0,
        x_um=(10.0, 20.0, 30.0, 40.0))
    grid = state.grid
    Nx = length(grid.x.centers)
    Ny = length(grid.y.centers)
    Nz = length(grid.z.centers)
    kz = isnan(Float64(slice_z_pos)) ? cld(Nz, 2) : MagnetoPhotonic._source_index(grid.z, slice_z_pos)
    jy = MagnetoPhotonic._source_index(grid.y, slice_y_pos)
    ix = (MagnetoPhotonic._source_index(grid.x, x_um[1] * 1e-6),
          MagnetoPhotonic._source_index(grid.x, x_um[2] * 1e-6),
          MagnetoPhotonic._source_index(grid.x, x_um[3] * 1e-6),
          MagnetoPhotonic._source_index(grid.x, x_um[4] * 1e-6))
    nf = Int(frame_count)
    h5 = HDF5.h5open(path, "cw")
    g = _ensure_group(h5, "pump")
    d = Dict{Symbol,HDF5.Dataset}()
    d[:Ez_xy] = HDF5.create_dataset(g, "Ez_xy", Float32, (Nx, Ny, nf); chunk=(Nx, Ny, 1))
    d[:Ez_xz] = HDF5.create_dataset(g, "Ez_xz", Float32, (Nx, Nz, nf); chunk=(Nx, Nz, 1))
    for key in (:Ez_yz_10, :Ez_yz_20, :Ez_yz_30, :Ez_yz_40)
        d[key] = HDF5.create_dataset(g, String(key), Float32, (Ny, Nz, nf); chunk=(Ny, Nz, 1))
    end
    Ez = state.fields.Ez
    return ProductionFrameWriter(h5, d, kz, jy, ix, 1, nf,
                                 similar(Ez, Float32, (Nx, Ny)),
                                 similar(Ez, Float32, (Nx, Nz)),
                                 similar(Ez, Float32, (Ny, Nz)),
                                 Matrix{Float32}(undef, Nx, Ny),
                                 Matrix{Float32}(undef, Nx, Nz),
                                 Matrix{Float32}(undef, Ny, Nz))
end

# Stage |field| through the writer's preallocated device + host buffers, then write the
# HDF5 slab — zero per-frame allocations (on CPU backends `dev` is a host Matrix and the
# same code path applies).
function _stage_abs_frame!(dset, frame_idx::Int, dev, host, plane)
    dev .= abs.(plane)
    copyto!(host, dev)
    dset[:, :, frame_idx] = host
    return nothing
end

function MagnetoPhotonic.record_production_frame!(w::ProductionFrameWriter, state)
    w.frame_idx <= w.frame_count || return w
    Ez = state.fields.Ez
    _stage_abs_frame!(w.dsets[:Ez_xy], w.frame_idx, w.dev_xy, w.host_xy, view(Ez, :, :, w.kz))
    _stage_abs_frame!(w.dsets[:Ez_xz], w.frame_idx, w.dev_xz, w.host_xz, view(Ez, :, w.jy, :))
    for (key, i) in zip((:Ez_yz_10, :Ez_yz_20, :Ez_yz_30, :Ez_yz_40), w.ix)
        _stage_abs_frame!(w.dsets[key], w.frame_idx, w.dev_yz, w.host_yz, view(Ez, i, :, :))
    end
    w.frame_idx += 1
    return w
end

MagnetoPhotonic.close_frame_writer!(w::ProductionFrameWriter) = (close(w.h5); w)

# Streaming probe-movie writer: per-shot |Ey|/|Ez| frames at the waveguide xy plane
# and the pre/post readout yz planes, in the reference GOLDSTD layout (Ey_xy/Ez_xy
# (Nx,Ny,nf), *_yz_pre/post (Ny,Nz,nf)). The goldstd writer later opens the same
# file with overwrite=false and skips these datasets via the existence guard.
mutable struct ProbeFrameWriter
    h5::HDF5.File
    dsets::Dict{Symbol,HDF5.Dataset}
    kz::Int
    ipre::Int
    ipost::Int
    frame_idx::Int
    frame_count::Int
    # Preallocated staging buffers, shared by the Ey/Ez planes (see ProductionFrameWriter).
    dev_xy::Any
    dev_yz::Any
    host_xy::Matrix{Float32}
    host_yz::Matrix{Float32}
end

function MagnetoPhotonic.probe_frame_writer(path::AbstractString, state; group::AbstractString,
        frame_count::Integer, slice_z_pos::Real=NaN, pre_x::Real, post_x::Real)
    grid = state.grid
    Nx = length(grid.x.centers)
    Ny = length(grid.y.centers)
    Nz = length(grid.z.centers)
    kz = isnan(Float64(slice_z_pos)) ? cld(Nz, 2) : MagnetoPhotonic._source_index(grid.z, slice_z_pos)
    ipre = MagnetoPhotonic._source_index(grid.x, pre_x)
    ipost = MagnetoPhotonic._source_index(grid.x, post_x)
    nf = Int(frame_count)
    h5 = HDF5.h5open(path, "cw")
    g = _ensure_group(h5, group)
    d = Dict{Symbol,HDF5.Dataset}()
    d[:Ey_xy] = HDF5.create_dataset(g, "Ey_xy", Float32, (Nx, Ny, nf); chunk=(Nx, Ny, 1))
    d[:Ez_xy] = HDF5.create_dataset(g, "Ez_xy", Float32, (Nx, Ny, nf); chunk=(Nx, Ny, 1))
    for key in (:Ey_yz_pre, :Ez_yz_pre, :Ey_yz_post, :Ez_yz_post)
        d[key] = HDF5.create_dataset(g, String(key), Float32, (Ny, Nz, nf); chunk=(Ny, Nz, 1))
    end
    Ez = state.fields.Ez
    return ProbeFrameWriter(h5, d, kz, ipre, ipost, 1, nf,
                            similar(Ez, Float32, (Nx, Ny)),
                            similar(Ez, Float32, (Ny, Nz)),
                            Matrix{Float32}(undef, Nx, Ny),
                            Matrix{Float32}(undef, Ny, Nz))
end

function MagnetoPhotonic.record_probe_frame!(w::ProbeFrameWriter, state)
    w.frame_idx <= w.frame_count || return w
    Ey = state.fields.Ey
    Ez = state.fields.Ez
    _stage_abs_frame!(w.dsets[:Ey_xy], w.frame_idx, w.dev_xy, w.host_xy, view(Ey, :, :, w.kz))
    _stage_abs_frame!(w.dsets[:Ez_xy], w.frame_idx, w.dev_xy, w.host_xy, view(Ez, :, :, w.kz))
    _stage_abs_frame!(w.dsets[:Ey_yz_pre], w.frame_idx, w.dev_yz, w.host_yz, view(Ey, w.ipre, :, :))
    _stage_abs_frame!(w.dsets[:Ez_yz_pre], w.frame_idx, w.dev_yz, w.host_yz, view(Ez, w.ipre, :, :))
    _stage_abs_frame!(w.dsets[:Ey_yz_post], w.frame_idx, w.dev_yz, w.host_yz, view(Ey, w.ipost, :, :))
    _stage_abs_frame!(w.dsets[:Ez_yz_post], w.frame_idx, w.dev_yz, w.host_yz, view(Ez, w.ipost, :, :))
    w.frame_idx += 1
    return w
end

MagnetoPhotonic.close_frame_writer!(w::ProbeFrameWriter) = (close(w.h5); w)

function MagnetoPhotonic.save_magnetization(path::AbstractString, state; overwrite::Bool=true)
    state.mag === nothing && throw(ArgumentError("state has no magnetization"))
    snap = MagnetoPhotonic.magnetization_snapshot(state)
    mode = overwrite ? "w" : "cw"
    HDF5.h5open(path, mode) do h5
        _write_any!(h5, "magnetization", snap)
    end
    return path
end

function MagnetoPhotonic.load_magnetization(path::AbstractString; group::AbstractString="magnetization")
    HDF5.h5open(path, "r") do h5
        return _read_node(h5[group])
    end
end

# Load the post-pump (switched) sublattice magnetization straight out of a *production*
# HDF5 file (the `shot_1/final_m_*_reduced_active` per-active-cell arrays), aligned to
# `state`'s active cells via the full-grid linear index (`pump/active_linear_index`).
# This mirrors the reference `run_probe_analysis_sim(h5_in)`, which reads the frozen
# magnetization out of the production file rather than re-running the pump. The grid
# stored in the file must match `state`'s grid, since the mapping is by linear index;
# unmatched cells fall back to the as-grown equilibrium recorded in the file metadata.
function MagnetoPhotonic.load_production_magnetization(path::AbstractString; state,
        group::AbstractString="shot_1")
    state.mag === nothing &&
        throw(ArgumentError("state has no magnetization to populate (n_material == 0)"))
    cells = Int.(MagnetoPhotonic.to_host(state.region.material_cells))
    n = length(cells)
    gNx = length(state.grid.x.centers)
    gNy = length(state.grid.y.centers)
    gNz = length(state.grid.z.centers)
    HDF5.h5open(path, "r") do h5
        meta = h5["metadata"]
        Nx = Int(HDF5.read(meta["Nx"])); Ny = Int(HDF5.read(meta["Ny"])); Nz = Int(HDF5.read(meta["Nz"]))
        (Nx, Ny, Nz) == (gNx, gNy, gNz) ||
            throw(ArgumentError("production h5 grid ($(Nx)x$(Ny)x$(Nz)) != state grid " *
                "($(gNx)x$(gNy)x$(gNz)); frozen magnetization can only be mapped by linear " *
                "index on an identical mesh (the package grid generator does not reproduce " *
                "the reference mesh — run the package's own production tier first)."))
        idx = Int.(HDF5.read(h5["pump"]["active_linear_index"]))
        g = h5[group]
        sTMx = Float64.(HDF5.read(g["final_m_TM_x_reduced_active"]))
        sTMy = Float64.(HDF5.read(g["final_m_TM_y_reduced_active"]))
        sTMz = Float64.(HDF5.read(g["final_m_TM_z_reduced_active"]))
        sREx = Float64.(HDF5.read(g["final_m_RE_x_reduced_active"]))
        sREy = Float64.(HDF5.read(g["final_m_RE_y_reduced_active"]))
        sREz = Float64.(HDF5.read(g["final_m_RE_z_reduced_active"]))
        im_tm = "initial_m_TM_x_reduced" in keys(meta) ? Float64(HDF5.read(meta["initial_m_TM_x_reduced"])) : 1.0
        im_re = "initial_m_RE_x_reduced" in keys(meta) ? Float64(HDF5.read(meta["initial_m_RE_x_reduced"])) : -1.0
        lut = Dict{Int,Int}(idx[i] => i for i in eachindex(idx))
        out = (m_TM_x=fill(im_tm, n), m_TM_y=zeros(n), m_TM_z=zeros(n),
               m_RE_x=fill(im_re, n), m_RE_y=zeros(n), m_RE_z=zeros(n))
        matched = 0
        for p in 1:n
            j = get(lut, cells[p], 0)
            if j != 0
                out.m_TM_x[p] = sTMx[j]; out.m_TM_y[p] = sTMy[j]; out.m_TM_z[p] = sTMz[j]
                out.m_RE_x[p] = sREx[j]; out.m_RE_y[p] = sREy[j]; out.m_RE_z[p] = sREz[j]
                matched += 1
            end
        end
        @info "load_production_magnetization" file=basename(path) active_cells=n matched=matched source_cells=length(idx)
        matched == 0 && @warn "no active-cell linear indices matched between the production h5 and the state; magnetization left at as-grown equilibrium"
        return out
    end
end

function _schema_visit!(rows, prefix, obj)
    for k in keys(obj)
        path = isempty(prefix) ? String(k) : "$prefix/$(String(k))"
        child = obj[k]
        if child isa HDF5.Dataset
            push!(rows, (path=path, shape=Tuple(size(child)), dtype=string(eltype(child))))
        else
            _schema_visit!(rows, path, child)
        end
    end
    return rows
end

function MagnetoPhotonic.h5_schema_signature(path::AbstractString)
    HDF5.h5open(path, "r") do h5
        return _schema_visit!(NamedTuple[], "", h5)
    end
end

# Collapse every HDF5 string flavor (variable-length String, fixed-length
# FixedString{N}, Cstring, …) to a single logical "String" so a fixed-length string
# of one capacity is not flagged as a type mismatch against a different capacity.
function _normalize_dtype(s::AbstractString)
    (occursin("String", s) || occursin("Cstring", s) || occursin("Char", s)) && return "String"
    return s
end

function MagnetoPhotonic.assert_h5_schema_compatible(candidate::AbstractString, reference::AbstractString;
                                                     require_shapes::Bool=true, require_types::Bool=true)
    cand = Dict(r.path => r for r in MagnetoPhotonic.h5_schema_signature(candidate))
    ref = Dict(r.path => r for r in MagnetoPhotonic.h5_schema_signature(reference))
    missing = setdiff(keys(ref), keys(cand))
    isempty(missing) || error("HDF5 schema missing datasets: $(sort!(collect(missing)))")
    bad = String[]
    for (path, rr) in ref
        cr = cand[path]
        require_shapes && cr.shape != rr.shape && push!(bad, "$path shape $(cr.shape) != $(rr.shape)")
        if require_types && _normalize_dtype(cr.dtype) != _normalize_dtype(rr.dtype)
            push!(bad, "$path dtype $(cr.dtype) != $(rr.dtype)")
        end
    end
    isempty(bad) || error("HDF5 schema mismatches: $bad")
    return true
end

end
