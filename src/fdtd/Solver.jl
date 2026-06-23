# Full coupled FDTD state and time step.
#
# A bare state (no CPML, NullModel, no ADE) reduces to the plain split-field Yee
# scheme. Supplying a MagnetoOpticModel + a rasterized region with material cells
# turns on, in order each step: Yee H, source, Yee E, diagonal ADE dispersion,
# magneto-optic gyration ADE, and (every `multiphysics_every` steps) the coupled
# 4TM + LLB update.

mutable struct FDTDState
    grid::Grid3D
    fields::FieldState
    inv_eps_x
    inv_eps_y
    inv_eps_z
    epsr
    params::FDTDParams
    dt::Float64
    backend::AbstractBackend
    compute_T::DataType
    inv_d_cell_x
    inv_d_cell_y
    inv_d_cell_z
    inv_d_dual_x
    inv_d_dual_y
    inv_d_dual_z
    n::Int
    source::Any
    cpml::Union{Nothing,CPMLState}
    region::Any
    model::AbstractPhysicsModel
    diag_poles
    ade_x::Union{Nothing,ADEState}
    ade_y::Union{Nothing,ADEState}
    ade_z::Union{Nothing,ADEState}
    mo_poles
    mo::Union{Nothing,MagnetoOpticADEState}
    mo_pos
    thermal::Union{Nothing,ThermalState}
    mag::Union{Nothing,MagnetizationState}
    absorption::Union{Nothing,AbsorptionState}
    lut::Any
    multiphysics_every::Int
    subcycles::Int
    brillouin_iters::Int
    multiphysics_enabled::Bool
    absorption_model::Symbol
    # Physical time. Equals n·dt while the EM solver runs, but diverges during a
    # Relax phase, which advances t by its own (coarser) dt_relax per step.
    t::Float64
    # EM step index of the last multiphysics consume, so the next window's dt_mp
    # is sized correctly (handles the final partial window of a pump phase).
    last_mp_n::Int
end

function FDTDState(grid::Grid3D, geo;
                   dt::Real=cfl_dt(grid, FDTDParams(); courant=0.45),
                   params::FDTDParams=FDTDParams(),
                   backend::Union{AbstractBackend,BackendConfig,Symbol,AbstractString}=CPUBackend(),
                   compute_precision=:auto,
                   T::Type=EM_FIELD_STORAGE_TYPE,
                   source=nothing,
                   n_pml=0,
                   cpml_kwargs=NamedTuple(),
                   model::AbstractPhysicsModel=NullModel(),
                   enable_dispersion::Bool=!(model isa NullModel),
                   enable_magneto_optic::Bool=false,
                   diag_poles=nothing,
                   mo_poles=nothing,
                   multiphysics_every::Int=1,
                   subcycles::Int=0,
                   brillouin_iters::Int=2,
                   enable_multiphysics::Bool=true,
                   absorption_model::Symbol=:cycle_average,
                   seed_tilt::Real=1e-4)
    b = backend isa AbstractBackend ? backend : MagnetoPhotonic.backend(backend)
    CT = resolve_compute_type(compute_precision, b)
    model = convert_compute_model(CT, model)
    fields = allocate_fields(grid; backend=b, T=T)
    dtf = Float64(dt)
    cpml = build_cpml(grid, n_pml, dtf, params; backend=b, T=CT, cpml_kwargs...)
    inv_d_cell_x = adapt_backend(b, CT.(grid.x.inv_d_cell))
    inv_d_cell_y = adapt_backend(b, CT.(grid.y.inv_d_cell))
    inv_d_cell_z = adapt_backend(b, CT.(grid.z.inv_d_cell))
    inv_d_dual_x = adapt_backend(b, CT.(grid.x.inv_d_dual))
    inv_d_dual_y = adapt_backend(b, CT.(grid.y.inv_d_dual))
    inv_d_dual_z = adapt_backend(b, CT.(grid.z.inv_d_dual))
    # Geometry exposes relative 1/eps_r. Store absolute 1/(eps0*eps_r) here so every
    # 3-D E update can use one multiply, matching the reference kernel. These three
    # full-grid volumes stay at field precision to avoid the WDDM memory-residency cliff;
    # kernels promote each load to CT for arithmetic.
    inv_eps0 = inv(Float64(params.eps0))
    inv_eps_x = adapt_backend(b, T.(geo.inv_eps_x .* inv_eps0))
    inv_eps_y = adapt_backend(b, T.(geo.inv_eps_y .* inv_eps0))
    inv_eps_z = adapt_backend(b, T.(geo.inv_eps_z .* inv_eps0))
    # Host-resident on purpose: the only consumer is the mode solver, which pulls a
    # YZ plane to host immediately (`_epsr_mode_plane` → `to_host`). Keeping this full
    # Float64 volume on the GPU wasted ~620 MB and pushed the 6 GiB WDDM card to 100%
    # residency (periodic paging stalls in the hot loop).
    epsr = CT.(geo.epsr)

    Nmat = hasproperty(geo, :n_material) ? geo.n_material : 0
    has_model = (model isa MagnetoOpticModel) && Nmat > 0

    dpoles = adapt_backend(b, DLPole{CT}[])
    ade_x = ade_y = ade_z = nothing
    mpoles = adapt_backend(b, DLPole{CT}[])
    mo = nothing
    mo_pos = adapt_backend(b, Int32[])
    thermal = nothing
    mag = nothing
    absorption = nothing
    lut = nothing

    cells_host = Int.(hasproperty(geo, :material_cells) ? geo.material_cells : Int[])
    fill_host = Float64.(hasproperty(geo, :material_fill) ? geo.material_fill : Float64[])
    # ADE patch screening factor: the semi-implicit denominator needs 1/(ε0·ε)
    # (the reference's act_inv convention). The legacy raster supplies 1/ε_r, so
    # scale by 1/ε0 here; the staggered reference geometry supplies act_inv_*
    # already in 1/(ε0·ε) and per component.
    inv_host = Float64.(hasproperty(geo, :material_inv_eps) ? geo.material_inv_eps : Float64[])
    has_stagger_lists = hasproperty(geo, :act_idx_x)
    if !has_stagger_lists
        inv_host = inv_host ./ Float64(params.eps0)
    end
    # Per-component dispersive lists: the reference geometry carries its own
    # staggered (idx, fill, 1/(ε0·ε)) triples per E component; the legacy raster
    # collapses to the collocated material list for all three components.
    idx_x_host = has_stagger_lists ? Int.(geo.act_idx_x) : cells_host
    idx_y_host = has_stagger_lists ? Int.(geo.act_idx_y) : cells_host
    idx_z_host = has_stagger_lists ? Int.(geo.act_idx_z) : cells_host
    f_x_host = has_stagger_lists ? Float64.(geo.act_f_x) : fill_host
    f_y_host = has_stagger_lists ? Float64.(geo.act_f_y) : fill_host
    f_z_host = has_stagger_lists ? Float64.(geo.act_f_z) : fill_host
    inv_x_host = has_stagger_lists ? Float64.(geo.act_inv_x) : inv_host
    inv_y_host = has_stagger_lists ? Float64.(geo.act_inv_y) : inv_host
    inv_z_host = has_stagger_lists ? Float64.(geo.act_inv_z) : inv_host
    pos_x_host = active_axis_position_map(cells_host, idx_x_host)
    pos_y_host = active_axis_position_map(cells_host, idx_y_host)
    pos_z_host = active_axis_position_map(cells_host, idx_z_host)
    # MO gyration acts on the Ez list (the reference's act_idx_mo = act_idx_z),
    # with a map from MO position to the all-list position of the m arrays.
    mo_idx_host = idx_z_host
    mo_f_host = f_z_host
    mo_pos_host = active_axis_position_map(mo_idx_host, cells_host)

    region = (;
        material_cells=adapt_backend(b, cells_host),
        material_fill=adapt_backend(b, CT.(fill_host)),
        material_inv_eps=adapt_backend(b, CT.(inv_host)),
        n_material=Nmat,
        ade_idx_x=adapt_backend(b, idx_x_host), ade_f_x=adapt_backend(b, CT.(f_x_host)), ade_inv_x=adapt_backend(b, CT.(inv_x_host)),
        ade_idx_y=adapt_backend(b, idx_y_host), ade_f_y=adapt_backend(b, CT.(f_y_host)), ade_inv_y=adapt_backend(b, CT.(inv_y_host)),
        ade_idx_z=adapt_backend(b, idx_z_host), ade_f_z=adapt_backend(b, CT.(f_z_host)), ade_inv_z=adapt_backend(b, CT.(inv_z_host)),
        pos_x=adapt_backend(b, pos_x_host), pos_y=adapt_backend(b, pos_y_host), pos_z=adapt_backend(b, pos_z_host),
        mo_idx=adapt_backend(b, mo_idx_host), mo_fill=adapt_backend(b, CT.(mo_f_host)),
        n_mo=length(mo_idx_host),
        cell_volumes=hasproperty(geo, :V_cell_all) ? Float64.(geo.V_cell_all) :
                     _cell_volumes_from_grid(grid, cells_host),
    )

    if has_model
        gd = model.params
        if enable_dispersion
            dpoles_host = diag_poles === nothing ? build_pump_poles(dtf, params.eps0, gd) : diag_poles
            dpoles = adapt_backend(b, collect(DLPole{CT}, dpoles_host))
            ade_x = allocate_ade_state(length(idx_x_host), dpoles; T=CT, backend=b)
            ade_y = allocate_ade_state(length(idx_y_host), dpoles; T=CT, backend=b)
            ade_z = allocate_ade_state(length(idx_z_host), dpoles; T=CT, backend=b)
        end
        thermal = ThermalState(Nmat, model; T=CT, backend=b)
        mag = MagnetizationState(Nmat, model; seed_tilt=seed_tilt, T=CT, backend=b)
        absorption = AbsorptionState(Nmat; backend=b)
        tm_lut, re_lut, T_min, inv_dT, lut_N = build_m_eq_lut(gd)
        lut = (adapt_backend(b, CT.(tm_lut)), adapt_backend(b, CT.(re_lut)), CT(T_min), CT(inv_dT), lut_N)
        if enable_magneto_optic
            mpoles_host = mo_poles === nothing ? build_probe_mo_poles(dtf, params.eps0, gd) : mo_poles
            mpoles = adapt_backend(b, collect(DLPole{CT}, mpoles_host))
            mo = allocate_mo_ade_state(length(mo_idx_host), mpoles; T=CT, backend=b)
            mo_pos = adapt_backend(b, mo_pos_host)
        end
    end

    subs = subcycles > 0 ? subcycles : max(1, multiphysics_every)
    return FDTDState(grid, fields,
                     inv_eps_x, inv_eps_y, inv_eps_z, epsr,
                     params, dtf, b, CT,
                     inv_d_cell_x, inv_d_cell_y, inv_d_cell_z,
                     inv_d_dual_x, inv_d_dual_y, inv_d_dual_z,
                     0, source, cpml, region, model,
                     dpoles, ade_x, ade_y, ade_z, mpoles, mo, mo_pos,
                     thermal, mag, absorption, lut, multiphysics_every, subs, brillouin_iters,
                     enable_multiphysics, absorption_model, 0.0, 0)
end

# --- step! per-kernel profiling (diagnostic; enabled per phase by run_phase!) -------
# When ON, each kernel group inside step! is bracketed by synchronize()+wall clock,
# giving true per-group GPU time (serialized — relative split is the signal). Every
# 1000 profiled steps it prints the bucket table plus host-GC time/allocation for the
# window and device free memory — the three numbers that separate "kernels are slow"
# from "host stalls between kernels" (GC pauses, WDDM paging). OFF: one Ref read/step.
const _STEP_PROF_ON = Ref(false)
const _STEP_PROF_T = Dict{Symbol,Float64}()
const _STEP_PROF_STEPS = Ref(0)
const _STEP_PROF_GC = Ref{Any}(nothing)
const _STEP_PROF_MARK = Ref(UInt64(0))
const _STEP_PROF_ORDER = (:H, :src, :E, :ADE, :MO, :pabs, :mp)

function step_profiling!(state::FDTDState, on::Bool)
    _STEP_PROF_ON[] = on
    if on
        empty!(_STEP_PROF_T)
        _STEP_PROF_STEPS[] = 0
        _STEP_PROF_GC[] = Base.gc_num()
    end
    return nothing
end

function _prof_mark!(state::FDTDState)
    synchronize(state.backend)
    _STEP_PROF_MARK[] = time_ns()
    return nothing
end

function _prof_tick!(state::FDTDState, key::Symbol)
    synchronize(state.backend)
    now = time_ns()
    _STEP_PROF_T[key] = get(_STEP_PROF_T, key, 0.0) + (now - _STEP_PROF_MARK[]) / 1.0e9
    _STEP_PROF_MARK[] = now
    return nothing
end

function _print_step_profile(state::FDTDState)
    n = _STEP_PROF_STEPS[]
    tot = sum(values(_STEP_PROF_T); init=0.0)
    @printf("  [step-profile] per-kernel GPU time, cumulative over %d steps (accounted %.3f s):\n", n, tot)
    for k in _STEP_PROF_ORDER
        t = get(_STEP_PROF_T, k, 0.0)
        @printf("    %-5s %9.3f s  %5.1f%%  %8.4f ms/step\n",
                String(k), t, tot > 0.0 ? 100.0 * t / tot : 0.0, n > 0 ? 1.0e3 * t / n : 0.0)
    end
    gc0 = _STEP_PROF_GC[]
    if gc0 !== nothing
        d = Base.GC_Diff(Base.gc_num(), gc0)
        @printf("    host (last window): GC pause %.3f s | %.1f MB allocated\n",
                d.total_time / 1.0e9, d.allocd / 1.0e6)
    end
    _STEP_PROF_GC[] = Base.gc_num()
    mi = device_memory_info(state.backend)
    mi === nothing || @printf("    vram: free %.2f GiB / total %.2f GiB\n", mi.free_GiB, mi.total_GiB)
    flush(stdout)
    return nothing
end

# Per-active-cell volumes dx·dy·dz from the column-major linear index (fallback
# when the geometry does not supply the reference's V_cell_all).
function _cell_volumes_from_grid(grid::Grid3D, cells::Vector{Int})
    Nx = length(grid.x.centers)
    Ny = length(grid.y.centers)
    dx = diff(grid.x.edges)
    dy = diff(grid.y.edges)
    dz = diff(grid.z.edges)
    vols = Vector{Float64}(undef, length(cells))
    for (n, li) in enumerate(cells)
        li0 = li - 1
        vols[n] = dx[li0 % Nx + 1] * dy[(li0 ÷ Nx) % Ny + 1] * dz[li0 ÷ (Nx * Ny) + 1]
    end
    return vols
end

# Type-stable hot loop (function barrier). FDTDState has several abstract/untyped fields
# (compute_T::DataType, untyped inv_eps_*/inv_d_*, backend::AbstractBackend, source/region/
# poles ::Any). Touching them per step boxed values and rebuilt Val(compute_T) on every
# kernel launch — the ~220 MB/1000-step host allocations and the GC pauses that stall the GPU.
# step! reads those fields ONCE and forwards them to a concretely-typed inner function, so
# every kernel launch specializes on the real types (no per-launch boxing) and Val(CT) is a
# compile-time constant; the lone dynamic dispatch is the barrier call, amortized over the step.
step!(state::FDTDState) = _step_typed!(state, state.backend, state.compute_T,
    state.inv_eps_x, state.inv_eps_y, state.inv_eps_z,
    state.inv_d_cell_x, state.inv_d_cell_y, state.inv_d_cell_z,
    state.inv_d_dual_x, state.inv_d_dual_y, state.inv_d_dual_z,
    state.source, state.region, state.diag_poles, state.mo_poles, state.mo_pos, state.model)

function _step_typed!(state::FDTDState, backend, ::Type{CT},
                      inv_eps_x, inv_eps_y, inv_eps_z,
                      inv_d_cell_x, inv_d_cell_y, inv_d_cell_z,
                      inv_d_dual_x, inv_d_dual_y, inv_d_dual_z,
                      source, region, diag_poles, mo_poles, mo_pos, model) where {CT}
    prof = _STEP_PROF_ON[]
    prof && _prof_mark!(state)
    update_H!(state.fields, state.grid, state.params, state.dt; cpml=state.cpml,
              backend=backend, compute_T=CT,
              inv_d_cell_x=inv_d_cell_x, inv_d_cell_y=inv_d_cell_y,
              inv_d_cell_z=inv_d_cell_z)
    prof && _prof_tick!(state, :H)

    if source !== nothing
        if source isa Tuple
            pulse, component, index = source
            inv_arr = component == :Ex ? inv_eps_x : component == :Ey ? inv_eps_y : inv_eps_z
            inject_soft!(state.fields, component, index, source_value(pulse, state.t);
                         inv_eps=inv_arr,
                         backend=backend, compute_T=CT)
        elseif source isa ModeSource
            inject!(state.fields, state.grid, source, state.t, state.params,
                    (inv_eps_x=inv_eps_x, inv_eps_y=inv_eps_y, inv_eps_z=inv_eps_z);
                    backend=backend, compute_T=CT)
        elseif source isa AbstractEMSource
            inject!(state.fields, state.grid, source, state.t, state.params,
                    (inv_eps_x=inv_eps_x, inv_eps_y=inv_eps_y, inv_eps_z=inv_eps_z))
        end
    end
    prof && _prof_tick!(state, :src)

    update_E!(state.fields, state.grid, state.params, state.dt,
              inv_eps_x, inv_eps_y, inv_eps_z; cpml=state.cpml,
              backend=backend, compute_T=CT,
              inv_d_dual_x=inv_d_dual_x, inv_d_dual_y=inv_d_dual_y,
              inv_d_dual_z=inv_d_dual_z)
    prof && _prof_tick!(state, :E)

    if state.ade_x !== nothing
        patch_E_dispersive!(state.fields.Ex, state.ade_x, region.ade_idx_x, region.ade_f_x, region.ade_inv_x, diag_poles, state.dt;
                            backend=backend, compute_T=CT)
        patch_E_dispersive!(state.fields.Ey, state.ade_y, region.ade_idx_y, region.ade_f_y, region.ade_inv_y, diag_poles, state.dt;
                            backend=backend, compute_T=CT)
        patch_E_dispersive!(state.fields.Ez, state.ade_z, region.ade_idx_z, region.ade_f_z, region.ade_inv_z, diag_poles, state.dt;
                            backend=backend, compute_T=CT)
    end
    prof && _prof_tick!(state, :ADE)

    if state.mo !== nothing && state.mag !== nothing
        gd = model.params
        # MO gyration on the Ez active list (the reference's act_idx_mo = act_idx_z),
        # with the map from MO position to the all-list position of the m arrays.
        # The staggered volumes already carry the full 1/(eps0*eps_r) screening factor.
        patch_E_mo_gyration!(state.fields.Ey, state.fields.Ez, state.mo,
                             region.mo_idx, region.mo_fill, mo_pos,
                             inv_eps_y, inv_eps_z,
                             state.mag.m_TM_x, state.mag.m_RE_x, mo_poles,
                             gd.Q_voigt_TM, gd.Q_voigt_RE, state.dt;
                             backend=backend, compute_T=CT)
    end
    prof && _prof_tick!(state, :MO)

    state.n += 1
    state.t += state.dt

    # Every EM step: integrate the absorbed power density into U_abs and the
    # multiphysics window sum (the reference's kernel_em_pabs_accumulate!). This
    # also runs with multiphysics FROZEN — the reference's probe loop accumulates
    # U_abs_probe every step, which supplies the physical film absorption for A.
    if state.absorption !== nothing
        accumulate_absorption!(state.absorption, state.fields, region, model, state.dt;
                               absorption_model=state.absorption_model, eps0=state.params.eps0,
                               backend=backend,
                               ade_x=state.ade_x, ade_y=state.ade_y, ade_z=state.ade_z)
    end
    prof && _prof_tick!(state, :pabs)
    if state.multiphysics_enabled && state.thermal !== nothing && state.mag !== nothing &&
       state.multiphysics_every > 0 && (state.n % state.multiphysics_every == 0)
        _consume_multiphysics_window!(state)
    end
    prof && _prof_tick!(state, :mp)
    if prof
        _STEP_PROF_STEPS[] += 1
        _STEP_PROF_STEPS[] % 1000 == 0 && _print_step_profile(state)
    end
    return state
end

# Run the 4TM+LLB over the EM steps accumulated since the last consume; dt_mp is
# sized from the actual window so a final partial window integrates correctly.
function _consume_multiphysics_window!(state::FDTDState)
    window = state.n - state.last_mp_n
    window > 0 || return state
    dt_mp = state.dt * window
    multiphysics_step!(state.thermal, state.mag, state.fields, state.region, state.model, state.lut, dt_mp;
                       subcycles=state.subcycles, absorption=state.absorption,
                       brillouin_iters=state.brillouin_iters,
                       backend=state.backend, compute_T=state.compute_T)
    state.last_mp_n = state.n
    return state
end

# Consume any partial multiphysics window at a phase boundary (the reference's
# `n == steps_pump` consume), so no accumulated pabs·dt is left dangling.
function flush_multiphysics!(state::FDTDState)
    (state.multiphysics_enabled && state.thermal !== nothing && state.mag !== nothing) || return state
    return _consume_multiphysics_window!(state)
end

function freeze_multiphysics!(state::FDTDState)
    state.multiphysics_enabled = false
    return state
end

function enable_multiphysics!(state::FDTDState)
    state.multiphysics_enabled = true
    state.last_mp_n = state.n   # don't bill the frozen interval to the next window
    return state
end

function reset_ade_states!(state::FDTDState)
    region = state.region
    if state.ade_x !== nothing
        state.ade_x = allocate_ade_state(length(region.ade_idx_x), state.diag_poles; T=state.compute_T, backend=state.backend)
        state.ade_y = allocate_ade_state(length(region.ade_idx_y), state.diag_poles; T=state.compute_T, backend=state.backend)
        state.ade_z = allocate_ade_state(length(region.ade_idx_z), state.diag_poles; T=state.compute_T, backend=state.backend)
    end
    if state.mo !== nothing
        state.mo = allocate_mo_ade_state(getproperty(region, :n_mo), state.mo_poles; T=state.compute_T, backend=state.backend)
    end
    return state
end

function configure_probe_mode!(state::FDTDState; lambda0::Real=532e-9,
                               reset_ade::Bool=true, freeze_magnetization::Bool=true,
                               enable_magneto_optic::Bool=state.mo !== nothing)
    state.model isa MagnetoOpticModel || return state
    gd = state.model.params
    p = FDTDParams(lambda0)
    state.params = p
    if state.ade_x !== nothing
        dpoles = build_probe_poles(state.dt, p.eps0, gd)
        state.diag_poles = adapt_backend(state.backend, collect(DLPole{state.compute_T}, dpoles))
    end
    if enable_magneto_optic
        mpoles = build_probe_mo_poles(state.dt, p.eps0, gd)
        state.mo_poles = adapt_backend(state.backend, collect(DLPole{state.compute_T}, mpoles))
        if state.mo === nothing
            region = state.region
            state.mo = allocate_mo_ade_state(getproperty(region, :n_mo), state.mo_poles; T=state.compute_T, backend=state.backend)
            state.mo_pos = adapt_backend(state.backend,
                active_axis_position_map(Int.(to_host(region.mo_idx)), Int.(to_host(region.material_cells))))
        end
    else
        state.mo = nothing
        state.mo_pos = adapt_backend(state.backend, Int32[])
    end
    reset_ade && reset_ade_states!(state)
    freeze_magnetization && freeze_multiphysics!(state)
    return state
end

# Returns the per-cell sublattice moments and active-cell index map. When the state
# has no magnetization (no active cells), every array comes back empty rather than
# `nothing`, so callers that index into the result stay safe on under-resolved grids.
function magnetization_snapshot(state::FDTDState)
    if state.mag === nothing
        return (;
            m_TM_x=Float64[], m_TM_y=Float64[], m_TM_z=Float64[],
            m_RE_x=Float64[], m_RE_y=Float64[], m_RE_z=Float64[],
            material_cells=copy(to_host(state.region.material_cells)),
            material_fill=copy(to_host(state.region.material_fill)),
        )
    end
    return (;
        m_TM_x=copy(to_host(state.mag.m_TM_x)),
        m_TM_y=copy(to_host(state.mag.m_TM_y)),
        m_TM_z=copy(to_host(state.mag.m_TM_z)),
        m_RE_x=copy(to_host(state.mag.m_RE_x)),
        m_RE_y=copy(to_host(state.mag.m_RE_y)),
        m_RE_z=copy(to_host(state.mag.m_RE_z)),
        material_cells=copy(to_host(state.region.material_cells)),
        material_fill=copy(to_host(state.region.material_fill)),
    )
end

function _mag_get(data, key::Symbol)
    data isa AbstractDict && return data[key]
    return getproperty(data, key)
end

function apply_magnetization!(state::FDTDState, data)
    state.mag === nothing && throw(ArgumentError("state has no MagnetizationState"))
    for key in (:m_TM_x, :m_TM_y, :m_TM_z, :m_RE_x, :m_RE_y, :m_RE_z)
        src = state.compute_T.(_mag_get(data, key))
        dest = getfield(state.mag, key)
        length(dest) == length(src) || throw(DimensionMismatch("magnetization $key length mismatch"))
        dest .= adapt_backend(state.backend, src)
    end
    return state
end

# Advance only the multiphysics (4TM + LLB) with no EM update — the relaxation /
# cool-down phase after the pump pulse. `absorption=nothing` guarantees ZERO
# heating, exactly like the reference's kernel_relax_multiphysics_fused! (which
# takes no absorbed-power argument at all). Passing the absorption state here
# would re-apply the stale end-of-pump E·J every relax step.
# Free every AbstractArray field of a struct / NamedTuple / Tuple (one level deep).
# free_device! is a no-op on host arrays, so this is safe to call on mixed containers.
function _free_struct_arrays!(s)
    s === nothing && return nothing
    if s isa Tuple
        for v in s
            v isa AbstractArray && free_device!(v)
        end
        return nothing
    end
    for f in fieldnames(typeof(s))
        v = getfield(s, f)
        v isa AbstractArray && free_device!(v)
    end
    return nothing
end

"""
    free_state!(state::FDTDState)

Eagerly release ALL of the state's device arrays back to the allocator (the
reference code's per-shot `free_fdtd_fields!` + `CUDA.unsafe_free!` discipline),
then reclaim pool reserve back to the driver. THE STATE IS UNUSABLE AFTERWARDS.

Call this between successive shots/experiments in one process. Without it, a
finished shot's ~5 GiB of device arrays stays rooted (e.g. inside a kept Result),
the next `init_state` allocates on top, and on a Windows-WDDM card the driver
silently demotes the overflow to system RAM — kernels touching demoted pages run
at PCIe speed (measured: pool reserved 9.4 GiB on a 6 GiB card, 3.9 GiB in shared
memory, multi-x slowdown of the following phase).
"""
function free_state!(state::FDTDState)
    _free_struct_arrays!(state.fields)
    for a in (state.inv_eps_x, state.inv_eps_y, state.inv_eps_z,
              state.inv_d_cell_x, state.inv_d_cell_y, state.inv_d_cell_z,
              state.inv_d_dual_x, state.inv_d_dual_y, state.inv_d_dual_z,
              state.diag_poles, state.mo_poles, state.mo_pos)
        a isa AbstractArray && free_device!(a)
    end
    if state.cpml !== nothing
        _free_struct_arrays!(state.cpml)      # the 24 psi slab arrays
        _free_struct_arrays!(state.cpml.x)    # per-axis inv_kappa/a/b
        _free_struct_arrays!(state.cpml.y)
        _free_struct_arrays!(state.cpml.z)
    end
    for s in (state.ade_x, state.ade_y, state.ade_z, state.mo,
              state.thermal, state.mag, state.absorption, state.region, state.lut)
        _free_struct_arrays!(s)
    end
    state.source isa ModeSource && _free_struct_arrays!(state.source)
    state.source = nothing
    # Drop any remaining unrooted garbage, then return freed pool pages to the OS so
    # WDDM residency pressure actually goes down before the next init_state.
    GC.gc(false)
    reclaim_device_memory!(state.backend)
    return nothing
end

function relax_step!(state::FDTDState, dt::Real; subcycles::Integer=state.subcycles)
    (state.thermal !== nothing && state.mag !== nothing) || return state
    multiphysics_step!(state.thermal, state.mag, state.fields, state.region, state.model, state.lut, dt;
                       subcycles=subcycles, absorption=nothing,
                       brillouin_iters=state.brillouin_iters,
                       backend=state.backend, compute_T=state.compute_T)
    return state
end

function run!(state::FDTDState, steps::Integer; callback=nothing)
    for _ in 1:steps
        step!(state)
        callback === nothing || callback(state)
    end
    return state
end
