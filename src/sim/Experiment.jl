abstract type Phase end

struct Pump <: Phase
    steps::Union{Nothing,Int}
    until::Union{Nothing,Float64}
    source::Any
    monitors::Vector{AbstractMonitor}
    log_every::Int
end

struct Relax <: Phase
    steps::Union{Nothing,Int}
    until::Union{Nothing,Float64}
    dt::Union{Nothing,Float64}
    subcycles::Union{Nothing,Int}
    monitors::Vector{AbstractMonitor}
end

struct Probe <: Phase
    lambda0::Float64
    steps::Union{Nothing,Int}
    until::Union{Nothing,Float64}
    source::Any
    monitors::Vector{AbstractMonitor}
    freeze_magnetization::Bool
    enable_magneto_optic::Bool
    reset_ade::Bool
    log_every::Int
end

Pump(; steps=nothing, until=nothing, source=nothing, monitors=AbstractMonitor[], log_every::Integer=0) =
    Pump(_maybe_int(steps), _maybe_float(until), source, AbstractMonitor[monitors...], Int(log_every))

Relax(; steps=nothing, until=nothing, dt=nothing, subcycles=nothing, monitors=AbstractMonitor[]) =
    Relax(_maybe_int(steps), _maybe_float(until), _maybe_float(dt),
          subcycles === nothing ? nothing : Int(subcycles), AbstractMonitor[monitors...])

function Probe(; lambda0=nothing, λ=nothing, steps=nothing, until=nothing, source=nothing,
               monitors=AbstractMonitor[], freeze_magnetization::Bool=true,
               enable_magneto_optic::Bool=true, reset_ade::Bool=true,
               log_every::Integer=0)
    lam = lambda0 === nothing ? (λ === nothing ? 532e-9 : λ) : lambda0
    return Probe(Float64(lam), _maybe_int(steps), _maybe_float(until), source,
                 AbstractMonitor[monitors...], freeze_magnetization,
                 enable_magneto_optic, reset_ade, Int(log_every))
end

struct Result{C,S,P,M,U}
    cfg::C
    state::S
    phase_results::P
    monitor_data::M
    summary::U
end

_maybe_int(x) = x === nothing ? nothing : Int(x)
_maybe_float(x) = x === nothing ? nothing : Float64(x)

function _format_hms(seconds::Real)
    total = max(0, floor(Int, Float64(seconds)))
    h = div(total, 3600)
    m = div(total % 3600, 60)
    s = total % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function _phase_timing_log!(phase_name::Symbol, state::FDTDState, n0::Integer,
                            nsteps::Integer, log_every::Integer, t_start_ns::UInt64,
                            last_log_time::Base.RefValue{UInt64})
    log_every > 0 || return nothing
    done = state.n - n0
    done > 0 && done % log_every == 0 || return nothing
    synchronize(state.backend)
    now = time_ns()
    total_s = (now - t_start_ns) / 1.0e9
    last_s = (now - last_log_time[]) / 1.0e9
    @printf("Phase %s Step %d/%d | Total Elapsed: %s | Last %d steps: %.3f s\n",
            String(phase_name), done, nsteps, _format_hms(total_s), log_every, last_s)
    flush(stdout)
    last_log_time[] = now
    return nothing
end

function _phase_steps(phase::Phase, dt::Real; default_steps::Integer=0)
    s = getproperty(phase, :steps)
    s !== nothing && return max(0, Int(s))
    u = getproperty(phase, :until)
    u !== nothing && return max(0, ceil(Int, Float64(u) / Float64(dt)))
    return max(0, Int(default_steps))
end

# `n_pml` may be a scalar (uniform PML) or a per-axis (npx, npy, npz) tuple; the source
# sits just inside the x-PML, so use the x-axis width.
_pml_x_cells(n_pml) = n_pml isa Integer ? Int(n_pml) : Int(n_pml[1])

function _source_index(grid::Grid3D, n_pml)
    jc = cld(length(grid.y.centers), 2)
    kc = cld(length(grid.z.centers), 2)
    return (max(2, _pml_x_cells(n_pml) + 2), jc, kc)
end

function _source_position(cfg, grid::Grid3D, source_cfg)
    source_cfg.position === nothing || return source_cfg.position
    idx = _source_index(grid, cfg.pml.cells)
    source_cfg.axis === :x && return grid.x.centers[idx[1]]
    source_cfg.axis === :y && return grid.y.centers[idx[2]]
    source_cfg.axis === :z && return grid.z.centers[idx[3]]
    return idx
end

function _epsr_mode_plane(epsr_volume, grid::Grid3D, axis::Symbol, position)
    epsr = to_host(epsr_volume)
    if axis === :x
        i = _source_index(grid.x, position)
        return epsr[i, :, :]
    elseif axis === :y
        j = _source_index(grid.y, position)
        return epsr[:, j, :]
    elseif axis === :z
        k = _source_index(grid.z, position)
        return epsr[:, :, k]
    end
    throw(ArgumentError("mode source axis must be :x, :y, or :z"))
end

function _mode_spacings(grid::Grid3D, axis::Symbol)
    if axis === :x
        return minimum(diff(grid.y.edges)), minimum(diff(grid.z.edges))
    elseif axis === :y
        return minimum(diff(grid.x.edges)), minimum(diff(grid.z.edges))
    elseif axis === :z
        return minimum(diff(grid.x.edges)), minimum(diff(grid.y.edges))
    end
    throw(ArgumentError("mode source axis must be :x, :y, or :z"))
end

function _source_from_config(cfg, grid::Grid3D, source_cfg; lambda0::Real=source_cfg.lambda0,
                             epsr_volume=nothing, backend::AbstractBackend=CPUBackend(),
                             compute_T::Type=default_compute_type(backend), amplitude_scale::Real=1.0)
    p = FDTDParams(lambda0)
    pulse = GaussianPulse(; amplitude=Float64(amplitude_scale) * source_cfg.amplitude,
                          t0=source_cfg.t0, tau=source_cfg.tau,
                          omega=2pi * p.c0 / Float64(lambda0), phase=source_cfg.phase)
    if source_cfg.kind === :soft
        return (pulse, source_cfg.component, _source_index(grid, cfg.pml.cells))
    elseif source_cfg.kind === :plane
        return PlaneSource(pulse, source_cfg.component; axis=source_cfg.axis,
                           position=_source_position(cfg, grid, source_cfg))
    elseif source_cfg.kind === :mode
        pos = _source_position(cfg, grid, source_cfg)
        if epsr_volume === nothing
            dy, dz = _mode_spacings(grid, source_cfg.axis)
            profile = gaussian_mode_profile(length(grid.y.centers), length(grid.z.centers), dy, dz)
            neff = source_cfg.neff_guess === nothing ? 1.0 : Float64(source_cfg.neff_guess)
        else
            plane = _epsr_mode_plane(epsr_volume, grid, source_cfg.axis, pos)
            dy, dz = _mode_spacings(grid, source_cfg.axis)
            sol = solve_waveguide_mode(plane, dy, dz, lambda0;
                                       neff_guess=source_cfg.neff_guess,
                                       max_iter=source_cfg.max_iter)
            profile = sol.profile
            neff = sol.neff
        end
        return ModeSource(adapt_backend(backend, compute_T.(profile)); pulse=pulse, neff=neff,
                          axis=source_cfg.axis, position=pos, component=source_cfg.component)
    end
    throw(ArgumentError("source kind must be :soft, :plane, or :mode"))
end

_scene_object(scene) = scene !== nothing && hasproperty(scene, :scene) ? scene.scene : scene

function _config_film_bounds(cfg::SimConfig)
    xlo, xhi = cfg.grid.xlim
    xfilm = cfg.device.x_film_start
    if !(xlo < xfilm < xhi)
        xfilm = 0.5 * (xlo + xhi)
    end
    thick = max(cfg.device.film_thickness, 2.0 * cfg.grid.fine_dx)
    return (xfilm - thick / 2, xfilm + thick / 2)
end

function _film_bounds_for_grid(cfg::SimConfig, scene)
    scene !== nothing && hasproperty(scene, :x_film_start) && hasproperty(scene, :x_film_end) &&
        return (scene.x_film_start, scene.x_film_end)
    return _config_film_bounds(cfg)
end

# `geo` overrides the scene rasterization with a prebuilt geometry (e.g. the
# verbatim reference staggered raster from not_gate_reference_geometry).
# `cpml_psi_T` sets the storage precision of the CPML convolution slabs (see build_cpml).
# Default Float32: it frees ~0.47 GiB so the 6 GiB card clears 0 B free (where WDDM thrashes
# and all kernels slow), speeding BOTH phases; the math stays Float64 (kernels cast ψ on
# read). Pass Float64 (or set MP_CPML_PSI=f64) for the full-precision slabs.
function init_state(cfg::SimConfig; model::AbstractPhysicsModel=MagnetoOpticModel(),
                    scene=nothing, enable_magneto_optic::Bool=true, diag_poles=nothing,
                    geo=nothing, cpml_psi_T::Type=Float32)
    p = FDTDParams(cfg.source.lambda0)
    b = backend(cfg)
    CT = compute_type(cfg)
    film_bounds = _film_bounds_for_grid(cfg, scene)
    grid = grid_from_config(cfg; film_region=film_bounds)
    if geo === nothing
        scn = scene === nothing ? pump_probe_scene(cfg, model; p=p) : _scene_object(scene)
        geo = rasterize(scn, grid; subpixel=cfg.grid.subpixel, model_stagger=cfg.grid.model_yee_stagger)
    end
    if model isa MagnetoOpticModel && enable_magneto_optic && getproperty(geo, :n_material) == 0
        @warn "init_state: the magneto-optic film rasterized to 0 cells — the device feature is " *
              "finer than the transverse grid, so 4TM/LLB/MO physics will be INACTIVE. Refine " *
              "GridConfig dy/dz (or widen the film) so a cell center falls inside the waveguide."
    end
    dt = cfl_dt(grid, p; courant=cfg.grid.courant)
    source = _source_from_config(cfg, grid, cfg.source; lambda0=cfg.source.lambda0,
                                 epsr_volume=geo.epsr, backend=b, compute_T=CT,
                                 amplitude_scale=cfg.model.pump_E_scale)
    cpml_kwargs = (order=cfg.pml.order, reflection=cfg.pml.reflection,
                   kappa_max=cfg.pml.kappa_max, alpha_max=cfg.pml.alpha_max,
                   psi_T=cpml_psi_T)
    return FDTDState(grid, geo; dt=dt, params=p, backend=b, compute_precision=CT,
                     T=cfg.precision, source=source, n_pml=cfg.pml.cells,
                     cpml_kwargs=cpml_kwargs, model=model, diag_poles=diag_poles,
                     enable_magneto_optic=enable_magneto_optic,
                     multiphysics_every=cfg.model.multiphysics_subcycle,
                     subcycles=cfg.model.multiphysics_subcycle,
                     brillouin_iters=cfg.model.brillouin_iters,
                     absorption_model=cfg.model.absorption_model)
end

function _monitor_view(state::FDTDState)
    return (;
        grid=state.grid,
        t=state.t,
        n=state.n,
        dt=state.dt,
        dimension=3,
        mode=:TM,
        state=state,
    )
end

_view_fields(sim) = hasproperty(sim, :fields) ? sim.fields : sim.state.fields

function _record_phase_monitors!(monitors, state::FDTDState)
    any(m -> _monitor_due(m, state.n), monitors) || return monitors
    if any(m -> _monitor_needs_sync(m, state.n), monitors)
        synchronize(state.backend)
    end
    view = _monitor_view(state)
    for m in monitors
        _monitor_due(m, state.n) && record!(m, view)
    end
    return monitors
end

function _monitor_key(m::AbstractMonitor, phase_index::Integer, monitor_index::Integer)
    return Symbol(lowercase(String(nameof(typeof(m)))) * "_p" * string(phase_index) * "_" * string(monitor_index))
end

function _phase_source!(state::FDTDState, phase::Pump, cfg::SimConfig)
    phase.source === nothing || (state.source = phase.source)
    return state
end

function _phase_source!(state::FDTDState, phase::Probe, cfg::SimConfig)
    configure_probe_mode!(state; lambda0=phase.lambda0, reset_ade=phase.reset_ade,
                          freeze_magnetization=phase.freeze_magnetization,
                          enable_magneto_optic=phase.enable_magneto_optic)
    state.source = phase.source === nothing ?
                   _source_from_config(cfg, state.grid, cfg.probe; lambda0=phase.lambda0,
                                       epsr_volume=state.epsr, backend=state.backend,
                                       compute_T=state.compute_T) :
                   phase.source
    return state
end

function _attach_mode_monitors!(monitors, source)
    source isa ModeSource || return monitors
    profile = to_host(source.profile)
    for m in monitors
        if m isa Transmission || m isa Reflection
            m.mode === nothing && (m.mode = profile)
            m.component = source.component
        elseif m isa Polarimetry
            m.mode === nothing && (m.mode = profile)
        end
    end
    return monitors
end

function _phase_source!(state::FDTDState, ::Relax, cfg::SimConfig)
    state.source = nothing
    return state
end

function run_phase!(state::FDTDState, phase::Pump, cfg::SimConfig; phase_index::Integer=1)
    _phase_source!(state, phase, cfg)
    _attach_mode_monitors!(phase.monitors, state.source)
    nsteps = _phase_steps(phase, state.dt; default_steps=cfg.steps)
    n0 = state.n
    t0 = state.t
    t_start_ns = time_ns()
    last_log_time = Ref(t_start_ns)
    # PROBE_PROFILE_KERNELS=1 also serializes the PUMP loop (sync after step! and
    # after monitors), giving a step-vs-readout split. The production path below now
    # flushes the WDDM queue on its own (wddm_sync_every), so the env var is purely
    # the profiling diagnostic again.
    prof_on = get(ENV, "PROBE_PROFILE_KERNELS", "0") == "1"
    prof_step = 0.0
    prof_readout = 0.0
    prof_on && @info "[profile] PROBE_PROFILE_KERNELS=1 — pump per-group timing ON (serialized per-step sync)"
    step_profiling!(state, prof_on)   # per-kernel split inside step! (H/src/E/ADE/MO/pabs/mp)
    # WDDM keep-alive: a bare synchronize() every `sync_every` steps prevents the
    # Windows driver from batching submissions and starving the GPU. KNOWN ISSUE
    # (open as of 2026-06-12): the one-off 140 → 90 s per 1000 steps improvement
    # measured for this on 2026-06-11 has NOT been reproduced — production runs
    # still log ~140 s/1000 steps with the keep-alive ON (reference: 92.7 s/1000).
    # See the full status note on wddm_sync_every in core/Backend.jl.
    sync_every = wddm_sync_every(state.backend)
    for i in 1:nsteps
        if prof_on
            ta = time_ns()
            step!(state)
            synchronize(state.backend); tb = time_ns()
            _record_phase_monitors!(phase.monitors, state)
            synchronize(state.backend); tc = time_ns()
            prof_step += (tb - ta) / 1.0e9
            prof_readout += (tc - tb) / 1.0e9
            if i % 1000 == 0
                tot = max(prof_step + prof_readout, eps())
                @info "[profile] pump per-group time (cumulative)" steps=i step_s=round(prof_step, digits=3) readout_s=round(prof_readout, digits=3) step_pct=round(100 * prof_step / tot, digits=1) readout_pct=round(100 * prof_readout / tot, digits=1)
            end
        else
            step!(state)
            _record_phase_monitors!(phase.monitors, state)
            sync_every > 0 && i % sync_every == 0 && synchronize(state.backend)
        end
        _phase_timing_log!(:pump, state, n0, nsteps, phase.log_every, t_start_ns, last_log_time)
    end
    # Consume any partial multiphysics window (the reference's `n == steps_pump`
    # consume) so the accumulated pabs·dt of the last few EM steps is not dropped.
    flush_multiphysics!(state)
    step_profiling!(state, false)
    return (phase=:pump, phase_index=phase_index, steps=nsteps, n_start=n0, n_end=state.n,
            t_start=t0, t_end=state.t)
end

function run_phase!(state::FDTDState, phase::Probe, cfg::SimConfig; phase_index::Integer=1)
    _phase_source!(state, phase, cfg)
    _attach_mode_monitors!(phase.monitors, state.source)
    nsteps = _phase_steps(phase, state.dt; default_steps=cfg.steps)
    n0 = state.n
    t0 = state.t
    # Optional per-group GPU-time attribution for the probe hot loop, mirroring the
    # reference driver's PROBE_PROFILE_KERNELS diagnostic. Enable by setting
    # `PROBE_PROFILE_KERNELS=1` before launching. When ON, the field update (`step!`)
    # and the readout (`_record_phase_monitors!`) are each bracketed by
    # synchronize() + wall clock, giving true per-group GPU time. This SERIALIZES the
    # pipeline, so absolute ms/step is inflated vs production — the relative split is
    # the signal. A cumulative breakdown prints every 1000 steps. Leave the env var
    # unset for production: the timing branch is skipped entirely (~free).
    prof_on = get(ENV, "PROBE_PROFILE_KERNELS", "0") == "1"
    prof_step = 0.0
    prof_readout = 0.0
    prof_on && @info "[profile] PROBE_PROFILE_KERNELS=1 — probe per-group timing ON (serialized; diagnostic only)"
    step_profiling!(state, prof_on)   # per-kernel split inside step! (H/src/E/ADE/MO/pabs/mp)
    t_start_ns = time_ns()
    last_log_time = Ref(t_start_ns)
    # WDDM keep-alive flush, same rationale (and same OPEN known-issue) as the pump loop.
    sync_every = wddm_sync_every(state.backend)
    for i in 1:nsteps
        if prof_on
            synchronize(state.backend); ta = time_ns()
            step!(state)
            synchronize(state.backend); tb = time_ns()
            _record_phase_monitors!(phase.monitors, state)
            synchronize(state.backend); tc = time_ns()
            prof_step += (tb - ta) / 1.0e9
            prof_readout += (tc - tb) / 1.0e9
            if i % 1000 == 0
                tot = max(prof_step + prof_readout, eps())
                @info "[profile] probe per-group GPU time (cumulative)" steps=i step_s=round(prof_step, digits=3) readout_s=round(prof_readout, digits=3) step_pct=round(100 * prof_step / tot, digits=1) readout_pct=round(100 * prof_readout / tot, digits=1)
            end
        else
            step!(state)
            _record_phase_monitors!(phase.monitors, state)
            sync_every > 0 && i % sync_every == 0 && synchronize(state.backend)
        end
        _phase_timing_log!(:probe, state, n0, nsteps, phase.log_every, t_start_ns, last_log_time)
    end
    step_profiling!(state, false)
    return (phase=:probe, phase_index=phase_index, steps=nsteps, n_start=n0, n_end=state.n,
            t_start=t0, t_end=state.t, lambda0=phase.lambda0)
end

function run_phase!(state::FDTDState, phase::Relax, cfg::SimConfig; phase_index::Integer=1)
    _phase_source!(state, phase, cfg)
    dt_relax = phase.dt === nothing ? state.dt : phase.dt
    nsteps = _phase_steps(phase, dt_relax; default_steps=0)
    n0 = state.n
    t0 = state.t
    subcycles = phase.subcycles === nothing ? state.subcycles : phase.subcycles
    # WDDM keep-alive, coarser than pump/probe: relax kernels touch only the active
    # film cells (µs-scale), where queue batching is efficient — a per-step sync would
    # be pure overhead. Flushing every 100 steps just bounds the submission queue.
    sync_every = wddm_sync_every(state.backend)
    for i in 1:nsteps
        relax_step!(state, dt_relax; subcycles=subcycles)
        state.n += 1
        # The relax phase has its own coarse time step: monitors must see
        # t = T_pump + n·dt_relax (the reference mag_time axis), not n·dt_EM.
        state.t += dt_relax
        _record_phase_monitors!(phase.monitors, state)
        sync_every > 0 && i % (100 * sync_every) == 0 && synchronize(state.backend)
    end
    state.last_mp_n = state.n   # relax steps consumed no EM window
    return (phase=:relax, phase_index=phase_index, steps=nsteps, n_start=n0, n_end=state.n,
            t_start=t0, t_end=state.t, dt=dt_relax)
end

function _result_summary(state::FDTDState)
    mag = state.mag
    switched_fraction = NaN
    if mag !== nothing
        m = to_host(mag.m_TM_x)
        switched_fraction = isempty(m) ? NaN : count(<(0.0), m) / length(m)
    end
    return (n=state.n, t=state.t, dt=state.dt,
            n_material=getproperty(state.region, :n_material),
            field_energy=field_energy(to_host(state.fields), state.grid, state.params),
            switched_fraction=switched_fraction)
end

function run_experiment(cfg::SimConfig; phases=[Pump(steps=cfg.steps)], model::AbstractPhysicsModel=MagnetoOpticModel(),
                        scene=nothing, enable_magneto_optic::Bool=true, on_init=nothing)
    state = init_state(cfg; model=model, scene=scene, enable_magneto_optic=enable_magneto_optic)
    on_init === nothing || on_init(state)
    # Return init-time temporaries (raster/scene uploads) held as pool reserve to the
    # driver before the hot loops start — VRAM headroom matters on saturated WDDM cards.
    reclaim_device_memory!(state.backend)
    phase_results = NamedTuple[]
    monitor_results = Dict{Symbol,Any}()
    for (pi, phase) in enumerate(phases)
        push!(phase_results, run_phase!(state, phase, cfg; phase_index=pi))
        for (mi, m) in enumerate(getproperty(phase, :monitors))
            monitor_results[_monitor_key(m, pi, mi)] = monitor_data(m)
        end
    end
    return Result(cfg, state, phase_results, monitor_results, _result_summary(state))
end
