# Replicate the validated reference run (Main_Research/pump_probe_switching_empirical_params.jl)
# with the generalized MagnetoPhotonic package, in two tiers that mirror the reference's
# two scripts: `production` (pump + relax, writes the production HDF5 + frozen magnetization)
# and `probe` (reloads the frozen magnetization, runs the 3-shot GOLD-STANDARD probe).
#
# The production/probe tiers replicate the reference run closely: bit-exact mesh
# (mesh=:reference, 4099x190x100, dt = 1.1218649561e-17 via courant=0.99), the verbatim
# analytic mode source (cos-taper profile at x = 2 µm, sin(ωt)·Gaussian(σ) pulse scaled by
# dt/dt_ref, neff = n_Si3N4·0.75), the reference equilibration dry-run, per-EM-step
# U_abs/pabs accumulation with window-averaged 4TM heating, single-step relax LLB at
# dt = 1 fs, streamed |Ez| movie frames in the reference (Nx,Ny,frames) layout, and the
# reference switching metrics. Compute precision is :f64 over Float32 field storage —
# the reference's validated mixed precision (:f32 compute destabilizes the ADE state).
# Remaining known divergences: source injection ordering inside one EM step (half-step
# bookkeeping), active-cell array ORDER (same set, different traversal), and GPU
# scheduling/rounding — so validation is physics-level + schema-shape-exact, not bit-exact.
#
# Big .h5 outputs are written to PackageValidation/ (override with PACKAGE_VALIDATION_DIR) so
# they can be git-ignored while the package source is published.
#
# Tiers:  julia --project=. examples/replicate_production.jl <smoke|calibrate|production|probe>
#   smoke       CPU pipeline check (tiny grid), exercises both writers + the frozen-m loader
#   calibrate   sweep pump_E_scale and report absorbed fluence / Te peak / switch fraction
#   production  GPU pump+relax on the 60um NOT-gate -> production HDF5 + frozen magnetization
#   probe       GPU 3-shot 532 nm readout (reloads frozen m from the production HDF5) -> GOLDSTD

using MagnetoPhotonic

try
    @eval using HDF5
catch err
    @warn "HDF5 is required for schema-exact production output; run `using HDF5` or add HDF5 to the environment" exception=(err, catch_backtrace())
end

const TIER = get(ARGS, 1, "smoke")

# Load the CUDA backend extension at TOP LEVEL for the GPU tiers, before any GPU code runs.
# Loading it lazily inside a function trips a world-age error: the already-running call
# cannot see the backend methods the extension adds.
if TIER in ("production", "probe", "spotcheck", "probespeed")
    try
        @eval using CUDA
    catch err
        @warn "CUDA could not be loaded; the :cuda backend will error" exception=(err, catch_backtrace())
    end
end

const PUMP_LAMBDA = 800e-9
const PROBE_LAMBDA = 532e-9
const PROBE_PRE_X = 33.0e-6
const PROBE_POST_X = 48.0e-6
const TARGET_F_ABS_MJCM2 = 0.5
const E_AMPLITUDE_BASE = 2.24e8
const PUMP_E_SCALE = 1.27
# Reference metadata ground truth: E_amplitude = E_amplitude_base × pump_E_scale
# = 2.8448e8 V/m. (The 0.7837 "scale_to_target" in the reference file is a
# post-run DIAGNOSTIC — sqrt(target/actual fluence) — never applied as input.)
const E_AMPLITUDE = E_AMPLITUDE_BASE * PUMP_E_SCALE
# Reference pump/probe envelopes: sin(ω t)·exp(−½((t−4σ)/σ)²) with σ = 40 fs
# (pump) / 24 fs (probe); both sources sit at x = 2 µm with neff = n_Si3N4·0.75.
const PUMP_SIGMA = 40e-15
const PROBE_SIGMA = 24e-15
const SOURCE_X = 2.0e-6
const WG_Z_CENTER = 0.2e-6        # waveguide mid-height (wg_height 0.40 µm)
const PUMP_ARM_Y = 0.75e-6        # NOT-gate input arm (pump)
const PROBE_ARM_Y = -0.75e-6      # NOT-gate lower arm (probe)
const EQUIL_STEPS = 10000         # reference equilibration dry-run @ 0.5 fs
const EQUIL_DT = 0.5e-15
const CORE_UABS_FRACTION = 0.5

# Reference production dt (read from the validated raw_v6 production HDF5). The pump
# window is the reference step count × dt; the probe window is the reference's literal
# probe_duration = 0.6 ps with steps = round(duration/dt) (run_probe_analysis_sim).
const REF_DT = 1.1218649561077176e-17
const PUMP_DURATION_S = 71310 * REF_DT     # ~8.00e-13 s pump optical window
const PROBE_DURATION_S = 0.6e-12           # reference probe_duration (0.6 ps)
const RELAX_STEPS = 15200                  # cool-down at a fixed coarse dt (dt-independent)
const RELAX_DT = 1e-15

# Big outputs land here so they can be git-ignored while the package source is published.
const OUTDIR = get(ENV, "PACKAGE_VALIDATION_DIR",
                   "D:/OneDrive - MIPA UNPAD/Thesis Preparation/Lab/PackageValidation")
mkpath(OUTDIR)
outpath(name::AbstractString) = joinpath(OUTDIR, name)

# dt is deterministic from the GridConfig (cfl_dt with the config courant); compute it from
# the grid alone (no field allocation) so phase durations map to integer step counts.
package_dt(cfg) = cfl_dt(MagnetoPhotonic.grid_from_config(cfg.grid), FDTDParams(); courant=cfg.grid.courant)
pkg_steps(duration::Real, dt::Real) = max(1, ceil(Int, Float64(duration) / Float64(dt)))

# CPML ψ-slab storage precision, overridable for A/B timing via MP_CPML_PSI=f32|f64.
# Default (env unset) is Float32 for BOTH phases: a controlled A/B (2026-06-13) showed the
# 6 GiB card hits a 0-free-VRAM WDDM cliff with Float64 ψ that slows ALL kernels, and
# Float32 ψ frees ~0.47 GiB to clear it — faster for the pump (H/E) and the probe (readout
# buffers stay resident). Math stays Float64 (kernels cast ψ on read). Set the env var to
# force Float64 for whichever phase runs, e.g. to re-measure the cliff or for A/B.
function _cpml_psi_T(default::Type)
    v = lowercase(strip(get(ENV, "MP_CPML_PSI", "")))
    (v == "f32" || v == "float32") && return Float32
    (v == "f64" || v == "float64") && return Float64
    return default
end

# CUDA is loaded at top level (above) for the GPU tiers; here we just verify a functional GPU.
function ensure_cuda()
    ok = MagnetoPhotonic.has_gpu()
    ok || @warn "No functional CUDA GPU detected; the :cuda backend will error. Ensure CUDA.jl is installed and a GPU is available."
    return ok
end

function log_cuda_memory_after_init(state)
    state.backend isa MagnetoPhotonic.CUDABackend || return nothing
    try
        @info "CUDA memory after init_state"
        if isdefined(CUDA, :pool_status)
            CUDA.pool_status()
        end
        if isdefined(CUDA, :memory_info)
            free_bytes, total_bytes = CUDA.memory_info()
            used_bytes = total_bytes - free_bytes
            @info "CUDA device memory" used_GiB=round(used_bytes / 2.0^30; digits=3) free_GiB=round(free_bytes / 2.0^30; digits=3) total_GiB=round(total_bytes / 2.0^30; digits=3)
        end
    catch err
        @warn "CUDA memory status failed" exception=(err, catch_backtrace())
    end
    return nothing
end

# Reference geometry: Sellmeier-dispersed core ε = n_Si3N4(λ)² (4.0972 @ 800 nm,
# 4.227 @ 532 nm), an SiO2 substrate filling z < 0, and a vacuum-base film whose
# metal response comes entirely from the ADE poles. `bare=true` reproduces the
# reference's film_disabled bare waveguide (gold-standard probe normalization).
function production_device(; model=MagnetoOpticModel(preset=:gdfeco),
                           lambda::Real=PUMP_LAMBDA, bare::Bool=false)
    p = FDTDParams(lambda)
    return not_gate_60um(units=:m, film_thickness=8e-3, model=model,
                         core_epsr=p.epsr_si3n4, substrate_epsr=p.epsr_sio2,
                         substrate_depth=2.0, bare=bare)
end

# Verbatim reference geometry (Yee-staggered ε + per-component ADE lists) on the
# reference grid; bare=true reproduces the film_disabled bare waveguide.
function reference_geometry(cfg, lambda::Real; bare::Bool=false)
    grid = MagnetoPhotonic.grid_from_config(cfg.grid)
    return not_gate_reference_geometry(grid, FDTDParams(lambda); film_disabled=bare)
end

# The reference pump does NOT fit poles to ε(800 nm): it uses three hard-coded
# empirical Drude–Lorentz poles (pump_probe_switching_empirical_params.jl L3548).
# Using a 2-pole fit instead changes the loss and skin depth at 800 nm — measured
# ~7.7× over-absorption. Reproduce the reference poles verbatim.
function reference_pump_poles(dt::Real)
    eps0 = FDTDParams().eps0
    return [create_pole(0.0, 1e14, (1e16)^2, dt, eps0),
            create_pole(1.5e15, 2e14, 2.5 * (1.5e15)^2, dt, eps0),
            create_pole(3.0e15, 5e14, 1.2 * (3.0e15)^2, dt, eps0)]
end

function tier_config(tier::AbstractString; backend_device=:cpu)
    if tier == "production" || tier == "probe"
        return gdfeco_pump_probe(
            # Exact validated reference NOT-gate mesh (4099x190x100): mesh=:reference uses
            # not_gate_reference_grid() (verbatim port of the reference grid builders), and
            # courant=0.99 makes the CFL dt bit-identical to the reference (1.1218649561e-17).
            # xlim/ylim/zlim are set to the reference extents so the bounds-derived code
            # (e.g. probe readout planes) is consistent; the geometry fields are ignored.
            grid=GridConfig(xlim=(0.0, 60e-6), ylim=(-3.0e-6, 3.0e-6), zlim=(-1.3e-6, 1.7e-6),
                            mesh=:reference, fine_region=(40e-6, 40.008e-6), courant=0.99,
                            subpixel=8, model_yee_stagger=true),
            source=SourceConfig(kind=:mode, lambda0=PUMP_LAMBDA, amplitude=E_AMPLITUDE,
                                tau=80e-15, t0=320e-15, component=:Ez,
                                position=1.0e-6, neff_guess=2.0),
            probe=ProbeConfig(kind=:mode, lambda0=PROBE_LAMBDA, amplitude=1.0e6,
                              tau=24e-15, t0=96e-15, component=:Ez,
                              position=1.0e-6, neff_guess=2.0),
            model=ModelConfig(absorption_model=:ade_work, multiphysics_subcycle=tier == "probe" ? 1 : 8,
                              brillouin_iters=2, Hsw0=0.0, pump_E_scale=1.0),
            # Reference anisotropic CPML: 40 cells along x (propagation) for the −60 dB budget,
            # 12 transverse. Slab-stored, so the thick x-PML costs only a few MB.
            pml=PMLConfig(cells=(40, 12, 12)),
            # :f64 compute everywhere (fields stay Float32 storage via SimConfig.precision) —
            # the reference's validated mixed precision. :f32 compute destabilizes the ADE
            # J = 2ΔP/dt − J_old recursion (2/dt ≈ 1.8e17 amplifies rounding) → Te blow-up.
            backend=BackendConfig(device=backend_device, compute_precision=:f64),
            steps=71310,
        )
    end
    return gdfeco_pump_probe(
        grid=GridConfig(xlim=(0.0, 3.0e-6), ylim=(-0.75e-6, 0.75e-6), zlim=(-0.30e-6, 0.30e-6),
                        mesh=:graded, dx=90e-9, dy=75e-9, dz=60e-9,
                        fine_dx=20e-9, fine_region=(1.55e-6, 1.70e-6),
                        stretch_ratio=1.12, courant=0.30),
        source=SourceConfig(kind=:mode, lambda0=PUMP_LAMBDA, amplitude=8.0e1,
                            tau=12e-15, t0=36e-15, component=:Ez,
                            position=0.30e-6, neff_guess=2.0),
        probe=ProbeConfig(kind=:mode, lambda0=PROBE_LAMBDA, amplitude=5.0,
                          tau=16e-15, t0=48e-15, component=:Ez,
                          position=0.30e-6, neff_guess=2.0),
        model=ModelConfig(absorption_model=:ade_work, multiphysics_subcycle=4,
                          brillouin_iters=2, Hsw0=0.0, pump_E_scale=1.0),
        pml=PMLConfig(cells=4),
        backend=BackendConfig(device=backend_device, compute_precision=:f64),
        steps=12,
    )
end

function smoke_device(model)
    p = FDTDParams(PUMP_LAMBDA)
    core = Material("Si3N4"; epsr=p.epsr_si3n4, color=:gray)
    return waveguide_device(
        ybranch((0.15e-6, 0.0), (0.75e-6, 0.0), (1.35e-6, 0.28e-6), (1.35e-6, -0.28e-6);
                width=0.28e-6, n=24),
        taper((1.35e-6, 0.0), (1.55e-6, 0.0); width_start=0.56e-6, width_end=0.28e-6),
        film_region(1.55e-6, 1.70e-6; width=0.28e-6),
        straight((1.70e-6, 0.0), (2.85e-6, 0.0); width=0.28e-6, name=:output);
        core=core, film_model=model, width=0.28e-6, height=0.22e-6, zmin=-0.11e-6,
    )
end

function production_metadata(; source_h5="", output_h5="", frame_skip=79,
                             steps_pump=71310, steps_relax=RELAX_STEPS, steps_probe=53500,
                             extras=NamedTuple())
    base = (;
        target_F_abs_mJcm2=TARGET_F_ABS_MJCM2,
        E_amplitude_base=E_AMPLITUDE_BASE,
        # Post-run diagnostic (sqrt(target/actual absorbed fluence)); the
        # production tier overrides it through `extras` after the pump.
        E_amplitude_scale_to_target_F_abs=NaN,
        E_amplitude=E_AMPLITUDE,
        pump_E_scale=PUMP_E_SCALE,
        fluence_scale_from_base=PUMP_E_SCALE^2,
        probe_amplitude_V_m=1.0e6,
        probe_duration_s=PROBE_DURATION_S,
        probe_lambda_nm=532.0,
        probe_pre_x_um_requested=33.0,
        probe_post_x_um_requested=48.0,
        probe_spectrum_bins=5,
        probe_trace_stride=10,
        probe_frame_target=903,
        frame_skip=frame_skip,
        steps_pump_nominal=steps_pump,
        steps_pump_run=steps_pump,
        steps_relax=steps_relax,
        steps_probe=steps_probe,
        pml_x=40, pml_y=12, pml_z=12,
        source_h5=source_h5,
        output_h5=output_h5,
        probe_tau_s=24e-15,
        eps_slice_z=WG_Z_CENTER,
        dt_relax_s=RELAX_DT,
    )
    return merge(base, extras)
end

# ---- Reference source replication --------------------------------------------------
# The reference injects an ANALYTIC cos-taper mode profile (not a solved eigenmode)
# at x = 2 µm with neff = n_Si3N4·0.75 and a sin(ωt)·Gaussian(σ) pulse scaled by
# (dt/dt_ref). Reproduced verbatim so the launched waveform matches the validated run.

function reference_mode_profile(grid; y_center::Real, z_center::Real=WG_Z_CENTER)
    Ny = length(grid.y.centers)
    Nz = length(grid.z.centers)
    w = zeros(Float64, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        dy = abs(grid.y.centers[j] - y_center)
        dz = abs(grid.z.centers[k] - z_center)
        w[j, k] = (dy <= 0.2e-6 ? cos(pi * dy / 0.5e-6) : cos(pi / 2.5) * exp(-(dy - 0.2e-6) / 100e-9)) *
                  (dz <= 0.2e-6 ? cos(pi * dz / 0.5e-6) : cos(pi / 2.5) * exp(-(dz - 0.2e-6) / 100e-9))
    end
    return w
end

# The reference's per-axis d_min is NOT the fine spacing: build_graded_grid takes
# min(minimum(diff(edges)), 1/maximum(inv_d_dual)) where inv_d_dual[1] is the
# boundary HALF-cell 1/(centers[1]−edges[1]) and the clamped outermost edge can
# leave a sub-d_fine cell. Using plain minimum(diff(edges)) made dt_ref ~1.52×
# too big → injected amplitude ×0.657 → absorbed dose ×0.43 (measured uniformly).
function reference_axis_d_min(axis)
    d_min_cell = minimum(diff(axis.edges))
    inv_dual_max = 1.0 / (axis.centers[1] - axis.edges[1])
    for i in 2:length(axis.centers)
        inv_dual_max = max(inv_dual_max, 1.0 / (axis.centers[i] - axis.centers[i - 1]))
    end
    return min(d_min_cell, 1.0 / inv_dual_max)
end

# dt_ref uses the 15 nm bulk dx (hard-coded in the reference) and the reference
# d_min of the two transverse axes; the per-step source amplitude is ×(dt/dt_ref).
function reference_dt_ref(grid, p)
    dymin = reference_axis_d_min(grid.y)
    dzmin = reference_axis_d_min(grid.z)
    return 0.99 / (p.c0 * sqrt(1.0 / (15e-9)^2 + 1.0 / dymin^2 + 1.0 / dzmin^2))
end

function reference_mode_source(state; amplitude::Real, lambda::Real, sigma::Real, y_center::Real)
    p = FDTDParams(lambda)
    dt_ref = reference_dt_ref(state.grid, p)
    pulse = CarrierGaussianPulse(amplitude=Float64(amplitude) * (state.dt / dt_ref),
                                 t0=4.0 * Float64(sigma), sigma=Float64(sigma),
                                 omega=2pi * p.c0 / Float64(lambda))
    prof = reference_mode_profile(state.grid; y_center=y_center)
    neff = sqrt(p.epsr_si3n4) * 0.75
    return ModeSource(adapt_backend(state.backend, state.compute_T.(prof));
                      pulse=pulse, neff=neff, axis=:x, position=SOURCE_X, component=:Ez)
end

# Streams probe movie frames straight into the GOLDSTD HDF5 at the planned step
# indices (host RAM cannot hold ~11 GB of in-memory frames on this machine; the
# reference streams too). Syncs only on frame steps.
struct ProbeFrameStreamer <: MagnetoPhotonic.AbstractMonitor
    writer::Any
    plan::Vector{Int}
    ptr::Base.RefValue{Int}
end

MagnetoPhotonic._monitor_due(m::ProbeFrameStreamer, n::Integer) =
    m.ptr[] <= length(m.plan) && Int(n) == m.plan[m.ptr[]]

function MagnetoPhotonic.record!(m::ProbeFrameStreamer, sim)
    record_probe_frame!(m.writer, sim.state)
    m.ptr[] += 1
    return m
end

MagnetoPhotonic.monitor_data(::ProbeFrameStreamer) = nothing

# ---- Output payloads ----------------------------------------------------------------

# Per-active-cell volumes dx_i·dy_j·dz_k from the column-major linear index
# (the reference's geo.V_cell_all; the mesh is graded, so volumes vary per cell).
function active_cell_volumes(state)
    cells = Int.(to_host(state.region.material_cells))
    g = state.grid
    Nx = length(g.x.centers)
    Ny = length(g.y.centers)
    dx = diff(g.x.edges)
    dy = diff(g.y.edges)
    dz = diff(g.z.edges)
    vols = Vector{Float64}(undef, length(cells))
    for (p, li) in enumerate(cells)
        li0 = li - 1
        i = li0 % Nx + 1
        j = (li0 ÷ Nx) % Ny + 1
        k = li0 ÷ (Nx * Ny) + 1
        vols[p] = dx[i] * dy[j] * dz[k]
    end
    return vols
end

# Reference print_absorbed_fluence: F_abs = Σ(U·V)/(ΣV/thickness), in mJ/cm².
function fluence_diagnostic(U_abs, vols; film_thickness::Real=8e-9, target::Real=TARGET_F_ABS_MJCM2)
    isempty(U_abs) && return (F_abs_mJcm2=NaN, local_F_abs_max_mJcm2=NaN,
                              local_U_abs_max=NaN, amp_scale_to_target=NaN)
    E_total = sum(U_abs .* vols)
    A_active = sum(vols) / film_thickness
    F = E_total / A_active / 10.0
    return (F_abs_mJcm2=F,
            local_F_abs_max_mJcm2=maximum(U_abs) * film_thickness / 10.0,
            local_U_abs_max=maximum(U_abs),
            amp_scale_to_target=F > 0.0 ? sqrt(Float64(target) / F) : Inf)
end

# Build the pump-group (post-pump) and shot_1 (post-relax) active payloads from a
# run_pump_relax result, with the reference metric definitions.
function production_payloads(res)
    state = res.state
    n = getproperty(state.region, :n_material)
    vols = active_cell_volumes(state)
    U_abs = res.U_abs
    peak = res.pabs_peak
    pump_active = merge(res.pump_snapshot, (;
        U_abs_J_m3=U_abs, U_abs_active_cells=U_abs,
        active_cell_volume_m3=vols,
        pabs_peak_W_m3=peak, pabs_peak_active_cells=peak,
        hot_cell_index=res.hot_cell_index,
    ))
    final = film_active_snapshot(state)
    metrics = switching_metrics(final.m_TM_x_reduced_active_cells,
                                final.m_RE_x_reduced_active_cells, U_abs;
                                core_fraction=CORE_UABS_FRACTION)
    cells_steps = Float64(n) * Float64(res.relax_steps)
    mlups = res.relax_elapsed_s > 0 ? cells_steps / res.relax_elapsed_s / 1e6 : NaN
    ns_per = cells_steps > 0 ? res.relax_elapsed_s * 1e9 / cells_steps : NaN
    shot_active = merge(final, metrics, (;
        final_Te_K_active=final.Te_active_cells,
        final_Tl_K_active=final.Tl_active_cells,
        final_Ts_TM_K_active=final.Ts_TM_active_cells,
        final_Ts_RE_K_active=final.Ts_RE_active_cells,
        final_m_TM_x_reduced_active=final.m_TM_x_reduced_active_cells,
        final_m_TM_y_reduced_active=final.m_TM_y_reduced_active_cells,
        final_m_TM_z_reduced_active=final.m_TM_z_reduced_active_cells,
        final_m_RE_x_reduced_active=final.m_RE_x_reduced_active_cells,
        final_m_RE_y_reduced_active=final.m_RE_y_reduced_active_cells,
        final_m_RE_z_reduced_active=final.m_RE_z_reduced_active_cells,
        final_U_abs_J_m3_active=U_abs,
        hot_cell_Te_K=res.hot.Te_K,
        hot_cell_mx_TM=res.hot.mx_TM,
        hot_cell_mx_RE=res.hot.mx_RE,
        switched_fraction_TM_time=res.swfrac.values,
        active_cell_MLUPS=mlups,
        ns_per_active_cell_step=ns_per,
    ))
    # film thickness along x: 40.008 µm − 40 µm (the reference's hard-coded 8 nm)
    fluence = fluence_diagnostic(U_abs, vols; film_thickness=8e-9)
    return pump_active, shot_active, fluence
end

# Pump + relax with the reference phase structure: optional equilibration dry-run
# (10000 × 0.5 fs pure 4TM+LLB), pump (optionally with a verbatim reference source
# and streamed movie frames), a post-pump active-cell snapshot (the reference's
# pump group is PRE-relax), then the deterministic relaxation with the reference's
# diagnostics (switched-fraction + hot-cell traces every `relax_every` steps).
function run_pump_relax(cfg, dev, model; pump_steps=nothing, pump_until=nothing,
                        relax_steps=20, relax_dt=RELAX_DT, pump_every=1, relax_every=1,
                        relax_subcycles=1, nan_guard=NaNGuard(10), log_every=0,
                        on_init=nothing, equilibrate::Bool=false, diag_poles=nothing,
                        geo=nothing,
                        pump_source_fn=nothing, frame_writer_fn=nothing, frame_every=0)
    # enable_magneto_optic=false: the reference pump loop runs NO magneto-optic
    # gyration (MO is probe-only); it also saves the MO ADE state's GPU memory.
    psi_T = _cpml_psi_T(Float32)   # pump default Float32 (off the 0-free-VRAM cliff); MP_CPML_PSI=f64 overrides
    @info "pump CPML ψ storage" psi_T=psi_T
    state = init_state(cfg; model=model, scene=dev, enable_magneto_optic=false,
                       diag_poles=diag_poles, geo=geo, cpml_psi_T=psi_T)
    on_init === nothing || on_init(state)
    # The reference records initial_m_* from the as-initialized LUT equilibrium,
    # BEFORE the equilibration dry-run (which the runtime |m| caps then trim).
    initial_m_TM = state.mag === nothing ? 1.0 :
        sum(Float64.(to_host(state.mag.m_TM_x))) / max(1, length(state.mag.m_TM_x))
    initial_m_RE = state.mag === nothing ? -1.0 :
        sum(Float64.(to_host(state.mag.m_RE_x))) / max(1, length(state.mag.m_RE_x))
    if equilibrate && state.thermal !== nothing
        @info "equilibration dry-run" steps=EQUIL_STEPS dt_fs=EQUIL_DT * 1e15
        for _ in 1:EQUIL_STEPS
            relax_step!(state, EQUIL_DT; subcycles=1)
        end
        MagnetoPhotonic.synchronize(state.backend)
    end

    pump_avg = FilmAverage(every=pump_every)
    monitors = Any[pump_avg, nan_guard]
    writer = frame_writer_fn === nothing ? nothing : frame_writer_fn(state)
    if writer !== nothing && frame_every > 0
        push!(monitors, CallbackMonitor(sim -> record_production_frame!(writer, sim.state);
                                        every=frame_every))
    end
    src = pump_source_fn === nothing ? nothing : pump_source_fn(state)
    pump = pump_until !== nothing ?
        Pump(until=pump_until, source=src, monitors=monitors, log_every=log_every) :
        Pump(steps=(pump_steps === nothing ? cfg.steps : pump_steps), source=src,
             monitors=monitors, log_every=log_every)
    run_phase!(state, pump, cfg)
    MagnetoPhotonic.synchronize(state.backend)
    writer === nothing || close_frame_writer!(writer)

    # Post-pump (pre-relax) snapshot: the reference's pump-group active arrays.
    pump_snapshot = film_active_snapshot(state)
    U_abs = state.absorption === nothing ? Float64[] : Float64.(to_host(state.absorption.U_abs))
    pabs_peak = state.absorption === nothing ? Float64[] : Float64.(to_host(state.absorption.peak))
    hot_idx = isempty(U_abs) ? 1 : argmax(U_abs)

    relax_avg = FilmAverage(every=relax_every)
    swfrac = SwitchedFraction(every=relax_every)
    hot = HotCellTrace(hot_idx; every=relax_every)
    relax = Relax(steps=relax_steps, dt=relax_dt, subcycles=relax_subcycles,
                  monitors=[relax_avg, swfrac, hot])
    t0 = time_ns()
    run_phase!(state, relax, cfg)
    MagnetoPhotonic.synchronize(state.backend)
    relax_elapsed = (time_ns() - t0) / 1e9

    return (; state,
            pump_data=monitor_data(pump_avg), relax_data=monitor_data(relax_avg),
            swfrac=monitor_data(swfrac), hot=monitor_data(hot),
            pump_snapshot, U_abs, pabs_peak, hot_cell_index=hot_idx,
            relax_steps, relax_elapsed_s=relax_elapsed,
            initial_m_TM_x=initial_m_TM, initial_m_RE_x=initial_m_RE,
            field_energy=field_energy(to_host(state.fields), state.grid, state.params))
end

function run_probe_shot(cfg, dev, model; label, is_reference=false, magdata=nothing,
                        mag_source_h5=nothing, steps=60, frame_target=0,
                        incident_energy=nothing, incident_sx=nothing, store_frames=true,
                        nan_guard=NaNGuard(10), log_every=0, on_init=nothing,
                        use_reference_source::Bool=false, geo=nothing,
                        frame_stream_path=nothing, frame_stream_group::AbstractString="")
    # Use the requested readout planes when they fit the domain; only fall back to a
    # fractional placement on a grid too small to host the requested plane (e.g. smoke).
    L = cfg.grid.xlim[2] - cfg.grid.xlim[1]
    pre_x = PROBE_PRE_X <= 0.9L ? PROBE_PRE_X : 0.45L
    post_x = PROBE_POST_X <= 0.9L ? PROBE_POST_X : 0.85L
    # Probe shots default to Float32 CPML ψ storage: it frees ~0.47 GiB so the per-step
    # readout DFT/trace buffers stay resident on the 6 GiB card (they otherwise demote to
    # PCIe at 0 B free). ψ is pure storage (kernels cast to Float64 on read), and this is
    # the configuration the GOLDSTD probe physics was validated in. Override with
    # MP_CPML_PSI=f64 to A/B against full-precision ψ. The pump defaults to Float64.
    psi_T = _cpml_psi_T(Float32)
    @info "probe CPML ψ storage" label=label psi_T=psi_T
    state = init_state(cfg; model=model, scene=dev, enable_magneto_optic=!is_reference,
                       geo=geo, cpml_psi_T=psi_T)
    on_init === nothing || on_init(state)
    if mag_source_h5 !== nothing
        apply_magnetization!(state, load_production_magnetization(mag_source_h5; state=state))
    elseif magdata !== nothing
        apply_magnetization!(state, magdata)
    end
    # Reference probe: analytic cos-taper profile on the LOWER NOT-gate arm
    # (y = −0.75 µm) used for BOTH injection and readout overlap, frames captured
    # at unique(round.(range(1, steps; length=903))) like the reference driver.
    probe_profile = use_reference_source ?
        reference_mode_profile(state.grid; y_center=PROBE_ARM_Y) : nothing
    frame_plan = frame_target > 0 ?
        unique(round.(Int, range(1, steps; length=min(frame_target, steps)))) : Int[]
    streaming = frame_stream_path !== nothing && !isempty(frame_plan)
    readout = ProbeReadout(pre=pre_x, post=post_x, mode=probe_profile,
                           lambda0=PROBE_LAMBDA, spectrum_bins=5, trace_stride=10,
                           frame_target=frame_target, frame_indices=frame_plan,
                           store_frames=store_frames && !streaming,
                           slice_z_position=use_reference_source ? WG_Z_CENTER : NaN,
                           final_step=steps,
                           probe_amplitude=cfg.probe.amplitude,
                           probe_tau=cfg.probe.tau)
    src = use_reference_source ?
        reference_mode_source(state; amplitude=cfg.probe.amplitude,
                              lambda=PROBE_LAMBDA, sigma=PROBE_SIGMA, y_center=PROBE_ARM_Y) :
        nothing
    monitors = Any[readout, nan_guard]
    writer = nothing
    if streaming
        writer = probe_frame_writer(frame_stream_path, state; group=frame_stream_group,
                                    frame_count=length(frame_plan),
                                    slice_z_pos=use_reference_source ? WG_Z_CENTER : NaN,
                                    pre_x=pre_x, post_x=post_x)
        push!(monitors, ProbeFrameStreamer(writer, frame_plan, Ref(1)))
    end
    phase = Probe(lambda0=PROBE_LAMBDA, steps=steps, source=src,
                  monitors=monitors,
                  freeze_magnetization=true, enable_magneto_optic=!is_reference,
                  log_every=log_every)
    run_phase!(state, phase, cfg)
    writer === nothing || close_frame_writer!(writer)
    # Physical film absorption (the reference's A): per-cell E·J work integrated
    # over the probe run × cell volume. Zero for the bare reference shot.
    absorbed = state.absorption === nothing ? 0.0 :
        sum(Float64.(to_host(state.absorption.U_abs)) .* state.region.cell_volumes)
    shot = probe_shot(readout; state_label=label, is_reference=is_reference,
                      incident_energy=incident_energy, incident_sx=incident_sx,
                      dt=state.dt, steps_probe=steps, absorbed_energy=absorbed)
    return shot, state
end

function run_smoke()
    model = MagnetoOpticModel(preset=:gdfeco)
    dev = smoke_device(model)
    cfg = tier_config("smoke")
    res = run_pump_relax(cfg, dev, model; pump_steps=12, relax_steps=20)
    @assert getproperty(res.state.region, :n_material) > 0
    @assert all(isfinite, to_host(res.state.fields.Ez))

    pump_active, shot_active, _ = production_payloads(res)
    prod_h5 = outpath("replicate_production_smoke.h5")
    gold_h5 = outpath("replicate_production_goldstd_smoke.h5")
    isfile(prod_h5) && rm(prod_h5)
    write_production_h5(prod_h5, res.state; pump=res.pump_data, relaxation=res.relax_data,
                        shot_1=res.relax_data, pump_active=pump_active, shot_active=shot_active,
                        metadata=production_metadata(output_h5=prod_h5))

    # Exercise the same frozen-magnetization path the GPU probe tier uses: reload the
    # switched magnetization straight out of the production HDF5 just written.
    ref, _ = run_probe_shot(cfg, dev, model; label="reference", is_reference=true, steps=12)
    init, _ = run_probe_shot(cfg, dev, model; label="initial", is_reference=false, steps=12,
                             incident_energy=ref.incident_energy_J, incident_sx=ref.Sx_inc)
    sw, sw_state = run_probe_shot(cfg, dev, model; label="switched", is_reference=false, steps=12,
                                  mag_source_h5=prod_h5,
                                  incident_energy=ref.incident_energy_J, incident_sx=ref.Sx_inc)
    write_goldstd_h5(gold_h5, (reference=ref, initial=init, switched=sw); state=sw_state,
                     metadata=production_metadata(source_h5=prod_h5, output_h5=gold_h5))
    @info "smoke complete" production_h5=prod_h5 goldstd_h5=gold_h5 n_material=getproperty(res.state.region, :n_material) switch_fraction=shot_active.final_switch_fraction field_energy=res.field_energy
end

function run_calibrate()
    model = MagnetoOpticModel(preset=:gdfeco)
    dev = smoke_device(model)
    for scale in (0.5, 0.8, 1.0, 1.27, 1.5)
        cfg = tier_config("smoke")
        cfg = SimConfig(; grid=cfg.grid, source=SourceConfig(; kind=cfg.source.kind, lambda0=cfg.source.lambda0,
                        amplitude=cfg.source.amplitude * scale, tau=cfg.source.tau, t0=cfg.source.t0,
                        phase=cfg.source.phase, component=cfg.source.component, axis=cfg.source.axis,
                        position=cfg.source.position, neff_guess=cfg.source.neff_guess,
                        max_iter=cfg.source.max_iter),
                        probe=cfg.probe, device=cfg.device, pml=cfg.pml, model=cfg.model,
                        output=cfg.output, steps=cfg.steps, backend=cfg.backend, precision=cfg.precision)
        res = run_pump_relax(cfg, dev, model; pump_steps=8, relax_steps=4)
        @info "calibration point" scale=scale field_energy=res.field_energy Te_peak=maximum(res.pump_data.Te_avg)
    end
end

function run_production()
    ensure_cuda()
    model = MagnetoOpticModel(preset=:gdfeco)
    dev = production_device(; model=model)
    cfg = tier_config("production"; backend_device=:cuda)
    dt = package_dt(cfg)
    steps_pump = pkg_steps(PUMP_DURATION_S, dt)
    # Subsample the film time series to the reference sample counts (713 pump,
    # 1520 relax) and the movie frames to the reference cadence (frame_skip 79).
    pump_every = max(1, fld(steps_pump, 713))
    relax_every = max(1, fld(RELAX_STEPS, 1520))
    frame_skip = max(1, floor(Int, steps_pump / 900))
    frame_count = cld(steps_pump, frame_skip)
    out = outpath("raw_v6_full_production_empirical_Hsw0_0T_Escale_1p27.h5")
    isfile(out) && rm(out)
    @info "production: physical-duration-driven pump" dt=dt steps_pump=steps_pump pump_duration_s=PUMP_DURATION_S frame_skip=frame_skip frame_count=frame_count
    res = run_pump_relax(cfg, dev, model; pump_until=PUMP_DURATION_S,
        relax_steps=RELAX_STEPS, relax_dt=RELAX_DT, pump_every=pump_every,
        relax_every=relax_every, relax_subcycles=1, equilibrate=true,
        diag_poles=reference_pump_poles(dt),
        geo=reference_geometry(cfg, PUMP_LAMBDA),
        nan_guard=NaNGuard([5000, 20000, 40000, 60000]),
        log_every=1000, on_init=log_cuda_memory_after_init,
        pump_source_fn=state -> reference_mode_source(state; amplitude=E_AMPLITUDE,
            lambda=PUMP_LAMBDA, sigma=PUMP_SIGMA, y_center=PUMP_ARM_Y),
        frame_writer_fn=state -> production_frame_writer(out, state;
            frame_count=frame_count, slice_z_pos=WG_Z_CENTER),
        frame_every=frame_skip)
    pump_active, shot_active, fluence = production_payloads(res)
    @info "PHASE 1 fluence diagnostics" F_abs_mJcm2=fluence.F_abs_mJcm2 U_abs_max=fluence.local_U_abs_max amp_scale_to_target=fluence.amp_scale_to_target
    extras = (;
        E_amplitude_scale_to_target_F_abs=fluence.amp_scale_to_target,
        absorbed_fluence_avg_mJcm2=fluence.F_abs_mJcm2,
        absorbed_fluence_local_max_mJcm2=fluence.local_F_abs_max_mJcm2,
        U_abs_local_max_Jm3=fluence.local_U_abs_max,
        initial_m_TM_x_reduced=res.initial_m_TM_x,
        initial_m_RE_x_reduced=res.initial_m_RE_x,
        relaxation_elapsed_s=res.relax_elapsed_s,
        shot_elapsed_s=res.relax_elapsed_s,
        phase2_elapsed_s=res.relax_elapsed_s,
        final_switch_fraction=shot_active.final_switch_fraction,
        final_core_switch_fraction=shot_active.final_core_switch_fraction,
    )
    write_production_h5(out, res.state; pump=res.pump_data, relaxation=res.relax_data,
                        shot_1=res.relax_data, pump_active=pump_active, shot_active=shot_active,
                        movies=(frame_count=frame_count,),
                        metadata=production_metadata(output_h5=out, steps_pump=steps_pump,
                                                     frame_skip=frame_skip, extras=extras),
                        overwrite=false)
    save_magnetization(outpath("frozen_switched_magnetization.h5"), res.state)
    @info "production pump/relax complete" output=out steps_pump=steps_pump switch_fraction=shot_active.final_switch_fraction core_switch_fraction=shot_active.final_core_switch_fraction
end

function run_probe()
    # Per-group GPU timing of the probe hot loop is available via the same env var the
    # reference driver uses: set `$env:PROBE_PROFILE_KERNELS = "1"` (PowerShell) before
    # launching to print a serialized step-vs-readout breakdown every 1000 steps.
    get(ENV, "PROBE_PROFILE_KERNELS", "0") == "1" &&
        @info "PROBE_PROFILE_KERNELS=1 detected — probe kernel profiling enabled"
    ensure_cuda()
    model = MagnetoOpticModel(preset=:gdfeco)
    # Probe geometry at 532 nm (Sellmeier core/substrate change with λ, like the
    # reference's geo_probe rebuild); the reference shot uses the BARE waveguide
    # (film replaced by continuous core — the reference's film_disabled path).
    dev = production_device(; model=model, lambda=PROBE_LAMBDA)
    dev_bare = production_device(; model=model, lambda=PROBE_LAMBDA, bare=true)
    cfg = tier_config("probe"; backend_device=:cuda)
    dt = package_dt(cfg)
    # Reference step mapping: steps_probe = max(1, round(Int, probe_duration/dt))
    # (run_probe_analysis_sim) → 53482 on the reference dt, not a ceil.
    steps_probe = max(1, round(Int, PROBE_DURATION_S / dt))
    # Frozen magnetization comes from a production HDF5 on the SAME grid (mirrors the
    # reference `run_probe_analysis_sim(h5_in)`). Defaults to the package's own production
    # output; override with MAG_SOURCE_H5 (must be a package production file, same mesh —
    # the reference raw_v6 file has a different mesh and will be rejected by the loader).
    source_h5 = get(ENV, "MAG_SOURCE_H5", outpath("raw_v6_full_production_empirical_Hsw0_0T_Escale_1p27.h5"))
    isfile(source_h5) ||
        error("frozen-magnetization source not found: $source_h5\n" *
              "Run the `production` tier first, or set MAG_SOURCE_H5 to a package production HDF5 on the same grid.")
    @info "probe: physical-duration-driven readout" dt=dt steps_probe=steps_probe mag_source=source_h5
    out = outpath("raw_v6_GOLDSTD_TRA_Hsw0_0T_Escale_1p27.h5")
    isfile(out) && rm(out)
    geo_probe = reference_geometry(cfg, PROBE_LAMBDA)
    geo_bare = reference_geometry(cfg, PROBE_LAMBDA; bare=true)
    # Reference memory discipline: free every finished shot's device arrays BEFORE the
    # next shot's init_state (the reference's free_fdtd_fields! + unsafe_free! per shot).
    # Without these frees each finished shot's ~5 GiB stays rooted, the pool reserve
    # climbs past the card (measured 9.4 GiB on 6 GiB) and WDDM demotes ~4 GiB to
    # system RAM — the next shot then runs at PCIe speed. Shot results (ref/init/sw)
    # are host-side NamedTuples, so freeing the states is safe.
    ref, ref_state = run_probe_shot(cfg, dev_bare, model; label="reference", is_reference=true,
                            steps=steps_probe, frame_target=0, use_reference_source=true,
                            geo=geo_bare,
                            nan_guard=NaNGuard([5000, 20000, 40000, 60000]),
                            log_every=1000, on_init=log_cuda_memory_after_init)
    MagnetoPhotonic.free_state!(ref_state); ref_state = nothing
    init, init_state = run_probe_shot(cfg, dev, model; label="initial", is_reference=false,
                             steps=steps_probe, frame_target=903,
                             use_reference_source=true, geo=geo_probe,
                             frame_stream_path=out, frame_stream_group="probe/initial",
                             incident_energy=ref.incident_energy_J, incident_sx=ref.Sx_inc,
                             nan_guard=NaNGuard([5000, 20000, 40000, 60000]),
                             log_every=1000, on_init=log_cuda_memory_after_init)
    MagnetoPhotonic.free_state!(init_state); init_state = nothing
    sw, sw_state = run_probe_shot(cfg, dev, model; label="switched", is_reference=false,
                                  mag_source_h5=source_h5, steps=steps_probe, frame_target=903,
                                  use_reference_source=true, geo=geo_probe,
                                  frame_stream_path=out, frame_stream_group="probe/switched",
                                  incident_energy=ref.incident_energy_J, incident_sx=ref.Sx_inc,
                                  nan_guard=NaNGuard([5000, 20000, 40000, 60000]),
                                  log_every=1000, on_init=log_cuda_memory_after_init)
    # Record the production run's switch fractions in the probe metadata, like the
    # reference's final_switch_fraction_from_shot_1.
    sf, csf = HDF5.h5open(source_h5, "r") do f
        g = f["shot_1"]
        (haskey(g, "final_switch_fraction") ? read(g["final_switch_fraction"]) : NaN,
         haskey(g, "final_core_switch_fraction") ? read(g["final_core_switch_fraction"]) : NaN)
    end
    write_goldstd_h5(out, (reference=ref, initial=init, switched=sw); state=sw_state,
                     metadata=production_metadata(source_h5=source_h5, output_h5=out, steps_probe=steps_probe,
                                                  extras=(final_switch_fraction=sf,
                                                          final_core_switch_fraction=csf)),
                     overwrite=false)
    @info "GOLDSTD probe complete" output=out steps_probe=steps_probe contrast=probe_contrast(init, sw)
end

# GPU regression probe for the f32 ADE blow-up: run the production config for a short
# physical window (default 250 fs; override SPOT_DURATION_FS) and print the film Te
# trajectory. With the validated :f64 compute the film must still sit at ~300.000 K at
# 237 fs (pump light cannot reach x = 40 µm yet); the broken f32 run showed 1111 K there.
function run_spotcheck()
    ensure_cuda()
    model = MagnetoOpticModel(preset=:gdfeco)
    dev = production_device(; model=model)
    cfg = tier_config("production"; backend_device=:cuda)
    duration = parse(Float64, get(ENV, "SPOT_DURATION_FS", "250")) * 1e-15
    res = run_pump_relax(cfg, dev, model; pump_until=duration, relax_steps=0,
        pump_every=100, relax_every=1, equilibrate=true, log_every=1000,
        diag_poles=reference_pump_poles(package_dt(cfg)),
        geo=reference_geometry(cfg, PUMP_LAMBDA),
        nan_guard=NaNGuard(5000), on_init=log_cuda_memory_after_init,
        pump_source_fn=state -> reference_mode_source(state; amplitude=E_AMPLITUDE,
            lambda=PUMP_LAMBDA, sigma=PUMP_SIGMA, y_center=PUMP_ARM_Y))
    te = res.pump_data.Te_avg
    mt = res.pump_data.mag_time
    for i in 1:max(1, length(te) ÷ 20):length(te)
        println("t=", round(mt[i] * 1e15, digits=1), " fs  Te_avg=", te[i])
    end
    println("final: t=", round(mt[end] * 1e15, digits=1), " fs  Te_avg=", te[end],
            "  Te_max_over_run=", maximum(te))
    println("U_abs mean=", isempty(res.U_abs) ? NaN : sum(res.U_abs) / length(res.U_abs),
            "  U_abs max=", isempty(res.U_abs) ? NaN : maximum(res.U_abs))
    println("initial m: m_TM_x=", res.initial_m_TM_x, "  m_RE_x=", res.initial_m_RE_x)
end

# GPU probe PERFORMANCE test: ONE 532 nm readout shot for a bounded step count
# (default 6000; override PROBE_SPEED_STEPS) with the per-kernel/step profiler ON and
# NO HDF5 output — no frozen-magnetization reload, no frame streaming (frame_target=0).
# Mirrors examples/perf_test_pump.jl so the probe readout cost is directly comparable to
# the pump. The magnetization is the as-initialized equilibrium: physics is irrelevant
# here (we don't normalize against a reference shot or write anything), and the readout
# monitor runs the identical per-step DFT + every-10 trace reduction regardless, so the
# step_s / readout_s split is representative of the real GOLDSTD probe.
function run_probespeed()
    get(ENV, "PROBE_PROFILE_KERNELS", "0") == "1" &&
        @info "PROBE_PROFILE_KERNELS=1 — probe per-group/per-kernel profiling enabled"
    ensure_cuda()
    model = MagnetoOpticModel(preset=:gdfeco)
    dev = production_device(; model=model, lambda=PROBE_LAMBDA)
    cfg = tier_config("probe"; backend_device=:cuda)
    geo_probe = reference_geometry(cfg, PROBE_LAMBDA)
    steps = parse(Int, get(ENV, "PROBE_SPEED_STEPS", "6000"))
    @info "probe perf test: bounded readout, profiled, NO HDF5" steps=steps
    shot, st = run_probe_shot(cfg, dev, model; label="speedtest", is_reference=false,
        steps=steps, frame_target=0, use_reference_source=true, geo=geo_probe,
        nan_guard=NaNGuard(5000), log_every=1000, on_init=log_cuda_memory_after_init)
    @info "probe perf test complete" steps=steps T=shot.T R=shot.R A=shot.A
    MagnetoPhotonic.free_state!(st)
end

if TIER == "smoke"
    run_smoke()
elseif TIER == "calibrate"
    run_calibrate()
elseif TIER == "production"
    run_production()
elseif TIER == "probe"
    run_probe()
elseif TIER == "spotcheck"
    run_spotcheck()
elseif TIER == "probespeed"
    run_probespeed()
else
    error("unknown tier $TIER; expected smoke, calibrate, production, probe, spotcheck, or probespeed")
end
