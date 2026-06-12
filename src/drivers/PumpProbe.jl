# High-level pump → relax → (probe) driver.
#
# Orchestrates the full coupled experiment the production code performs, at a
# scale set by `config` (use a small GridConfig for CPU runs; the production
# 60 µm device is GPU-scale). Phases:
#   1. Pump  — pulsed FDTD with CPML + diagonal ADE + MO gyration + 4TM/LLB.
#   2. Relax — fields off, only 4TM/LLB cool-down (rebuilds |m| on the new branch).
#   3. Probe — optional polarimetry shot (Faraday/Kerr) once magnetization is frozen.

# A straight Si3N4 waveguide core spanning x, with a thin GdFeCo film slab across
# it marked by `model` (so rasterize tags those cells as magneto-optic).
function pump_probe_scene(config::SimConfig, model::AbstractPhysicsModel; p::FDTDParams=FDTDParams(config.source.lambda0))
    xlo, xhi = config.grid.xlim
    ww = config.device.wg_width
    wh = config.device.wg_height
    core = Material("Si3N4"; epsr=p.epsr_si3n4, color=:gray)
    film = Material("GdFeCo"; epsr=1.0, model=model, color=:orange)
    scene = Scene()
    add_shape!(scene, Box(xlo, xhi, -ww / 2, ww / 2, -wh / 2, wh / 2), core)
    xfilm = clamp(config.device.x_film_start, xlo + 2 * config.grid.dx, xhi - 2 * config.grid.dx)
    if !(xlo < config.device.x_film_start < xhi)
        xfilm = 0.5 * (xlo + xhi)   # production film x is outside a small demo grid
    end
    thick = max(config.device.film_thickness, 2.0 * config.grid.dx)
    add_shape!(scene, Box(xfilm - thick / 2, xfilm + thick / 2, -ww / 2, ww / 2, -wh / 2, wh / 2), film)
    return scene
end

function run_pump_probe_sim(config::SimConfig=SimConfig(); model::AbstractPhysicsModel=MagnetoOpticModel(),
                            pump_steps::Integer=config.steps, relax_steps::Integer=0,
                            relax_dt::Real=1e-15, relax_subcycles::Integer=4,
                            thermal_kick::Real=0.0,
                            n_pml::Integer=8, enable_magneto_optic::Bool=true,
                            scene=nothing, h5_filename=nothing)
    p = FDTDParams(config.source.lambda0)
    b = backend(config)
    CT = compute_type(config)
    grid = grid_from_config(config; film_region=_film_bounds_for_grid(config, scene))
    scn = scene === nothing ? pump_probe_scene(config, model; p=p) : _scene_object(scene)
    geo = rasterize(scn, grid; subpixel=1)

    dt = cfl_dt(grid, p; courant=config.grid.courant)
    pulse = GaussianPulse(; amplitude=config.source.amplitude, t0=config.source.t0, tau=config.source.tau,
                          omega=2pi * p.c0 / config.source.lambda0, phase=config.source.phase)
    # Source plane near the x-min boundary, transversely centred.
    jc = cld(length(grid.y.centers), 2)
    kc = cld(length(grid.z.centers), 2)
    src_index = (max(2, n_pml + 2), jc, kc)

    st = FDTDState(grid, geo; dt=dt, params=p, source=(pulse, config.source.component, src_index),
                   backend=b, compute_precision=CT, T=config.precision, model=model, n_pml=n_pml,
                   enable_magneto_optic=enable_magneto_optic,
                   multiphysics_every=config.model.multiphysics_subcycle,
                   subcycles=config.model.multiphysics_subcycle,
                   brillouin_iters=config.model.brillouin_iters,
                   absorption_model=config.model.absorption_model)

    has_mag = st.mag !== nothing
    m_TM_x0 = has_mag ? copy(to_host(st.mag.m_TM_x)) : Float64[]
    m_RE_x0 = has_mag ? copy(to_host(st.mag.m_RE_x)) : Float64[]

    energies = Float64[]
    Te_peak = Float64[]
    function cb(s)
        push!(energies, field_energy(to_host(s.fields), s.grid, s.params))
        s.thermal !== nothing && push!(Te_peak, maximum(to_host(s.thermal.Te)))
    end

    # Phase 1: pump
    run!(st, pump_steps; callback=cb)

    # Optional thermal kick: deposit the absorbed pump energy directly as a hot
    # electron/spin transient. Lets the relaxation phase exhibit switching without
    # calibrating the (GPU-scale) EM fluence — the production code reaches this
    # hot state through thousands of pump steps at full power instead.
    if thermal_kick > 0 && has_mag
        tk = Float64(thermal_kick)
        fill!(st.thermal.Te, tk); fill!(st.thermal.Tl, tk)
        fill!(st.thermal.Ts_TM, tk); fill!(st.thermal.Ts_RE, tk)
    end

    # Phase 2: relax (no EM)
    for _ in 1:relax_steps
        relax_step!(st, relax_dt)
    end

    # Switching diagnostics: fraction of material cells whose FeCo x-component flipped.
    switched_fraction = 0.0
    if has_mag && !isempty(m_TM_x0)
        m_TM_x_now = to_host(st.mag.m_TM_x)
        flips = 0
        for n in eachindex(m_TM_x0)
            (sign(m_TM_x_now[n]) != sign(m_TM_x0[n]) && abs(m_TM_x_now[n]) > 0.1) && (flips += 1)
        end
        switched_fraction = flips / length(m_TM_x0)
    end

    result = (state=st, grid=grid, geo=geo, model=model, energies=energies, Te_peak=Te_peak,
              n_material=geo.n_material,
              m_TM_x0=m_TM_x0, m_RE_x0=m_RE_x0,
              m_TM_x=has_mag ? copy(to_host(st.mag.m_TM_x)) : Float64[],
              m_RE_x=has_mag ? copy(to_host(st.mag.m_RE_x)) : Float64[],
              Te=st.thermal === nothing ? Float64[] : copy(to_host(st.thermal.Te)),
              switched_fraction=switched_fraction)
    h5_filename === nothing || save_state(String(h5_filename), result)
    return result
end
