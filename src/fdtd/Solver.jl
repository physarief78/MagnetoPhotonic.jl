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
    inv_eps_x::Array{Float64,3}
    inv_eps_y::Array{Float64,3}
    inv_eps_z::Array{Float64,3}
    params::FDTDParams
    dt::Float64
    n::Int
    source::Union{Nothing,Tuple{GaussianPulse,Symbol,NTuple{3,Int}}}
    cpml::Union{Nothing,CPMLState}
    region::Any
    model::AbstractPhysicsModel
    diag_poles::Vector{DLPole}
    ade_x::Union{Nothing,ADEState}
    ade_y::Union{Nothing,ADEState}
    ade_z::Union{Nothing,ADEState}
    mo_poles::Vector{DLPole}
    mo::Union{Nothing,MagnetoOpticADEState}
    mo_pos::Vector{Int}
    thermal::Union{Nothing,ThermalState}
    mag::Union{Nothing,MagnetizationState}
    lut::Any
    multiphysics_every::Int
    subcycles::Int
    brillouin_iters::Int
    absorption_model::Symbol
end

function FDTDState(grid::Grid3D, geo;
                   dt::Real=cfl_dt(grid, FDTDParams(); courant=0.45),
                   params::FDTDParams=FDTDParams(),
                   backend::AbstractBackend=CPUBackend(),
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
                   absorption_model::Symbol=:cycle_average,
                   seed_tilt::Real=1e-4)
    fields = allocate_fields(grid; backend=backend, T=T)
    dtf = Float64(dt)
    cpml = (n_pml isa Integer ? n_pml > 0 : any(>(0), n_pml)) ?
           build_cpml(grid, n_pml, dtf, params; cpml_kwargs...) : nothing

    Nmat = hasproperty(geo, :n_material) ? geo.n_material : 0
    has_model = (model isa MagnetoOpticModel) && Nmat > 0

    dpoles = DLPole{Float64}[]
    ade_x = ade_y = ade_z = nothing
    mpoles = DLPole{Float64}[]
    mo = nothing
    mo_pos = Int[]
    thermal = nothing
    mag = nothing
    lut = nothing

    if has_model
        gd = model.params
        if enable_dispersion
            dpoles = diag_poles === nothing ? build_pump_poles(dtf, params.eps0, gd) : collect(DLPole{Float64}, diag_poles)
            ade_x = allocate_ade_state(Nmat, dpoles)
            ade_y = allocate_ade_state(Nmat, dpoles)
            ade_z = allocate_ade_state(Nmat, dpoles)
        end
        thermal = ThermalState(Nmat, model)
        mag = MagnetizationState(Nmat, model; seed_tilt=seed_tilt)
        lut = build_m_eq_lut(gd)
        if enable_magneto_optic
            mpoles = mo_poles === nothing ? build_probe_mo_poles(dtf, params.eps0, gd) : collect(DLPole{Float64}, mo_poles)
            mo = allocate_mo_ade_state(Nmat, mpoles)
            mo_pos = collect(1:Nmat)
        end
    end

    subs = subcycles > 0 ? subcycles : max(1, multiphysics_every)
    return FDTDState(grid, fields,
                     Float64.(geo.inv_eps_x), Float64.(geo.inv_eps_y), Float64.(geo.inv_eps_z),
                     params, dtf, 0, source, cpml, geo, model,
                     dpoles, ade_x, ade_y, ade_z, mpoles, mo, mo_pos,
                     thermal, mag, lut, multiphysics_every, subs, brillouin_iters, absorption_model)
end

function step!(state::FDTDState)
    update_H!(state.fields, state.grid, state.params, state.dt; cpml=state.cpml)

    if state.source !== nothing
        pulse, component, index = state.source
        inv_arr = component == :Ex ? state.inv_eps_x : component == :Ey ? state.inv_eps_y : state.inv_eps_z
        inject_soft!(state.fields, component, index, gaussian_pulse_value(pulse, state.n * state.dt);
                     eps0=state.params.eps0, inv_eps=inv_arr)
    end

    update_E!(state.fields, state.grid, state.params, state.dt,
              state.inv_eps_x, state.inv_eps_y, state.inv_eps_z; cpml=state.cpml)

    if state.ade_x !== nothing
        region = state.region
        patch_E_dispersive!(state.fields.Ex, state.ade_x, region.material_cells, region.material_fill, region.material_inv_eps, state.diag_poles, state.dt)
        patch_E_dispersive!(state.fields.Ey, state.ade_y, region.material_cells, region.material_fill, region.material_inv_eps, state.diag_poles, state.dt)
        patch_E_dispersive!(state.fields.Ez, state.ade_z, region.material_cells, region.material_fill, region.material_inv_eps, state.diag_poles, state.dt)
    end

    if state.mo !== nothing && state.mag !== nothing
        gd = state.model.params
        region = state.region
        patch_E_mo_gyration!(state.fields.Ey, state.fields.Ez, state.mo,
                             region.material_cells, region.material_fill, state.mo_pos,
                             state.inv_eps_y, state.inv_eps_z,
                             state.mag.m_TM_x, state.mag.m_RE_x, state.mo_poles,
                             gd.Q_voigt_TM, gd.Q_voigt_RE, state.dt)
    end

    state.n += 1

    if state.thermal !== nothing && state.mag !== nothing && (state.n % state.multiphysics_every == 0)
        dt_mp = state.dt * state.multiphysics_every
        multiphysics_step!(state.thermal, state.mag, state.fields, state.region, state.model, state.lut, dt_mp;
                           subcycles=state.subcycles, absorption_model=state.absorption_model,
                           brillouin_iters=state.brillouin_iters, eps0=state.params.eps0)
    end
    return state
end

# Advance only the multiphysics (4TM + LLB) with no EM update — the relaxation /
# cool-down phase after the pump pulse, where the fields are zero.
function relax_step!(state::FDTDState, dt::Real; subcycles::Integer=state.subcycles)
    (state.thermal !== nothing && state.mag !== nothing) || return state
    # pabs_scale = 0: the post-pump cool-down evolves only T and m (no EM heating).
    multiphysics_step!(state.thermal, state.mag, state.fields, state.region, state.model, state.lut, dt;
                       subcycles=subcycles, absorption_model=state.absorption_model,
                       brillouin_iters=state.brillouin_iters, eps0=state.params.eps0, pabs_scale=0.0)
    return state
end

function run!(state::FDTDState, steps::Integer; callback=nothing)
    for _ in 1:steps
        step!(state)
        callback === nothing || callback(state)
    end
    return state
end
