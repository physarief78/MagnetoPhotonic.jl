Base.@kwdef struct GridConfig
    xlim::Tuple{Float64,Float64} = (0.0, 60e-6)
    ylim::Tuple{Float64,Float64} = (-3e-6, 3e-6)
    zlim::Tuple{Float64,Float64} = (-3e-6, 3e-6)
    mesh::Symbol = :uniform
    dx::Float64 = 50e-9
    dy::Float64 = 50e-9
    dz::Float64 = 50e-9
    fine_dx::Float64 = 8e-9
    fine_dy::Float64 = fine_dx
    fine_dz::Float64 = fine_dx
    fine_region::Any = nothing
    fine_buffer::Float64 = 0.25e-6
    grade_y::Bool = false
    grade_z::Bool = false
    stretch_ratio::Float64 = 1.08
    courant::Float64 = 0.45
    subpixel::Int = 1
    model_yee_stagger::Bool = false
end

Base.@kwdef struct SourceConfig
    kind::Symbol = :soft
    lambda0::Float64 = 800e-9
    amplitude::Float64 = 1.0
    tau::Float64 = 80e-15
    t0::Float64 = 4.0 * tau
    phase::Float64 = 0.0
    component::Symbol = :Ez
    index::Tuple{Int,Int,Int} = (2, 2, 2)
    axis::Symbol = :x
    position::Any = nothing
    neff_guess::Any = nothing   # nothing ⇒ auto-seed at the core index (fundamental mode)
    max_iter::Int = 20
end

Base.@kwdef struct DeviceConfig
    wg_width::Float64 = 0.40e-6
    wg_height::Float64 = 0.40e-6
    film_thickness::Float64 = 8e-9
    x_film_start::Float64 = 40e-6
    x_film_end::Float64 = 40.008e-6
    not_gate_length::Float64 = 60e-6
end

Base.@kwdef struct PMLConfig
    # Scalar for a uniform PML, or a per-axis (npx, npy, npz) tuple. The reference uses an
    # anisotropic (40, 12, 12) PML: a thick 40-cell absorber along the propagation axis (x)
    # where the −60 dB budget of a thin PML doesn't hold, and 12 cells transverse.
    cells::Union{Int,NTuple{3,Int}} = 10
    order::Float64 = 3.0
    reflection::Float64 = 1e-8
    kappa_max::Float64 = 5.0
    alpha_max::Float64 = 0.05
end

Base.@kwdef struct ModelConfig
    multiphysics_subcycle::Int = 1
    absorption_model::Symbol = :cycle_average   # :cycle_average | :ade_work
    brillouin_iters::Int = 2
    Hsw0::Float64 = 0.0
    pump_E_scale::Float64 = 1.0
end

Base.@kwdef struct ProbeConfig
    kind::Symbol = :mode
    lambda0::Float64 = 532e-9
    amplitude::Float64 = 1.0
    tau::Float64 = 80e-15
    t0::Float64 = 4.0 * tau
    phase::Float64 = 0.0
    component::Symbol = :Ez
    axis::Symbol = :x
    position::Any = nothing
    neff_guess::Any = nothing   # nothing ⇒ auto-seed at the core index (fundamental mode)
    max_iter::Int = 20
end

Base.@kwdef struct OutputConfig
    output_dir::String = "outputs"
    basename::String = "magnetophotonic_run"
    save_fields::Bool = true
    save_monitors::Bool = true
end

struct BackendConfig
    device::Symbol
    compute_precision::Any
    workgroupsize::Any
end

function BackendConfig(; device::Symbol=:cpu, compute_precision=:auto, workgroupsize=nothing)
    device in (:cpu, :cuda) || throw(ArgumentError("backend device must be :cpu or :cuda"))
    resolve_compute_type(compute_precision, backend(device))
    if workgroupsize !== nothing
        workgroupsize isa Integer || throw(ArgumentError("workgroupsize must be an integer or nothing"))
        workgroupsize > 0 || throw(ArgumentError("workgroupsize must be positive"))
    end
    return BackendConfig(device, compute_precision, workgroupsize)
end

Base.convert(::Type{BackendConfig}, x::Symbol) = BackendConfig(device=x)
Base.convert(::Type{BackendConfig}, x::AbstractString) = BackendConfig(device=Symbol(x))
Base.convert(::Type{BackendConfig}, x::BackendConfig) = x

Base.@kwdef struct SimConfig
    grid::GridConfig = GridConfig()
    source::SourceConfig = SourceConfig()
    probe::ProbeConfig = ProbeConfig()
    device::DeviceConfig = DeviceConfig()
    pml::PMLConfig = PMLConfig()
    model::ModelConfig = ModelConfig()
    output::OutputConfig = OutputConfig()
    steps::Int = 10
    backend::BackendConfig = BackendConfig()
    precision::DataType = EM_FIELD_STORAGE_TYPE
end

Base.@kwdef struct RenderConfig
    output_dir::String = "outputs"
    plane::Symbol = :xy
    every::Int = 10
    colormap::Symbol = :viridis
    field_component::Symbol = :Ez
end
