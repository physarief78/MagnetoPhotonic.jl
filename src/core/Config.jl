Base.@kwdef struct GridConfig
    xlim::Tuple{Float64,Float64} = (0.0, 60e-6)
    ylim::Tuple{Float64,Float64} = (-3e-6, 3e-6)
    zlim::Tuple{Float64,Float64} = (-3e-6, 3e-6)
    dx::Float64 = 50e-9
    dy::Float64 = 50e-9
    dz::Float64 = 50e-9
    fine_dx::Float64 = 8e-9
    stretch_ratio::Float64 = 1.08
    courant::Float64 = 0.45
end

Base.@kwdef struct SourceConfig
    lambda0::Float64 = 800e-9
    amplitude::Float64 = 1.0
    tau::Float64 = 80e-15
    t0::Float64 = 4.0 * tau
    phase::Float64 = 0.0
    component::Symbol = :Ez
    index::Tuple{Int,Int,Int} = (2, 2, 2)
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
    cells::Int = 10
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

Base.@kwdef struct SimConfig
    grid::GridConfig = GridConfig()
    source::SourceConfig = SourceConfig()
    device::DeviceConfig = DeviceConfig()
    pml::PMLConfig = PMLConfig()
    model::ModelConfig = ModelConfig()
    steps::Int = 10
    backend::Symbol = :cpu
    precision::DataType = EM_FIELD_STORAGE_TYPE
end

Base.@kwdef struct RenderConfig
    output_dir::String = "outputs"
    plane::Symbol = :xy
    every::Int = 10
    colormap::Symbol = :viridis
    field_component::Symbol = :Ez
end
