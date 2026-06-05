mutable struct Simulation{D,M,G,F,IE,B,CP,A}
    dimension::Int
    mode::Symbol
    grid::G
    fields::F
    inv_eps::IE
    params::FDTDParams
    dt::Float64
    boundary::B
    sources::Vector{AbstractEMSource}
    n::Int
    t::Float64
    cpml::CP
    ade::A
    poles::Vector{DLPole{Float64}}
    active_indices::Vector{Int}
    active_fill::Vector{Float64}
    active_inv_eps::Vector{Float64}
end

function _cell_limits(cell, dimension::Integer)
    if dimension == 1
        L = cell isa Tuple ? Float64(cell[1]) : Float64(cell)
        return ((0.0, L),)
    elseif dimension == 2
        length(cell) == 2 || throw(ArgumentError("2-D cell must be (Lx, Ly) or ((x0,x1),(y0,y1))"))
        return (cell[1] isa Tuple ? cell[1] : (0.0, Float64(cell[1])),
                cell[2] isa Tuple ? cell[2] : (0.0, Float64(cell[2])))
    elseif dimension == 3
        length(cell) == 3 || throw(ArgumentError("3-D cell must have three extents"))
        return (cell[1] isa Tuple ? cell[1] : (0.0, Float64(cell[1])),
                cell[2] isa Tuple ? cell[2] : (0.0, Float64(cell[2])),
                cell[3] isa Tuple ? cell[3] : (0.0, Float64(cell[3])))
    else
        throw(ArgumentError("dimension must be 1, 2, or 3"))
    end
end

function _grid_from_cell(cell, dimension::Integer, dx::Real)
    lim = _cell_limits(cell, dimension)
    dimension == 1 && return uniform_grid(lim[1], dx)
    dimension == 2 && return uniform_grid(lim[1], lim[2], dx)
    return uniform_grid(lim[1], lim[2], lim[3], dx)
end

function _geo_for_grid(scene::Scene, grid::Grid1D; kwargs...)
    return rasterize_1d(scene, grid)
end

function _geo_for_grid(scene::Scene, grid::Grid2D; kwargs...)
    return rasterize_2d(scene, grid; kwargs...)
end

function _geo_for_grid(scene::Scene, grid::Grid3D; kwargs...)
    g = rasterize(scene, grid; subpixel=get(kwargs, :subpixel, 1))
    return (; g..., inv_eps=g.inv_eps_z,
            active_indices=hasproperty(g, :pole_active_cells) ? g.pole_active_cells : Int[],
            active_fill=hasproperty(g, :pole_active_fill) ? g.pole_active_fill : Float64[],
            active_inv_eps=hasproperty(g, :pole_active_inv_eps) ? g.pole_active_inv_eps : Float64[],
            poles=hasproperty(g, :pole_poles) ? g.pole_poles : Any[])
end

function _build_sim_cpml(boundary::AbstractBoundary, grid, dt::Real, p::FDTDParams)
    boundary isa PML || return nothing
    if grid isa Grid1D
        return build_cpml(grid, _pml_cells(boundary, grid.x), dt, p)
    elseif grid isa Grid2D
        return build_cpml(grid, (_pml_cells(boundary, grid.x), _pml_cells(boundary, grid.y)), dt, p)
    elseif grid isa Grid3D
        return build_cpml(grid, (_pml_cells(boundary, grid.x), _pml_cells(boundary, grid.y), _pml_cells(boundary, grid.z)), dt, p)
    end
    return nothing
end

function _allocate_sim_ade(dimension::Integer, mode::Symbol, active_indices, poles)
    (isempty(poles) || isempty(active_indices)) && return nothing
    n = length(active_indices)
    if dimension == 1
        return (z=allocate_ade_state(n, poles),)
    elseif dimension == 2
        mode === :TM && return (z=allocate_ade_state(n, poles),)
        return (x=allocate_ade_state(n, poles), y=allocate_ade_state(n, poles))
    elseif dimension == 3
        return (x=allocate_ade_state(n, poles), y=allocate_ade_state(n, poles), z=allocate_ade_state(n, poles))
    end
    return nothing
end

function Simulation(; cell=(1.0,), resolution=nothing, dx=nothing, geometry=Scene(),
                    sources=AbstractEMSource[], boundary::AbstractBoundary=PEC(),
                    dimension::Integer=length(cell), mode::Symbol=:TM, courant::Real=0.5,
                    params::FDTDParams=FDTDParams(), subpixel::Integer=1, T::Type=Float64)
    dxf = dx === nothing ? (resolution === nothing ? 1 / 20 : 1 / Float64(resolution)) : Float64(dx)
    grid = _grid_from_cell(cell, dimension, dxf)
    geo = _geo_for_grid(geometry, grid; subpixel=subpixel)
    dt = cfl_dt(grid, params; courant=courant)
    fields = dimension == 1 ? allocate_fields(grid; T=T) :
             dimension == 2 ? allocate_fields(grid; mode=mode, T=T) :
             allocate_fields(grid; T=T)
    inv_eps = dimension == 3 ? (inv_eps_x=geo.inv_eps_x, inv_eps_y=geo.inv_eps_y, inv_eps_z=geo.inv_eps_z) : geo.inv_eps
    poles = hasproperty(geo, :poles) ? DLPole{Float64}[geo.poles...] : DLPole{Float64}[]
    active_indices = hasproperty(geo, :active_indices) ? collect(Int, geo.active_indices) : Int[]
    active_fill = hasproperty(geo, :active_fill) ? collect(Float64, geo.active_fill) : Float64[]
    active_inv_eps = hasproperty(geo, :active_inv_eps) ? collect(Float64, geo.active_inv_eps) : Float64[]
    cpml = _build_sim_cpml(boundary, grid, dt, params)
    ade = _allocate_sim_ade(dimension, mode, active_indices, poles)
    srcs = AbstractEMSource[sources...]
    return Simulation{Int(dimension), mode, typeof(grid), typeof(fields), typeof(inv_eps),
                      typeof(boundary), typeof(cpml), typeof(ade)}(
        Int(dimension), mode, grid, fields, inv_eps, params, dt, boundary, srcs,
        0, 0.0, cpml, ade, poles, active_indices, active_fill, active_inv_eps)
end
