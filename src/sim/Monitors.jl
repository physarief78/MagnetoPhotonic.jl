abstract type AbstractMonitor end

mutable struct PointMonitor <: AbstractMonitor
    component::Symbol
    position::Any
    t::Vector{Float64}
    values::Vector{Float64}
end

PointMonitor(component::Symbol, position) = PointMonitor(component, position, Float64[], Float64[])

mutable struct FieldMonitor <: AbstractMonitor
    component::Symbol
    every::Int
    frames::Vector{Any}
end

FieldMonitor(component::Symbol=:Ez; every::Integer=1) = FieldMonitor(component, Int(every), Any[])

mutable struct FluxMonitor <: AbstractMonitor
    axis::Symbol
    position::Any
    t::Vector{Float64}
    flux::Vector{Float64}
end

FluxMonitor(axis::Symbol, position) = FluxMonitor(axis, position, Float64[], Float64[])

mutable struct DFTMonitor <: AbstractMonitor
    component::Symbol
    position::Any
    t::Vector{Float64}
    values::Vector{Float64}
end

DFTMonitor(component::Symbol, position) = DFTMonitor(component, position, Float64[], Float64[])

function _field_array(fields, component::Symbol)
    return getfield(fields, component)
end

_cell_width(axis::Axis1D, i::Integer) = axis.edges[i + 1] - axis.edges[i]

_axis_sample_weights(axis::Axis1D, position::Integer) = ((_source_index(axis, position), 1.0),)
_axis_sample_weights(axis::Axis1D, position) = _axis_weights(axis, position)

function _sample_array(arr, grid::Grid1D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position)
        v += wi * Float64(arr[i])
    end
    return v
end

function _sample_array(arr, grid::Grid2D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position[1]),
        (j, wj) in _axis_sample_weights(grid.y, position[2])
        v += wi * wj * Float64(arr[i, j])
    end
    return v
end

function _sample_array(arr, grid::Grid3D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position[1]),
        (j, wj) in _axis_sample_weights(grid.y, position[2]),
        (k, wk) in _axis_sample_weights(grid.z, position[3])
        v += wi * wj * wk * Float64(arr[i, j, k])
    end
    return v
end

function _sample_field(fields, grid, component::Symbol, position)
    return _sample_array(_field_array(fields, component), grid, position)
end

function _plane_coordinate(position, axis::Symbol)
    if position isa Tuple
        axis === :x && return position[1]
        axis === :y && return position[2]
        axis === :z && return position[3]
    end
    return position
end

function _plane_index(grid::Grid1D, axis::Symbol, position)
    axis === :x || throw(ArgumentError("1-D flux axis must be :x"))
    return _source_index(grid.x, _plane_coordinate(position, axis))
end

function _plane_index(grid::Grid2D, axis::Symbol, position)
    if axis === :x
        return _source_index(grid.x, _plane_coordinate(position, axis))
    elseif axis === :y
        return _source_index(grid.y, _plane_coordinate(position, axis))
    end
    throw(ArgumentError("2-D flux axis must be :x or :y"))
end

function _plane_index(grid::Grid3D, axis::Symbol, position)
    if axis === :x
        return _source_index(grid.x, _plane_coordinate(position, axis))
    elseif axis === :y
        return _source_index(grid.y, _plane_coordinate(position, axis))
    elseif axis === :z
        return _source_index(grid.z, _plane_coordinate(position, axis))
    end
    throw(ArgumentError("3-D flux axis must be :x, :y or :z"))
end

function record!(m::PointMonitor, sim)
    push!(m.t, sim.t)
    push!(m.values, _sample_field(sim.fields, sim.grid, m.component, m.position))
    return m
end

function record!(m::DFTMonitor, sim)
    push!(m.t, sim.t)
    push!(m.values, _sample_field(sim.fields, sim.grid, m.component, m.position))
    return m
end

function record!(m::FieldMonitor, sim)
    sim.n % m.every == 0 || return m
    push!(m.frames, copy(_field_array(sim.fields, m.component)))
    return m
end

function record!(m::FluxMonitor, sim)
    push!(m.t, sim.t)
    if sim.dimension == 1
        i = _plane_index(sim.grid, m.axis, m.position)
        push!(m.flux, -Float64(sim.fields.Ez[i]) * Float64(sim.fields.Hy[i]))
    elseif sim.dimension == 2
        idx = _plane_index(sim.grid, m.axis, m.position)
        flux = 0.0
        if m.axis === :x
            for j in eachindex(sim.grid.y.centers)
                dy = _cell_width(sim.grid.y, j)
                if sim.mode === :TM
                    flux += -Float64(sim.fields.Ez[idx, j]) * Float64(sim.fields.Hy[idx, j]) * dy
                else
                    flux += Float64(sim.fields.Ey[idx, j]) * Float64(sim.fields.Hz[idx, j]) * dy
                end
            end
        elseif m.axis === :y
            for i in eachindex(sim.grid.x.centers)
                dx = _cell_width(sim.grid.x, i)
                if sim.mode === :TM
                    flux += Float64(sim.fields.Ez[i, idx]) * Float64(sim.fields.Hx[i, idx]) * dx
                else
                    flux += -Float64(sim.fields.Ex[i, idx]) * Float64(sim.fields.Hz[i, idx]) * dx
                end
            end
        else
            throw(ArgumentError("2-D flux axis must be :x or :y"))
        end
        push!(m.flux, flux)
    else
        idx = _plane_index(sim.grid, m.axis, m.position)
        flux = 0.0
        if m.axis === :x
            for j in eachindex(sim.grid.y.centers), k in eachindex(sim.grid.z.centers)
                area = _cell_width(sim.grid.y, j) * _cell_width(sim.grid.z, k)
                flux += (Float64(sim.fields.Ey[idx, j, k]) * Float64(sim.fields.Hz[idx, j, k]) -
                         Float64(sim.fields.Ez[idx, j, k]) * Float64(sim.fields.Hy[idx, j, k])) * area
            end
        elseif m.axis === :y
            for i in eachindex(sim.grid.x.centers), k in eachindex(sim.grid.z.centers)
                area = _cell_width(sim.grid.x, i) * _cell_width(sim.grid.z, k)
                flux += (Float64(sim.fields.Ez[i, idx, k]) * Float64(sim.fields.Hx[i, idx, k]) -
                         Float64(sim.fields.Ex[i, idx, k]) * Float64(sim.fields.Hz[i, idx, k])) * area
            end
        elseif m.axis === :z
            for i in eachindex(sim.grid.x.centers), j in eachindex(sim.grid.y.centers)
                area = _cell_width(sim.grid.x, i) * _cell_width(sim.grid.y, j)
                flux += (Float64(sim.fields.Ex[i, j, idx]) * Float64(sim.fields.Hy[i, j, idx]) -
                         Float64(sim.fields.Ey[i, j, idx]) * Float64(sim.fields.Hx[i, j, idx])) * area
            end
        else
            throw(ArgumentError("3-D flux axis must be :x, :y or :z"))
        end
        push!(m.flux, flux)
    end
    return m
end

monitor_data(m::PointMonitor) = (t=m.t, values=m.values)
monitor_data(m::DFTMonitor) = (t=m.t, values=m.values, spectrum=compute_spectrum(m.t, m.values))
monitor_data(m::FieldMonitor) = m.frames
monitor_data(m::FluxMonitor) = (t=m.t, flux=m.flux, total=isempty(m.t) ? 0.0 : sum(m.flux) * (length(m.t) > 1 ? m.t[2] - m.t[1] : 0.0))
