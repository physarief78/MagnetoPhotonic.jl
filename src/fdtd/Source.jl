Base.@kwdef struct GaussianPulse
    amplitude::Float64 = 1.0
    t0::Float64 = 160e-15
    tau::Float64 = 40e-15
    omega::Float64 = 2pi * 299792458.0 / 800e-9
    phase::Float64 = 0.0
end

gaussian_pulse_value(p::GaussianPulse, t::Real) =
    p.amplitude * exp(-((Float64(t) - p.t0) / p.tau)^2) * sin(p.omega * (Float64(t) - p.t0) + p.phase)

Base.@kwdef struct ContinuousSource
    amplitude::Float64 = 1.0
    omega::Float64 = 2pi * 299792458.0 / 800e-9
    phase::Float64 = 0.0
end

source_value(p::GaussianPulse, t::Real) = gaussian_pulse_value(p, t)
source_value(s::ContinuousSource, t::Real) = s.amplitude * sin(s.omega * Float64(t) + s.phase)

abstract type AbstractEMSource end

struct PointSource{P} <: AbstractEMSource
    pulse::P
    component::Symbol
    position::Any
end

struct PlaneSource{P} <: AbstractEMSource
    pulse::P
    component::Symbol
    axis::Symbol
    position::Any
end

PlaneSource(pulse, component::Symbol; axis::Symbol=:x, position=0.0) = PlaneSource{typeof(pulse)}(pulse, component, axis, position)

_nearest_index(axis::Axis1D, x::Real) = clamp(searchsortedfirst(axis.centers, Float64(x)), 1, length(axis.centers))
_source_index(axis::Axis1D, i::Integer) = clamp(Int(i), 1, length(axis.centers))
_source_index(axis::Axis1D, x::Real) = _nearest_index(axis, x)

function _axis_weights(axis::Axis1D, x::Real)
    xf = Float64(x)
    N = length(axis.centers)
    if xf <= axis.centers[1]
        return ((1, 1.0),)
    elseif xf >= axis.centers[end]
        return ((N, 1.0),)
    end
    hi = searchsortedfirst(axis.centers, xf)
    lo = hi - 1
    w_hi = (xf - axis.centers[lo]) / (axis.centers[hi] - axis.centers[lo])
    return ((lo, 1.0 - w_hi), (hi, w_hi))
end

function inject_soft!(fields::FieldState, component::Symbol, index::NTuple{3,Int}, value::Real;
                      eps0::Real=1.0, inv_eps=nothing)
    arr = getfield(fields, component)
    arr[index...] += value
    ie = inv_eps === nothing ? 1.0 : Float64(inv_eps[index...])
    dval = Float64(value) * Float64(eps0) / ie
    if component == :Ex
        fields.Dx[index...] += dval
    elseif component == :Ey
        fields.Dy[index...] += dval
    elseif component == :Ez
        fields.Dz[index...] += dval
    end
    return fields
end

function inject!(fields::Fields1D, grid::Grid1D, src::PointSource, t::Real, p::FDTDParams, inv_eps)
    src.component == :Ez || throw(ArgumentError("1-D source component must be :Ez"))
    val = source_value(src.pulse, t)
    for (i, w) in _axis_weights(grid.x, src.position)
        dv = w * val
        fields.Ez[i] += dv
        fields.Dz[i] += dv * p.eps0 / Float64(inv_eps[i])
    end
    return fields
end

function inject!(fields::Fields1D, grid::Grid1D, src::PlaneSource, t::Real, p::FDTDParams, inv_eps)
    return inject!(fields, grid, PointSource(src.pulse, src.component, src.position), t, p, inv_eps)
end

function inject!(fields::Fields2D, grid::Grid2D, src::PointSource, t::Real, p::FDTDParams, inv_eps)
    pos = src.position
    val = source_value(src.pulse, t)
    wx = _axis_weights(grid.x, pos isa Tuple ? pos[1] : pos)
    wy = _axis_weights(grid.y, pos isa Tuple ? pos[2] : 0.0)
    for (i, wi) in wx, (j, wj) in wy
        _inject_2d_cell!(fields, src.component, i, j, wi * wj * val, p, inv_eps)
    end
    return fields
end

function inject!(fields::Fields2D, grid::Grid2D, src::PlaneSource, t::Real, p::FDTDParams, inv_eps)
    val = source_value(src.pulse, t)
    if src.axis === :x
        for (i, wi) in _axis_weights(grid.x, src.position), j in eachindex(grid.y.centers)
            _inject_2d_cell!(fields, src.component, i, j, wi * val, p, inv_eps)
        end
    elseif src.axis === :y
        for i in eachindex(grid.x.centers), (j, wj) in _axis_weights(grid.y, src.position)
            _inject_2d_cell!(fields, src.component, i, j, wj * val, p, inv_eps)
        end
    else
        throw(ArgumentError("2-D PlaneSource axis must be :x or :y"))
    end
    return fields
end

function _inject_2d_cell!(fields::Fields2D, component::Symbol, i::Int, j::Int, val::Real, p::FDTDParams, inv_eps)
    if component == :Ez
        fields.Ez[i, j] += val
        fields.Dz[i, j] += Float64(val) * p.eps0 / Float64(inv_eps[i, j])
    elseif component == :Ex
        fields.Ex[i, j] += val
        fields.Dx[i, j] += Float64(val) * p.eps0 / Float64(inv_eps[i, j])
    elseif component == :Ey
        fields.Ey[i, j] += val
        fields.Dy[i, j] += Float64(val) * p.eps0 / Float64(inv_eps[i, j])
    elseif component == :Hz
        fields.Hz[i, j] += val
    elseif component == :Hx
        fields.Hx[i, j] += val
    elseif component == :Hy
        fields.Hy[i, j] += val
    else
        throw(ArgumentError("unsupported 2-D source component $component"))
    end
    return fields
end

function inject!(fields::FieldState, grid::Grid3D, src::PointSource, t::Real, p::FDTDParams, inv_eps)
    pos = src.position
    val = source_value(src.pulse, t)
    if pos isa NTuple{3,Int}
        _inject_3d_cell!(fields, src.component, pos..., val, p, inv_eps)
    else
        for (i, wi) in _axis_weights(grid.x, pos[1]),
            (j, wj) in _axis_weights(grid.y, pos[2]),
            (k, wk) in _axis_weights(grid.z, pos[3])
            _inject_3d_cell!(fields, src.component, i, j, k, wi * wj * wk * val, p, inv_eps)
        end
    end
    return fields
end

function inject!(fields::FieldState, grid::Grid3D, src::PlaneSource, t::Real, p::FDTDParams, inv_eps)
    val = source_value(src.pulse, t)
    if src.axis === :x
        for (i, wi) in _axis_weights(grid.x, src.position), j in eachindex(grid.y.centers), k in eachindex(grid.z.centers)
            _inject_3d_cell!(fields, src.component, i, j, k, wi * val, p, inv_eps)
        end
    elseif src.axis === :y
        for i in eachindex(grid.x.centers), (j, wj) in _axis_weights(grid.y, src.position), k in eachindex(grid.z.centers)
            _inject_3d_cell!(fields, src.component, i, j, k, wj * val, p, inv_eps)
        end
    elseif src.axis === :z
        for i in eachindex(grid.x.centers), j in eachindex(grid.y.centers), (k, wk) in _axis_weights(grid.z, src.position)
            _inject_3d_cell!(fields, src.component, i, j, k, wk * val, p, inv_eps)
        end
    else
        throw(ArgumentError("3-D PlaneSource axis must be :x, :y or :z"))
    end
    return fields
end

function _inject_3d_cell!(fields::FieldState, component::Symbol, i::Int, j::Int, k::Int, val::Real, p::FDTDParams, inv_eps)
    if component == :Ex
        fields.Ex[i, j, k] += val
        fields.Dx[i, j, k] += Float64(val) * p.eps0 / Float64(inv_eps.inv_eps_x[i, j, k])
    elseif component == :Ey
        fields.Ey[i, j, k] += val
        fields.Dy[i, j, k] += Float64(val) * p.eps0 / Float64(inv_eps.inv_eps_y[i, j, k])
    elseif component == :Ez
        fields.Ez[i, j, k] += val
        fields.Dz[i, j, k] += Float64(val) * p.eps0 / Float64(inv_eps.inv_eps_z[i, j, k])
    elseif component == :Hx
        fields.Hx[i, j, k] += val
    elseif component == :Hy
        fields.Hy[i, j, k] += val
    elseif component == :Hz
        fields.Hz[i, j, k] += val
    else
        throw(ArgumentError("unsupported 3-D source component $component"))
    end
    return fields
end
