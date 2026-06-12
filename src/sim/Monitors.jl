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

# Transmission/Reflection. When `mode` is supplied (a transverse profile array over
# the recording plane) the monitor records the mode-overlap amplitude
# a(t) = ∫ E_component·mode dA at the plane (the production "mode_w" projection);
# `monitor_data` then reports the modal energy ∫a²dt and its ratio to `incident`.
# When `mode === nothing` it falls back to the raw plane-integrated Poynting flux.
mutable struct Transmission <: AbstractMonitor
    axis::Symbol
    position::Any
    component::Symbol
    mode::Any
    incident::Float64
    t::Vector{Float64}
    values::Vector{Float64}
end

mutable struct Reflection <: AbstractMonitor
    axis::Symbol
    position::Any
    component::Symbol
    mode::Any
    incident::Float64
    t::Vector{Float64}
    values::Vector{Float64}
end

mutable struct Absorption <: AbstractMonitor
    t::Vector{Float64}
    values::Vector{Float64}
end

mutable struct AbsorbedPower <: AbstractMonitor
    t::Vector{Float64}
    values::Vector{Float64}
end

# Polarimetry. Records the (mode-projected, if `mode` given, else plane-mean)
# Ey/Ez amplitude traces; `monitor_data` forms the complex phasors via a single-
# frequency DFT at `omega` and returns the Jones rotation/ellipticity. Using the
# phasor (not an instantaneous real field) is what makes the ellipticity meaningful.
mutable struct Polarimetry <: AbstractMonitor
    axis::Symbol
    position::Any
    mode::Any
    omega::Float64
    t::Vector{Float64}
    ey::Vector{Float64}
    ez::Vector{Float64}
end

mutable struct SwitchedFraction <: AbstractMonitor
    every::Int
    t::Vector{Float64}
    values::Vector{Float64}
end

# Single-cell trajectory probe (the reference's hot-cell diagnostic): records
# Te, m_TM_x, m_RE_x of one active cell (typically argmax U_abs) every `every`
# steps, alongside the monitor time axis.
mutable struct HotCellTrace <: AbstractMonitor
    cell::Int
    every::Int
    t::Vector{Float64}
    Te_K::Vector{Float64}
    mx_TM::Vector{Float64}
    mx_RE::Vector{Float64}
end

# Generic strided callback: calls `f(sim)` every `every` steps with the monitor
# view (sim.state, sim.n, sim.t, …). Used e.g. to stream movie frames to HDF5
# during the pump phase without baking I/O into the solver.
mutable struct CallbackMonitor <: AbstractMonitor
    f::Any
    every::Int
end

mutable struct FilmAverage <: AbstractMonitor
    every::Int
    data::Dict{Symbol,Vector{Float64}}
end

mutable struct Progress <: AbstractMonitor
    every::Int
    steps::Vector{Int}
    t::Vector{Float64}
end

mutable struct NaNGuard <: AbstractMonitor
    every::Int
    milestones::Vector{Int}
end

Transmission(; at, axis::Symbol=:x, component::Symbol=:Ez, mode=nothing, incident::Real=1.0) =
    Transmission(axis, at, component, mode, Float64(incident), Float64[], Float64[])
Reflection(; at, axis::Symbol=:x, component::Symbol=:Ez, mode=nothing, incident::Real=1.0) =
    Reflection(axis, at, component, mode, Float64(incident), Float64[], Float64[])
Absorption() = Absorption(Float64[], Float64[])
AbsorbedPower() = AbsorbedPower(Float64[], Float64[])
Polarimetry(; at=nothing, axis::Symbol=:x, mode=nothing, omega::Real=2pi * 299792458.0 / 532e-9) =
    Polarimetry(axis, at, mode, Float64(omega), Float64[], Float64[], Float64[])
SwitchedFraction(; every::Integer=1) = SwitchedFraction(Int(every), Float64[], Float64[])
HotCellTrace(cell::Integer; every::Integer=1) =
    HotCellTrace(Int(cell), Int(every), Float64[], Float64[], Float64[], Float64[])
CallbackMonitor(f; every::Integer=1) = CallbackMonitor(f, Int(every))
const _FILM_AVERAGE_KEYS = (
    :mag_time,
    :mx_TM, :mx_RE,
    :m_TM_x_reduced, :m_TM_y_reduced, :m_TM_z_reduced,
    :m_RE_x_reduced, :m_RE_y_reduced, :m_RE_z_reduced,
    :M_TM_x_Apm, :M_TM_y_Apm, :M_TM_z_Apm,
    :M_RE_x_Apm, :M_RE_y_Apm, :M_RE_z_Apm,
    :M_net_x_Apm, :M_net_y_Apm, :M_net_z_Apm,
    :Te_avg, :Tl_avg, :Ts_avg, :Te_K, :Tl_K, :Ts_TM_K, :Ts_RE_K,
    :mag_TM_norm_avg, :mag_RE_norm_avg,
)
FilmAverage(; every::Integer=1) =
    FilmAverage(Int(every), Dict(k => Float64[] for k in _FILM_AVERAGE_KEYS))
Progress(every::Integer=100) = Progress(Int(every), Int[], Float64[])
NaNGuard(every::Integer=1) = NaNGuard(Int(every), Int[])
NaNGuard(milestones::AbstractVector) = NaNGuard(0, sort!(Int.(collect(milestones))))

_monitor_due(::AbstractMonitor, n::Integer) = true
_monitor_due(m::FieldMonitor, n::Integer) = n % m.every == 0
_monitor_due(m::FilmAverage, n::Integer) = n % m.every == 0
_monitor_due(m::Progress, n::Integer) = n % m.every == 0
_monitor_due(m::SwitchedFraction, n::Integer) = m.every <= 1 || n % m.every == 0
_monitor_due(m::HotCellTrace, n::Integer) = m.every <= 1 || n % m.every == 0
_monitor_due(m::CallbackMonitor, n::Integer) = m.every > 0 && n % m.every == 0
_monitor_due(m::NaNGuard, n::Integer) =
    isempty(m.milestones) ? (m.every > 0 && n % m.every == 0) : (Int(n) in m.milestones)

_monitor_needs_sync(m::AbstractMonitor, n::Integer) = _monitor_due(m, n)
_monitor_needs_sync(m::Progress, n::Integer) = false

function _field_array(fields, component::Symbol)
    return getfield(fields, component)
end

_cell_width(axis::Axis1D, i::Integer) = axis.edges[i + 1] - axis.edges[i]

_axis_sample_weights(axis::Axis1D, position::Integer) = ((_source_index(axis, position), 1.0),)
_axis_sample_weights(axis::Axis1D, position) = _axis_weights(axis, position)

function _sample_array(arr, grid::Grid1D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position)
        v += wi * Float64(scalar_to_host(arr, i))
    end
    return v
end

function _sample_array(arr, grid::Grid2D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position[1]),
        (j, wj) in _axis_sample_weights(grid.y, position[2])
        v += wi * wj * Float64(scalar_to_host(arr, i, j))
    end
    return v
end

function _sample_array(arr, grid::Grid3D, position)
    v = 0.0
    for (i, wi) in _axis_sample_weights(grid.x, position[1]),
        (j, wj) in _axis_sample_weights(grid.y, position[2]),
        (k, wk) in _axis_sample_weights(grid.z, position[3])
        v += wi * wj * wk * Float64(scalar_to_host(arr, i, j, k))
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
    push!(m.values, _sample_field(_view_fields(sim), sim.grid, m.component, m.position))
    return m
end

function record!(m::DFTMonitor, sim)
    push!(m.t, sim.t)
    push!(m.values, _sample_field(_view_fields(sim), sim.grid, m.component, m.position))
    return m
end

function record!(m::FieldMonitor, sim)
    sim.n % m.every == 0 || return m
    push!(m.frames, copy(to_host(_field_array(_view_fields(sim), m.component))))
    return m
end

function record!(m::FluxMonitor, sim)
    push!(m.t, sim.t)
    fields = _view_fields(sim)
    if sim.dimension == 1
        i = _plane_index(sim.grid, m.axis, m.position)
        push!(m.flux, -Float64(scalar_to_host(fields.Ez, i)) * Float64(scalar_to_host(fields.Hy, i)))
    elseif sim.dimension == 2
        idx = _plane_index(sim.grid, m.axis, m.position)
        flux = 0.0
        if m.axis === :x
            Ez = plane_to_host(fields.Ez, :x, idx)
            Hy = plane_to_host(fields.Hy, :x, idx)
            Ey = sim.mode === :TM ? nothing : plane_to_host(fields.Ey, :x, idx)
            Hz = sim.mode === :TM ? nothing : plane_to_host(fields.Hz, :x, idx)
            for j in eachindex(sim.grid.y.centers)
                dy = _cell_width(sim.grid.y, j)
                if sim.mode === :TM
                    flux += -Float64(Ez[j]) * Float64(Hy[j]) * dy
                else
                    flux += Float64(Ey[j]) * Float64(Hz[j]) * dy
                end
            end
        elseif m.axis === :y
            Ez = plane_to_host(fields.Ez, :y, idx)
            Hx = plane_to_host(fields.Hx, :y, idx)
            Ex = sim.mode === :TM ? nothing : plane_to_host(fields.Ex, :y, idx)
            Hz = sim.mode === :TM ? nothing : plane_to_host(fields.Hz, :y, idx)
            for i in eachindex(sim.grid.x.centers)
                dx = _cell_width(sim.grid.x, i)
                if sim.mode === :TM
                    flux += Float64(Ez[i]) * Float64(Hx[i]) * dx
                else
                    flux += -Float64(Ex[i]) * Float64(Hz[i]) * dx
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
            Ey = plane_to_host(fields.Ey, :x, idx)
            Hz = plane_to_host(fields.Hz, :x, idx)
            Ez = plane_to_host(fields.Ez, :x, idx)
            Hy = plane_to_host(fields.Hy, :x, idx)
            for j in eachindex(sim.grid.y.centers), k in eachindex(sim.grid.z.centers)
                area = _cell_width(sim.grid.y, j) * _cell_width(sim.grid.z, k)
                flux += (Float64(Ey[j, k]) * Float64(Hz[j, k]) -
                         Float64(Ez[j, k]) * Float64(Hy[j, k])) * area
            end
        elseif m.axis === :y
            Ez = plane_to_host(fields.Ez, :y, idx)
            Hx = plane_to_host(fields.Hx, :y, idx)
            Ex = plane_to_host(fields.Ex, :y, idx)
            Hz = plane_to_host(fields.Hz, :y, idx)
            for i in eachindex(sim.grid.x.centers), k in eachindex(sim.grid.z.centers)
                area = _cell_width(sim.grid.x, i) * _cell_width(sim.grid.z, k)
                flux += (Float64(Ez[i, k]) * Float64(Hx[i, k]) -
                         Float64(Ex[i, k]) * Float64(Hz[i, k])) * area
            end
        elseif m.axis === :z
            Ex = plane_to_host(fields.Ex, :z, idx)
            Hy = plane_to_host(fields.Hy, :z, idx)
            Ey = plane_to_host(fields.Ey, :z, idx)
            Hx = plane_to_host(fields.Hx, :z, idx)
            for i in eachindex(sim.grid.x.centers), j in eachindex(sim.grid.y.centers)
                area = _cell_width(sim.grid.x, i) * _cell_width(sim.grid.y, j)
                flux += (Float64(Ex[i, j]) * Float64(Hy[i, j]) -
                         Float64(Ey[i, j]) * Float64(Hx[i, j])) * area
            end
        else
            throw(ArgumentError("3-D flux axis must be :x, :y or :z"))
        end
        push!(m.flux, flux)
    end
    return m
end

function _instant_flux(sim, axis::Symbol, position)
    tmp = FluxMonitor(axis, position)
    record!(tmp, sim)
    return isempty(tmp.flux) ? 0.0 : tmp.flux[end]
end

# Mode-overlap amplitude a = ∫ E_component · mode dA over the recording plane.
function _mode_overlap(fields, grid::Grid3D, axis::Symbol, position, mode, component::Symbol)
    idx = _plane_index(grid, axis, position)
    arr = plane_to_host(_field_array(fields, component), axis, idx)
    a = 0.0
    if axis === :x
        @inbounds for j in eachindex(grid.y.centers), k in eachindex(grid.z.centers)
            a += Float64(arr[j, k]) * Float64(mode[j, k]) * _cell_width(grid.y, j) * _cell_width(grid.z, k)
        end
    elseif axis === :y
        @inbounds for i in eachindex(grid.x.centers), k in eachindex(grid.z.centers)
            a += Float64(arr[i, k]) * Float64(mode[i, k]) * _cell_width(grid.x, i) * _cell_width(grid.z, k)
        end
    else
        @inbounds for i in eachindex(grid.x.centers), j in eachindex(grid.y.centers)
            a += Float64(arr[i, j]) * Float64(mode[i, j]) * _cell_width(grid.x, i) * _cell_width(grid.y, j)
        end
    end
    return a
end

function _mode_overlap(fields, grid::Grid2D, axis::Symbol, position, mode, component::Symbol)
    idx = _plane_index(grid, axis, position)
    arr = plane_to_host(_field_array(fields, component), axis, idx)
    a = 0.0
    if axis === :x
        @inbounds for j in eachindex(grid.y.centers)
            a += Float64(arr[j]) * Float64(mode[j]) * _cell_width(grid.y, j)
        end
    else
        @inbounds for i in eachindex(grid.x.centers)
            a += Float64(arr[i]) * Float64(mode[i]) * _cell_width(grid.x, i)
        end
    end
    return a
end

function record!(m::Transmission, sim)
    push!(m.t, sim.t)
    fields = _view_fields(sim)
    v = m.mode === nothing ? _instant_flux(sim, m.axis, m.position) :
        _mode_overlap(fields, sim.grid, m.axis, m.position, m.mode, m.component)
    push!(m.values, v)
    return m
end

function record!(m::Reflection, sim)
    push!(m.t, sim.t)
    fields = _view_fields(sim)
    v = m.mode === nothing ? -_instant_flux(sim, m.axis, m.position) :
        _mode_overlap(fields, sim.grid, m.axis, m.position, m.mode, m.component)
    push!(m.values, v)
    return m
end

function _state_from_monitor_view(sim)
    hasproperty(sim, :state) ? sim.state : nothing
end

function _absorbed_power_value(sim)
    st = _state_from_monitor_view(sim)
    if st !== nothing && st.model isa MagnetoOpticModel && hasproperty(st.region, :material_cells)
        gd = st.model.params
        cells = to_host(st.region.material_cells)
        fillv = to_host(st.region.material_fill)
        fields = to_host(st.fields)
        total = 0.0
        # Match the heating model actually used by the run: :ade_work uses the local
        # E·J_ADE work, :cycle_average uses ½ω ε0 ε'' |E|².
        if st.absorption_model === :ade_work && st.ade_x !== nothing && st.ade_y !== nothing && st.ade_z !== nothing
            Jx = to_host(st.ade_x.J); Jy = to_host(st.ade_y.J); Jz = to_host(st.ade_z.J)
            px = Int.(to_host(st.region.pos_x)); py = Int.(to_host(st.region.pos_y)); pz = Int.(to_host(st.region.pos_z))
            npoles = size(Jx, 2)
            @inbounds for n in eachindex(cells)
                li = cells[n]
                jx = 0.0; jy = 0.0; jz = 0.0
                for p in 1:npoles
                    px[n] > 0 && (jx += Float64(Jx[px[n], p]))
                    py[n] > 0 && (jy += Float64(Jy[py[n], p]))
                    pz[n] > 0 && (jz += Float64(Jz[pz[n], p]))
                end
                ej = Float64(fields.Ex[li]) * jx + Float64(fields.Ey[li]) * jy + Float64(fields.Ez[li]) * jz
                total += max(ej, 0.0) * Float64(fillv[n])
            end
        else
            pabs_factor = 0.5 * gd.omega_pump * st.params.eps0 * gd.eps_imag_pump
            @inbounds for n in eachindex(cells)
                li = cells[n]
                e2 = Float64(fields.Ex[li])^2 + Float64(fields.Ey[li])^2 + Float64(fields.Ez[li])^2
                total += pabs_factor * e2 * Float64(fillv[n])
            end
        end
        return total
    end
    return field_energy(to_host(_view_fields(sim)), sim.grid)
end

function record!(m::AbsorbedPower, sim)
    push!(m.t, sim.t)
    push!(m.values, _absorbed_power_value(sim))
    return m
end

function record!(m::Absorption, sim)
    push!(m.t, sim.t)
    push!(m.values, _absorbed_power_value(sim))
    return m
end

function _mean_plane_components(fields, grid::Grid3D, position)
    idx = position === nothing ? length(grid.x.centers) : _plane_index(grid, :x, position)
    Ey = plane_to_host(fields.Ey, :x, idx)
    Ez = plane_to_host(fields.Ez, :x, idx)
    ey = 0.0
    ez = 0.0
    n = 0
    for j in eachindex(grid.y.centers), k in eachindex(grid.z.centers)
        ey += Float64(Ey[j, k])
        ez += Float64(Ez[j, k])
        n += 1
    end
    return ey / max(n, 1), ez / max(n, 1)
end

function record!(m::Polarimetry, sim)
    push!(m.t, sim.t)
    fields = _view_fields(sim)
    if m.mode === nothing
        ey, ez = _mean_plane_components(fields, sim.grid, m.position)
    else
        ey = _mode_overlap(fields, sim.grid, m.axis, m.position, m.mode, :Ey)
        ez = _mode_overlap(fields, sim.grid, m.axis, m.position, m.mode, :Ez)
    end
    push!(m.ey, ey)
    push!(m.ez, ez)
    return m
end

function record!(m::SwitchedFraction, sim)
    st = _state_from_monitor_view(sim)
    push!(m.t, sim.t)
    if st === nothing || st.mag === nothing
        push!(m.values, NaN)
    else
        mx = st.mag.m_TM_x
        push!(m.values, isempty(mx) ? NaN : reduce_count_negative(mx) / length(mx))
    end
    return m
end

function record!(m::HotCellTrace, sim)
    st = _state_from_monitor_view(sim)
    push!(m.t, sim.t)
    if st === nothing || st.mag === nothing || st.thermal === nothing ||
       !(1 <= m.cell <= length(st.mag.m_TM_x))
        push!(m.Te_K, NaN); push!(m.mx_TM, NaN); push!(m.mx_RE, NaN)
    else
        push!(m.Te_K, Float64(scalar_to_host(st.thermal.Te, m.cell)))
        push!(m.mx_TM, Float64(scalar_to_host(st.mag.m_TM_x, m.cell)))
        push!(m.mx_RE, Float64(scalar_to_host(st.mag.m_RE_x, m.cell)))
    end
    return m
end

record!(m::CallbackMonitor, sim) = (m.f(sim); m)

_mean_or_nan(v) = isempty(v) ? NaN : sum(Float64, v) / length(v)
_push_film!(m::FilmAverage, key::Symbol, value) = (push!(m.data[key], Float64(value)); m)

function _film_state_values(st)
    mag = st.mag
    th = st.thermal
    gd = st.model isa MagnetoOpticModel ? st.model.params : nothing
    if mag === nothing
        return nothing
    end
    mxT = mag.m_TM_x; myT = mag.m_TM_y; mzT = mag.m_TM_z
    mxR = mag.m_RE_x; myR = mag.m_RE_y; mzR = mag.m_RE_z
    mtmx = reduce_mean(mxT); mtmy = reduce_mean(myT); mtmz = reduce_mean(mzT)
    mrex = reduce_mean(mxR); mrey = reduce_mean(myR); mrez = reduce_mean(mzR)
    normT = reduce_norm_mean(mxT, myT, mzT)
    normR = reduce_norm_mean(mxR, myR, mzR)
    Te = Tl = TsT = TsR = NaN
    if th !== nothing
        Te = reduce_mean(th.Te)
        Tl = reduce_mean(th.Tl)
        TsT = reduce_mean(th.Ts_TM)
        TsR = reduce_mean(th.Ts_RE)
    end
    MsT = gd === nothing ? 1.0 : gd.Ms_TM
    MsR = gd === nothing ? 1.0 : gd.Ms_RE
    MT = (MsT * mtmx, MsT * mtmy, MsT * mtmz)
    MR = (MsR * mrex, MsR * mrey, MsR * mrez)
    return (;
        mx_TM=mtmx, mx_RE=mrex,
        m_TM_x_reduced=mtmx, m_TM_y_reduced=mtmy, m_TM_z_reduced=mtmz,
        m_RE_x_reduced=mrex, m_RE_y_reduced=mrey, m_RE_z_reduced=mrez,
        M_TM_x_Apm=MT[1], M_TM_y_Apm=MT[2], M_TM_z_Apm=MT[3],
        M_RE_x_Apm=MR[1], M_RE_y_Apm=MR[2], M_RE_z_Apm=MR[3],
        M_net_x_Apm=MT[1] + MR[1], M_net_y_Apm=MT[2] + MR[2], M_net_z_Apm=MT[3] + MR[3],
        Te_avg=Te, Tl_avg=Tl, Ts_avg=0.5 * (TsT + TsR),
        Te_K=Te, Tl_K=Tl, Ts_TM_K=TsT, Ts_RE_K=TsR,
        mag_TM_norm_avg=normT, mag_RE_norm_avg=normR,
    )
end

function record!(m::FilmAverage, sim)
    sim.n % m.every == 0 || return m
    st = _state_from_monitor_view(sim)
    _push_film!(m, :mag_time, sim.t)
    vals = st === nothing ? nothing : _film_state_values(st)
    if vals === nothing
        for k in _FILM_AVERAGE_KEYS
            k === :mag_time || _push_film!(m, k, NaN)
        end
        return m
    end
    for k in _FILM_AVERAGE_KEYS
        k === :mag_time || _push_film!(m, k, getproperty(vals, k))
    end
    return m
end

function record!(m::Progress, sim)
    if sim.n % m.every == 0
        push!(m.steps, sim.n)
        push!(m.t, sim.t)
    end
    return m
end

function _nan_guard_check!(label::AbstractString, arr, n::Integer)
    any_nonfinite(arr) && error("NaNGuard detected a non-finite value in $label at step $n")
    return nothing
end

function record!(m::NaNGuard, sim)
    _monitor_due(m, sim.n) || return m
    fields = _view_fields(sim)
    for component in (:Ex, :Ey, :Ez, :Hx, :Hy, :Hz)
        hasproperty(fields, component) || continue
        _nan_guard_check!(String(component), _field_array(fields, component), sim.n)
    end
    st = _state_from_monitor_view(sim)
    if st !== nothing
        if st.mag !== nothing
            for component in (:m_TM_x, :m_TM_y, :m_TM_z, :m_RE_x, :m_RE_y, :m_RE_z)
                _nan_guard_check!(String(component), getfield(st.mag, component), sim.n)
            end
        end
        if st.thermal !== nothing
            for component in (:Te, :Tl, :Ts_TM, :Ts_RE)
                _nan_guard_check!(String(component), getfield(st.thermal, component), sim.n)
            end
        end
    end
    return m
end

monitor_data(m::PointMonitor) = (t=m.t, values=m.values)
monitor_data(m::DFTMonitor) = (t=m.t, values=m.values, spectrum=compute_spectrum(m.t, m.values))
monitor_data(m::FieldMonitor) = m.frames
monitor_data(m::FluxMonitor) = (t=m.t, flux=m.flux, total=isempty(m.t) ? 0.0 : sum(m.flux) * (length(m.t) > 1 ? m.t[2] - m.t[1] : 0.0))
function _tr_summary(m, sign::Real)
    dt = length(m.t) > 1 ? m.t[2] - m.t[1] : 0.0
    if m.mode === nothing
        energy = sum(m.values) * dt                       # ∫ flux dt (signed)
    else
        energy = sum(abs2, m.values) * dt                 # modal energy ∝ ∫a²dt
    end
    ratio = sign * energy / max(m.incident, eps(Float64))
    return (t=m.t, values=m.values, modal=(m.mode !== nothing), energy=energy, ratio=ratio)
end
monitor_data(m::Transmission) = merge(_tr_summary(m, 1.0), (transmission=_tr_summary(m, 1.0).ratio,))
monitor_data(m::Reflection) = merge(_tr_summary(m, 1.0), (reflection=_tr_summary(m, 1.0).ratio,))
monitor_data(m::Absorption) = (t=m.t, values=m.values)
monitor_data(m::AbsorbedPower) = (t=m.t, values=m.values)
monitor_data(m::FilmAverage) = NamedTuple{_FILM_AVERAGE_KEYS}(Tuple(m.data[k] for k in _FILM_AVERAGE_KEYS))
function monitor_data(m::Polarimetry)
    if isempty(m.t)
        return (t=m.t, ey=m.ey, ez=m.ez, rotation_deg=NaN, ellipticity_deg=NaN)
    end
    Ey = sum(m.ey[i] * cis(m.omega * m.t[i]) for i in eachindex(m.t))   # single-freq DFT phasor
    Ez = sum(m.ez[i] * cis(m.omega * m.t[i]) for i in eachindex(m.t))
    ang = probe_jones_angles_deg(Ey, Ez)
    return (t=m.t, ey=m.ey, ez=m.ez, rotation_deg=ang.rotation_deg, ellipticity_deg=ang.ellipticity_deg)
end
monitor_data(m::SwitchedFraction) = (t=m.t, values=m.values)
monitor_data(m::HotCellTrace) = (t=m.t, Te_K=m.Te_K, mx_TM=m.mx_TM, mx_RE=m.mx_RE, cell=m.cell)
monitor_data(::CallbackMonitor) = nothing
monitor_data(m::Progress) = (steps=m.steps, t=m.t)
monitor_data(::NaNGuard) = nothing

# Always returns the full set of keys, with empty arrays when the state carries no
# magnetization (e.g. an under-resolved grid with zero active cells). Keeping the
# shape stable lets downstream writers/accessors run uniformly instead of crashing
# on a missing field.
function film_active_snapshot(state::FDTDState)
    gd = state.model isa MagnetoOpticModel ? state.model.params : FerrimagnetParameters()
    mag = state.mag
    mxT = mag === nothing ? Float64[] : Float64.(to_host(mag.m_TM_x))
    myT = mag === nothing ? Float64[] : Float64.(to_host(mag.m_TM_y))
    mzT = mag === nothing ? Float64[] : Float64.(to_host(mag.m_TM_z))
    mxR = mag === nothing ? Float64[] : Float64.(to_host(mag.m_RE_x))
    myR = mag === nothing ? Float64[] : Float64.(to_host(mag.m_RE_y))
    mzR = mag === nothing ? Float64[] : Float64.(to_host(mag.m_RE_z))
    Te = state.thermal === nothing ? fill(NaN, length(mxT)) : Float64.(to_host(state.thermal.Te))
    Tl = state.thermal === nothing ? fill(NaN, length(mxT)) : Float64.(to_host(state.thermal.Tl))
    TsT = state.thermal === nothing ? fill(NaN, length(mxT)) : Float64.(to_host(state.thermal.Ts_TM))
    TsR = state.thermal === nothing ? fill(NaN, length(mxT)) : Float64.(to_host(state.thermal.Ts_RE))
    cells = mag === nothing ? Int[] : Int.(to_host(state.region.material_cells))
    normT = sqrt.(mxT.^2 .+ myT.^2 .+ mzT.^2)
    normR = sqrt.(mxR.^2 .+ myR.^2 .+ mzR.^2)
    return (;
        m_TM_x_reduced_active_cells=mxT, m_TM_y_reduced_active_cells=myT, m_TM_z_reduced_active_cells=mzT,
        m_RE_x_reduced_active_cells=mxR, m_RE_y_reduced_active_cells=myR, m_RE_z_reduced_active_cells=mzR,
        M_TM_x_Apm_active_cells=gd.Ms_TM .* mxT, M_TM_y_Apm_active_cells=gd.Ms_TM .* myT, M_TM_z_Apm_active_cells=gd.Ms_TM .* mzT,
        M_RE_x_Apm_active_cells=gd.Ms_RE .* mxR, M_RE_y_Apm_active_cells=gd.Ms_RE .* myR, M_RE_z_Apm_active_cells=gd.Ms_RE .* mzR,
        M_net_x_Apm_active_cells=gd.Ms_TM .* mxT .+ gd.Ms_RE .* mxR,
        M_net_y_Apm_active_cells=gd.Ms_TM .* myT .+ gd.Ms_RE .* myR,
        M_net_z_Apm_active_cells=gd.Ms_TM .* mzT .+ gd.Ms_RE .* mzR,
        mag_TM_norm_active_cells=normT, mag_RE_norm_active_cells=normR,
        Te_active_cells=Te, Tl_active_cells=Tl, Ts_TM_active_cells=TsT, Ts_RE_active_cells=TsR,
        active_linear_index=cells,
    )
end

_masked_mean(v, mask, cnt) = cnt > 0 ? sum(v[mask]) / cnt : NaN

# Final-state switching diagnostics, with the reference production definitions:
# a cell counts as switched only when BOTH sublattices reversed (m_TM_x < 0 AND
# m_RE_x > 0); the diagnostic "core" is the cells with U_abs ≥ core_fraction of
# the local U_abs maximum.
function switching_metrics(m_TM_x::AbstractVector, m_RE_x::AbstractVector,
                           U_abs::AbstractVector; core_fraction::Real=0.5)
    n = length(m_TM_x)
    n > 0 || return (;
        final_switch_fraction=NaN, final_tm_reversed_fraction=NaN, final_re_reversed_fraction=NaN,
        final_switched_cell_mean_m_TM_x=NaN, final_switched_cell_mean_m_RE_x=NaN,
        final_unswitched_cell_mean_m_TM_x=NaN, final_unswitched_cell_mean_m_RE_x=NaN,
        final_mixture_check_m_TM_x=NaN, final_mixture_check_m_RE_x=NaN,
        core_Uabs_fraction_of_max=Float64(core_fraction), final_core_cell_count=0,
        final_core_switch_fraction=NaN, final_core_mean_m_TM_x=NaN, final_core_mean_m_RE_x=NaN,
        hot_cell_index=1,
    )
    mTM = Float64.(m_TM_x)
    mRE = Float64.(m_RE_x)
    switched = (mTM .< 0.0) .& (mRE .> 0.0)
    unswitched = .!switched
    sw_count = count(switched)
    un_count = n - sw_count
    sw_frac = sw_count / n
    sw_mean_TM = _masked_mean(mTM, switched, sw_count)
    sw_mean_RE = _masked_mean(mRE, switched, sw_count)
    un_mean_TM = _masked_mean(mTM, unswitched, un_count)
    un_mean_RE = _masked_mean(mRE, unswitched, un_count)
    U = Float64.(U_abs)
    Umax = isempty(U) ? 0.0 : maximum(U)
    core = U .>= Float64(core_fraction) * Umax
    core_count = count(core)
    core_sw = core_count > 0 ? count(switched .& core) / core_count : NaN
    return (;
        final_switch_fraction=sw_frac,
        final_tm_reversed_fraction=count(<(0.0), mTM) / n,
        final_re_reversed_fraction=count(>(0.0), mRE) / n,
        final_switched_cell_mean_m_TM_x=sw_mean_TM,
        final_switched_cell_mean_m_RE_x=sw_mean_RE,
        final_unswitched_cell_mean_m_TM_x=un_mean_TM,
        final_unswitched_cell_mean_m_RE_x=un_mean_RE,
        final_mixture_check_m_TM_x=sw_frac * sw_mean_TM + (1.0 - sw_frac) * un_mean_TM,
        final_mixture_check_m_RE_x=sw_frac * sw_mean_RE + (1.0 - sw_frac) * un_mean_RE,
        core_Uabs_fraction_of_max=Float64(core_fraction),
        final_core_cell_count=core_count,
        final_core_switch_fraction=core_sw,
        final_core_mean_m_TM_x=_masked_mean(mTM, core, core_count),
        final_core_mean_m_RE_x=_masked_mean(mRE, core, core_count),
        hot_cell_index=isempty(U) ? 1 : argmax(U),
    )
end
