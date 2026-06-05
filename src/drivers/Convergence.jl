# Dimension-agnostic convergence driver for the general EM layer. It runs the
# same pulsed source/probe experiment at decreasing cell size and compares
# successive probe traces after resampling onto a common time axis.

function _resample(t::Vector{Float64}, y::Vector{Float64}, tq::Vector{Float64})
    out = similar(tq)
    nt = length(t)
    @inbounds for (i, q) in enumerate(tq)
        if q <= t[1]
            out[i] = y[1]
        elseif q >= t[nt]
            out[i] = y[nt]
        else
            lo, hi = 1, nt
            while hi - lo > 1
                mid = (lo + hi) >> 1
                t[mid] <= q ? (lo = mid) : (hi = mid)
            end
            w = (q - t[lo]) / (t[hi] - t[lo])
            out[i] = (1.0 - w) * y[lo] + w * y[hi]
        end
    end
    return out
end

function _conv_cell(L::Float64, dimension::Integer)
    dimension == 1 && return (L,)
    dimension == 2 && return (L, L)
    dimension == 3 && return (L, L, L)
    throw(ArgumentError("dimension must be 1, 2, or 3"))
end

function _conv_source_probe(L::Float64, dimension::Integer, mode::Symbol, pulse)
    if dimension == 1
        return PointSource(pulse, :Ez, 0.30L), (:Ez, 0.70L)
    elseif dimension == 2
        comp = mode === :TE ? :Hz : :Ez
        return PlaneSource(pulse, comp; axis=:x, position=0.30L), (comp, (0.70L, 0.50L))
    else
        return PlaneSource(pulse, :Ez; axis=:x, position=0.30L), (:Ez, (0.70L, 0.50L, 0.50L))
    end
end

function _conv_geometry(L::Float64, dimension::Integer, p::FDTDParams, dt::Float64, dispersive::Bool)
    scene = Scene()
    dispersive || return scene
    gd = MagnetoOpticModel().params
    poles = build_pump_poles(dt, p.eps0, gd)
    mat = Material("dispersive slab"; epsr=1.0, poles=poles)
    if dimension == 1
        add_shape!(scene, Box(0.45L, 0.55L, -1.0, 1.0, -1.0, 1.0), mat)
    elseif dimension == 2
        add_shape!(scene, Box(0.45L, 0.55L, 0.0, L, -1.0, 1.0), mat)
    else
        add_shape!(scene, Box(0.45L, 0.55L, 0.0, L, 0.0, L), mat)
    end
    return scene
end

function convergence_study(; dimension::Integer=2, mode::Symbol=:TM, dispersive::Bool=false,
                           pml::Bool=false, dxs=(50e-9, 35e-9, 22e-9), L::Real=1.0e-6,
                           lambda0::Real=800e-9, tau::Real=18e-15, T_max::Real=90e-15,
                           courant::Real=0.4, n_pml::Integer=6, nsample::Integer=400)
    p = FDTDParams(lambda0)
    Lf = Float64(L)
    traces = NamedTuple[]
    energies = Float64[]
    for dx in dxs
        cell = _conv_cell(Lf, dimension)
        grid = _grid_from_cell(cell, dimension, dx)
        dt = cfl_dt(grid, p; courant=courant)
        pulse = GaussianPulse(; amplitude=1.0, tau=Float64(tau), t0=4.0 * Float64(tau),
                              omega=2pi * p.c0 / Float64(lambda0))
        src, probe = _conv_source_probe(Lf, dimension, mode, pulse)
        scene = _conv_geometry(Lf, dimension, p, dt, dispersive)
        boundary = pml ? PML(n_pml) : PEC()
        sim = Simulation(; cell=cell, dx=dx, geometry=scene, sources=[src], boundary=boundary,
                         dimension=dimension, mode=mode, courant=courant, params=p, subpixel=2, T=Float64)
        mon = PointMonitor(probe[1], probe[2])
        nsteps = max(1, round(Int, Float64(T_max) / sim.dt))
        run!(sim, nsteps; monitors=[mon])
        push!(traces, (dx=Float64(dx), t=copy(mon.t), trace=copy(mon.values)))
        push!(energies, field_energy(sim.fields, sim.grid, p))
    end

    tq = collect(range(0.0, Float64(T_max); length=nsample))
    resampled = [_resample(tr.t, tr.trace, tq) for tr in traces]
    cauchy = Float64[]
    for i in 1:(length(traces) - 1)
        d = resampled[i] .- resampled[i + 1]
        push!(cauchy, sqrt(sum(abs2, d) / length(d)))
    end
    orders = Float64[]
    for i in 1:(length(cauchy) - 1)
        push!(orders, log(cauchy[i] / cauchy[i + 1]) / log(Float64(dxs[i]) / Float64(dxs[i + 1])))
    end
    spectrum = dispersive ? compute_spectrum(traces[end].t, traces[end].trace) : nothing
    return (dimension=Int(dimension), mode=mode, dispersive=dispersive, pml=pml,
            dxs=collect(Float64, dxs), traces=traces, t_common=tq,
            cauchy_l2=cauchy, orders=orders, final_energy=energies, spectrum=spectrum)
end

run_convergence_study(; kwargs...) = convergence_study(; dimension=2, mode=:TM, kwargs...)
run_convergence_study_3D(; kwargs...) = convergence_study(; dimension=3, mode=:TM, kwargs...)
run_convergence_study_3D_dispersive(; kwargs...) = convergence_study(; dimension=3, mode=:TM, dispersive=true, kwargs...)
