function _patch_sim_ade!(sim::Simulation{1})
    sim.ade === nothing && return sim
    patch_E_dispersive!(sim.fields.Ez, sim.ade.z, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    return sim
end

function _patch_sim_ade!(sim::Simulation{2,:TM})
    sim.ade === nothing && return sim
    patch_E_dispersive!(sim.fields.Ez, sim.ade.z, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    return sim
end

function _patch_sim_ade!(sim::Simulation{2})
    sim.ade === nothing && return sim
    patch_E_dispersive!(sim.fields.Ex, sim.ade.x, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    patch_E_dispersive!(sim.fields.Ey, sim.ade.y, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    return sim
end

function _patch_sim_ade!(sim::Simulation{3})
    sim.ade === nothing && return sim
    patch_E_dispersive!(sim.fields.Ex, sim.ade.x, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    patch_E_dispersive!(sim.fields.Ey, sim.ade.y, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    patch_E_dispersive!(sim.fields.Ez, sim.ade.z, sim.active_indices, sim.active_fill, sim.active_inv_eps, sim.poles, sim.dt)
    return sim
end

function _inject_sources!(sim::Simulation)
    for src in sim.sources
        inject!(sim.fields, sim.grid, src, sim.t, sim.params, sim.inv_eps)
    end
    return sim
end

function _finish_step!(sim::Simulation)
    sim.n += 1
    sim.t = sim.n * sim.dt
    return sim
end

function step!(sim::Simulation{1})
    update_H_1d!(sim.fields, sim.grid, sim.params, sim.dt; cpml=sim.cpml)
    _inject_sources!(sim)
    update_E_1d!(sim.fields, sim.grid, sim.params, sim.dt, sim.inv_eps; cpml=sim.cpml)
    _patch_sim_ade!(sim)
    apply_boundary!(sim.fields, sim.grid, sim.boundary, sim.dt, sim.params)
    return _finish_step!(sim)
end

function step!(sim::Simulation{2,M}) where {M}
    update_H_2d!(sim.fields, sim.grid, sim.params, sim.dt; mode=M, cpml=sim.cpml)
    _inject_sources!(sim)
    update_E_2d!(sim.fields, sim.grid, sim.params, sim.dt, sim.inv_eps; mode=M, cpml=sim.cpml)
    _patch_sim_ade!(sim)
    apply_boundary!(sim.fields, sim.grid, sim.boundary, sim.dt, sim.params)
    return _finish_step!(sim)
end

function step!(sim::Simulation{3})
    update_H!(sim.fields, sim.grid, sim.params, sim.dt; cpml=sim.cpml)
    _inject_sources!(sim)
    update_E!(sim.fields, sim.grid, sim.params, sim.dt,
              sim.inv_eps.inv_eps_x, sim.inv_eps.inv_eps_y, sim.inv_eps.inv_eps_z; cpml=sim.cpml)
    _patch_sim_ade!(sim)
    apply_boundary!(sim.fields, sim.grid, sim.boundary, sim.dt, sim.params)
    return _finish_step!(sim)
end

function run!(sim::Simulation, steps::Integer; monitors=AbstractMonitor[], callback=nothing)
    mons = AbstractMonitor[monitors...]
    for _ in 1:steps
        step!(sim)
        for m in mons
            record!(m, sim)
        end
        callback === nothing || callback(sim)
    end
    return sim
end

function run!(sim::Simulation; until=nothing, steps=nothing, monitors=AbstractMonitor[], callback=nothing)
    nsteps = steps === nothing ? (until === nothing ? 1 : max(1, round(Int, Float64(until) / sim.dt))) : Int(steps)
    return run!(sim, nsteps; monitors=monitors, callback=callback)
end
