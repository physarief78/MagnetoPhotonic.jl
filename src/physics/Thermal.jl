# Four-temperature model (electron, lattice, TM-spin, RE-spin), ported from
# `update_4tm_raw_v6_device`. The electron bath is integrated in the energy
# variable u_e = ½ γ_e T_e² to guarantee T_e ≥ 1 K.

mutable struct ThermalState{A}
    Te::A
    Tl::A
    Ts_TM::A
    Ts_RE::A
end

Adapt.@adapt_structure ThermalState

function ThermalState(N::Integer, model::MagnetoOpticModel; T=Float64, backend::AbstractBackend=CPUBackend())
    T0 = T(model.params.T0)
    return ThermalState(
        fill_backend(backend, T0, N),
        fill_backend(backend, T0, N),
        fill_backend(backend, T0, N),
        fill_backend(backend, T0, N),
    )
end

@inline function update_Te_energy(te, tl, tsTM, tsRE, pabs, gd::FerrimagnetParameters, dt::Real)
    ue = 0.5 * gd.gamma_e * te^2
    q = -gd.Gel * (te - tl) - gd.Ges_TM * (te - tsTM) - gd.Ges_RE * (te - tsRE) + pabs
    ue_new = max(0.5 * gd.gamma_e, ue + Float64(dt) * q)
    return sqrt(2.0 * ue_new / gd.gamma_e)
end

# Advance the four baths one step. Q_spin_TM/RE are the magnetic-work power
# densities fed back from the LLB step (zero when magnetization is frozen).
@inline function update_4tm(te, tl, ts_TM, ts_RE, pabs, Q_spin_TM, Q_spin_RE, gd::FerrimagnetParameters, dt::Real)
    tau_th = 2.0e-12
    te_new = update_Te_energy(te, tl, ts_TM, ts_RE, pabs, gd, dt)
    tl_new = tl + Float64(dt) * (gd.Gel * (te - tl) * gd.inv_Cl) - Float64(dt) * (tl - gd.T0) / tau_th
    ts_TM_new = ts_TM + Float64(dt) * (gd.Ges_TM * (te - ts_TM) - Q_spin_TM) * gd.inv_Cs_TM
    ts_RE_new = ts_RE + Float64(dt) * (gd.Ges_RE * (te - ts_RE) - Q_spin_RE) * gd.inv_Cs_RE
    return te_new, tl_new, max(1.0, ts_TM_new), max(1.0, ts_RE_new)
end

# Per-active-cell absorbed-energy bookkeeping, mirroring the reference production
# accumulators (U_abs_gpu / pabs_window_sum_gpu / pabs_peak_gpu): U_abs integrates
# max(E·J, 0)·dt every EM step, window_sum holds the same integral since the last
# multiphysics consume (the 4TM reads window_sum/dt_mp and resets it), peak tracks
# the running max instantaneous power density. Always Float64, like the reference.
mutable struct AbsorptionState{A}
    U_abs::A
    window_sum::A
    peak::A
end

Adapt.@adapt_structure AbsorptionState

function AbsorptionState(N::Integer; backend::AbstractBackend=CPUBackend())
    return AbsorptionState(
        fill_backend(backend, 0.0, N),
        fill_backend(backend, 0.0, N),
        fill_backend(backend, 0.0, N),
    )
end

# Pure-thermal advance (no magnetic back-coupling). `pabs` is a per-cell vector.
function thermal_step!(state::ThermalState, pabs, model::MagnetoOpticModel, dt::Real)
    gd = model.params
    @inbounds for i in eachindex(state.Te)
        te, tl, tsT, tsR = update_4tm(Float64(state.Te[i]), Float64(state.Tl[i]),
                                      Float64(state.Ts_TM[i]), Float64(state.Ts_RE[i]),
                                      Float64(pabs[i]), 0.0, 0.0, gd, dt)
        state.Te[i] = te; state.Tl[i] = tl; state.Ts_TM[i] = tsT; state.Ts_RE[i] = tsR
    end
    return state
end

thermal_step!(state::ThermalState, pabs, ::NullModel, dt::Real) = state
