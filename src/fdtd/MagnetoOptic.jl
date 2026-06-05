struct MagnetoOpticADEState{T}
    Py_from_z::Matrix{T}
    Jy_from_z::Matrix{T}
    Pz_from_y::Matrix{T}
    Jz_from_y::Matrix{T}
    E_old_y::Vector{T}
    E_old_z::Vector{T}
end

function allocate_mo_ade_state(N_active::Integer, poles::AbstractVector; T=Float64)
    np = length(poles)
    return MagnetoOpticADEState(
        zeros(T, N_active, np),
        zeros(T, N_active, np),
        zeros(T, N_active, np),
        zeros(T, N_active, np),
        zeros(T, N_active),
        zeros(T, N_active),
    )
end

function patch_E_mo_gyration!(Ey, Ez, state::MagnetoOpticADEState, active_idx::AbstractVector{<:Integer},
                              fill::AbstractVector, material_pos::AbstractVector{<:Integer}, inv_eps_y, inv_eps_z,
                              m_TM_x::AbstractVector, m_RE_x::AbstractVector, poles::AbstractVector{<:DLPole},
                              Q_voigt_TM::Real, Q_voigt_RE::Real, dt::Real)
    inv_alpha = 2.0 / Float64(dt)
    @inbounds for n in eachindex(active_idx)
        idx = active_idx[n]
        mpos = material_pos[n]
        mpos > 0 || continue
        f = Float64(fill[n])
        qeff = Float64(Q_voigt_TM) * Float64(m_TM_x[mpos]) + Float64(Q_voigt_RE) * Float64(m_RE_x[mpos])
        Ey_drive = Float64(Ey[idx])
        Ez_drive = Float64(Ez[idx])
        Ey_old = state.E_old_y[n]
        Ez_old = state.E_old_z[n]
        sum_P_y = 0.0
        sum_P_z = 0.0

        for p in eachindex(poles)
            pole = poles[p]
            beta = pole.C3 * f * qeff
            Py_old = state.Py_from_z[n, p]
            Jy_old = state.Jy_from_z[n, p]
            Py_new = muladd(pole.C1, Py_old, muladd(pole.C2, Jy_old, beta * Ez_old)) + beta * Ez_drive
            sum_P_y += Py_new

            Pz_old = state.Pz_from_y[n, p]
            Jz_old = state.Jz_from_y[n, p]
            Pz_new = muladd(pole.C1, Pz_old, muladd(pole.C2, Jz_old, beta * Ey_old)) + beta * Ey_drive
            sum_P_z += Pz_new
        end

        Ey_final = Ey_drive - sum_P_y * Float64(inv_eps_y[idx])
        Ez_final = Ez_drive + sum_P_z * Float64(inv_eps_z[idx])
        if isfinite(Ey_final) && isfinite(Ez_final)
            Ey[idx] = Ey_final
            Ez[idx] = Ez_final
            state.E_old_y[n] = Ey_final
            state.E_old_z[n] = Ez_final
            for p in eachindex(poles)
                pole = poles[p]
                beta = pole.C3 * f * qeff
                Py_old = state.Py_from_z[n, p]
                Jy_old = state.Jy_from_z[n, p]
                Py_new = muladd(pole.C1, Py_old, muladd(pole.C2, Jy_old, beta * Ez_old)) + beta * Ez_drive
                state.Py_from_z[n, p] = Py_new
                state.Jy_from_z[n, p] = (Py_new - Py_old) * inv_alpha - Jy_old

                Pz_old = state.Pz_from_y[n, p]
                Jz_old = state.Jz_from_y[n, p]
                Pz_new = muladd(pole.C1, Pz_old, muladd(pole.C2, Jz_old, beta * Ey_old)) + beta * Ey_drive
                state.Pz_from_y[n, p] = Pz_new
                state.Jz_from_y[n, p] = (Pz_new - Pz_old) * inv_alpha - Jz_old
            end
        end
    end
    return Ey, Ez
end
