struct MagnetoOpticADEState{T,A2,A1}
    Py_from_z::A2
    Jy_from_z::A2
    Pz_from_y::A2
    Jz_from_y::A2
    E_old_y::A1
    E_old_z::A1
end

MagnetoOpticADEState(Py_from_z, Jy_from_z, Pz_from_y, Jz_from_y, E_old_y, E_old_z) =
    MagnetoOpticADEState{eltype(Py_from_z),typeof(Py_from_z),typeof(E_old_y)}(
        Py_from_z, Jy_from_z, Pz_from_y, Jz_from_y, E_old_y, E_old_z)

Adapt.@adapt_structure MagnetoOpticADEState

function allocate_mo_ade_state(N_active::Integer, poles::AbstractVector; T=Float64, backend::AbstractBackend=CPUBackend())
    np = length(poles)
    A2 = typeof(zeros_backend(backend, T, N_active, np))
    A1 = typeof(zeros_backend(backend, T, N_active))
    return MagnetoOpticADEState{T,A2,A1}(
        zeros_backend(backend, T, N_active, np),
        zeros_backend(backend, T, N_active, np),
        zeros_backend(backend, T, N_active, np),
        zeros_backend(backend, T, N_active, np),
        zeros_backend(backend, T, N_active),
        zeros_backend(backend, T, N_active),
    )
end

function patch_E_mo_gyration!(Ey, Ez, state::MagnetoOpticADEState, active_idx::AbstractVector{<:Integer},
                              fill::AbstractVector, material_pos::AbstractVector{<:Integer}, inv_eps_y, inv_eps_z,
                              m_TM_x::AbstractVector, m_RE_x::AbstractVector, poles::AbstractVector{<:DLPole},
                              Q_voigt_TM::Real, Q_voigt_RE::Real, dt::Real;
                              eps0::Real=FDTDParams().eps0,
                              backend::AbstractBackend=CPUBackend(), compute_T::Type=default_compute_type(backend))
    return _ka_patch_E_mo_gyration!(backend, Ey, Ez, state, active_idx, fill, material_pos,
                                    inv_eps_y, inv_eps_z, m_TM_x, m_RE_x, poles,
                                    Q_voigt_TM, Q_voigt_RE, dt, compute_T)
end
