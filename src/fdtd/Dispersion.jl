using LinearAlgebra
using Printf

struct DLPole{T<:AbstractFloat}
    C1::T
    C2::T
    C3::T
end

Base.convert(::Type{DLPole{T}}, p::DLPole) where {T<:AbstractFloat} = DLPole{T}(T(p.C1), T(p.C2), T(p.C3))

function create_pole(omega0::Real, gamma::Real, delta_eps_omega0_sq::Real, dt::Real, eps0::Real)
    alpha = Float64(dt) / 2.0
    D = 1.0 + alpha * Float64(gamma) + (alpha * Float64(omega0))^2
    C1 = (1.0 + alpha * Float64(gamma) - (alpha * Float64(omega0))^2) / D
    C2 = (2.0 * alpha) / D
    C3 = (alpha^2 * Float64(eps0) * Float64(delta_eps_omega0_sq)) / D
    return DLPole{Float64}(C1, C2, C3)
end

_chi_cont_pole(omega0::Float64, gamma::Float64, S::Float64, omega::Float64) =
    S / (omega0^2 - omega^2 - im * gamma * omega)

function discrete_pole_chi(pole::DLPole, dt::Real, eps0::Real, omega::Real; warm::Int=4000, meas::Int=2000)
    beta = pole.C3
    P = 0.0
    J = 0.0
    Eprev = 0.0
    Pv = Vector{Float64}(undef, meas + 1)
    k = 0
    for n in 0:(warm + meas)
        E = cos(Float64(omega) * Float64(dt) * n)
        psi = muladd(pole.C1, P, muladd(pole.C2, J, beta * Eprev))
        Pnew = psi + beta * E
        Jnew = (Pnew - P) * (2.0 / Float64(dt)) - J
        P, J, Eprev = Pnew, Jnew, E
        if n >= warm
            k += 1
            Pv[k] = Pnew
        end
    end
    ns = collect(warm:(warm + meas))
    t = ns .* Float64(dt)
    nyq = [isodd(n) ? -1.0 : 1.0 for n in ns]
    M = hcat(cos.(Float64(omega) .* t), sin.(Float64(omega) .* t), nyq, ones(length(ns)))
    coef = M \ Pv
    return (coef[1] + im * coef[2]) / Float64(eps0)
end

function _fit_pole_strengths(pos, omega::Float64, tgt::ComplexF64)
    U = [_chi_cont_pole(Float64(p[1]), Float64(p[2]), 1.0, omega) for p in pos]
    A = vcat(reshape(real.(U), 1, :), reshape(imag.(U), 1, :))
    return A \ [real(tgt); imag(tgt)]
end

const _PROBE_POLE_POSITIONS = ((0.0, 3.0e14), (4.0e15, 8.0e14))
const _PROBE_MO_POLE_POSITIONS = ((4.0e15, 8.0e14), (5.0e15, 1.2e15))

function _build_probe_pole_set(dt::Real, eps0::Real, omega::Real, target::ComplexF64, name::AbstractString;
                               pos=_PROBE_POLE_POSITIONS, require_passive::Bool=false, verbose::Bool=false)
    S = _fit_pole_strengths(pos, Float64(omega), target)
    poles = [create_pole(Float64(pos[i][1]), Float64(pos[i][2]), Float64(S[i]), Float64(dt), Float64(eps0)) for i in eachindex(pos)]
    if verbose
        discrete = sum(discrete_pole_chi(p, dt, eps0, omega) for p in poles)
        relerr = abs(discrete - target) / max(abs(target), eps(Float64))
        @printf("%s ADE fit @ %.3e rad/s: relerr=%.3f%%\n", name, omega, 100.0 * relerr)
    end
    min_S = minimum(S)
    if min_S < 0.0 && require_passive
        error("$name pole fit produced a negative oscillator strength (min S = $min_S).")
    end
    return poles
end

function build_pump_poles(dt::Real, eps0::Real, gd; verbose::Bool=false)
    epsd = ComplexF64(gd.eps_real_pump, gd.eps_imag_pump)
    return _build_probe_pole_set(dt, eps0, Float64(gd.omega_pump), epsd - 1.0 + 0.0im, "pump diagonal";
                                 pos=_PROBE_POLE_POSITIONS, require_passive=false, verbose=verbose)
end

function build_probe_poles(dt::Real, eps0::Real, gd)
    epsd = ComplexF64(gd.eps_real_probe, gd.eps_imag_probe)
    return _build_probe_pole_set(dt, eps0, Float64(gd.omega_probe), epsd - 1.0 + 0.0im, "probe diagonal";
                                 pos=_PROBE_POLE_POSITIONS, require_passive=true)
end

function build_probe_mo_poles(dt::Real, eps0::Real, gd)
    epsd = ComplexF64(gd.eps_real_probe, gd.eps_imag_probe)
    return _build_probe_pole_set(dt, eps0, Float64(gd.omega_probe), -im * epsd, "probe MO off-diagonal";
                                 pos=_PROBE_MO_POLE_POSITIONS, require_passive=true)
end

struct ADEState{T,A2,A1}
    P::A2
    J::A2
    E_old::A1
end

ADEState(P, J, E_old) = ADEState{eltype(P),typeof(P),typeof(E_old)}(P, J, E_old)

Adapt.@adapt_structure ADEState

function allocate_ade_state(N_active::Integer, poles::AbstractVector; T=Float64, backend::AbstractBackend=CPUBackend())
    P = zeros_backend(backend, T, N_active, length(poles))
    J = zeros_backend(backend, T, N_active, length(poles))
    E_old = zeros_backend(backend, T, N_active)
    return ADEState{T,typeof(P),typeof(E_old)}(P, J, E_old)
end

function patch_E_dispersive!(E_arr, state::ADEState, active_idx::AbstractVector{<:Integer}, fill::AbstractVector, inv_eps::AbstractVector, poles::AbstractVector{<:DLPole}, dt::Real;
                             backend::AbstractBackend=CPUBackend(), compute_T::Type=default_compute_type(backend))
    return _ka_patch_E_dispersive!(backend, E_arr, state, active_idx, fill, inv_eps, poles, dt, compute_T)
end
