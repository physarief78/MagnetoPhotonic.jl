const EM_FIELD_STORAGE_TYPE = Float32

struct FDTDParams{T<:AbstractFloat}
    c0::T
    mu0::T
    eps0::T
    n_si3n4::T
    epsr_si3n4::T
    n_sio2::T
    epsr_sio2::T
    epsr_vac::T
end

function FDTDParams{T}(; c0::Real=299792458.0, mu0::Real=4pi * 1e-7,
                       eps0::Real=1.0 / ((4pi * 1e-7) * 299792458.0^2),
                       n_si3n4::Real=2.0, epsr_si3n4::Real=Float64(n_si3n4)^2,
                       n_sio2::Real=1.444, epsr_sio2::Real=Float64(n_sio2)^2,
                       epsr_vac::Real=1.0) where {T<:AbstractFloat}
    return FDTDParams{T}(
        T(c0), T(mu0), T(eps0), T(n_si3n4), T(epsr_si3n4),
        T(n_sio2), T(epsr_sio2), T(epsr_vac),
    )
end

FDTDParams(; kwargs...) = FDTDParams{Float64}(; kwargs...)

function get_n_sio2(lambda_m::Real)
    l_um = Float64(lambda_m) * 1e6
    l2 = l_um^2
    n2 = 1.0 +
         (0.6961663 * l2) / (l2 - 0.00467914826) +
         (0.4079426 * l2) / (l2 - 0.0135120631) +
         (0.8974794 * l2) / (l2 - 97.9340025)
    return sqrt(n2)
end

function get_n_si3n4(lambda_m::Real)
    l_um = Float64(lambda_m) * 1e6
    l2 = l_um^2
    n2 = 1.0 +
         (3.0249 * l2) / (l2 - 0.13534^2) +
         (40314.0 * l2) / (l2 - 1239.842^2)
    return sqrt(n2)
end

function FDTDParams(lambda0::Real)
    n_si = get_n_si3n4(lambda0)
    n_ox = get_n_sio2(lambda0)
    return FDTDParams{Float64}(;
        c0=299792458.0,
        mu0=4pi * 1e-7,
        eps0=1.0 / ((4pi * 1e-7) * 299792458.0^2),
        n_si3n4=n_si,
        epsr_si3n4=n_si^2,
        n_sio2=n_ox,
        epsr_sio2=n_ox^2,
        epsr_vac=1.0,
    )
end

function cfl_dt(dx::Real, dy::Real, dz::Real, p::FDTDParams=FDTDParams(); courant::Real=0.99)
    denom = p.c0 * sqrt(inv(Float64(dx))^2 + inv(Float64(dy))^2 + inv(Float64(dz))^2)
    return Float64(courant) / denom
end
