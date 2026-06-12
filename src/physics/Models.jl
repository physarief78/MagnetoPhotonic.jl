abstract type AbstractPhysicsModel end

struct NullModel <: AbstractPhysicsModel end

struct FerrimagnetParameters{T<:AbstractFloat}
    gamma_e::T
    Cl::T
    Cs_TM::T
    Cs_RE::T
    Gel::T
    Ges_TM::T
    Ges_RE::T
    Gamma_E::T
    Hsw0::T
    Tsw::T
    dTsw::T
    gamma_gyro::T
    Ms_TM::T
    Ms_RE::T
    mu_TM::T
    mu_RE::T
    alpha0_TM::T
    alpha0_RE::T
    J_TMRE::T
    J_TMTM::T
    J_RERE::T
    K_TM::T
    K_RE::T
    T_Curie::T
    T_comp::T
    T0::T
    TcT::T
    TcR::T
    inv_TcT::T
    inv_TcR::T
    beta_T::T
    beta_R::T
    floor_T::T
    floor_R::T
    val300_T::T
    val300_R::T
    inv_val300_T::T
    inv_val300_R::T
    GammaT0::T
    GammaT_hot::T
    GammaR0::T
    GammaR_hot::T
    Tsw_long::T
    dTsw_long::T
    inv_dTsw_long::T
    ellT::T
    ellR::T
    inv_ellT::T
    inv_ellR::T
    GammaEx::T
    eta_quench::T
    inv_eta_quench::T
    GammaT_tail::T
    eta_tail::T
    inv_eta_tail::T
    mTM_tail_target::T
    field_to_K::T
    GTlT::T
    GTlR::T
    GsTR::T
    S_TM::T
    S_RE::T
    Q_voigt_TM::T
    Q_voigt_RE::T
    omega_pump::T
    eps_imag_pump::T
    eps_real_pump::T
    omega_probe::T
    eps_imag_probe::T
    eps_real_probe::T
    kb::T
    moment_ratio_TM_RE::T
    inv_Ms_sum::T
    inv_T_Curie::T
    inv_Cl::T
    inv_Cs_TM::T
    inv_Cs_RE::T
    ferri_RE_scale::T
    re_exchange_lag::T
    inv_mu_TM::T
    inv_mu_RE::T
    inv_Ms_TM::T
    inv_Ms_RE::T
    twelve_J_TMTM::T
    twelve_J_RERE::T
    two_K_over_Ms_TM::T
    two_K_over_Ms_RE::T
    inv_tau_TM_demag::T
    inv_tau_RE_demag::T
    moment_ratio_RE_TM::T
    HK_T_field::T
    HK_R_field::T
    HAF_T_field::T
    HAF_R_field::T
    alphaT_hot::T
    alphaR_hot::T
    seed_tilt::T
end

const GdFeCoParameters = FerrimagnetParameters

function FerrimagnetParameters{T}(;
    c0::Real=299792458.0,
    kb::Real=1.380649e-23,
    muB::Real=9.2740100783e-24,
    gamma_e::Real=300.0,
    Cl::Real=2.30e6,
    Gel::Real=6.0e17,
    T0::Real=300.0,
    T_Curie::Real=540.0,
    T_comp::Real=250.0,
    TcT::Real=620.0,
    TcR::Real=760.0,
    beta_T::Real=0.50,
    beta_R::Real=0.50,
    floor_T::Real=0.05,
    floor_R::Real=0.05,
    Ms_TM::Real=1.10e6,
    Ms_RE::Real=0.55e6,
    mu_TM::Real=1.92 * muB,
    mu_RE::Real=7.63 * muB,
    S_TM::Real=1.0,
    S_RE::Real=3.5,
    tau_TM_demag::Real=100e-15,
    tau_RE_demag::Real=430e-15,
    Cs_TM::Real=5.0e4,
    Cs_RE::Real=2.0e4,
    Ges_TM=nothing,
    Ges_RE=nothing,
    Gamma_E::Real=1.0 / 140e-15,
    gamma_gyro::Real=1.76085963023e11,
    alpha0_TM::Real=0.05,
    alpha0_RE::Real=0.05,
    J_TMRE::Real=-4.77e-21,
    J_TMTM::Real=3.46e-21,
    J_RERE::Real=1.389e-21,
    K_total::Real=3.8e4,
    K_TM=nothing,
    K_RE=nothing,
    Hsw0::Real=0.0,
    Tsw=nothing,
    dTsw::Real=50.0,
    Tsw_long=nothing,
    dTsw_long::Real=55.0,
    GammaT0::Real=1.0 / 5.0e-12,
    GammaT_hot=nothing,
    GammaR0::Real=1.0 / 20.0e-12,
    GammaR_hot=nothing,
    GammaEx=nothing,
    eta_quench::Real=1.0,
    GammaT_tail::Real=0.0,
    eta_tail::Real=1.0,
    mTM_tail_target::Real=0.0,
    GTlT=nothing,
    GTlR=nothing,
    GsTR::Real=0.90e17,
    lambda_pump::Real=800e-9,
    lambda_probe::Real=532e-9,
    n_pump::Real=3.50,
    k_pump::Real=3.73,
    n_probe::Real=2.73,
    k_probe::Real=4.60,
    Q_voigt_TM::Real=0.020,
    Q_voigt_RE::Real=0.006,
    ellT::Real=1.0,
    ellR=nothing,
    field_to_K=nothing,
    ferri_RE_scale=nothing,
    re_exchange_lag=nothing,
    seed_tilt::Real=1e-4,
) where {T<:AbstractFloat}
    Ges_TM_v = Ges_TM === nothing ? Float64(Cs_TM) / Float64(tau_TM_demag) : Float64(Ges_TM)
    Ges_RE_v = Ges_RE === nothing ? Float64(Cs_RE) / Float64(tau_RE_demag) : Float64(Ges_RE)
    K_TM_v = K_TM === nothing ? 0.5 * Float64(K_total) : Float64(K_TM)
    K_RE_v = K_RE === nothing ? 0.5 * Float64(K_total) : Float64(K_RE)
    Tsw_v = Tsw === nothing ? Float64(T_Curie) : Float64(Tsw)
    Tsw_long_v = Tsw_long === nothing ? Float64(T_Curie) : Float64(Tsw_long)
    GammaT_hot_v = GammaT_hot === nothing ? 1.0 / Float64(tau_TM_demag) : Float64(GammaT_hot)
    GammaR_hot_v = GammaR_hot === nothing ? 1.0 / Float64(tau_RE_demag) : Float64(GammaR_hot)
    GammaEx_v = GammaEx === nothing ? Float64(Gamma_E) : Float64(GammaEx)
    ellR_v = ellR === nothing ? Float64(mu_RE) / Float64(mu_TM) : Float64(ellR)
    field_to_K_v = field_to_K === nothing ? Float64(mu_TM) / Float64(kb) : Float64(field_to_K)
    GTlT_v = GTlT === nothing ? Ges_TM_v : Float64(GTlT)
    GTlR_v = GTlR === nothing ? Ges_RE_v : Float64(GTlR)
    ferri_RE_scale_v = ferri_RE_scale === nothing ? Float64(Ms_RE) / Float64(Ms_TM) : Float64(ferri_RE_scale)
    re_exchange_lag_v = re_exchange_lag === nothing ? Float64(mu_TM) / Float64(mu_RE) : Float64(re_exchange_lag)

    val300_T = max(Float64(floor_T), sqrt(max(0.0, 1.0 - Float64(T0) / Float64(TcT))))
    val300_R = max(Float64(floor_R), sqrt(max(0.0, 1.0 - Float64(T0) / Float64(TcR))))
    omega_pump = 2pi * Float64(c0) / Float64(lambda_pump)
    omega_probe = 2pi * Float64(c0) / Float64(lambda_probe)
    eps_real_pump = Float64(n_pump)^2 - Float64(k_pump)^2
    eps_imag_pump = 2.0 * Float64(n_pump) * Float64(k_pump)
    eps_real_probe = Float64(n_probe)^2 - Float64(k_probe)^2
    eps_imag_probe = 2.0 * Float64(n_probe) * Float64(k_probe)
    HK_T = 2.0 * K_TM_v / Float64(Ms_TM)
    HK_R = 2.0 * K_RE_v / Float64(Ms_RE)
    HAF_T = abs(Float64(J_TMRE)) / Float64(mu_TM)
    HAF_R = abs(Float64(J_TMRE)) / Float64(mu_RE)

    vals = (
        gamma_e, Cl, Cs_TM, Cs_RE, Gel, Ges_TM_v, Ges_RE_v, Gamma_E,
        Hsw0, Tsw_v, dTsw, gamma_gyro, Ms_TM, Ms_RE, mu_TM, mu_RE,
        alpha0_TM, alpha0_RE, J_TMRE, J_TMTM, J_RERE, K_TM_v, K_RE_v,
        T_Curie, T_comp, T0, TcT, TcR, 1 / Float64(TcT), 1 / Float64(TcR),
        beta_T, beta_R, floor_T, floor_R, val300_T, val300_R, 1 / val300_T, 1 / val300_R,
        GammaT0, GammaT_hot_v, GammaR0, GammaR_hot_v, Tsw_long_v, dTsw_long, 1 / Float64(dTsw_long),
        ellT, ellR_v, 1 / Float64(ellT), 1 / ellR_v, GammaEx_v, eta_quench, 1 / Float64(eta_quench),
        GammaT_tail, eta_tail, 1 / Float64(eta_tail), mTM_tail_target, field_to_K_v,
        GTlT_v, GTlR_v, GsTR, S_TM, S_RE, Q_voigt_TM, Q_voigt_RE,
        omega_pump, eps_imag_pump, eps_real_pump, omega_probe, eps_imag_probe, eps_real_probe,
        kb, Float64(mu_TM) / Float64(mu_RE), 1 / (Float64(Ms_TM) + Float64(Ms_RE)),
        1 / Float64(T_Curie), 1 / Float64(Cl), 1 / Float64(Cs_TM), 1 / Float64(Cs_RE),
        ferri_RE_scale_v, re_exchange_lag_v, 1 / Float64(mu_TM), 1 / Float64(mu_RE),
        1 / Float64(Ms_TM), 1 / Float64(Ms_RE), 12 * Float64(J_TMTM), 12 * Float64(J_RERE),
        2 * K_TM_v / Float64(Ms_TM), 2 * K_RE_v / Float64(Ms_RE),
        1 / Float64(tau_TM_demag), 1 / Float64(tau_RE_demag), Float64(mu_RE) / Float64(mu_TM),
        HK_T, HK_R, HAF_T, HAF_R, alpha0_TM, alpha0_RE, seed_tilt,
    )
    return FerrimagnetParameters{T}(map(T, vals)...)
end

FerrimagnetParameters(; kwargs...) = FerrimagnetParameters{Float64}(; kwargs...)

function ferrimagnet(preset::Symbol=:gdfeco; overrides...)
    preset === :gdfeco || throw(ArgumentError("unknown ferrimagnet preset $preset"))
    return FerrimagnetParameters(; overrides...)
end

gdfeco_parameters(; overrides...) = ferrimagnet(:gdfeco; overrides...)

struct MagnetoOpticModel{P<:FerrimagnetParameters} <: AbstractPhysicsModel
    params::P
end

function MagnetoOpticModel(; preset::Symbol=:gdfeco, params=nothing, overrides...)
    params === nothing || return MagnetoOpticModel(params)
    return MagnetoOpticModel(ferrimagnet(preset; overrides...))
end

function convert_compute_params(::Type{T}, gd::FerrimagnetParameters) where {T<:AbstractFloat}
    vals = ntuple(i -> T(getfield(gd, i)), fieldcount(typeof(gd)))
    return FerrimagnetParameters{T}(vals...)
end

convert_compute_model(::Type{T}, model::MagnetoOpticModel) where {T<:AbstractFloat} =
    MagnetoOpticModel(convert_compute_params(T, model.params))
convert_compute_model(::Type, model::AbstractPhysicsModel) = model

optical_coupling(::NullModel, args...) = 0.0
optical_coupling(model::MagnetoOpticModel, m_TM_x::Real, m_RE_x::Real) =
    model.params.Q_voigt_TM * Float64(m_TM_x) + model.params.Q_voigt_RE * Float64(m_RE_x)

absorbed_power_density(::NullModel, args...) = 0.0

function absorbed_power_density(model::MagnetoOpticModel, E2::Real;
                                omega::Real=model.params.omega_pump,
                                eps_imag::Real=model.params.eps_imag_pump,
                                eps0::Real=FDTDParams().eps0)
    return 0.5 * Float64(omega) * Float64(eps0) * Float64(eps_imag) * Float64(E2)
end
