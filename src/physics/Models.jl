abstract type AbstractPhysicsModel end

struct NullModel <: AbstractPhysicsModel end

struct GdFeCoParameters{T<:AbstractFloat}
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

function GdFeCoParameters{T}(; Hsw0::Real=0.0, T_Curie::Real=540.0, Q_voigt_TM::Real=0.020,
                             Q_voigt_RE::Real=0.006, alpha0_TM::Real=0.05, alpha0_RE::Real=0.05) where {T<:AbstractFloat}
    c0 = 299792458.0
    kb = 1.380649e-23
    muB = 9.2740100783e-24
    mu_TM = 1.92 * muB
    mu_RE = 7.63 * muB
    gamma_e = 300.0
    Cl = 2.30e6
    Gel = 6.0e17
    T0 = 300.0
    T_comp = 250.0
    Ms_TM = 1.10e6
    Ms_RE = 0.55e6
    TcT = 620.0
    TcR = 760.0
    beta_T = 0.50
    beta_R = 0.50
    floor_T = 0.05
    floor_R = 0.05
    val300_T = max(floor_T, sqrt(max(0.0, 1.0 - T0 / TcT)))
    val300_R = max(floor_R, sqrt(max(0.0, 1.0 - T0 / TcR)))
    tau_TM_demag = 100e-15
    tau_RE_demag = 430e-15
    Cs_TM = 5.0e4
    Cs_RE = 2.0e4
    Ges_TM = Cs_TM / tau_TM_demag
    Ges_RE = Cs_RE / tau_RE_demag
    Gamma_E = 1.0 / 140e-15
    gamma_gyro = 1.76085963023e11
    J_TMRE = -4.77e-21
    J_TMTM = 3.46e-21
    J_RERE = 1.389e-21
    K_total = 3.8e4
    K_TM = 0.5 * K_total
    K_RE = 0.5 * K_total
    lambda_pump = 800e-9
    lambda_probe = 532e-9
    omega_pump = 2pi * c0 / lambda_pump
    omega_probe = 2pi * c0 / lambda_probe
    n_pump, k_pump = 3.50, 3.73
    n_probe, k_probe = 2.73, 4.60
    eps_real_pump = n_pump^2 - k_pump^2
    eps_imag_pump = 2.0 * n_pump * k_pump
    eps_real_probe = n_probe^2 - k_probe^2
    eps_imag_probe = 2.0 * n_probe * k_probe
    Tsw = Float64(T_Curie)
    dTsw = 50.0
    Tsw_long = Float64(T_Curie)
    dTsw_long = 55.0
    ellT = 1.0
    ellR = mu_RE / mu_TM
    GammaT0 = 1.0 / 5.0e-12
    GammaT_hot = 1.0 / tau_TM_demag
    GammaR0 = 1.0 / 20.0e-12
    GammaR_hot = 1.0 / tau_RE_demag
    GammaEx = Gamma_E
    eta_quench = 1.0
    GammaT_tail = 0.0
    eta_tail = 1.0
    mTM_tail_target = 0.0
    GTlT = Ges_TM
    GTlR = Ges_RE
    GsTR = 0.90e17
    HK_T = 2.0 * K_TM / Ms_TM
    HK_R = 2.0 * K_RE / Ms_RE
    HAF_T = abs(J_TMRE) / mu_TM
    HAF_R = abs(J_TMRE) / mu_RE

    vals = (
        gamma_e, Cl, Cs_TM, Cs_RE, Gel, Ges_TM, Ges_RE, Gamma_E,
        Float64(Hsw0), Tsw, dTsw, gamma_gyro, Ms_TM, Ms_RE, mu_TM, mu_RE,
        Float64(alpha0_TM), Float64(alpha0_RE), J_TMRE, J_TMTM, J_RERE, K_TM, K_RE,
        Float64(T_Curie), T_comp, T0, TcT, TcR, 1 / TcT, 1 / TcR, beta_T, beta_R,
        floor_T, floor_R, val300_T, val300_R, 1 / val300_T, 1 / val300_R,
        GammaT0, GammaT_hot, GammaR0, GammaR_hot, Tsw_long, dTsw_long, 1 / dTsw_long,
        ellT, ellR, 1 / ellT, 1 / ellR, GammaEx, eta_quench, 1 / eta_quench,
        GammaT_tail, eta_tail, 1 / eta_tail, mTM_tail_target, mu_TM / kb,
        GTlT, GTlR, GsTR, 1.0, 3.5, Float64(Q_voigt_TM), Float64(Q_voigt_RE),
        omega_pump, eps_imag_pump, eps_real_pump, omega_probe, eps_imag_probe, eps_real_probe,
        kb, mu_TM / mu_RE, 1 / (Ms_TM + Ms_RE), 1 / Float64(T_Curie), 1 / Cl, 1 / Cs_TM, 1 / Cs_RE,
        Ms_RE / Ms_TM, mu_TM / mu_RE, 1 / mu_TM, 1 / mu_RE, 1 / Ms_TM, 1 / Ms_RE,
        12 * J_TMTM, 12 * J_RERE, 2 * K_TM / Ms_TM, 2 * K_RE / Ms_RE,
        1 / tau_TM_demag, 1 / tau_RE_demag, mu_RE / mu_TM, HK_T, HK_R, HAF_T, HAF_R,
        Float64(alpha0_TM), Float64(alpha0_RE), 1e-4,
    )
    return GdFeCoParameters{T}(map(T, vals)...)
end

GdFeCoParameters(; kwargs...) = GdFeCoParameters{Float64}(; kwargs...)

struct MagnetoOpticModel{P<:GdFeCoParameters} <: AbstractPhysicsModel
    params::P
end

MagnetoOpticModel(; kwargs...) = MagnetoOpticModel(GdFeCoParameters(; kwargs...))

optical_coupling(::NullModel, args...) = 0.0
optical_coupling(model::MagnetoOpticModel, m_TM_x::Real, m_RE_x::Real) =
    model.params.Q_voigt_TM * Float64(m_TM_x) + model.params.Q_voigt_RE * Float64(m_RE_x)

absorbed_power_density(::NullModel, args...) = 0.0

function absorbed_power_density(model::MagnetoOpticModel, E2::Real; omega::Real=model.params.omega_pump, eps_imag::Real=model.params.eps_imag_pump, eps0::Real=FDTDParams().eps0)
    return 0.5 * Float64(omega) * Float64(eps0) * Float64(eps_imag) * Float64(E2)
end
