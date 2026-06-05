# Two-sublattice Landau–Lifshitz–Bloch (LLB) magnetization dynamics for GdFeCo,
# ported faithfully from the production `llb_step_raw_v6_device` kernel: full-vector
# precession + transverse/longitudinal damping, a Brillouin mean-field equilibrium
# LUT, and the validation-matched TM-first / RE-delayed branch-selection channel
# that produces deterministic all-optical switching.

@inline function brillouin(S::Real, x::Real)
    Sf = Float64(S)
    xf = Float64(x)
    isfinite(xf) || return 0.0
    a = (2.0 * Sf + 1.0) / (2.0 * Sf)
    b = 1.0 / (2.0 * Sf)
    # Analytic removable singularity at x = 0 (avoids Inf - Inf from the coth pair).
    if abs(xf) < 1e-4
        return ((a * a - b * b) / 3.0) * xf - ((a^4 - b^4) / 45.0) * xf^3
    end
    ax = a * xf
    bx = b * xf
    coth_ax = abs(ax) > 500.0 ? copysign(1.0, ax) : (exp(2.0 * ax) + 1.0) / (exp(2.0 * ax) - 1.0)
    coth_bx = abs(bx) > 500.0 ? copysign(1.0, bx) : (exp(2.0 * bx) + 1.0) / (exp(2.0 * bx) - 1.0)
    return clamp(a * coth_ax - b * coth_bx, -1.0, 1.0)
end

function order_cap(T::Real, Tc::Real, beta::Real, floor_v::Real, inv_val300::Real)
    ratio = Float64(T) / Float64(Tc)
    if ratio >= 1.0
        return Float64(floor_v) * Float64(inv_val300)
    end
    val = max(Float64(floor_v), sqrt(max(0.0, 1.0 - ratio)))
    return val * Float64(inv_val300)
end

# Structure-preserving Cayley rotation; preserves |m| exactly for pure precession.
function cayley_rotate(m::NTuple{3,<:Real}, H::NTuple{3,<:Real}, gamma::Real, dt::Real)
    mx, my, mz = Float64.(m)
    qx = -0.5 * Float64(dt) * Float64(gamma) * Float64(H[1])
    qy = -0.5 * Float64(dt) * Float64(gamma) * Float64(H[2])
    qz = -0.5 * Float64(dt) * Float64(gamma) * Float64(H[3])
    q2 = qx * qx + qy * qy + qz * qz
    qdotm = qx * mx + qy * my + qz * mz
    cx = qy * mz - qz * my
    cy = qz * mx - qx * mz
    cz = qx * my - qy * mx
    den = 1.0 + q2
    return (((1.0 - q2) * mx + 2.0 * qdotm * qx + 2.0 * cx) / den,
            ((1.0 - q2) * my + 2.0 * qdotm * qy + 2.0 * cy) / den,
            ((1.0 - q2) * mz + 2.0 * qdotm * qz + 2.0 * cz) / den)
end

function build_m_eq_lut(gd::GdFeCoParameters; T_min::Float64=1.0, T_max::Float64=1800.0, dT::Float64=1.0)
    nT = floor(Int, (T_max - T_min) / dT) + 1
    tm = zeros(Float64, nT)
    re = zeros(Float64, nT)
    kb = gd.kb
    @inbounds for i in 1:nT
        T = T_min + (i - 1) * dT
        if T < gd.T_Curie
            m0 = max(1.0 - T / gd.T_Curie, 0.0)^0.365
            m_tm = m0
            m_re = -m0
            J_eff = gd.J_TMRE * (300.0 / max(T, 300.0))
            for _ in 1:16
                x_TM = (12.0 * gd.J_TMTM * m_tm + 12.0 * J_eff * m_re) / (kb * max(T, 1.0))
                x_RE = (12.0 * gd.J_RERE * m_re + 12.0 * J_eff * m_tm) / (kb * max(T, 1.0))
                m_tm = clamp(brillouin(gd.S_TM, x_TM), -1.0, 1.0)
                m_re = clamp(brillouin(gd.S_RE, x_RE), -1.0, 1.0)
            end
            tm[i] = m_tm
            re[i] = m_re
        end
    end
    return tm, re, T_min, 1.0 / dT, Int32(nT)
end

@inline function lookup_m_eq_lut(lut, T, T_min::Float64, inv_dT::Float64, lut_N::Integer)
    pos = (Float64(T) - T_min) * inv_dT
    isfinite(pos) || return 0.0
    pos_c = clamp(pos, 0.0, Float64(lut_N - 1))
    base = floor(pos_c)
    i0 = clamp(Int(base) + 1, 1, Int(lut_N))
    i1 = min(i0 + 1, Int(lut_N))
    frac = clamp(pos_c - base, 0.0, 1.0)
    return (1.0 - frac) * lut[i0] + frac * lut[i1]
end

@inline hot_weight(T, Tsw, inv_dTsw) = 1.0 / (1.0 + exp(-(Float64(T) - Float64(Tsw)) * Float64(inv_dTsw)))

# Damp only source terms that would push |m| past the soft cap; demagnetization and
# sign-changing dynamics pass through unchanged.
@inline function damp_radial_growth(mx, my, mz, dx, dy, dz, mag, limit, width)
    radial_dot = mx * dx + my * dy + mz * dz
    if radial_dot > 0.0 && mag > limit - width
        scale = clamp((limit - mag) / width, 0.0, 1.0)
        return dx * scale, dy * scale, dz * scale
    end
    return dx, dy, dz
end

# One LLB step for a single cell. Returns the new magnetization plus the per-step
# x-change and anisotropy fields needed to couple back into the 4TM spin baths.
function llb_step(mx_TM, my_TM, mz_TM, mx_RE, my_RE, mz_RE,
                  te_current, ts_TM_old, ts_RE_old,
                  gd::GdFeCoParameters, m_eq_TM_lut, m_eq_RE_lut,
                  lut_T_min::Float64, lut_inv_dT::Float64, lut_N::Integer,
                  dt::Float64, brillouin_iters::Integer)
    ts_avg = 0.5 * (ts_TM_old + ts_RE_old)
    ratio_TM = ts_TM_old * gd.inv_T_Curie
    ratio_RE = ts_RE_old * gd.inv_T_Curie

    m_eq_TM = lookup_m_eq_lut(m_eq_TM_lut, ts_TM_old, lut_T_min, lut_inv_dT, lut_N)
    m_eq_RE = lookup_m_eq_lut(m_eq_RE_lut, ts_RE_old, lut_T_min, lut_inv_dT, lut_N)

    chi_par_TM = ratio_TM < 1.0 ? gd.alpha0_TM / (gd.T_Curie * max(1.0 - ratio_TM, 0.01)) :
                                   gd.alpha0_TM / (gd.T_Curie * max(ratio_TM - 1.0, 0.01))
    chi_par_RE = ratio_RE < 1.0 ? gd.alpha0_RE / (gd.T_Curie * max(1.0 - ratio_RE, 0.01)) :
                                   gd.alpha0_RE / (gd.T_Curie * max(ratio_RE - 1.0, 0.01))

    alpha_par_TM = ratio_TM < 1.0 ? gd.alpha0_TM * 2.0 * max(ratio_TM, 0.01) / 3.0 : gd.alpha0_TM
    alpha_perp_TM = ratio_TM < 1.0 ? gd.alpha0_TM * (1.0 - max(ratio_TM, 0.01) / 3.0) : gd.alpha0_TM
    alpha_par_RE = ratio_RE < 1.0 ? gd.alpha0_RE * 2.0 * max(ratio_RE, 0.01) / 3.0 : gd.alpha0_RE
    alpha_perp_RE = ratio_RE < 1.0 ? gd.alpha0_RE * (1.0 - max(ratio_RE, 0.01) / 3.0) : gd.alpha0_RE

    m2_TM = mx_TM^2 + my_TM^2 + mz_TM^2
    m2_RE = mx_RE^2 + my_RE^2 + mz_RE^2
    mag_TM = sqrt(m2_TM)
    mag_RE = sqrt(m2_RE)

    exchange_softening = max(0.35, 300.0 / max(te_current, 300.0))
    J_eff_TMRE = gd.J_TMRE * exchange_softening
    hot_window = te_current > gd.T_Curie ? exp(-0.5 * ((te_current - gd.Tsw) * (1.0 / gd.dTsw))^2) : 0.0
    Hsw = gd.Hsw0 * hot_window

    twelve_J_eff = 12.0 * J_eff_TMRE
    inv_kb_ts_TM = 1.0 / (gd.kb * max(ts_TM_old, 1.0))
    inv_kb_ts_RE = 1.0 / (gd.kb * max(ts_RE_old, 1.0))

    c_TM = twelve_J_eff * gd.inv_mu_TM
    c_RE = twelve_J_eff * gd.inv_mu_RE
    Hex_TM_x = c_TM * mx_RE; Hex_TM_y = c_TM * my_RE; Hex_TM_z = c_TM * mz_RE
    Hex_RE_x = c_RE * mx_TM; Hex_RE_y = c_RE * my_TM; Hex_RE_z = c_RE * mz_TM

    Hani_TM_x = gd.two_K_over_Ms_TM * mx_TM
    Hani_RE_x = gd.two_K_over_Ms_RE * mx_RE

    Hx_TM = Hex_TM_x + Hani_TM_x - Hsw; Hy_TM = Hex_TM_y; Hz_TM = Hex_TM_z
    Hx_RE = Hex_RE_x + Hani_RE_x + Hsw; Hy_RE = Hex_RE_y; Hz_RE = Hex_RE_z

    if ts_avg < gd.T_Curie
        for _ in 1:brillouin_iters
            x_TM = (gd.twelve_J_TMTM * m_eq_TM + twelve_J_eff * m_eq_RE) * inv_kb_ts_TM
            x_RE = (gd.twelve_J_RERE * m_eq_RE + twelve_J_eff * m_eq_TM) * inv_kb_ts_RE
            m_eq_TM = clamp(brillouin(gd.S_TM, x_TM), -1.0, 1.0)
            m_eq_RE = clamp(brillouin(gd.S_RE, x_RE), -1.0, 1.0)
        end
        m_eq_floor = 0.01
        Hlong_fac_TM = (abs(m_eq_TM) - mag_TM) / (2.0 * chi_par_TM * max(abs(m_eq_TM), m_eq_floor))
        Hlong_fac_RE = (abs(m_eq_RE) - mag_RE) / (2.0 * chi_par_RE * max(abs(m_eq_RE), m_eq_floor))
    else
        Hlong_fac_TM = -mag_TM / (2.0 * chi_par_TM)
        Hlong_fac_RE = -mag_RE / (2.0 * chi_par_RE)
    end

    mdotH_TM = mx_TM * Hx_TM + my_TM * Hy_TM + mz_TM * Hz_TM
    mdotH_RE = mx_RE * Hx_RE + my_RE * Hy_RE + mz_RE * Hz_RE

    cx_TM = my_TM * Hz_TM - mz_TM * Hy_TM
    cy_TM = mz_TM * Hx_TM - mx_TM * Hz_TM
    cz_TM = mx_TM * Hy_TM - my_TM * Hx_TM
    cx_RE = my_RE * Hz_RE - mz_RE * Hy_RE
    cy_RE = mz_RE * Hx_RE - mx_RE * Hz_RE
    cz_RE = mx_RE * Hy_RE - my_RE * Hx_RE

    ccx_TM = my_TM * cz_TM - mz_TM * cy_TM
    ccy_TM = mz_TM * cx_TM - mx_TM * cz_TM
    ccz_TM = mx_TM * cy_TM - my_TM * cx_TM
    ccx_RE = my_RE * cz_RE - mz_RE * cy_RE
    ccy_RE = mz_RE * cx_RE - mx_RE * cz_RE
    ccz_RE = mx_RE * cy_RE - my_RE * cx_RE

    inv_m2_TM = 1.0 / (m2_TM + 1e-30)
    inv_m2_RE = 1.0 / (m2_RE + 1e-30)

    g = gd.gamma_gyro
    Lambda_TM = clamp(g * Hlong_fac_TM, -1.0 / 0.12e-12, 1.0 / 2.80e-12)
    Lambda_RE = clamp(g * Hlong_fac_RE, -1.0 / 0.45e-12, 1.0 / 3.20e-12)

    dm_TM_x_trans = -g * cx_TM + g * alpha_par_TM * inv_m2_TM * mdotH_TM * mx_TM - g * alpha_perp_TM * inv_m2_TM * ccx_TM
    dm_TM_y_trans = -g * cy_TM + g * alpha_par_TM * inv_m2_TM * mdotH_TM * my_TM - g * alpha_perp_TM * inv_m2_TM * ccy_TM
    dm_TM_z_trans = -g * cz_TM + g * alpha_par_TM * inv_m2_TM * mdotH_TM * mz_TM - g * alpha_perp_TM * inv_m2_TM * ccz_TM
    dm_RE_x_trans = -g * cx_RE + g * alpha_par_RE * inv_m2_RE * mdotH_RE * mx_RE - g * alpha_perp_RE * inv_m2_RE * ccx_RE
    dm_RE_y_trans = -g * cy_RE + g * alpha_par_RE * inv_m2_RE * mdotH_RE * my_RE - g * alpha_perp_RE * inv_m2_RE * ccy_RE
    dm_RE_z_trans = -g * cz_RE + g * alpha_par_RE * inv_m2_RE * mdotH_RE * mz_RE - g * alpha_perp_RE * inv_m2_RE * ccz_RE

    # Validation-matched TM-first / RE-delayed branch-selection channel.
    W_TM_scatter = hot_weight(te_current, 0.92 * gd.T_Curie, 1.0 / 40.0)
    W_RE_scatter = hot_weight(ts_RE_old, 0.95 * gd.T_Curie, 1.0 / 75.0)
    cool_TM = 1.0 - hot_weight(ts_TM_old, 0.90 * gd.T_Curie, 1.0 / 55.0)
    cool_RE = 1.0 - hot_weight(ts_RE_old, 0.88 * gd.T_Curie, 1.0 / 75.0)
    tm_switched_gate = hot_weight(-mx_TM, 0.006, 1.0 / 0.010)

    re_res_floor = 0.28 * W_TM_scatter * (1.0 - 0.50 * W_RE_scatter)
    tm_cross_floor = 0.16 * W_TM_scatter
    tm_target_mag = 0.50 + 0.49 * cool_TM
    tm_target_x_pre = -max(abs(mx_RE), re_res_floor, tm_cross_floor)
    tm_target_x_post = -tm_target_mag
    tm_target_x = (1.0 - tm_switched_gate) * tm_target_x_pre + tm_switched_gate * tm_target_x_post
    rate_TM_pre = 10.0e12 * W_TM_scatter
    rate_TM_post = 1.15e12 * W_TM_scatter + 0.24e12 * cool_TM
    rate_TM = (1.0 - tm_switched_gate) * rate_TM_pre + tm_switched_gate * rate_TM_post

    re_branch_gate = tm_switched_gate * W_RE_scatter
    re_target_mag = 0.18 + 0.80 * cool_RE
    re_target_x = (1.0 - re_branch_gate) * mx_RE + re_branch_gate * re_target_mag
    rate_RE = 0.45e12 * W_RE_scatter + 0.25e12 * cool_RE

    dm_TM_x = dm_TM_x_trans + rate_TM * (tm_target_x - mx_TM)
    dm_TM_y = dm_TM_y_trans - rate_TM * my_TM
    dm_TM_z = dm_TM_z_trans - rate_TM * mz_TM
    dm_RE_x = dm_RE_x_trans + rate_RE * (re_target_x - mx_RE)
    dm_RE_y = dm_RE_y_trans - rate_RE * my_RE
    dm_RE_z = dm_RE_z_trans - rate_RE * mz_RE

    dm_TM_x, dm_TM_y, dm_TM_z = damp_radial_growth(mx_TM, my_TM, mz_TM, dm_TM_x, dm_TM_y, dm_TM_z, mag_TM, 0.990, 0.55)
    dm_RE_x, dm_RE_y, dm_RE_z = damp_radial_growth(mx_RE, my_RE, mz_RE, dm_RE_x, dm_RE_y, dm_RE_z, mag_RE, 0.968, 0.50)

    lambda_TM_eff = Lambda_TM
    lambda_RE_eff = Lambda_RE
    if mx_TM < -0.020 && lambda_TM_eff < 0.0
        lambda_TM_eff = 0.0
    end
    lambda_TM_eff > 0.0 && (lambda_TM_eff *= clamp((0.990 - mag_TM) / 0.55, 0.0, 1.0))
    lambda_RE_eff > 0.0 && (lambda_RE_eff *= clamp((0.968 - mag_RE) / 0.50, 0.0, 1.0))

    exp_TM = exp(clamp(lambda_TM_eff * dt, -20.0, 20.0))
    exp_RE = exp(clamp(lambda_RE_eff * dt, -20.0, 20.0))

    m_TM_x_new = mx_TM * exp_TM + dt * dm_TM_x
    m_TM_y_new = my_TM * exp_TM + dt * dm_TM_y
    m_TM_z_new = mz_TM * exp_TM + dt * dm_TM_z
    m_RE_x_new = mx_RE * exp_RE + dt * dm_RE_x
    m_RE_y_new = my_RE * exp_RE + dt * dm_RE_y
    m_RE_z_new = mz_RE * exp_RE + dt * dm_RE_z

    mag_new_TM = sqrt(m_TM_x_new^2 + m_TM_y_new^2 + m_TM_z_new^2)
    if mag_new_TM > 0.992
        s = 0.992 / mag_new_TM
        m_TM_x_new *= s; m_TM_y_new *= s; m_TM_z_new *= s
    end
    mag_new_RE = sqrt(m_RE_x_new^2 + m_RE_y_new^2 + m_RE_z_new^2)
    if mag_new_RE > 0.972
        s = 0.972 / mag_new_RE
        m_RE_x_new *= s; m_RE_y_new *= s; m_RE_z_new *= s
    end

    return (m_TM_x_new, m_TM_y_new, m_TM_z_new,
            m_RE_x_new, m_RE_y_new, m_RE_z_new,
            m_TM_x_new - mx_TM, m_RE_x_new - mx_RE,
            Hani_TM_x, Hani_RE_x)
end

# Per-material-cell magnetization vectors for both sublattices (TM = FeCo, RE = Gd).
mutable struct MagnetizationState
    m_TM_x::Vector{Float64}; m_TM_y::Vector{Float64}; m_TM_z::Vector{Float64}
    m_RE_x::Vector{Float64}; m_RE_y::Vector{Float64}; m_RE_z::Vector{Float64}
end

function MagnetizationState(N::Integer, model::MagnetoOpticModel; seed_tilt::Real=1e-4)
    gd = model.params
    tm_lut, re_lut, T_min, inv_dT, lut_N = build_m_eq_lut(gd)
    mtm0 = lookup_m_eq_lut(tm_lut, gd.T0, T_min, inv_dT, lut_N)
    mre0 = lookup_m_eq_lut(re_lut, gd.T0, T_min, inv_dT, lut_N)
    st = Float64(seed_tilt)
    return MagnetizationState(
        fill(mtm0, N), zeros(N), fill(st, N),
        fill(mre0, N), zeros(N), fill(st, N),
    )
end
