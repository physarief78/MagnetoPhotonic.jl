_component_code(::Val{:Ex}) = Int32(1)
_component_code(::Val{:Ey}) = Int32(2)
_component_code(::Val{:Ez}) = Int32(3)
_component_code(::Val{:Hx}) = Int32(4)
_component_code(::Val{:Hy}) = Int32(5)
_component_code(::Val{:Hz}) = Int32(6)
_component_code(component::Symbol) = _component_code(Val(component))

@inline function _pml_slab(idx, N, npml)
    npml <= Int32(0) && return Int32(0), Int32(0)
    idx <= npml && return Int32(idx), Int32(1)
    idx > N - npml && return Int32(idx - (N - npml)), Int32(2)
    return Int32(0), Int32(0)
end

@inline function _cpml_term_x!(lo, hi, idx, j, k, N, npml, b, a, deriv, ::Val{CT}) where {CT}
    js, region = _pml_slab(idx, N, npml)
    if region == Int32(1)
        p = muladd(CT(b[idx]), CT(lo[js, j, k]), CT(a[idx]) * deriv)
        lo[js, j, k] = p
        return p
    elseif region == Int32(2)
        p = muladd(CT(b[idx]), CT(hi[js, j, k]), CT(a[idx]) * deriv)
        hi[js, j, k] = p
        return p
    end
    return zero(CT)
end

@inline function _cpml_term_y!(lo, hi, i, idx, k, N, npml, b, a, deriv, ::Val{CT}) where {CT}
    js, region = _pml_slab(idx, N, npml)
    if region == Int32(1)
        p = muladd(CT(b[idx]), CT(lo[i, js, k]), CT(a[idx]) * deriv)
        lo[i, js, k] = p
        return p
    elseif region == Int32(2)
        p = muladd(CT(b[idx]), CT(hi[i, js, k]), CT(a[idx]) * deriv)
        hi[i, js, k] = p
        return p
    end
    return zero(CT)
end

@inline function _cpml_term_z!(lo, hi, i, j, idx, N, npml, b, a, deriv, ::Val{CT}) where {CT}
    js, region = _pml_slab(idx, N, npml)
    if region == Int32(1)
        p = muladd(CT(b[idx]), CT(lo[i, j, js]), CT(a[idx]) * deriv)
        lo[i, j, js] = p
        return p
    elseif region == Int32(2)
        p = muladd(CT(b[idx]), CT(hi[i, j, js]), CT(a[idx]) * deriv)
        hi[i, j, js] = p
        return p
    end
    return zero(CT)
end

# Read-only arrays are @Const so the GPU compiler can issue them through the
# read-only data cache (__ldg), like a hand-written CUDA stencil would.
@kernel function yee_H_3d_kernel!(@Const(Ex), @Const(Ey), @Const(Ez), Hx, Hy, Hz,
                                  @Const(inv_d_cell_x), @Const(inv_d_cell_y), @Const(inv_d_cell_z),
                                  @Const(x_inv_kappa), @Const(x_a), @Const(x_b),
                                  @Const(y_inv_kappa), @Const(y_a), @Const(y_b),
                                  @Const(z_inv_kappa), @Const(z_a), @Const(z_b),
                                  psi_Hxy_lo, psi_Hxy_hi, psi_Hxz_lo, psi_Hxz_hi,
                                  psi_Hyz_lo, psi_Hyz_hi, psi_Hyx_lo, psi_Hyx_hi,
                                  psi_Hzx_lo, psi_Hzx_hi, psi_Hzy_lo, psi_Hzy_hi,
                                  Npml_x, Npml_y, Npml_z, Nx, Ny, Nz,
                                  dt_mu0, ::Val{CT}) where {CT}
    I = @index(Global, Cartesian)
    i = I[1]
    j = I[2]
    k = I[3]

    dEz_dy = (CT(Ez[i, j + 1, k]) - CT(Ez[i, j, k])) * CT(inv_d_cell_y[j])
    dEy_dz = (CT(Ey[i, j, k + 1]) - CT(Ey[i, j, k])) * CT(inv_d_cell_z[k])
    dEx_dz = (CT(Ex[i, j, k + 1]) - CT(Ex[i, j, k])) * CT(inv_d_cell_z[k])
    dEz_dx = (CT(Ez[i + 1, j, k]) - CT(Ez[i, j, k])) * CT(inv_d_cell_x[i])
    dEy_dx = (CT(Ey[i + 1, j, k]) - CT(Ey[i, j, k])) * CT(inv_d_cell_x[i])
    dEx_dy = (CT(Ex[i, j + 1, k]) - CT(Ex[i, j, k])) * CT(inv_d_cell_y[j])

    pHxy = _cpml_term_y!(psi_Hxy_lo, psi_Hxy_hi, i, j, k, Ny, Npml_y, y_b, y_a, dEz_dy, Val(CT))
    pHxz = _cpml_term_z!(psi_Hxz_lo, psi_Hxz_hi, i, j, k, Nz, Npml_z, z_b, z_a, dEy_dz, Val(CT))
    Hx[i, j, k] = CT(Hx[i, j, k]) -
                  CT(dt_mu0) * ((dEz_dy * CT(y_inv_kappa[j]) + pHxy) -
                                (dEy_dz * CT(z_inv_kappa[k]) + pHxz))

    pHyz = _cpml_term_z!(psi_Hyz_lo, psi_Hyz_hi, i, j, k, Nz, Npml_z, z_b, z_a, dEx_dz, Val(CT))
    pHyx = _cpml_term_x!(psi_Hyx_lo, psi_Hyx_hi, i, j, k, Nx, Npml_x, x_b, x_a, dEz_dx, Val(CT))
    Hy[i, j, k] = CT(Hy[i, j, k]) -
                  CT(dt_mu0) * ((dEx_dz * CT(z_inv_kappa[k]) + pHyz) -
                                (dEz_dx * CT(x_inv_kappa[i]) + pHyx))

    pHzx = _cpml_term_x!(psi_Hzx_lo, psi_Hzx_hi, i, j, k, Nx, Npml_x, x_b, x_a, dEy_dx, Val(CT))
    pHzy = _cpml_term_y!(psi_Hzy_lo, psi_Hzy_hi, i, j, k, Ny, Npml_y, y_b, y_a, dEx_dy, Val(CT))
    Hz[i, j, k] = CT(Hz[i, j, k]) -
                  CT(dt_mu0) * ((dEy_dx * CT(x_inv_kappa[i]) + pHzx) -
                                (dEx_dy * CT(y_inv_kappa[j]) + pHzy))
end

@kernel function yee_E_3d_kernel!(Ex, Ey, Ez, @Const(Hx), @Const(Hy), @Const(Hz), Dx, Dy, Dz,
                                  @Const(inv_eps_x), @Const(inv_eps_y), @Const(inv_eps_z),
                                  @Const(inv_d_dual_x), @Const(inv_d_dual_y), @Const(inv_d_dual_z),
                                  @Const(x_inv_kappa), @Const(x_a), @Const(x_b),
                                  @Const(y_inv_kappa), @Const(y_a), @Const(y_b),
                                  @Const(z_inv_kappa), @Const(z_a), @Const(z_b),
                                  psi_Dxy_lo, psi_Dxy_hi, psi_Dxz_lo, psi_Dxz_hi,
                                  psi_Dyz_lo, psi_Dyz_hi, psi_Dyx_lo, psi_Dyx_hi,
                                  psi_Dzx_lo, psi_Dzx_hi, psi_Dzy_lo, psi_Dzy_hi,
                                  Npml_x, Npml_y, Npml_z, Nx, Ny, Nz,
                                  dt, inv_eps0, ::Val{CT}) where {CT}
    I = @index(Global, Cartesian)
    i = I[1] + 1
    j = I[2] + 1
    k = I[3] + 1

    dHz_dy = (CT(Hz[i, j, k]) - CT(Hz[i, j - 1, k])) * CT(inv_d_dual_y[j])
    dHy_dz = (CT(Hy[i, j, k]) - CT(Hy[i, j, k - 1])) * CT(inv_d_dual_z[k])
    dHx_dz = (CT(Hx[i, j, k]) - CT(Hx[i, j, k - 1])) * CT(inv_d_dual_z[k])
    dHz_dx = (CT(Hz[i, j, k]) - CT(Hz[i - 1, j, k])) * CT(inv_d_dual_x[i])
    dHy_dx = (CT(Hy[i, j, k]) - CT(Hy[i - 1, j, k])) * CT(inv_d_dual_x[i])
    dHx_dy = (CT(Hx[i, j, k]) - CT(Hx[i, j - 1, k])) * CT(inv_d_dual_y[j])

    pDxy = _cpml_term_y!(psi_Dxy_lo, psi_Dxy_hi, i, j, k, Ny, Npml_y, y_b, y_a, dHz_dy, Val(CT))
    pDxz = _cpml_term_z!(psi_Dxz_lo, psi_Dxz_hi, i, j, k, Nz, Npml_z, z_b, z_a, dHy_dz, Val(CT))
    cx_term = (dHz_dy * CT(y_inv_kappa[j]) + pDxy) -
              (dHy_dz * CT(z_inv_kappa[k]) + pDxz)

    pDyz = _cpml_term_z!(psi_Dyz_lo, psi_Dyz_hi, i, j, k, Nz, Npml_z, z_b, z_a, dHx_dz, Val(CT))
    pDyx = _cpml_term_x!(psi_Dyx_lo, psi_Dyx_hi, i, j, k, Nx, Npml_x, x_b, x_a, dHz_dx, Val(CT))
    cy_term = (dHx_dz * CT(z_inv_kappa[k]) + pDyz) -
              (dHz_dx * CT(x_inv_kappa[i]) + pDyx)

    pDzx = _cpml_term_x!(psi_Dzx_lo, psi_Dzx_hi, i, j, k, Nx, Npml_x, x_b, x_a, dHy_dx, Val(CT))
    pDzy = _cpml_term_y!(psi_Dzy_lo, psi_Dzy_hi, i, j, k, Ny, Npml_y, y_b, y_a, dHx_dy, Val(CT))
    cz_term = (dHy_dx * CT(x_inv_kappa[i]) + pDzx) -
              (dHx_dy * CT(y_inv_kappa[j]) + pDzy)

    dx_new = CT(Dx[i, j, k]) + CT(dt) * cx_term
    dy_new = CT(Dy[i, j, k]) + CT(dt) * cy_term
    dz_new = CT(Dz[i, j, k]) + CT(dt) * cz_term
    Dx[i, j, k] = dx_new
    Dy[i, j, k] = dy_new
    Dz[i, j, k] = dz_new
    Ex[i, j, k] = dx_new * CT(inv_eps_x[i, j, k]) * CT(inv_eps0)
    Ey[i, j, k] = dy_new * CT(inv_eps_y[i, j, k]) * CT(inv_eps0)
    Ez[i, j, k] = dz_new * CT(inv_eps_z[i, j, k]) * CT(inv_eps0)
end

@kernel function inject_soft_3d_kernel!(Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz,
                                        inv_eps, code, i, j, k, value, eps0,
                                        ::Val{CT}) where {CT}
    val = CT(value)
    @inbounds if code == Int32(1)
        Ex[i, j, k] = CT(Ex[i, j, k]) + val
        Dx[i, j, k] = CT(Dx[i, j, k]) + val * CT(eps0) / CT(inv_eps[i, j, k])
    elseif code == Int32(2)
        Ey[i, j, k] = CT(Ey[i, j, k]) + val
        Dy[i, j, k] = CT(Dy[i, j, k]) + val * CT(eps0) / CT(inv_eps[i, j, k])
    elseif code == Int32(3)
        Ez[i, j, k] = CT(Ez[i, j, k]) + val
        Dz[i, j, k] = CT(Dz[i, j, k]) + val * CT(eps0) / CT(inv_eps[i, j, k])
    elseif code == Int32(4)
        Hx[i, j, k] = CT(Hx[i, j, k]) + val
    elseif code == Int32(5)
        Hy[i, j, k] = CT(Hy[i, j, k]) + val
    elseif code == Int32(6)
        Hz[i, j, k] = CT(Hz[i, j, k]) + val
    end
end

@kernel function inject_mode_x_3d_kernel!(Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz,
                                          profile, inv_eps, code, i, value, eps0,
                                          hcorr, ::Val{CT}) where {CT}
    I = @index(Global, Cartesian)
    j = I[1]
    k = I[2]
    amp = CT(value) * CT(profile[j, k])
    if code == Int32(2)
        Ey[i, j, k] = CT(Ey[i, j, k]) + amp
        Dy[i, j, k] = CT(Dy[i, j, k]) + amp * CT(eps0) / CT(inv_eps[i, j, k])
        Hz[i, j, k] = CT(Hz[i, j, k]) + CT(hcorr) * amp
    elseif code == Int32(3)
        Ez[i, j, k] = CT(Ez[i, j, k]) + amp
        Dz[i, j, k] = CT(Dz[i, j, k]) + amp * CT(eps0) / CT(inv_eps[i, j, k])
        Hy[i, j, k] = CT(Hy[i, j, k]) - CT(hcorr) * amp
    end
end

@kernel function ade_patch_kernel!(E_arr, P, J, E_old, active_idx, fill, inv_eps,
                                   poles, npoles, inv_alpha, ::Val{CT}) where {CT}
    n = @index(Global)
    @inbounds begin
    idx = active_idx[n]
    Eprev = CT(E_old[n])
    f = CT(fill[n])
    ie = CT(inv_eps[n])
    sum_psi = zero(CT)
    sum_beta = zero(CT)

    for pidx in 1:npoles
        pole = poles[pidx]
        beta = CT(pole.C3) * f
        psi = muladd(CT(pole.C1), CT(P[n, pidx]), muladd(CT(pole.C2), CT(J[n, pidx]), beta * Eprev))
        sum_psi += psi
        sum_beta += beta
    end

    den = one(CT) + sum_beta * ie
    Etemp = CT(E_arr[idx])
    if isfinite(Etemp) && isfinite(sum_psi) && isfinite(den) && abs(den) >= CT(1e-30)
        Efinal = (Etemp - sum_psi * ie) / den
        E_arr[idx] = Efinal
        E_old[n] = Efinal
        for pidx in 1:npoles
            pole = poles[pidx]
            beta = CT(pole.C3) * f
            Pold = CT(P[n, pidx])
            Jold = CT(J[n, pidx])
            psi = muladd(CT(pole.C1), Pold, muladd(CT(pole.C2), Jold, beta * Eprev))
            Pnew = psi + beta * Efinal
            P[n, pidx] = Pnew
            J[n, pidx] = (Pnew - Pold) * CT(inv_alpha) - Jold
        end
    end
    end  # @inbounds
end

@kernel function mo_gyration_kernel!(Ey, Ez, Py_from_z, Jy_from_z, Pz_from_y, Jz_from_y,
                                     E_old_y, E_old_z, active_idx, fill, material_pos,
                                     inv_eps_y, inv_eps_z, m_TM_x, m_RE_x, poles, npoles,
                                     Q_voigt_TM, Q_voigt_RE, inv_alpha, inv_eps0, ::Val{CT}) where {CT}
    n = @index(Global)
    @inbounds begin
    idx = active_idx[n]
    mpos = material_pos[n]
    if mpos > 0
        f = CT(fill[n])
        qeff = CT(Q_voigt_TM) * CT(m_TM_x[mpos]) + CT(Q_voigt_RE) * CT(m_RE_x[mpos])
        Ey_drive = CT(Ey[idx])
        Ez_drive = CT(Ez[idx])
        Ey_old = CT(E_old_y[n])
        Ez_old = CT(E_old_z[n])
        sum_P_y = zero(CT)
        sum_P_z = zero(CT)

        for pidx in 1:npoles
            pole = poles[pidx]
            beta = CT(pole.C3) * f * qeff
            Py_old = CT(Py_from_z[n, pidx])
            Jy_old = CT(Jy_from_z[n, pidx])
            Py_new = muladd(CT(pole.C1), Py_old, muladd(CT(pole.C2), Jy_old, beta * Ez_old)) + beta * Ez_drive
            sum_P_y += Py_new

            Pz_old = CT(Pz_from_y[n, pidx])
            Jz_old = CT(Jz_from_y[n, pidx])
            Pz_new = muladd(CT(pole.C1), Pz_old, muladd(CT(pole.C2), Jz_old, beta * Ey_old)) + beta * Ey_drive
            sum_P_z += Pz_new
        end

        # The polarization feedback is screened by 1/(ε0·ε): the volumes carry the
        # staggered 1/ε, the ε0 factor arrives via inv_eps0 (reference act_inv).
        Ey_final = Ey_drive - sum_P_y * CT(inv_eps_y[idx]) * CT(inv_eps0)
        Ez_final = Ez_drive + sum_P_z * CT(inv_eps_z[idx]) * CT(inv_eps0)
        if isfinite(Ey_final) && isfinite(Ez_final)
            Ey[idx] = Ey_final
            Ez[idx] = Ez_final
            E_old_y[n] = Ey_final
            E_old_z[n] = Ez_final
            for pidx in 1:npoles
                pole = poles[pidx]
                beta = CT(pole.C3) * f * qeff
                Py_old = CT(Py_from_z[n, pidx])
                Jy_old = CT(Jy_from_z[n, pidx])
                Py_new = muladd(CT(pole.C1), Py_old, muladd(CT(pole.C2), Jy_old, beta * Ez_old)) + beta * Ez_drive
                Py_from_z[n, pidx] = Py_new
                Jy_from_z[n, pidx] = (Py_new - Py_old) * CT(inv_alpha) - Jy_old

                Pz_old = CT(Pz_from_y[n, pidx])
                Jz_old = CT(Jz_from_y[n, pidx])
                Pz_new = muladd(CT(pole.C1), Pz_old, muladd(CT(pole.C2), Jz_old, beta * Ey_old)) + beta * Ey_drive
                Pz_from_y[n, pidx] = Pz_new
                Jz_from_y[n, pidx] = (Pz_new - Pz_old) * CT(inv_alpha) - Jz_old
            end
        end
    end
    end  # @inbounds
end

# Per-EM-step absorbed-power accumulation (mirrors the reference's
# kernel_em_pabs_accumulate!): pabs is the positive local E·J_ADE work for
# :ade_work (the per-cell ADE J already carries the fill factor through
# β = C3·f, so no extra fill multiplier — matching the reference) or the
# cycle-average ½ω ε0 ε''|E|² otherwise. U_abs and window_sum integrate
# pabs·dt; peak tracks the running max. All arithmetic in Float64.
@kernel function pabs_accumulate_kernel!(U_abs, window_sum, peak,
                                         Ex, Ey, Ez, cells, pos_x, pos_y, pos_z,
                                         Jx, Jy, Jz, npoles, pabs_factor, dt,
                                         ::Val{HAS_ADE_WORK}) where {HAS_ADE_WORK}
    n = @index(Global)
    @inbounds begin
    li = cells[n]
    ex = Float64(Ex[li])
    ey = Float64(Ey[li])
    ez = Float64(Ez[li])
    if HAS_ADE_WORK
        # The per-component ADE J arrays are indexed by their own active lists;
        # pos_* maps this all-list cell into each (0 = component inactive here),
        # exactly the reference's ade_current_sum_device.
        jx = 0.0
        jy = 0.0
        jz = 0.0
        px = pos_x[n]
        py = pos_y[n]
        pz = pos_z[n]
        for pidx in 1:npoles
            if px > 0
                jx += Float64(Jx[px, pidx])
            end
            if py > 0
                jy += Float64(Jy[py, pidx])
            end
            if pz > 0
                jz += Float64(Jz[pz, pidx])
            end
        end
        pabs = max(ex * jx + ey * jy + ez * jz, 0.0)
    else
        pabs = Float64(pabs_factor) * (ex * ex + ey * ey + ez * ez)
    end
    pdt = pabs * Float64(dt)
    U_abs[n] += pdt
    window_sum[n] += pdt
    if pabs > peak[n]
        peak[n] = pabs
    end
    end  # @inbounds
end

# 4TM+LLB advance. The absorbed power is consumed from the per-cell window
# sum accumulated by pabs_accumulate_kernel! (pabs = window_sum/dt_mp, then
# the window resets), mirroring the reference's
# kernel_pump_multiphysics_subcycled!. With HAS_PABS=false (relaxation /
# cool-down) no heating enters at all, like kernel_relax_multiphysics_fused!.
@kernel function multiphysics_kernel!(Te, Tl, Ts_TM, Ts_RE,
                                      m_TM_x, m_TM_y, m_TM_z, m_RE_x, m_RE_y, m_RE_z,
                                      window_sum, gd, tm_lut, re_lut,
                                      T_min, inv_dT, lut_N, dt, subcycles, brillouin_iters,
                                      ::Val{HAS_PABS}, ::Val{CT}) where {HAS_PABS,CT}
    n = @index(Global)
    @inbounds begin
    # The stiff 4TM+LLB ODE always integrates in Float64 (matching the reference),
    # independent of the EM compute precision CT. This also keeps the subcycle loop
    # type-stable: llb_step returns Float64, so the loop variables must be Float64 —
    # otherwise they'd switch Float32→Float64 between iterations and box on the GPU.
    if HAS_PABS
        pabs = Float64(window_sum[n]) / Float64(dt)
        window_sum[n] = 0.0
    else
        pabs = 0.0
    end

    te = Float64(Te[n])
    tl = Float64(Tl[n])
    tsT = Float64(Ts_TM[n])
    tsR = Float64(Ts_RE[n])
    mtx = Float64(m_TM_x[n])
    mty = Float64(m_TM_y[n])
    mtz = Float64(m_TM_z[n])
    mrx = Float64(m_RE_x[n])
    mry = Float64(m_RE_y[n])
    mrz = Float64(m_RE_z[n])
    sub_dt = Float64(dt) / Float64(subcycles)

    for _ in 1:subcycles
        (mtx, mty, mtz, mrx, mry, mrz, dm_TM_x, dm_RE_x, Hani_TM_x, Hani_RE_x) =
            llb_step(mtx, mty, mtz, mrx, mry, mrz, te, tsT, tsR, gd,
                     tm_lut, re_lut, T_min, inv_dT, lut_N, sub_dt, brillouin_iters)
        Q_spin_TM = gd.Ms_TM * Hani_TM_x * (dm_TM_x / sub_dt)
        Q_spin_RE = gd.Ms_RE * Hani_RE_x * (dm_RE_x / sub_dt)
        te, tl, tsT, tsR = update_4tm(te, tl, tsT, tsR, pabs, Q_spin_TM, Q_spin_RE, gd, sub_dt)
    end

    Te[n] = te
    Tl[n] = tl
    Ts_TM[n] = tsT
    Ts_RE[n] = tsR
    m_TM_x[n] = mtx
    m_TM_y[n] = mty
    m_TM_z[n] = mtz
    m_RE_x[n] = mrx
    m_RE_y[n] = mry
    m_RE_z[n] = mrz
    end  # @inbounds
end

@kernel function probe_dft_accumulate_kernel!(dft_Ey_pre, dft_Ez_pre, dft_Hy_pre, dft_Hz_pre,
                                              dft_Ey_post, dft_Ez_post, dft_Hy_post, dft_Hz_post,
                                              energy_pre, energy_post,
                                              Ey, Ez, Hy, Hz, omega_bins,
                                              ix_pre, ix_post, t, dt, nb)
    I = @index(Global, Cartesian)
    j = I[1]
    k = I[2]
    @inbounds begin
        eyp = Float64(Ey[ix_pre, j, k])
        ezp = Float64(Ez[ix_pre, j, k])
        hyp = Float64(Hy[ix_pre, j, k])
        hzp = Float64(Hz[ix_pre, j, k])
        eyq = Float64(Ey[ix_post, j, k])
        ezq = Float64(Ez[ix_post, j, k])
        hyq = Float64(Hy[ix_post, j, k])
        hzq = Float64(Hz[ix_post, j, k])
        dtt = Float64(dt)
        tt = Float64(t)
        energy_pre[j, k] += (eyp * hzp - ezp * hyp) * dtt
        energy_post[j, k] += (eyq * hzq - ezq * hyq) * dtt
        for q in 1:nb
            phase = Float64(omega_bins[q]) * tt
            c = cos(phase) * dtt
            s = -sin(phase) * dtt
            dft_Ey_pre[j, k, q] += ComplexF64(eyp * c, eyp * s)
            dft_Ez_pre[j, k, q] += ComplexF64(ezp * c, ezp * s)
            dft_Hy_pre[j, k, q] += ComplexF64(hyp * c, hyp * s)
            dft_Hz_pre[j, k, q] += ComplexF64(hzp * c, hzp * s)
            dft_Ey_post[j, k, q] += ComplexF64(eyq * c, eyq * s)
            dft_Ez_post[j, k, q] += ComplexF64(ezq * c, ezq * s)
            dft_Hy_post[j, k, q] += ComplexF64(hyq * c, hyq * s)
            dft_Hz_post[j, k, q] += ComplexF64(hzq * c, hzq * s)
        end
    end
end

@kernel function probe_trace_plane_kernel!(work_Ey_pre, work_Ez_pre, work_Ey_post, work_Ez_post,
                                           Ey, Ez, mode_w, area_yz, ix_pre, ix_post)
    I = @index(Global, Cartesian)
    j = I[1]
    k = I[2]
    @inbounds begin
        w = Float64(mode_w[j, k]) * Float64(area_yz[j, k])
        work_Ey_pre[j, k] = Float64(Ey[ix_pre, j, k]) * w
        work_Ez_pre[j, k] = Float64(Ez[ix_pre, j, k]) * w
        work_Ey_post[j, k] = Float64(Ey[ix_post, j, k]) * w
        work_Ez_post[j, k] = Float64(Ez[ix_post, j, k]) * w
    end
end

# KernelAbstractions launches issued through one backend stream are ordered; host
# synchronization is deferred until a monitor or phase boundary actually reads data.

# Reference-tuned GPU workgroup shapes. The reference launches its Yee kernels with
# threads_3d_shape = (32, 4, 2) (pump_probe_switching_empirical_params.jl:3178 default;
# the drivers don't override it). The 32-wide x-dimension matters: Julia arrays are
# column-major, so each 32-thread warp reads 32 CONSECUTIVE x-addresses — one coalesced
# transaction per array stream. An (8,8,4) shape splits a warp across 4 y-rows, fragmenting
# every stencil load into 4 scattered segments — measured ~134 vs ~92 ms/step on the
# production grid (serialized per-step profile, 2026-06-11). KA's fully dynamic default is
# worse still (flat ndrange decomposition). CPU keeps KA's dynamic sizing.
#
# !!! KNOWN ISSUE (open as of 2026-06-12): the workgroup fix is necessary but NOT
# sufficient. Later per-kernel profiling on the same grid put these KA Yee kernels at
# H 55 + E 67 ms/step vs the reference's H 37.5 + E 49.4 (see the codegen analysis in
# ext/MagnetoPhotonicCUDAExt.jl), and end-to-end production runs still log
# ~140 s per 1000 steps vs the reference's 92.7. The package-vs-reference speed gap
# is NOT closed; treat any "~90 ms/step" figure in this file's history as stale.
_workgroup(backend::AbstractBackend, shape::Tuple) = is_gpu_backend(backend) ? shape : nothing
_ka_kernel(f, dev, ::Nothing) = f(dev)
_ka_kernel(f, dev, wg::Tuple) = f(dev, wg)

# Kernel-object cache. KA's `f(dev, workgroup)` lifts the workgroup tuple into a
# StaticSize type parameter, so each construction is dynamically typed and allocates;
# rebuilding at every launch put steady garbage on the ~71k-step hot loop, and the
# periodic host GC pauses starved the GPU (visible as utilization dips). Build each
# (kernel, backend, workgroup) once and reuse. The hot loop is single-threaded, so a
# plain Dict with dynamic values is fine — the per-launch dynamic dispatch is ~ns
# against a ~90 ms Maxwell step.
const _KA_KERNEL_CACHE = Dict{Tuple{Any,Any,Any},Any}()
function _ka_kernel_cached(f, backend::AbstractBackend, wg)
    key = (f, backend, wg)
    kern = get(_KA_KERNEL_CACHE, key, nothing)
    kern === nothing || return kern
    kern = _ka_kernel(f, ka_device(backend), wg)
    _KA_KERNEL_CACHE[key] = kern
    return kern
end
const _WG_3D = (32, 4, 2)
const _WG_2D = (16, 16)
const _WG_1D = (256,)

function _ka_update_H_3d!(backend::AbstractBackend, fields::FieldState, grid::Grid3D,
                          p::FDTDParams, dt::Real, cpml, ::Type{CT};
                          inv_d_cell_x=grid.x.inv_d_cell,
                          inv_d_cell_y=grid.y.inv_d_cell,
                          inv_d_cell_z=grid.z.inv_d_cell) where {CT}
    Nx, Ny, Nz = size(fields.Ex)
    (Nx > 1 && Ny > 1 && Nz > 1) || return fields
    kernel = _ka_kernel_cached(yee_H_3d_kernel!, backend, _workgroup(backend, _WG_3D))
    kernel(fields.Ex, fields.Ey, fields.Ez, fields.Hx, fields.Hy, fields.Hz,
           inv_d_cell_x, inv_d_cell_y, inv_d_cell_z,
           cpml.x.inv_kappa, cpml.x.a, cpml.x.b,
           cpml.y.inv_kappa, cpml.y.a, cpml.y.b,
           cpml.z.inv_kappa, cpml.z.a, cpml.z.b,
           cpml.psi_Hxy_lo, cpml.psi_Hxy_hi, cpml.psi_Hxz_lo, cpml.psi_Hxz_hi,
           cpml.psi_Hyz_lo, cpml.psi_Hyz_hi, cpml.psi_Hyx_lo, cpml.psi_Hyx_hi,
           cpml.psi_Hzx_lo, cpml.psi_Hzx_hi, cpml.psi_Hzy_lo, cpml.psi_Hzy_hi,
           cpml.Npml_x, cpml.Npml_y, cpml.Npml_z, Int32(Nx), Int32(Ny), Int32(Nz),
           CT(dt) / CT(p.mu0), Val(CT); ndrange=(Nx - 1, Ny - 1, Nz - 1))
    return fields
end

function _ka_update_E_3d!(backend::AbstractBackend, fields::FieldState, grid::Grid3D,
                          p::FDTDParams, dt::Real, inv_eps_x, inv_eps_y, inv_eps_z,
                          cpml, ::Type{CT};
                          inv_d_dual_x=grid.x.inv_d_dual,
                          inv_d_dual_y=grid.y.inv_d_dual,
                          inv_d_dual_z=grid.z.inv_d_dual) where {CT}
    Nx, Ny, Nz = size(fields.Ex)
    (Nx > 1 && Ny > 1 && Nz > 1) || return fields
    kernel = _ka_kernel_cached(yee_E_3d_kernel!, backend, _workgroup(backend, _WG_3D))
    kernel(fields.Ex, fields.Ey, fields.Ez, fields.Hx, fields.Hy, fields.Hz,
           fields.Dx, fields.Dy, fields.Dz,
           inv_eps_x, inv_eps_y, inv_eps_z,
           inv_d_dual_x, inv_d_dual_y, inv_d_dual_z,
           cpml.x.inv_kappa, cpml.x.a, cpml.x.b,
           cpml.y.inv_kappa, cpml.y.a, cpml.y.b,
           cpml.z.inv_kappa, cpml.z.a, cpml.z.b,
           cpml.psi_Dxy_lo, cpml.psi_Dxy_hi, cpml.psi_Dxz_lo, cpml.psi_Dxz_hi,
           cpml.psi_Dyz_lo, cpml.psi_Dyz_hi, cpml.psi_Dyx_lo, cpml.psi_Dyx_hi,
           cpml.psi_Dzx_lo, cpml.psi_Dzx_hi, cpml.psi_Dzy_lo, cpml.psi_Dzy_hi,
           cpml.Npml_x, cpml.Npml_y, cpml.Npml_z, Int32(Nx), Int32(Ny), Int32(Nz),
           CT(dt), inv(CT(p.eps0)), Val(CT); ndrange=(Nx - 1, Ny - 1, Nz - 1))
    return fields
end

function _ka_inject_soft_3d!(backend::AbstractBackend, fields::FieldState, component::Symbol,
                             index::NTuple{3,Int}, value::Real, eps0::Real, inv_eps,
                             ::Type{CT}) where {CT}
    kernel = _ka_kernel_cached(inject_soft_3d_kernel!, backend, nothing)
    kernel(fields.Ex, fields.Ey, fields.Ez, fields.Hx, fields.Hy, fields.Hz,
           fields.Dx, fields.Dy, fields.Dz, inv_eps, _component_code(component),
           Int32(index[1]), Int32(index[2]), Int32(index[3]), CT(value), CT(eps0),
           Val(CT); ndrange=1)
    return fields
end

function _ka_inject_mode_x_3d!(backend::AbstractBackend, fields::FieldState, src::ModeSource,
                               grid::Grid3D, value::Real, eps0::Real, inv_eps,
                               ::Type{CT}) where {CT}
    src.axis === :x || throw(ArgumentError("KA ModeSource injection currently supports axis=:x"))
    src.component in (:Ey, :Ez) || throw(ArgumentError("ModeSource electric component must be :Ey or :Ez"))
    i = _source_index(grid.x, src.position)
    eta0 = sqrt(FDTDParams().mu0 / FDTDParams().eps0)
    hcorr = src.neff / eta0
    Ny, Nz = size(src.profile)
    kernel = _ka_kernel_cached(inject_mode_x_3d_kernel!, backend, _workgroup(backend, _WG_2D))
    kernel(fields.Ex, fields.Ey, fields.Ez, fields.Hx, fields.Hy, fields.Hz,
           fields.Dx, fields.Dy, fields.Dz, src.profile, inv_eps,
           _component_code(src.component), Int32(i), CT(value), CT(eps0), CT(hcorr),
           Val(CT); ndrange=(Ny, Nz))
    return fields
end

function _ka_patch_E_dispersive!(backend::AbstractBackend, E_arr, state::ADEState,
                                 active_idx, fill, inv_eps, poles::AbstractVector,
                                 dt::Real, ::Type{CT}) where {CT}
    N = length(active_idx)
    (N > 0 && length(poles) > 0) || return E_arr
    kernel = _ka_kernel_cached(ade_patch_kernel!, backend, _workgroup(backend, _WG_1D))
    kernel(E_arr, state.P, state.J, state.E_old, active_idx, fill, inv_eps,
           poles, Int32(length(poles)), CT(2) / CT(dt), Val(CT); ndrange=N)
    return E_arr
end

function _ka_patch_E_mo_gyration!(backend::AbstractBackend, Ey, Ez, state::MagnetoOpticADEState,
                                  active_idx, fill, material_pos, inv_eps_y, inv_eps_z,
                                  m_TM_x, m_RE_x, poles::AbstractVector,
                                  Q_voigt_TM::Real, Q_voigt_RE::Real, dt::Real, eps0::Real,
                                  ::Type{CT}) where {CT}
    N = length(active_idx)
    (N > 0 && length(poles) > 0) || return Ey, Ez
    kernel = _ka_kernel_cached(mo_gyration_kernel!, backend, _workgroup(backend, _WG_1D))
    kernel(Ey, Ez, state.Py_from_z, state.Jy_from_z, state.Pz_from_y, state.Jz_from_y,
           state.E_old_y, state.E_old_z, active_idx, fill, material_pos, inv_eps_y, inv_eps_z,
           m_TM_x, m_RE_x, poles, Int32(length(poles)), CT(Q_voigt_TM), CT(Q_voigt_RE),
           CT(2) / CT(dt), inv(CT(eps0)), Val(CT); ndrange=N)
    return Ey, Ez
end

# Per-EM-step launcher for the absorbed-power accumulators (every Maxwell step,
# like the reference's kernel_em_pabs_accumulate! launch).
function _ka_pabs_accumulate!(backend::AbstractBackend, absorption::AbsorptionState,
                              fields, region, model::MagnetoOpticModel, dt::Real;
                              absorption_model::Symbol=:cycle_average,
                              eps0::Real=FDTDParams().eps0,
                              ade_x=nothing, ade_y=nothing, ade_z=nothing)
    N = length(region.material_cells)
    N > 0 || return absorption
    gd = model.params
    pabs_factor = 0.5 * Float64(gd.omega_pump) * Float64(eps0) * Float64(gd.eps_imag_pump)
    has_ade_work = absorption_model === :ade_work &&
                   ade_x !== nothing && ade_y !== nothing && ade_z !== nothing &&
                   length(getfield(ade_x, :J)) > 0
    Jx = has_ade_work ? ade_x.J : absorption.U_abs
    Jy = has_ade_work ? ade_y.J : absorption.U_abs
    Jz = has_ade_work ? ade_z.J : absorption.U_abs
    npoles = has_ade_work ? size(ade_x.J, 2) : 0
    kernel = _ka_kernel_cached(pabs_accumulate_kernel!, backend, _workgroup(backend, _WG_1D))
    kernel(absorption.U_abs, absorption.window_sum, absorption.peak,
           fields.Ex, fields.Ey, fields.Ez, region.material_cells,
           region.pos_x, region.pos_y, region.pos_z,
           Jx, Jy, Jz, Int32(npoles), pabs_factor, Float64(dt),
           Val(has_ade_work); ndrange=N)
    return absorption
end

function _ka_multiphysics_step!(backend::AbstractBackend, thermal::ThermalState, mag::MagnetizationState,
                                fields, region, model::MagnetoOpticModel, lut, dt::Real;
                                subcycles::Integer=1, absorption=nothing,
                                brillouin_iters::Integer=2,
                                compute_T::Type=default_compute_type(backend), _ignored...)
    N = length(region.material_cells)
    N > 0 || return thermal
    tm_lut, re_lut, T_min, inv_dT, lut_N = lut
    gd = model.params
    has_pabs = absorption !== nothing
    window = has_pabs ? absorption.window_sum : thermal.Te
    kernel = _ka_kernel_cached(multiphysics_kernel!, backend, _workgroup(backend, _WG_1D))
    kernel(thermal.Te, thermal.Tl, thermal.Ts_TM, thermal.Ts_RE,
           mag.m_TM_x, mag.m_TM_y, mag.m_TM_z, mag.m_RE_x, mag.m_RE_y, mag.m_RE_z,
           window, gd, tm_lut, re_lut, compute_T(T_min), compute_T(inv_dT), Int32(lut_N),
           compute_T(dt), Int32(subcycles), Int32(brillouin_iters),
           Val(has_pabs), Val(compute_T); ndrange=N)
    return thermal
end

function _ka_probe_dft_accumulate!(backend::AbstractBackend, fields,
                                   dft_Ey_pre, dft_Ez_pre, dft_Hy_pre, dft_Hz_pre,
                                   dft_Ey_post, dft_Ez_post, dft_Hy_post, dft_Hz_post,
                                   energy_pre, energy_post, omega_bins,
                                   ix_pre::Integer, ix_post::Integer, t::Real, dt::Real)
    Ny, Nz = size(energy_pre)
    (Ny > 0 && Nz > 0) || return nothing
    nb = length(omega_bins)
    nb > 0 || return nothing
    kernel = _ka_kernel_cached(probe_dft_accumulate_kernel!, backend, _workgroup(backend, _WG_2D))
    kernel(dft_Ey_pre, dft_Ez_pre, dft_Hy_pre, dft_Hz_pre,
           dft_Ey_post, dft_Ez_post, dft_Hy_post, dft_Hz_post,
           energy_pre, energy_post, fields.Ey, fields.Ez, fields.Hy, fields.Hz,
           omega_bins, Int32(ix_pre), Int32(ix_post), Float64(t), Float64(dt), Int32(nb);
           ndrange=(Ny, Nz))
    return nothing
end

function _ka_probe_trace_plane!(backend::AbstractBackend,
                                work_Ey_pre, work_Ez_pre, work_Ey_post, work_Ez_post,
                                fields, mode_w, area_yz, ix_pre::Integer, ix_post::Integer)
    Ny, Nz = size(work_Ey_pre)
    (Ny > 0 && Nz > 0) || return nothing
    kernel = _ka_kernel_cached(probe_trace_plane_kernel!, backend, _workgroup(backend, _WG_2D))
    kernel(work_Ey_pre, work_Ez_pre, work_Ey_post, work_Ez_post,
           fields.Ey, fields.Ez, mode_w, area_yz, Int32(ix_pre), Int32(ix_post);
           ndrange=(Ny, Nz))
    return nothing
end
