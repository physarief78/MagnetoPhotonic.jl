# Multiphysics coupling: EM-absorbed power -> 4TM heat -> LLB magnetization.
#
# `region` is the NamedTuple returned by `rasterize` (it supplies material_cells,
# material_fill, material_inv_eps). `lut` is (tm_lut, re_lut, T_min, inv_dT, lut_N)
# from `build_m_eq_lut`. Mirrors the production `coupled_4tm_llb_raw_v6_device`
# fused kernel: LLB advances first using the old spin temperatures, then the 4TM
# baths advance with the LLB anisotropy work fed back as Q_spin.

# Pure-thermal / no-model fallback.
multiphysics_step!(thermal::ThermalState, ::Nothing, args...; kwargs...) = thermal
multiphysics_step!(thermal::ThermalState, mag, fields, region, ::NullModel, lut, dt::Real; kwargs...) = thermal

function multiphysics_step!(thermal::ThermalState, mag::MagnetizationState, fields, region,
                            model::MagnetoOpticModel, lut, dt::Real;
                            subcycles::Integer=1, absorption_model::Symbol=:cycle_average,
                            pabs_scale::Real=1.0, brillouin_iters::Integer=2, eps0::Real=FDTDParams().eps0)
    subcycles > 0 || throw(ArgumentError("subcycles must be positive"))
    gd = model.params
    tm_lut, re_lut, T_min, inv_dT, lut_N = lut
    cells = region.material_cells
    fillv = region.material_fill
    sub_dt = Float64(dt) / subcycles
    pabs_factor = 0.5 * gd.omega_pump * Float64(eps0) * gd.eps_imag_pump * Float64(pabs_scale)

    @inbounds for n in eachindex(cells)
        li = cells[n]
        ex = Float64(fields.Ex[li]); ey = Float64(fields.Ey[li]); ez = Float64(fields.Ez[li])
        # Cycle-averaged optical absorption (W/m^3): ½ ω ε0 ε'' |E|². (The original's
        # alternative :ade_work model — positive local E·J_ADE — is not yet ported; it
        # falls back to this cycle-average form.)
        pabs = pabs_factor * (ex * ex + ey * ey + ez * ez) * fillv[n]

        te = Float64(thermal.Te[n]); tl = Float64(thermal.Tl[n])
        tsT = Float64(thermal.Ts_TM[n]); tsR = Float64(thermal.Ts_RE[n])
        mtx = mag.m_TM_x[n]; mty = mag.m_TM_y[n]; mtz = mag.m_TM_z[n]
        mrx = mag.m_RE_x[n]; mry = mag.m_RE_y[n]; mrz = mag.m_RE_z[n]

        for _ in 1:subcycles
            (mtx, mty, mtz, mrx, mry, mrz, dm_TM_x, dm_RE_x, Hani_TM_x, Hani_RE_x) =
                llb_step(mtx, mty, mtz, mrx, mry, mrz, te, tsT, tsR, gd,
                         tm_lut, re_lut, T_min, inv_dT, lut_N, sub_dt, brillouin_iters)
            Q_spin_TM = gd.Ms_TM * Hani_TM_x * (dm_TM_x / sub_dt)
            Q_spin_RE = gd.Ms_RE * Hani_RE_x * (dm_RE_x / sub_dt)
            te, tl, tsT, tsR = update_4tm(te, tl, tsT, tsR, pabs, Q_spin_TM, Q_spin_RE, gd, sub_dt)
        end

        thermal.Te[n] = te; thermal.Tl[n] = tl; thermal.Ts_TM[n] = tsT; thermal.Ts_RE[n] = tsR
        mag.m_TM_x[n] = mtx; mag.m_TM_y[n] = mty; mag.m_TM_z[n] = mtz
        mag.m_RE_x[n] = mrx; mag.m_RE_y[n] = mry; mag.m_RE_z[n] = mrz
    end
    return thermal
end
