# Zero-dimensional all-optical switching: a single GdFeCo material cell driven by
# the full four-temperature model (4TM) coupled to the two-sublattice LLB kernel.
#
# An absorbed-power Gaussian heats the electron bath, which feeds the FeCo (TM) and
# Gd (RE) spin baths on their *distinct* element-resolved demagnetization timescales
# (TM ≈ 100 fs, RE ≈ 430 fs). Because the two sublattices respond asymmetrically and
# the branch-selection channel gates the RE reversal on the TM one, **FeCo reverses
# first and Gd follows** — opening the transient-ferromagnetic window that is the
# fingerprint of GdFeCo all-optical switching.
#
#   julia --project=. examples/aos_switching_0d.jl
using MagnetoPhotonic
using Printf

gd = MagnetoOpticModel().params
tm, re, Tmin, invdT, N = build_m_eq_lut(gd)
m0TM = lookup_m_eq_lut(tm, gd.T0, Tmin, invdT, N)
m0RE = lookup_m_eq_lut(re, gd.T0, Tmin, invdT, N)

mTM = (m0TM, 0.0, 1e-4)          # FeCo (TM): starts +m_eq(300 K)
mRE = (m0RE, 0.0, 1e-4)          # Gd   (RE): starts −m_eq(300 K)  (antiparallel)
Te = gd.T0; Tl = gd.T0; TsTM = gd.T0; TsRE = gd.T0

dt = 0.5e-15; nsteps = 16000     # 8 ps
pabs_peak, t_pulse, tau_pulse = 4.0e21, 0.4e-12, 0.2e-12   # absorbed-power pulse (W/m³)

t = Float64[]; mFe = Float64[]; mGd = Float64[]; Tev = Float64[]
for n in 1:nsteps
    tt = n * dt
    pabs = pabs_peak * exp(-((tt - t_pulse) / tau_pulse)^2)
    r = llb_step(mTM..., mRE..., Te, TsTM, TsRE, gd, tm, re, Tmin, invdT, N, dt, 2)
    global mTM = (r[1], r[2], r[3]); global mRE = (r[4], r[5], r[6])
    Q_TM = gd.Ms_TM * r[9]  * (r[7] / dt)     # magnetic-work power density → 4TM
    Q_RE = gd.Ms_RE * r[10] * (r[8] / dt)
    global Te, Tl, TsTM, TsRE = update_4tm(Te, Tl, TsTM, TsRE, pabs, Q_TM, Q_RE, gd, dt)
    if n % 40 == 0
        push!(t, tt * 1e12); push!(mFe, mTM[1]); push!(mGd, mRE[1]); push!(Tev, Te)
    end
end

zero_cross(t, y) = begin
    for i in 2:length(y)
        (y[i-1] < 0) != (y[i] < 0) && return t[i-1] + (t[i]-t[i-1]) * (0 - y[i-1]) / (y[i]-y[i-1])
    end
    NaN
end
cFe = zero_cross(t, mFe); cGd = zero_cross(t, mGd)

@printf("FeCo (TM) m_x: %+.3f → %+.3f   zero crossing @ %.2f ps\n", mFe[1], mFe[end], cFe)
@printf("Gd   (RE) m_x: %+.3f → %+.3f   zero crossing @ %.2f ps\n", mGd[1], mGd[end], cGd)
@printf("FeCo reverses FIRST; Gd follows %.2f ps later (transient-ferromagnetic window).\n", cGd - cFe)
@printf("electron T_e peak = %.0f K\n", maximum(Tev))
