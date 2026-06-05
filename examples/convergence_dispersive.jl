using MagnetoPhotonic

# Convergence study with a dielectric slab + transmitted-spectrum readout.
cv = run_convergence_study_3D_dispersive(; dxs=(50e-9, 35e-9, 22e-9), L=1.0e-6, T_max=120e-15)
println("Δx (nm)     : ", round.(cv.dxs .* 1e9, digits=1))
println("Cauchy L2   : ", round.(cv.cauchy_l2, sigdigits=4))
println("obs. order  : ", round.(cv.orders, digits=2))
peak_THz = cv.spectrum.freq[argmax(cv.spectrum.amplitude)] / 1e12
println("spectral peak ≈ ", round(peak_THz, digits=1), " THz  (source ≈ 375 THz)")
