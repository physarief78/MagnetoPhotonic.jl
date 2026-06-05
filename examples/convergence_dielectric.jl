using MagnetoPhotonic

# Grid-convergence (Cauchy L2) study for free-space pulse propagation.
cv = run_convergence_study_3D(; dxs=(50e-9, 35e-9, 22e-9), L=1.0e-6, T_max=120e-15)
println("Δx (nm)     : ", round.(cv.dxs .* 1e9, digits=1))
println("Cauchy L2   : ", round.(cv.cauchy_l2, sigdigits=4))
println("obs. order  : ", round.(cv.orders, digits=2), "   (Yee scheme ≈ 2)")
