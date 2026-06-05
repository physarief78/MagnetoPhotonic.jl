using MagnetoPhotonic

println("3-D pure-EM convergence through the general Simulation API")
cv = convergence_study(; dimension=3, mode=:TM, pml=true, n_pml=4, dxs=(80e-9, 60e-9, 45e-9),
                       L=0.8e-6, T_max=45e-15, nsample=200)
println("dx (nm)   = ", round.(cv.dxs .* 1e9; digits=2))
println("Cauchy L2 = ", cv.cauchy_l2)
println("orders    = ", cv.orders)
