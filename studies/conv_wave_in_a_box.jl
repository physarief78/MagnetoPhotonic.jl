using MagnetoPhotonic

println("2-D TM reflecting-box convergence")
tm = convergence_study(; dimension=2, mode=:TM, pml=false, dxs=(50e-9, 35e-9, 22e-9), T_max=90e-15)
println("dx (nm)   = ", round.(tm.dxs .* 1e9; digits=2))
println("Cauchy L2 = ", tm.cauchy_l2)
println("orders    = ", tm.orders)

println("\n2-D TE reflecting-box convergence")
te = convergence_study(; dimension=2, mode=:TE, pml=false, dxs=(50e-9, 35e-9, 22e-9), T_max=90e-15)
println("dx (nm)   = ", round.(te.dxs .* 1e9; digits=2))
println("Cauchy L2 = ", te.cauchy_l2)
println("orders    = ", te.orders)

