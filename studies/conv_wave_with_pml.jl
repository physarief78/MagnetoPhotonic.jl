using MagnetoPhotonic

println("2-D TM PML convergence")
cv = convergence_study(; dimension=2, mode=:TM, pml=true, n_pml=8, dxs=(50e-9, 35e-9, 22e-9), T_max=90e-15)
println("dx (nm)      = ", round.(cv.dxs .* 1e9; digits=2))
println("Cauchy L2    = ", cv.cauchy_l2)
println("orders       = ", cv.orders)
println("final energy = ", cv.final_energy)

println("\n2-D TM dispersive slab with PML")
cvd = convergence_study(; dimension=2, mode=:TM, dispersive=true, pml=true, n_pml=8,
                        dxs=(50e-9, 35e-9, 22e-9), T_max=90e-15)
peak = cvd.spectrum.freq[argmax(cvd.spectrum.amplitude)]
println("Cauchy L2    = ", cvd.cauchy_l2)
println("spectrum peak THz = ", round(peak / 1e12; digits=2))

