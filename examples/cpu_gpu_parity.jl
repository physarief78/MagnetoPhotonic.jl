# Backend / precision parity check.
#
# Runs the SAME small 3-D coupled pump experiment on the CPU and (if a CUDA device
# is present) on the GPU, at both Float32 and Float64 compute precision, and reports
# how closely the final fields and magnetization agree. This is the validation that
# the unified KernelAbstractions kernels produce the same physics on every backend.
#
#   julia --project=. examples/cpu_gpu_parity.jl
using MagnetoPhotonic
using Printf

# Load CUDA if it is available so the GPU rows light up on a CUDA machine.
try
    @eval using CUDA
catch
end

function parity_run(device, precision)
    cfg = SimConfig(
        grid   = GridConfig(xlim=(0.0, 1.6e-6), ylim=(-0.4e-6, 0.4e-6), zlim=(-0.4e-6, 0.4e-6),
                            dx=0.1e-6, courant=0.3),
        source = SourceConfig(component=:Ez, amplitude=1e2, tau=15e-15, t0=45e-15),
        device = DeviceConfig(wg_width=0.3e-6, wg_height=0.3e-6, film_thickness=0.2e-6,
                              x_film_start=0.8e-6, x_film_end=1.0e-6),
        model  = ModelConfig(multiphysics_subcycle=4, absorption_model=:ade_work),
        pml    = PMLConfig(cells=4), steps=80,
        backend = BackendConfig(device=device, compute_precision=precision),
    )
    res = run_experiment(cfg; phases=[Pump(steps=80)])
    return (Ez = Float64.(to_host(res.state.fields.Ez)),
            mTM = Float64.(to_host(res.state.mag.m_TM_x)))
end

reldiff(a, b) = maximum(abs.(a .- b)) / max(maximum(abs, a), eps())

println("Backend / precision parity — final pump-phase fields (32x8x8 cells, 80 steps):\n")
ref = parity_run(:cpu, :f64)
@printf("  %-16s max|Ez|=%.6e   (reference)\n", "cpu / Float64", maximum(abs, ref.Ez))

r32 = parity_run(:cpu, :f32)
@printf("  %-16s max|Ez|=%.6e   relΔEz=%.2e  relΔm=%.2e\n", "cpu / Float32",
        maximum(abs, r32.Ez), reldiff(ref.Ez, r32.Ez), reldiff(ref.mTM, r32.mTM))

if has_gpu()
    g32 = parity_run(:cuda, :f32)
    @printf("  %-16s max|Ez|=%.6e   relΔEz=%.2e  relΔm=%.2e\n", "cuda / Float32",
            maximum(abs, g32.Ez), reldiff(ref.Ez, g32.Ez), reldiff(ref.mTM, g32.mTM))
    g64 = parity_run(:cuda, :f64)
    @printf("  %-16s max|Ez|=%.6e   relΔEz=%.2e  relΔm=%.2e\n", "cuda / Float64",
            maximum(abs, g64.Ez), reldiff(ref.Ez, g64.Ez), reldiff(ref.mTM, g64.mTM))
else
    println("\n  CUDA device not available on this host — GPU rows skipped.")
    println("  Run on a CUDA machine (with CUDA.jl installed) to populate the cuda/* rows.")
end
