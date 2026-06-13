# Bounded production-pump PERFORMANCE test — writes NO .h5 (and no movie frames).
#
# Runs the production-exact pump (4099x190x100 reference mesh, reference analytic
# mode source, reference empirical ADE poles, :f64 compute) for a fixed number of
# EM steps, then stops. It reuses the `spotcheck` tier of replicate_production.jl,
# which attaches only the cheap FilmAverage + NaNGuard monitors — NO ProbeReadout,
# NO frame writer, NO HDF5 — so the per-1000-step time reflects the bare EM hot loop
# (H / src / E / ADE / MO / pabs / 4TM-LLB), isolated from the probe-readout cost.
# Any incidental output is redirected to PackageValidation_PerfTest/ as insurance,
# so your finished production .h5 in PackageValidation/ is never touched.
#
# Usage (from MagnetoPhotonic.jl/):
#   julia --project=. examples/perf_test_prod.jl [steps]
#     steps   number of pump EM steps to run (default 6000 ≈ 6 timing blocks)
#
#   The pump dt is fixed by the reference mesh (courant=0.99), so `steps` maps to a
#   femtosecond window deterministically; the run stops after exactly `steps`.
#
# Two modes (the difference matters — read both numbers differently):
#
#   1) TRUE PRODUCTION CADENCE (default — leave PROBE_PROFILE_KERNELS unset):
#        julia --project=. examples/perf_test_prod.jl 6000
#      Prints one honest wall-clock line every 1000 steps:
#        "Phase pump Step 1000/6000 | Total Elapsed: ... | Last 1000 steps: X.XXX s"
#      X is the real s/1000 steps. The FIRST block includes one-time kernel JIT, so
#      read blocks 2..6 for steady state. Compare to the reference's 92.7 s/1000.
#
#   2) PER-KERNEL BREAKDOWN (diagnostic — set the env var first):
#        $env:PROBE_PROFILE_KERNELS = "1"
#        julia --project=. examples/perf_test_prod.jl 6000
#        Remove-Item Env:\PROBE_PROFILE_KERNELS        # turn it back off afterwards
#      Adds the [step-profile] table (H/src/E/ADE/MO/pabs/mp ms/step + host GC + free
#      VRAM). NOTE: this inserts ~7 device syncs PER STEP, which serializes the
#      pipeline and INFLATES the absolute "Last 1000 steps" time — use it only for
#      the RELATIVE split (which kernel dominates) and the GC / free-VRAM lines, not
#      for the headline s/1000 number.

const STEPS = isempty(ARGS) ? 6000 : parse(Int, ARGS[1])

# dt in femtoseconds for the reference mesh (REF_DT = 1.1218649561077176e-17 s,
# courant=0.99). `until = (STEPS - 0.5)*dt` makes _phase_steps' ceil() land on
# exactly STEPS pump steps.
const _REF_DT_FS = 1.1218649561077176e-2
ENV["SPOT_DURATION_FS"] = string((STEPS - 0.5) * _REF_DT_FS)

# Belt-and-suspenders: the spotcheck tier writes no .h5, but also redirect the output
# dir so nothing can land next to the real production data.
get!(ENV, "PACKAGE_VALIDATION_DIR",
     abspath(joinpath(@__DIR__, "..", "..", "PackageValidation_PerfTest")))

@info "perf test: production pump, NO HDF5 / NO frames" steps = STEPS profiling = get(ENV, "PROBE_PROFILE_KERNELS", "0") outdir = ENV["PACKAGE_VALIDATION_DIR"]

# replicate_production.jl reads its tier from ARGS[1] at include time; `spotcheck`
# is the bounded pump-only path (relax_steps=0, no writers).
empty!(ARGS)
push!(ARGS, "spotcheck")
include(joinpath(@__DIR__, "replicate_production.jl"))
