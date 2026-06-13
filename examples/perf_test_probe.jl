# Performance-test driver: runs ONE production-exact 532 nm probe shot (same
# 4099x190x100 reference mesh, same :f64 compute, same reference mode source) for a
# BOUNDED step count with the per-kernel step profiler ON, then prints T/R/A. It reuses
# the `probespeed` tier of replicate_production.jl, which writes NO HDF5 output and does
# NOT reload frozen magnetization or stream movie frames — so your completed GOLDSTD .h5
# cannot be touched, and the timing isolates the readout monitor + EM step. Any
# incidental output is redirected to PackageValidation_PerfTest/.
#
# Usage:
#   julia --project=. examples/perf_test_probe.jl [steps]
#
#   steps   probe steps to simulate (default 6000 ≈ 6 profile blocks), so the readout
#           cost lines up 1:1 with examples/perf_test_pump.jl (also ~6000 steps).
#
# CPML ψ storage precision (A/B): defaults to Float32 ψ (frees ~0.47 GiB so the readout
# buffers stay resident — measured 56→2 ms/step vs Float64). To time the Float64-ψ variant,
# set the env var:
#     $env:MP_CPML_PSI = "f64"   # then run; unset (or "f32") for the default
# The "probe CPML ψ storage" log line and the post-init pool usage (≈4.80 GiB f32 vs
# ≈5.25 GiB f64) confirm which ran.
#
# What to read in the output, every 1000 steps:
#   [profile] probe per-group time — step_s vs readout_s. After the trig-hoist + fused
#             trace-reduction fixes, readout_s should be small (single-digit s/1000), so
#             the probe's Last-1000-steps wall approaches the pump's ~90 s.
#   [step-profile] bucket table — H/E should match the pump (≈37 / ≈53 ms).
#   host line   — GC pause and MB allocated (should be far below the pre-fix ~5 s/236 MB).
#   vram line   — free device memory.

const STEPS = isempty(ARGS) ? "6000" : ARGS[1]

ENV["PROBE_PROFILE_KERNELS"] = "1"
ENV["PROBE_SPEED_STEPS"] = STEPS
get!(ENV, "PACKAGE_VALIDATION_DIR",
     abspath(joinpath(@__DIR__, "..", "..", "PackageValidation_PerfTest")))

@info "perf test: production-exact probe, profiled" steps = STEPS outdir = ENV["PACKAGE_VALIDATION_DIR"]

# replicate_production.jl reads its tier from ARGS[1] at include time.
empty!(ARGS)
push!(ARGS, "probespeed")
include(joinpath(@__DIR__, "replicate_production.jl"))
