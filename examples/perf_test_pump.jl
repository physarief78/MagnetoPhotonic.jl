# Performance-test driver: runs the production-exact pump (same 4099x190x100 reference
# mesh, same sources, same :f64 compute) for a BOUNDED duration with the per-kernel
# step profiler ON, then prints the Te spot-check numbers. It reuses the `spotcheck`
# tier of replicate_production.jl, which writes NO HDF5 output — your completed
# production .h5 in PackageValidation/ cannot be touched by this run. As extra
# insurance, any incidental outputs are redirected to PackageValidation_PerfTest/.
#
# Usage:
#   julia --project=. examples/perf_test_pump.jl [duration_fs]
#
#   duration_fs  pump window to simulate, in femtoseconds (default 100).
#                100 fs ≈ 8,900 steps ≈ 9 profile blocks; 250 fs (the regression
#                default) additionally makes the Te@237fs physics check meaningful.
#
# CPML ψ storage precision (A/B): defaults to Float32 ψ (the validated peak config — it
# clears the 6 GiB card off 0-free-VRAM). To time the Float64-ψ variant, set the env var:
#     $env:MP_CPML_PSI = "f64"   # then run; unset (or "f32") for the default
# The "pump CPML ψ storage" log line and the post-init pool usage (≈4.80 GiB f32 vs
# ≈5.25 GiB f64) confirm which ran.
#
# What to read in the output, every 1000 steps:
#   [step-profile] bucket table — compare H/E against the reference baseline
#                  (H 37.5 / E 49.4 ms per step, profiled).
#   host line     — GC pause time and MB allocated in the window (hot loop should
#                  be near allocation-free after the kernel-object cache).
#   vram line     — free device memory (should now be > 0.5 GiB, not 0).
#
# !!! STATUS (2026-06-12): the package-vs-reference speed gap is STILL OPEN.
# Production-config runs log ~140 s per 1000 steps vs the reference's 92.7 s/1000
# (reference H5 metadata/pump_elapsed_s = 6613 s / 71310 steps). Earlier claims of
# "~90 s per 1000 steps after the WDDM keep-alive" did not reproduce. Last profiled
# KA Yee split: H 55 + E 67 ms/step (vs reference 37.5/49.4); a verbatim native
# @cuda port of the reference Yee kernels exists in ext/MagnetoPhotonicCUDAExt.jl
# but has not yet been validated to close the end-to-end gap. Use this script to
# re-measure after any change, and update this header with what you find.

const DURATION_FS = isempty(ARGS) ? "100" : ARGS[1]

ENV["PROBE_PROFILE_KERNELS"] = "1"
ENV["SPOT_DURATION_FS"] = DURATION_FS
get!(ENV, "PACKAGE_VALIDATION_DIR",
     abspath(joinpath(@__DIR__, "..", "..", "PackageValidation_PerfTest")))

@info "perf test: production-exact pump, profiled" duration_fs = DURATION_FS outdir = ENV["PACKAGE_VALIDATION_DIR"]

# replicate_production.jl reads its tier from ARGS[1] at include time.
empty!(ARGS)
push!(ARGS, "spotcheck")
include(joinpath(@__DIR__, "replicate_production.jl"))
