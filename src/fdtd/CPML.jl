struct CPMLProfiles
    kappa::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
end

struct SlabCPML
    x::CPMLProfiles
    y::CPMLProfiles
    z::CPMLProfiles
end

function build_cpml_profiles(N::Integer, N_pml::Integer, dx::Real, dt::Real, p::FDTDParams=FDTDParams(); order=3.0, reflection=1e-8, kappa_max=1.0, alpha_max=0.05)
    eta = sqrt(p.mu0 / p.eps0)
    sigma_max = -(order + 1.0) * log(reflection) / (2.0 * eta * (N_pml * Float64(dx)))
    kappa = ones(Float64, N)
    a = zeros(Float64, N)
    b = zeros(Float64, N)
    for i in 1:N
        dist = i <= N_pml ? (N_pml - i + 1.0) / N_pml : (i > N - N_pml ? (i - (N - N_pml)) / N_pml : 0.0)
        if dist > 0.0
            sigma = sigma_max * dist^order
            kappa[i] = 1.0 + (kappa_max - 1.0) * dist^order
            alpha = alpha_max * (1.0 - dist)^order
            b[i] = exp(-(sigma / (kappa[i] * p.eps0) + alpha / p.eps0) * Float64(dt))
            denom = sigma * kappa[i] + kappa[i]^2 * alpha
            a[i] = denom == 0.0 ? 0.0 : (sigma / denom) * (b[i] - 1.0) / Float64(dx)
        end
    end
    return CPMLProfiles(kappa, a, b)
end

function build_cpml_profiles_nonuniform(axis::Axis1D, N_pml::Integer, dt::Real, p::FDTDParams=FDTDParams(); order=3.0, reflection=1e-8, kappa_max=5.0, alpha_max=0.05)
    edges = axis.edges
    N = length(edges) - 1
    eta = sqrt(p.mu0 / p.eps0)
    th_lo = edges[N_pml + 1] - edges[1]
    th_hi = edges[N + 1] - edges[N - N_pml + 1]
    sigma_max_lo = -(order + 1.0) * log(reflection) / (2.0 * eta * th_lo)
    sigma_max_hi = -(order + 1.0) * log(reflection) / (2.0 * eta * th_hi)
    kappa = ones(Float64, N)
    a = zeros(Float64, N)
    b = zeros(Float64, N)

    for i in 1:N
        x_c = (edges[i] + edges[i + 1]) / 2.0
        dist = 0.0
        sigma_max = 0.0
        dx = edges[i + 1] - edges[i]
        if i <= N_pml
            dist = (edges[N_pml + 1] - x_c) / th_lo
            sigma_max = sigma_max_lo
        elseif i > N - N_pml
            dist = (x_c - edges[N - N_pml + 1]) / th_hi
            sigma_max = sigma_max_hi
        end
        if dist > 0.0
            sigma = sigma_max * dist^order
            kappa[i] = 1.0 + (kappa_max - 1.0) * dist^order
            alpha = alpha_max * (1.0 - dist)^order
            b[i] = exp(-(sigma / (kappa[i] * p.eps0) + alpha / p.eps0) * Float64(dt))
            denom = sigma * kappa[i] + kappa[i]^2 * alpha
            a[i] = denom == 0.0 ? 0.0 : (sigma / denom) * (b[i] - 1.0) / dx
        end
    end
    return CPMLProfiles(kappa, a, b)
end

# ----------------------------------------------------------------------------
# CPMLState: per-axis CFS-PML coefficients (inv_kappa, a, b) plus thin lo/hi
# convolution-memory slabs applied inside the Yee H/E updates.
#
# Coefficient convention follows the original production kernel: `a` multiplies
# the ALREADY spacing-scaled derivative (dE * inv_d), so `a` carries NO 1/dx
# factor (unlike the legacy build_cpml_profiles_* helpers above which target a
# different call site). In the bulk dist == 0 ⇒ kappa = 1, a = b = 0, so the
# update reduces exactly to plain Yee and the psi arrays stay zero.
# ----------------------------------------------------------------------------
struct CPMLAxis{A}
    inv_kappa::A
    a::A
    b::A
end

Adapt.@adapt_structure CPMLAxis

function _cpml_axis(edges::Vector{Float64}, N_pml::Integer, dt::Float64, p::FDTDParams;
                    backend::AbstractBackend=CPUBackend(), T::Type=Float64,
                    order::Float64=3.0, reflection::Float64=1e-8, kappa_max::Float64=5.0, alpha_max::Float64=0.05)
    N = length(edges) - 1
    inv_kappa = ones(T, N)
    a = zeros(T, N)
    b = zeros(T, N)
    if N_pml <= 0 || N < 2 * N_pml + 1
        return CPMLAxis(adapt_backend(backend, inv_kappa), adapt_backend(backend, a), adapt_backend(backend, b))
    end
    eta = sqrt(p.mu0 / p.eps0)
    th_lo = edges[N_pml + 1] - edges[1]
    th_hi = edges[N + 1] - edges[N - N_pml + 1]
    sigma_max_lo = -(order + 1.0) * log(reflection) / (2.0 * eta * th_lo)
    sigma_max_hi = -(order + 1.0) * log(reflection) / (2.0 * eta * th_hi)
    for i in 1:N
        x_c = (edges[i] + edges[i + 1]) / 2.0
        dist = 0.0
        sigma_max = 0.0
        if i <= N_pml
            dist = (edges[N_pml + 1] - x_c) / th_lo
            sigma_max = sigma_max_lo
        elseif i > N - N_pml
            dist = (x_c - edges[N - N_pml + 1]) / th_hi
            sigma_max = sigma_max_hi
        end
        if dist > 0.0
            sigma = sigma_max * dist^order
            kappa = 1.0 + (kappa_max - 1.0) * dist^order
            alpha = alpha_max * (1.0 - dist)^order
            inv_kappa[i] = T(1.0 / kappa)
            b[i] = T(exp(-(sigma / (kappa * p.eps0) + alpha / p.eps0) * dt))
            denom = sigma * kappa + kappa^2 * alpha
            a[i] = T(denom == 0.0 ? 0.0 : (sigma / denom) * (Float64(b[i]) - 1.0))
        end
    end
    return CPMLAxis(adapt_backend(backend, inv_kappa), adapt_backend(backend, a), adapt_backend(backend, b))
end

mutable struct CPMLState{AX,AY,AZ,A}
    x::AX
    y::AY
    z::AZ
    Npml_x::Int32
    Npml_y::Int32
    Npml_z::Int32
    # H-update convolution memory (named psi_H<component><deriv-axis>_<side>)
    psi_Hxy_lo::A; psi_Hxy_hi::A
    psi_Hxz_lo::A; psi_Hxz_hi::A
    psi_Hyz_lo::A; psi_Hyz_hi::A
    psi_Hyx_lo::A; psi_Hyx_hi::A
    psi_Hzx_lo::A; psi_Hzx_hi::A
    psi_Hzy_lo::A; psi_Hzy_hi::A
    # E/D-update convolution memory (named psi_D<component><deriv-axis>_<side>)
    psi_Dxy_lo::A; psi_Dxy_hi::A
    psi_Dxz_lo::A; psi_Dxz_hi::A
    psi_Dyz_lo::A; psi_Dyz_hi::A
    psi_Dyx_lo::A; psi_Dyx_hi::A
    psi_Dzx_lo::A; psi_Dzx_hi::A
    psi_Dzy_lo::A; psi_Dzy_hi::A
end

Adapt.@adapt_structure CPMLState

_cpml_width(N::Integer, np::Integer) = (np > 0 && N >= 2 * np + 1) ? Int(np) : 0

# `T` sets the precision of the per-axis coefficient vectors (inv_kappa/a/b), which
# feed the EM update and stay Float64 on the production path. `psi_T` sets the storage
# precision of the 24 convolution-memory SLABS, the largest CPML allocation: on the
# 4099x190x100 reference grid (PML 40/12/12) they total ~0.94 GiB in Float64 (the
# Nx×Ny×wz z-slabs alone are ~75 MB each × 8), halving to ~0.47 GiB in Float32.
#
# DEFAULT IS Float32 — settled by a controlled back-to-back A/B 2026-06-13. ψ is pure
# storage (every kernel casts to Float64 on read and writes the Float64 result back), so
# Float32 ψ leaves the convolution math in Float64 and is reference-faithful (the
# reference stores ψ in Float32). On the 6 GiB card the win is decisive: Float64 ψ pushes
# device memory to EXACTLY 0 B free, where the Windows WDDM driver thrashes residency and
# every kernel slows; Float32 ψ frees ~0.47 GiB (slabs 0.94→0.47), tipping the card off
# that cliff. Measured same-session: pump H 41.5→39.0 / E 62.0→55.6 ms (~104→96 s/1000),
# and the probe readout collapses 56→2 ms/step (its ComplexF64 DFT buffers go from
# demoted-to-PCIe back to resident). An earlier "Float32 is ~4 ms slower for the pump"
# note was a THERMAL confound across separate sessions — the controlled A/B reversed it.
# Pump spotcheck physics was bit-identical between Float32 and Float64 ψ. Override per run
# with the MP_CPML_PSI=f32|f64 env var (see _cpml_psi_T in examples/replicate_production.jl);
# pass `psi_T=Float64` here for the full-precision slabs.
function build_cpml(grid::Grid3D, n_pml, dt::Real, p::FDTDParams=FDTDParams();
                    backend::AbstractBackend=CPUBackend(), T::Type=Float64,
                    psi_T::Type=Float32, kwargs...)
    npx, npy, npz = n_pml isa Integer ? (n_pml, n_pml, n_pml) : (n_pml[1], n_pml[2], n_pml[3])
    ax = _cpml_axis(grid.x.edges, npx, Float64(dt), p; backend=backend, T=T, kwargs...)
    ay = _cpml_axis(grid.y.edges, npy, Float64(dt), p; backend=backend, T=T, kwargs...)
    az = _cpml_axis(grid.z.edges, npz, Float64(dt), p; backend=backend, T=T, kwargs...)
    Nx, Ny, Nz = length(grid.x.centers), length(grid.y.centers), length(grid.z.centers)
    wx = _cpml_width(Nx, npx)
    wy = _cpml_width(Ny, npy)
    wz = _cpml_width(Nz, npz)
    sx() = zeros_backend(backend, psi_T, wx, Ny, Nz)
    sy() = zeros_backend(backend, psi_T, Nx, wy, Nz)
    sz() = zeros_backend(backend, psi_T, Nx, Ny, wz)
    return CPMLState(ax, ay, az, Int32(wx), Int32(wy), Int32(wz),
        sy(), sy(), sz(), sz(), sz(), sz(), sx(), sx(), sx(), sx(), sy(), sy(),
        sy(), sy(), sz(), sz(), sz(), sz(), sx(), sx(), sx(), sx(), sy(), sy())
end
