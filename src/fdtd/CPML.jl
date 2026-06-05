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
# CPMLState: per-axis CFS-PML coefficients (inv_kappa, a, b) plus the twelve
# convolution memory (psi) arrays applied inside the Yee H/E updates.
#
# Coefficient convention follows the original production kernel: `a` multiplies
# the ALREADY spacing-scaled derivative (dE * inv_d), so `a` carries NO 1/dx
# factor (unlike the legacy build_cpml_profiles_* helpers above which target a
# different call site). In the bulk dist == 0 ⇒ kappa = 1, a = b = 0, so the
# update reduces exactly to plain Yee and the psi arrays stay zero.
# ----------------------------------------------------------------------------
struct CPMLAxis
    inv_kappa::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
end

function _cpml_axis(edges::Vector{Float64}, N_pml::Integer, dt::Float64, p::FDTDParams;
                    order::Float64=3.0, reflection::Float64=1e-8, kappa_max::Float64=5.0, alpha_max::Float64=0.05)
    N = length(edges) - 1
    inv_kappa = ones(Float64, N)
    a = zeros(Float64, N)
    b = zeros(Float64, N)
    if N_pml <= 0 || N < 2 * N_pml + 1
        return CPMLAxis(inv_kappa, a, b)
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
            inv_kappa[i] = 1.0 / kappa
            b[i] = exp(-(sigma / (kappa * p.eps0) + alpha / p.eps0) * dt)
            denom = sigma * kappa + kappa^2 * alpha
            a[i] = denom == 0.0 ? 0.0 : (sigma / denom) * (b[i] - 1.0)
        end
    end
    return CPMLAxis(inv_kappa, a, b)
end

mutable struct CPMLState
    x::CPMLAxis
    y::CPMLAxis
    z::CPMLAxis
    # H-update convolution memory (named psi_H<component><deriv-axis>)
    psi_Hxy::Array{Float64,3}; psi_Hxz::Array{Float64,3}
    psi_Hyz::Array{Float64,3}; psi_Hyx::Array{Float64,3}
    psi_Hzx::Array{Float64,3}; psi_Hzy::Array{Float64,3}
    # E/D-update convolution memory (named psi_D<component><deriv-axis>)
    psi_Dxy::Array{Float64,3}; psi_Dxz::Array{Float64,3}
    psi_Dyz::Array{Float64,3}; psi_Dyx::Array{Float64,3}
    psi_Dzx::Array{Float64,3}; psi_Dzy::Array{Float64,3}
end

function build_cpml(grid::Grid3D, n_pml, dt::Real, p::FDTDParams=FDTDParams(); kwargs...)
    npx, npy, npz = n_pml isa Integer ? (n_pml, n_pml, n_pml) : (n_pml[1], n_pml[2], n_pml[3])
    ax = _cpml_axis(grid.x.edges, npx, Float64(dt), p; kwargs...)
    ay = _cpml_axis(grid.y.edges, npy, Float64(dt), p; kwargs...)
    az = _cpml_axis(grid.z.edges, npz, Float64(dt), p; kwargs...)
    Nx, Ny, Nz = length(grid.x.centers), length(grid.y.centers), length(grid.z.centers)
    z3() = zeros(Float64, Nx, Ny, Nz)
    return CPMLState(ax, ay, az,
        z3(), z3(), z3(), z3(), z3(), z3(),
        z3(), z3(), z3(), z3(), z3(), z3())
end
