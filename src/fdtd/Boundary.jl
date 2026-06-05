abstract type AbstractBoundary end

struct PEC <: AbstractBoundary end
struct Periodic <: AbstractBoundary end

struct PML <: AbstractBoundary
    thickness::Float64
end

PML(thickness::Integer) = PML(Float64(thickness))

_pml_cells(boundary::PML, axis::Axis1D) =
    boundary.thickness >= 1 ? Int(round(boundary.thickness)) : max(1, round(Int, boundary.thickness / axis.d_min))

_pml_cells(::PEC, axis::Axis1D) = 0
_pml_cells(::Periodic, axis::Axis1D) = 0

mutable struct CPMLState1D
    x::CPMLAxis
    psi_Hyx::Vector{Float64}
    psi_Dzx::Vector{Float64}
end

mutable struct CPMLState2D
    x::CPMLAxis
    y::CPMLAxis
    psi_Hxy::Array{Float64,2}
    psi_Hyx::Array{Float64,2}
    psi_Hzx::Array{Float64,2}
    psi_Hzy::Array{Float64,2}
    psi_Dxy::Array{Float64,2}
    psi_Dyx::Array{Float64,2}
    psi_Dzx::Array{Float64,2}
    psi_Dzy::Array{Float64,2}
end

function build_cpml(grid::Grid1D, n_pml, dt::Real, p::FDTDParams=FDTDParams(); kwargs...)
    npx = n_pml isa Integer ? n_pml : n_pml[1]
    ax = _cpml_axis(grid.x.edges, npx, Float64(dt), p; kwargs...)
    Nx = length(grid.x.centers)
    return CPMLState1D(ax, zeros(Float64, Nx), zeros(Float64, Nx))
end

function build_cpml(grid::Grid2D, n_pml, dt::Real, p::FDTDParams=FDTDParams(); kwargs...)
    npx, npy = n_pml isa Integer ? (n_pml, n_pml) : (n_pml[1], n_pml[2])
    ax = _cpml_axis(grid.x.edges, npx, Float64(dt), p; kwargs...)
    ay = _cpml_axis(grid.y.edges, npy, Float64(dt), p; kwargs...)
    Nx, Ny = length(grid.x.centers), length(grid.y.centers)
    z2() = zeros(Float64, Nx, Ny)
    return CPMLState2D(ax, ay, z2(), z2(), z2(), z2(), z2(), z2(), z2(), z2())
end

function apply_boundary!(fields::Fields1D, grid::Grid1D, boundary::AbstractBoundary, dt::Real, p::FDTDParams)
    if boundary isa PEC
        fields.Ez[1] = 0.0
        fields.Ez[end] = 0.0
        fields.Dz[1] = 0.0
        fields.Dz[end] = 0.0
    elseif boundary isa Periodic
        fields.Ez[1] = fields.Ez[end - 1]
        fields.Ez[end] = fields.Ez[2]
        fields.Dz[1] = fields.Dz[end - 1]
        fields.Dz[end] = fields.Dz[2]
        fields.Hy[1] = fields.Hy[end - 1]
        fields.Hy[end] = fields.Hy[2]
    end
    return fields
end

function apply_boundary!(fields::Fields2D, grid::Grid2D, boundary::AbstractBoundary, dt::Real, p::FDTDParams)
    if boundary isa PEC
        for arr in (fields.Ex, fields.Ey, fields.Ez, fields.Dx, fields.Dy, fields.Dz)
            arr[1, :] .= 0.0; arr[end, :] .= 0.0; arr[:, 1] .= 0.0; arr[:, end] .= 0.0
        end
    elseif boundary isa Periodic
        for arr in (fields.Ex, fields.Ey, fields.Ez, fields.Dx, fields.Dy, fields.Dz,
                    fields.Hx, fields.Hy, fields.Hz)
            arr[1, :] .= arr[end - 1, :]; arr[end, :] .= arr[2, :]
            arr[:, 1] .= arr[:, end - 1]; arr[:, end] .= arr[:, 2]
        end
    end
    return fields
end

function apply_boundary!(fields::FieldState, grid::Grid3D, boundary::AbstractBoundary, dt::Real, p::FDTDParams)
    if boundary isa PEC
        for arr in (fields.Ex, fields.Ey, fields.Ez, fields.Dx, fields.Dy, fields.Dz)
            arr[1, :, :] .= 0.0; arr[end, :, :] .= 0.0
            arr[:, 1, :] .= 0.0; arr[:, end, :] .= 0.0
            arr[:, :, 1] .= 0.0; arr[:, :, end] .= 0.0
        end
    elseif boundary isa Periodic
        for arr in (fields.Ex, fields.Ey, fields.Ez, fields.Dx, fields.Dy, fields.Dz,
                    fields.Hx, fields.Hy, fields.Hz)
            arr[1, :, :] .= arr[end - 1, :, :]; arr[end, :, :] .= arr[2, :, :]
            arr[:, 1, :] .= arr[:, end - 1, :]; arr[:, end, :] .= arr[:, 2, :]
            arr[:, :, 1] .= arr[:, :, end - 1]; arr[:, :, end] .= arr[:, :, 2]
        end
    end
    return fields
end
