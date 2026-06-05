struct FieldState{T,A}
    Ex::A
    Ey::A
    Ez::A
    Hx::A
    Hy::A
    Hz::A
    Dx::A
    Dy::A
    Dz::A
end

struct Fields1D{T,A}
    Ez::A
    Hy::A
    Dz::A
end

struct Fields2D{T,A}
    mode::Symbol
    Ex::A
    Ey::A
    Ez::A
    Hx::A
    Hy::A
    Hz::A
    Dx::A
    Dy::A
    Dz::A
end

function allocate_fields(grid::Grid1D; backend::AbstractBackend=CPUBackend(), T::Type=EM_FIELD_STORAGE_TYPE)
    N = length(grid.x.centers)
    return Fields1D{T,typeof(zeros_backend(backend, T, N))}(
        zeros_backend(backend, T, N),
        zeros_backend(backend, T, N),
        zeros_backend(backend, T, N),
    )
end

function allocate_fields(grid::Grid2D; mode::Symbol=:TM, backend::AbstractBackend=CPUBackend(), T::Type=EM_FIELD_STORAGE_TYPE)
    dims = (length(grid.x.centers), length(grid.y.centers))
    arrays = ntuple(_ -> zeros_backend(backend, T, dims...), 9)
    return Fields2D{T,typeof(arrays[1])}(mode, arrays...)
end

function allocate_fields(grid::Grid3D; backend::AbstractBackend=CPUBackend(), T::Type=EM_FIELD_STORAGE_TYPE)
    dims = (length(grid.x.centers), length(grid.y.centers), length(grid.z.centers))
    arrays = ntuple(_ -> zeros_backend(backend, T, dims...), 9)
    return FieldState{T,typeof(arrays[1])}(arrays...)
end

function field_energy(fields::Fields1D, grid::Grid1D, p::FDTDParams=FDTDParams(); area::Real=1.0)
    energy = 0.0
    @inbounds for i in eachindex(grid.x.centers)
        dx = grid.x.edges[i + 1] - grid.x.edges[i]
        energy += 0.5 * (p.eps0 * Float64(fields.Ez[i])^2 + p.mu0 * Float64(fields.Hy[i])^2) * dx * Float64(area)
    end
    return energy
end

function field_energy(fields::Fields2D, grid::Grid2D, p::FDTDParams=FDTDParams(); thickness::Real=1.0)
    energy = 0.0
    @inbounds for i in eachindex(grid.x.centers), j in eachindex(grid.y.centers)
        area = (grid.x.edges[i + 1] - grid.x.edges[i]) * (grid.y.edges[j + 1] - grid.y.edges[j]) * Float64(thickness)
        e2 = Float64(fields.Ex[i, j])^2 + Float64(fields.Ey[i, j])^2 + Float64(fields.Ez[i, j])^2
        h2 = Float64(fields.Hx[i, j])^2 + Float64(fields.Hy[i, j])^2 + Float64(fields.Hz[i, j])^2
        energy += 0.5 * (p.eps0 * e2 + p.mu0 * h2) * area
    end
    return energy
end

function field_energy(fields::FieldState, grid::Grid3D, p::FDTDParams=FDTDParams())
    energy = 0.0
    @inbounds for i in eachindex(grid.x.centers), j in eachindex(grid.y.centers), k in eachindex(grid.z.centers)
        vol = (grid.x.edges[i + 1] - grid.x.edges[i]) *
              (grid.y.edges[j + 1] - grid.y.edges[j]) *
              (grid.z.edges[k + 1] - grid.z.edges[k])
        e2 = Float64(fields.Ex[i, j, k])^2 + Float64(fields.Ey[i, j, k])^2 + Float64(fields.Ez[i, j, k])^2
        h2 = Float64(fields.Hx[i, j, k])^2 + Float64(fields.Hy[i, j, k])^2 + Float64(fields.Hz[i, j, k])^2
        energy += 0.5 * (p.eps0 * e2 + p.mu0 * h2) * vol
    end
    return energy
end
