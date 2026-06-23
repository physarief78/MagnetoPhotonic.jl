module MagnetoPhotonicCUDAExt

using MagnetoPhotonic
import Adapt
import CUDA

# Each method below is MORE SPECIFIC (::CUDABackend / ::CuArray) than the generic
# `AbstractBackend` fallbacks in core, so these ADD methods rather than overwrite
# them — safe to precompile as an extension.
MagnetoPhotonic.zeros_backend(::MagnetoPhotonic.CUDABackend, ::Type{T}, dims::Integer...) where {T} =
    CUDA.zeros(T, Int.(dims)...)

MagnetoPhotonic.ka_device(::MagnetoPhotonic.CUDABackend) = CUDA.CUDABackend()
MagnetoPhotonic.array_type(::MagnetoPhotonic.CUDABackend) = CUDA.CuArray
MagnetoPhotonic.adapt_backend(::MagnetoPhotonic.CUDABackend, x) = Adapt.adapt(CUDA.CuArray, x)
MagnetoPhotonic.to_host(x::CUDA.CuArray) = Array(x)

MagnetoPhotonic.device_memory_info(::MagnetoPhotonic.CUDABackend) =
    (free_GiB = CUDA.free_memory() / 2^30, total_GiB = CUDA.total_memory() / 2^30)

MagnetoPhotonic.reclaim_device_memory!(::MagnetoPhotonic.CUDABackend) = (CUDA.reclaim(); nothing)

MagnetoPhotonic.free_device!(x::CUDA.CuArray) = (CUDA.unsafe_free!(x); nothing)

# ======================================================================================
# Native @cuda fast path for the two Maxwell kernels (verbatim port of the reference
# kernel_update_H!/kernel_update_E! from pump_probe_switching_empirical_params.jl).
#
# Why: profiled on the production grid (4099×190×100, RTX 3050), the KernelAbstractions
# Yee kernels cost H 55 + E 67 ms/step vs the reference's H 37.5 + E 49.4 — same math,
# same (32,4,2) workgroup, same array eltypes. The gap is codegen: KA's Int64 Cartesian
# indexing + dynamic-ndrange masking, and PML term-functions evaluated for every cell.
# The reference kernels use Int32 indices and one `in_pml` branch so bulk warps (95% of
# cells) skip all psi work. Porting them verbatim should match the reference kernels by
# construction. Dispatch: these methods override _ka_update_*_3d! only for CUDABackend
# with Float64 compute (the production-validated mode); anything else falls back to the
# generic KA path via invoke. CPU and all non-Maxwell kernels remain KernelAbstractions.
#
# !!! KNOWN ISSUE (open as of 2026-06-12): this port has NOT yet been validated to
# close the end-to-end gap — production-config runs still log ~140 s per 1000 steps
# vs the reference's 92.7 s/1000. Either this fast path is not engaging in the real
# run (check the dispatch conditions actually hit: CUDABackend + Float64 compute),
# or the remaining time is outside the Yee pair (per-step launches/syncs, monitors,
# WDDM residency/paging). Re-profile with examples/perf_test_pump.jl and verify with
# an @info once at first dispatch before trusting any speedup claim.
# ======================================================================================

function _ref_yee_H!(Hx, Hy, Hz, Ex, Ey, Ez,
                     pHx_y_lo, pHx_y_hi, pHx_z_lo, pHx_z_hi,
                     pHy_x_lo, pHy_x_hi, pHy_z_lo, pHy_z_hi,
                     pHz_x_lo, pHz_x_hi, pHz_y_lo, pHz_y_hi,
                     inv_kx, inv_ky, inv_kz, ax, ay, az, bx, by, bz,
                     dt_mu0, inv_dx_cell, inv_dy_cell, inv_dz_cell,
                     Nx::Int32, Ny::Int32, Nz::Int32, N_pml_x::Int32, N_pml_y::Int32, N_pml_z::Int32)
    ix = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y + CUDA.threadIdx().y
    iz = (CUDA.blockIdx().z - Int32(1)) * CUDA.blockDim().z + CUDA.threadIdx().z

    @inbounds if ix < Nx && iy < Ny && iz < Nz
        dEz_dy = Ez[ix, iy+Int32(1), iz] - Ez[ix, iy, iz]
        dEy_dz = Ey[ix, iy, iz+Int32(1)] - Ey[ix, iy, iz]
        dEx_dz = Ex[ix, iy, iz+Int32(1)] - Ex[ix, iy, iz]
        dEz_dx = Ez[ix+Int32(1), iy, iz] - Ez[ix, iy, iz]
        dEy_dx = Ey[ix+Int32(1), iy, iz] - Ey[ix, iy, iz]
        dEx_dy = Ex[ix, iy+Int32(1), iz] - Ex[ix, iy, iz]

        in_pml = (ix <= N_pml_x) || (ix > Nx - N_pml_x) ||
                 (iy <= N_pml_y) || (iy > Ny - N_pml_y) ||
                 (iz <= N_pml_z) || (iz > Nz - N_pml_z)

        if in_pml
            FT = eltype(Hx)
            dEz_dy_dx = Float64(dEz_dy) * inv_dy_cell[iy]
            term_y = 0.0
            if iy <= N_pml_y
                psi = muladd(by[iy], Float64(pHx_y_lo[ix, iy, iz]), ay[iy] * dEz_dy_dx)
                pHx_y_lo[ix, iy, iz] = psi; term_y = psi
            elseif iy > Ny - N_pml_y
                si = iy - (Ny - N_pml_y)
                psi = muladd(by[iy], Float64(pHx_y_hi[ix, si, iz]), ay[iy] * dEz_dy_dx)
                pHx_y_hi[ix, si, iz] = psi; term_y = psi
            end

            dEy_dz_dx = Float64(dEy_dz) * inv_dz_cell[iz]
            term_z = 0.0
            if iz <= N_pml_z
                psi = muladd(bz[iz], Float64(pHx_z_lo[ix, iy, iz]), az[iz] * dEy_dz_dx)
                pHx_z_lo[ix, iy, iz] = psi; term_z = psi
            elseif iz > Nz - N_pml_z
                si = iz - (Nz - N_pml_z)
                psi = muladd(bz[iz], Float64(pHx_z_hi[ix, iy, si]), az[iz] * dEy_dz_dx)
                pHx_z_hi[ix, iy, si] = psi; term_z = psi
            end
            Hx[ix, iy, iz] = FT(Float64(Hx[ix, iy, iz]) - dt_mu0 * ((dEz_dy_dx * inv_ky[iy] - dEy_dz_dx * inv_kz[iz]) + (term_y - term_z)))

            dEx_dz_dx = Float64(dEx_dz) * inv_dz_cell[iz]
            term_z2 = 0.0
            if iz <= N_pml_z
                psi = muladd(bz[iz], Float64(pHy_z_lo[ix, iy, iz]), az[iz] * dEx_dz_dx)
                pHy_z_lo[ix, iy, iz] = psi; term_z2 = psi
            elseif iz > Nz - N_pml_z
                si = iz - (Nz - N_pml_z)
                psi = muladd(bz[iz], Float64(pHy_z_hi[ix, iy, si]), az[iz] * dEx_dz_dx)
                pHy_z_hi[ix, iy, si] = psi; term_z2 = psi
            end

            dEz_dx_dx = Float64(dEz_dx) * inv_dx_cell[ix]
            term_x = 0.0
            if ix <= N_pml_x
                psi = muladd(bx[ix], Float64(pHy_x_lo[ix, iy, iz]), ax[ix] * dEz_dx_dx)
                pHy_x_lo[ix, iy, iz] = psi; term_x = psi
            elseif ix > Nx - N_pml_x
                si = ix - (Nx - N_pml_x)
                psi = muladd(bx[ix], Float64(pHy_x_hi[si, iy, iz]), ax[ix] * dEz_dx_dx)
                pHy_x_hi[si, iy, iz] = psi; term_x = psi
            end
            Hy[ix, iy, iz] = FT(Float64(Hy[ix, iy, iz]) - dt_mu0 * ((dEx_dz_dx * inv_kz[iz] - dEz_dx_dx * inv_kx[ix]) + (term_z2 - term_x)))

            dEy_dx_dx = Float64(dEy_dx) * inv_dx_cell[ix]
            term_x2 = 0.0
            if ix <= N_pml_x
                psi = muladd(bx[ix], Float64(pHz_x_lo[ix, iy, iz]), ax[ix] * dEy_dx_dx)
                pHz_x_lo[ix, iy, iz] = psi; term_x2 = psi
            elseif ix > Nx - N_pml_x
                si = ix - (Nx - N_pml_x)
                psi = muladd(bx[ix], Float64(pHz_x_hi[si, iy, iz]), ax[ix] * dEy_dx_dx)
                pHz_x_hi[si, iy, iz] = psi; term_x2 = psi
            end

            dEx_dy_dx = Float64(dEx_dy) * inv_dy_cell[iy]
            term_y2 = 0.0
            if iy <= N_pml_y
                psi = muladd(by[iy], Float64(pHz_y_lo[ix, iy, iz]), ay[iy] * dEx_dy_dx)
                pHz_y_lo[ix, iy, iz] = psi; term_y2 = psi
            elseif iy > Ny - N_pml_y
                si = iy - (Ny - N_pml_y)
                psi = muladd(by[iy], Float64(pHz_y_hi[ix, si, iz]), ay[iy] * dEx_dy_dx)
                pHz_y_hi[ix, si, iz] = psi; term_y2 = psi
            end
            Hz[ix, iy, iz] = FT(Float64(Hz[ix, iy, iz]) - dt_mu0 * ((dEy_dx_dx * inv_kx[ix] - dEx_dy_dx * inv_ky[iy]) + (term_x2 - term_y2)))
        else
            Hx[ix, iy, iz] -= dt_mu0 * (dEz_dy * inv_dy_cell[iy] - dEy_dz * inv_dz_cell[iz])
            Hy[ix, iy, iz] -= dt_mu0 * (dEx_dz * inv_dz_cell[iz] - dEz_dx * inv_dx_cell[ix])
            Hz[ix, iy, iz] -= dt_mu0 * (dEy_dx * inv_dx_cell[ix] - dEx_dy * inv_dy_cell[iy])
        end
    end
    return nothing
end

function _ref_yee_E!(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz,
                     pEx_y_lo, pEx_y_hi, pEx_z_lo, pEx_z_hi,
                     pEy_x_lo, pEy_x_hi, pEy_z_lo, pEy_z_hi,
                     pEz_x_lo, pEz_x_hi, pEz_y_lo, pEz_y_hi,
                     inv_kx, inv_ky, inv_kz, ax, ay, az, bx, by, bz,
                     inv_eps_x, inv_eps_y, inv_eps_z,
                     dt, inv_dx_dual, inv_dy_dual, inv_dz_dual,
                     Nx::Int32, Ny::Int32, Nz::Int32, N_pml_x::Int32, N_pml_y::Int32, N_pml_z::Int32)
    # FDTDState arrays carry absolute 1/(eps0*eps_r), matching the reference.
    ix = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y + CUDA.threadIdx().y
    iz = (CUDA.blockIdx().z - Int32(1)) * CUDA.blockDim().z + CUDA.threadIdx().z

    @inbounds if Int32(1) < ix <= Nx && Int32(1) < iy <= Ny && Int32(1) < iz <= Nz
        dHz_dy = Hz[ix, iy, iz] - Hz[ix, iy-Int32(1), iz]
        dHy_dz = Hy[ix, iy, iz] - Hy[ix, iy, iz-Int32(1)]
        dHx_dz = Hx[ix, iy, iz] - Hx[ix, iy, iz-Int32(1)]
        dHz_dx = Hz[ix, iy, iz] - Hz[ix-Int32(1), iy, iz]
        dHy_dx = Hy[ix, iy, iz] - Hy[ix-Int32(1), iy, iz]
        dHx_dy = Hx[ix, iy, iz] - Hx[ix, iy-Int32(1), iz]

        in_pml = (ix <= N_pml_x) || (ix > Nx - N_pml_x) ||
                 (iy <= N_pml_y) || (iy > Ny - N_pml_y) ||
                 (iz <= N_pml_z) || (iz > Nz - N_pml_z)

        if in_pml
            FT = eltype(Dx)
            dHz_dy_dx = Float64(dHz_dy) * inv_dy_dual[iy]
            term_y = 0.0
            if iy <= N_pml_y
                psi = muladd(by[iy], Float64(pEx_y_lo[ix, iy, iz]), ay[iy] * dHz_dy_dx)
                pEx_y_lo[ix, iy, iz] = psi; term_y = psi
            elseif iy > Ny - N_pml_y
                si = iy - (Ny - N_pml_y)
                psi = muladd(by[iy], Float64(pEx_y_hi[ix, si, iz]), ay[iy] * dHz_dy_dx)
                pEx_y_hi[ix, si, iz] = psi; term_y = psi
            end

            dHy_dz_dx = Float64(dHy_dz) * inv_dz_dual[iz]
            term_z = 0.0
            if iz <= N_pml_z
                psi = muladd(bz[iz], Float64(pEx_z_lo[ix, iy, iz]), az[iz] * dHy_dz_dx)
                pEx_z_lo[ix, iy, iz] = psi; term_z = psi
            elseif iz > Nz - N_pml_z
                si = iz - (Nz - N_pml_z)
                psi = muladd(bz[iz], Float64(pEx_z_hi[ix, iy, si]), az[iz] * dHy_dz_dx)
                pEx_z_hi[ix, iy, si] = psi; term_z = psi
            end
            dx_new = Float64(Dx[ix, iy, iz]) + dt * ((dHz_dy_dx * inv_ky[iy] - dHy_dz_dx * inv_kz[iz]) + (term_y - term_z))
            Dx[ix, iy, iz] = FT(dx_new)
            Ex[ix, iy, iz] = FT(dx_new * inv_eps_x[ix, iy, iz])

            dHx_dz_dx = Float64(dHx_dz) * inv_dz_dual[iz]
            term_z2 = 0.0
            if iz <= N_pml_z
                psi = muladd(bz[iz], Float64(pEy_z_lo[ix, iy, iz]), az[iz] * dHx_dz_dx)
                pEy_z_lo[ix, iy, iz] = psi; term_z2 = psi
            elseif iz > Nz - N_pml_z
                si = iz - (Nz - N_pml_z)
                psi = muladd(bz[iz], Float64(pEy_z_hi[ix, iy, si]), az[iz] * dHx_dz_dx)
                pEy_z_hi[ix, iy, si] = psi; term_z2 = psi
            end

            dHz_dx_dx = Float64(dHz_dx) * inv_dx_dual[ix]
            term_x = 0.0
            if ix <= N_pml_x
                psi = muladd(bx[ix], Float64(pEy_x_lo[ix, iy, iz]), ax[ix] * dHz_dx_dx)
                pEy_x_lo[ix, iy, iz] = psi; term_x = psi
            elseif ix > Nx - N_pml_x
                si = ix - (Nx - N_pml_x)
                psi = muladd(bx[ix], Float64(pEy_x_hi[si, iy, iz]), ax[ix] * dHz_dx_dx)
                pEy_x_hi[si, iy, iz] = psi; term_x = psi
            end
            dy_new = Float64(Dy[ix, iy, iz]) + dt * ((dHx_dz_dx * inv_kz[iz] - dHz_dx_dx * inv_kx[ix]) + (term_z2 - term_x))
            Dy[ix, iy, iz] = FT(dy_new)
            Ey[ix, iy, iz] = FT(dy_new * inv_eps_y[ix, iy, iz])

            dHy_x_dx = Float64(dHy_dx) * inv_dx_dual[ix]
            term_x2 = 0.0
            if ix <= N_pml_x
                psi = muladd(bx[ix], Float64(pEz_x_lo[ix, iy, iz]), ax[ix] * dHy_x_dx)
                pEz_x_lo[ix, iy, iz] = psi; term_x2 = psi
            elseif ix > Nx - N_pml_x
                si = ix - (Nx - N_pml_x)
                psi = muladd(bx[ix], Float64(pEz_x_hi[si, iy, iz]), ax[ix] * dHy_x_dx)
                pEz_x_hi[si, iy, iz] = psi; term_x2 = psi
            end

            dHx_y_dx = Float64(dHx_dy) * inv_dy_dual[iy]
            term_y2 = 0.0
            if iy <= N_pml_y
                psi = muladd(by[iy], Float64(pEz_y_lo[ix, iy, iz]), ay[iy] * dHx_y_dx)
                pEz_y_lo[ix, iy, iz] = psi; term_y2 = psi
            elseif iy > Ny - N_pml_y
                si = iy - (Ny - N_pml_y)
                psi = muladd(by[iy], Float64(pEz_y_hi[ix, si, iz]), ay[iy] * dHx_y_dx)
                pEz_y_hi[ix, si, iz] = psi; term_y2 = psi
            end
            dz_new = Float64(Dz[ix, iy, iz]) + dt * ((dHy_x_dx * inv_kx[ix] - dHx_y_dx * inv_ky[iy]) + (term_x2 - term_y2))
            Dz[ix, iy, iz] = FT(dz_new)
            Ez[ix, iy, iz] = FT(dz_new * inv_eps_z[ix, iy, iz])
        else
            Dx[ix, iy, iz] += dt * (dHz_dy * inv_dy_dual[iy] - dHy_dz * inv_dz_dual[iz])
            Ex[ix, iy, iz] = Dx[ix, iy, iz] * inv_eps_x[ix, iy, iz]

            Dy[ix, iy, iz] += dt * (dHx_dz * inv_dz_dual[iz] - dHz_dx * inv_dx_dual[ix])
            Ey[ix, iy, iz] = Dy[ix, iy, iz] * inv_eps_y[ix, iy, iz]

            Dz[ix, iy, iz] += dt * (dHy_dx * inv_dx_dual[ix] - dHx_dy * inv_dy_dual[iy])
            Ez[ix, iy, iz] = Dz[ix, iy, iz] * inv_eps_z[ix, iy, iz]
        end
    end
    return nothing
end

const _REF_THREADS_3D = (32, 4, 2)   # the reference's tuned threads_3d_shape

function MagnetoPhotonic._ka_update_H_3d!(backend::MagnetoPhotonic.CUDABackend,
        fields::MagnetoPhotonic.FieldState, grid::MagnetoPhotonic.Grid3D,
        p::MagnetoPhotonic.FDTDParams, dt::Real, cpml, ::Type{CT};
        inv_d_cell_x=grid.x.inv_d_cell, inv_d_cell_y=grid.y.inv_d_cell,
        inv_d_cell_z=grid.z.inv_d_cell) where {CT}
    if CT !== Float64 || cpml === nothing
        return invoke(MagnetoPhotonic._ka_update_H_3d!,
            Tuple{MagnetoPhotonic.AbstractBackend, MagnetoPhotonic.FieldState, MagnetoPhotonic.Grid3D,
                  MagnetoPhotonic.FDTDParams, Real, Any, Type{CT}},
            backend, fields, grid, p, dt, cpml, CT;
            inv_d_cell_x=inv_d_cell_x, inv_d_cell_y=inv_d_cell_y, inv_d_cell_z=inv_d_cell_z)
    end
    Nx, Ny, Nz = size(fields.Ex)
    (Nx > 1 && Ny > 1 && Nz > 1) || return fields
    blocks = (cld(Nx, _REF_THREADS_3D[1]), cld(Ny, _REF_THREADS_3D[2]), cld(Nz, _REF_THREADS_3D[3]))
    CUDA.@cuda threads=_REF_THREADS_3D blocks=blocks _ref_yee_H!(
        fields.Hx, fields.Hy, fields.Hz, fields.Ex, fields.Ey, fields.Ez,
        cpml.psi_Hxy_lo, cpml.psi_Hxy_hi, cpml.psi_Hxz_lo, cpml.psi_Hxz_hi,
        cpml.psi_Hyx_lo, cpml.psi_Hyx_hi, cpml.psi_Hyz_lo, cpml.psi_Hyz_hi,
        cpml.psi_Hzx_lo, cpml.psi_Hzx_hi, cpml.psi_Hzy_lo, cpml.psi_Hzy_hi,
        cpml.x.inv_kappa, cpml.y.inv_kappa, cpml.z.inv_kappa,
        cpml.x.a, cpml.y.a, cpml.z.a, cpml.x.b, cpml.y.b, cpml.z.b,
        Float64(dt) / Float64(p.mu0), inv_d_cell_x, inv_d_cell_y, inv_d_cell_z,
        Int32(Nx), Int32(Ny), Int32(Nz),
        Int32(cpml.Npml_x), Int32(cpml.Npml_y), Int32(cpml.Npml_z))
    return fields
end

function MagnetoPhotonic._ka_update_E_3d!(backend::MagnetoPhotonic.CUDABackend,
        fields::MagnetoPhotonic.FieldState, grid::MagnetoPhotonic.Grid3D,
        p::MagnetoPhotonic.FDTDParams, dt::Real, inv_eps_x, inv_eps_y, inv_eps_z,
        cpml, ::Type{CT};
        inv_d_dual_x=grid.x.inv_d_dual, inv_d_dual_y=grid.y.inv_d_dual,
        inv_d_dual_z=grid.z.inv_d_dual) where {CT}
    if CT !== Float64 || cpml === nothing
        return invoke(MagnetoPhotonic._ka_update_E_3d!,
            Tuple{MagnetoPhotonic.AbstractBackend, MagnetoPhotonic.FieldState, MagnetoPhotonic.Grid3D,
                  MagnetoPhotonic.FDTDParams, Real, Any, Any, Any, Any, Type{CT}},
            backend, fields, grid, p, dt, inv_eps_x, inv_eps_y, inv_eps_z, cpml, CT;
            inv_d_dual_x=inv_d_dual_x, inv_d_dual_y=inv_d_dual_y, inv_d_dual_z=inv_d_dual_z)
    end
    Nx, Ny, Nz = size(fields.Ex)
    (Nx > 1 && Ny > 1 && Nz > 1) || return fields
    blocks = (cld(Nx, _REF_THREADS_3D[1]), cld(Ny, _REF_THREADS_3D[2]), cld(Nz, _REF_THREADS_3D[3]))
    # dt as Float32, matching the reference's dt32 argument bit-for-bit.
    CUDA.@cuda threads=_REF_THREADS_3D blocks=blocks _ref_yee_E!(
        fields.Dx, fields.Dy, fields.Dz, fields.Ex, fields.Ey, fields.Ez,
        fields.Hx, fields.Hy, fields.Hz,
        cpml.psi_Dxy_lo, cpml.psi_Dxy_hi, cpml.psi_Dxz_lo, cpml.psi_Dxz_hi,
        cpml.psi_Dyx_lo, cpml.psi_Dyx_hi, cpml.psi_Dyz_lo, cpml.psi_Dyz_hi,
        cpml.psi_Dzx_lo, cpml.psi_Dzx_hi, cpml.psi_Dzy_lo, cpml.psi_Dzy_hi,
        cpml.x.inv_kappa, cpml.y.inv_kappa, cpml.z.inv_kappa,
        cpml.x.a, cpml.y.a, cpml.z.a, cpml.x.b, cpml.y.b, cpml.z.b,
        inv_eps_x, inv_eps_y, inv_eps_z,
        Float32(dt), inv_d_dual_x, inv_d_dual_y, inv_d_dual_z,
        Int32(Nx), Int32(Ny), Int32(Nz),
        Int32(cpml.Npml_x), Int32(cpml.Npml_y), Int32(cpml.Npml_z))
    return fields
end

# Flip the core flag once, at load time, instead of overwriting `has_gpu()`.
function __init__()
    MagnetoPhotonic._GPU_FUNCTIONAL[] = CUDA.functional()
    return nothing
end

end
