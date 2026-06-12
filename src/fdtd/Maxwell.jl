# Non-uniform Yee updates with optional CFS-CPML.
#
# When `cpml === nothing` these reduce to the plain split-field Yee scheme. When
# a CPMLState is supplied, each spatial derivative d is replaced by
#   d * inv_kappa  +  psi,   psi <- b*psi + a*d
# which absorbs at the domain boundaries. In the bulk (a=b=0, inv_kappa=1) the
# CPML branch is identical to the plain update, so a single code path suffices.

function update_H_1d!(fields::Fields1D, grid::Grid1D, p::FDTDParams, dt::Real; cpml=nothing)
    dt_mu0 = Float64(dt) / p.mu0
    @inbounds for i in 1:(length(grid.x.centers) - 1)
        dEz_dx = (Float64(fields.Ez[i + 1]) - Float64(fields.Ez[i])) * grid.x.inv_d_cell[i]
        if cpml === nothing
            fields.Hy[i] += dt_mu0 * dEz_dx
        else
            px = muladd(cpml.x.b[i], cpml.psi_Hyx[i], cpml.x.a[i] * dEz_dx)
            cpml.psi_Hyx[i] = px
            fields.Hy[i] += dt_mu0 * (dEz_dx * cpml.x.inv_kappa[i] + px)
        end
    end
    return fields
end

function update_E_1d!(fields::Fields1D, grid::Grid1D, p::FDTDParams, dt::Real, inv_eps; cpml=nothing)
    inv_eps0 = 1.0 / p.eps0
    @inbounds for i in 2:length(grid.x.centers)
        dHy_dx = (Float64(fields.Hy[i]) - Float64(fields.Hy[i - 1])) * grid.x.inv_d_dual[i]
        term = dHy_dx
        if cpml !== nothing
            px = muladd(cpml.x.b[i], cpml.psi_Dzx[i], cpml.x.a[i] * dHy_dx)
            cpml.psi_Dzx[i] = px
            term = dHy_dx * cpml.x.inv_kappa[i] + px
        end
        fields.Dz[i] += Float64(dt) * term
        fields.Ez[i] = fields.Dz[i] * Float64(inv_eps[i]) * inv_eps0
    end
    return fields
end

function update_H_2d!(fields::Fields2D, grid::Grid2D, p::FDTDParams, dt::Real; mode::Symbol=fields.mode, cpml=nothing)
    Nx, Ny = size(fields.Ez)
    dt_mu0 = Float64(dt) / p.mu0
    @inbounds if mode === :TM
        for j in 1:(Ny - 1), i in 1:(Nx - 1)
            dEz_dy = (Float64(fields.Ez[i, j + 1]) - Float64(fields.Ez[i, j])) * grid.y.inv_d_cell[j]
            dEz_dx = (Float64(fields.Ez[i + 1, j]) - Float64(fields.Ez[i, j])) * grid.x.inv_d_cell[i]
            if cpml === nothing
                fields.Hx[i, j] -= dt_mu0 * dEz_dy
                fields.Hy[i, j] += dt_mu0 * dEz_dx
            else
                py = muladd(cpml.y.b[j], cpml.psi_Hxy[i, j], cpml.y.a[j] * dEz_dy)
                px = muladd(cpml.x.b[i], cpml.psi_Hyx[i, j], cpml.x.a[i] * dEz_dx)
                cpml.psi_Hxy[i, j] = py
                cpml.psi_Hyx[i, j] = px
                fields.Hx[i, j] -= dt_mu0 * (dEz_dy * cpml.y.inv_kappa[j] + py)
                fields.Hy[i, j] += dt_mu0 * (dEz_dx * cpml.x.inv_kappa[i] + px)
            end
        end
    elseif mode === :TE
        for j in 1:(Ny - 1), i in 1:(Nx - 1)
            dEy_dx = (Float64(fields.Ey[i + 1, j]) - Float64(fields.Ey[i, j])) * grid.x.inv_d_cell[i]
            dEx_dy = (Float64(fields.Ex[i, j + 1]) - Float64(fields.Ex[i, j])) * grid.y.inv_d_cell[j]
            if cpml === nothing
                fields.Hz[i, j] -= dt_mu0 * (dEy_dx - dEx_dy)
            else
                px = muladd(cpml.x.b[i], cpml.psi_Hzx[i, j], cpml.x.a[i] * dEy_dx)
                py = muladd(cpml.y.b[j], cpml.psi_Hzy[i, j], cpml.y.a[j] * dEx_dy)
                cpml.psi_Hzx[i, j] = px
                cpml.psi_Hzy[i, j] = py
                fields.Hz[i, j] -= dt_mu0 * ((dEy_dx * cpml.x.inv_kappa[i] + px) -
                                             (dEx_dy * cpml.y.inv_kappa[j] + py))
            end
        end
    else
        throw(ArgumentError("2-D mode must be :TM or :TE"))
    end
    return fields
end

function update_E_2d!(fields::Fields2D, grid::Grid2D, p::FDTDParams, dt::Real, inv_eps; mode::Symbol=fields.mode, cpml=nothing)
    Nx, Ny = size(fields.Ez)
    inv_eps0 = 1.0 / p.eps0
    @inbounds if mode === :TM
        for j in 2:Ny, i in 2:Nx
            dHy_dx = (Float64(fields.Hy[i, j]) - Float64(fields.Hy[i - 1, j])) * grid.x.inv_d_dual[i]
            dHx_dy = (Float64(fields.Hx[i, j]) - Float64(fields.Hx[i, j - 1])) * grid.y.inv_d_dual[j]
            if cpml !== nothing
                px = muladd(cpml.x.b[i], cpml.psi_Dzx[i, j], cpml.x.a[i] * dHy_dx)
                py = muladd(cpml.y.b[j], cpml.psi_Dzy[i, j], cpml.y.a[j] * dHx_dy)
                cpml.psi_Dzx[i, j] = px
                cpml.psi_Dzy[i, j] = py
                dHy_dx = dHy_dx * cpml.x.inv_kappa[i] + px
                dHx_dy = dHx_dy * cpml.y.inv_kappa[j] + py
            end
            fields.Dz[i, j] += Float64(dt) * (dHy_dx - dHx_dy)
            fields.Ez[i, j] = fields.Dz[i, j] * Float64(inv_eps[i, j]) * inv_eps0
        end
    elseif mode === :TE
        for j in 2:Ny, i in 2:Nx
            dHz_dy = (Float64(fields.Hz[i, j]) - Float64(fields.Hz[i, j - 1])) * grid.y.inv_d_dual[j]
            dHz_dx = (Float64(fields.Hz[i, j]) - Float64(fields.Hz[i - 1, j])) * grid.x.inv_d_dual[i]
            if cpml !== nothing
                py = muladd(cpml.y.b[j], cpml.psi_Dxy[i, j], cpml.y.a[j] * dHz_dy)
                px = muladd(cpml.x.b[i], cpml.psi_Dyx[i, j], cpml.x.a[i] * dHz_dx)
                cpml.psi_Dxy[i, j] = py
                cpml.psi_Dyx[i, j] = px
                dHz_dy = dHz_dy * cpml.y.inv_kappa[j] + py
                dHz_dx = dHz_dx * cpml.x.inv_kappa[i] + px
            end
            fields.Dx[i, j] += Float64(dt) * dHz_dy
            fields.Dy[i, j] -= Float64(dt) * dHz_dx
            fields.Ex[i, j] = fields.Dx[i, j] * Float64(inv_eps[i, j]) * inv_eps0
            fields.Ey[i, j] = fields.Dy[i, j] * Float64(inv_eps[i, j]) * inv_eps0
        end
    else
        throw(ArgumentError("2-D mode must be :TM or :TE"))
    end
    return fields
end

function update_H!(fields::FieldState, grid::Grid3D, p::FDTDParams, dt::Real;
                   cpml=nothing, backend::AbstractBackend=CPUBackend(),
                   compute_T::Type=default_compute_type(backend),
                   inv_d_cell_x=grid.x.inv_d_cell,
                   inv_d_cell_y=grid.y.inv_d_cell,
                   inv_d_cell_z=grid.z.inv_d_cell)
    if cpml !== nothing
        return _ka_update_H_3d!(backend, fields, grid, p, dt, cpml, compute_T;
                                inv_d_cell_x=inv_d_cell_x,
                                inv_d_cell_y=inv_d_cell_y,
                                inv_d_cell_z=inv_d_cell_z)
    end
    Nx, Ny, Nz = size(fields.Ex)
    dt_mu0 = Float64(dt) / p.mu0
    icx, icy, icz = grid.x.inv_d_cell, grid.y.inv_d_cell, grid.z.inv_d_cell
    @inbounds for k in 1:(Nz - 1), j in 1:(Ny - 1), i in 1:(Nx - 1)
        dEz_dy = (Float64(fields.Ez[i, j + 1, k]) - Float64(fields.Ez[i, j, k])) * icy[j]
        dEy_dz = (Float64(fields.Ey[i, j, k + 1]) - Float64(fields.Ey[i, j, k])) * icz[k]
        dEx_dz = (Float64(fields.Ex[i, j, k + 1]) - Float64(fields.Ex[i, j, k])) * icz[k]
        dEz_dx = (Float64(fields.Ez[i + 1, j, k]) - Float64(fields.Ez[i, j, k])) * icx[i]
        dEy_dx = (Float64(fields.Ey[i + 1, j, k]) - Float64(fields.Ey[i, j, k])) * icx[i]
        dEx_dy = (Float64(fields.Ex[i, j + 1, k]) - Float64(fields.Ex[i, j, k])) * icy[j]

        fields.Hx[i, j, k] -= dt_mu0 * (dEz_dy - dEy_dz)
        fields.Hy[i, j, k] -= dt_mu0 * (dEx_dz - dEz_dx)
        fields.Hz[i, j, k] -= dt_mu0 * (dEy_dx - dEx_dy)
    end
    return fields
end

function update_E!(fields::FieldState, grid::Grid3D, p::FDTDParams, dt::Real, inv_eps_x, inv_eps_y, inv_eps_z;
                   cpml=nothing, backend::AbstractBackend=CPUBackend(),
                   compute_T::Type=default_compute_type(backend),
                   inv_d_dual_x=grid.x.inv_d_dual,
                   inv_d_dual_y=grid.y.inv_d_dual,
                   inv_d_dual_z=grid.z.inv_d_dual)
    if cpml !== nothing
        return _ka_update_E_3d!(backend, fields, grid, p, dt, inv_eps_x, inv_eps_y, inv_eps_z,
                                cpml, compute_T; inv_d_dual_x=inv_d_dual_x,
                                inv_d_dual_y=inv_d_dual_y,
                                inv_d_dual_z=inv_d_dual_z)
    end
    Nx, Ny, Nz = size(fields.Ex)
    dtv = Float64(dt)
    inv_eps0 = 1.0 / p.eps0
    idx, idy, idz = grid.x.inv_d_dual, grid.y.inv_d_dual, grid.z.inv_d_dual
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
        dHz_dy = (Float64(fields.Hz[i, j, k]) - Float64(fields.Hz[i, j - 1, k])) * idy[j]
        dHy_dz = (Float64(fields.Hy[i, j, k]) - Float64(fields.Hy[i, j, k - 1])) * idz[k]
        dHx_dz = (Float64(fields.Hx[i, j, k]) - Float64(fields.Hx[i, j, k - 1])) * idz[k]
        dHz_dx = (Float64(fields.Hz[i, j, k]) - Float64(fields.Hz[i - 1, j, k])) * idx[i]
        dHy_dx = (Float64(fields.Hy[i, j, k]) - Float64(fields.Hy[i - 1, j, k])) * idx[i]
        dHx_dy = (Float64(fields.Hx[i, j, k]) - Float64(fields.Hx[i, j - 1, k])) * idy[j]

        cx_term = dHz_dy - dHy_dz
        cy_term = dHx_dz - dHz_dx
        cz_term = dHy_dx - dHx_dy

        fields.Dx[i, j, k] += dtv * cx_term
        fields.Ex[i, j, k] = fields.Dx[i, j, k] * Float64(inv_eps_x[i, j, k]) * inv_eps0
        fields.Dy[i, j, k] += dtv * cy_term
        fields.Ey[i, j, k] = fields.Dy[i, j, k] * Float64(inv_eps_y[i, j, k]) * inv_eps0
        fields.Dz[i, j, k] += dtv * cz_term
        fields.Ez[i, j, k] = fields.Dz[i, j, k] * Float64(inv_eps_z[i, j, k]) * inv_eps0
    end
    return fields
end
