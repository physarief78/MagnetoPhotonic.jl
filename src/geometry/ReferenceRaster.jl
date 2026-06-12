# Verbatim port of the reference production geometry builder
# (pump_probe_switching_empirical_params.jl `build_simulation_geometry`):
# Yee-STAGGERED permittivity volumes (separate inv_eps_x/y/z evaluated at each E
# component's staggered position) and PER-COMPONENT dispersive active-cell lists
# (act_idx_x/y/z with their own fills and 1/(ε0·ε) factors), plus the union list
# act_idx_all = unique(vcat(x, y, z)) in the reference's first-occurrence order.
#
# Conventions (must match both codes):
#  - Ex samples at (x_center, y_center, z_edge); Ey at (x_edge, y_center+dy/2,
#    z_edge); Ez at (x_edge, y_center, z_center) — exactly the reference's
#    cx_*/cy_*/cz_* choices, with cell-sized sub_n×sub_n averaging windows.
#  - Inside the film x-band the waveguide fraction becomes the DISPERSIVE film
#    (base ε = vacuum, ADE supplies the response); outside it is Si3N4 core.
#  - The substrate (z < 0) is SiO2 everywhere.
#  - act_inv_* = 1/(ε0·ε_component) — the screening factor consumed by the ADE
#    patch denominator (the reference's act_inv convention).
#  - The returned inv_eps_x/y/z volumes use the PACKAGE Yee convention 1/ε_r
#    (the E-update kernel multiplies by 1/ε0 separately) so the Maxwell update
#    is unchanged; only the staggered values differ.

# The reference rounds its polygons through Point2f (Float32) before the
# point-in-polygon tests; reproduce that rounding for bit-equal fills.
_poly_f32(poly) = [Vec2(Float64(Float32(p.x)), Float64(Float32(p.y))) for p in poly]

# NOT-gate waveguide polygons in µm, exactly as the reference builds them.
function not_gate_reference_polygons(; wg_width::Real=0.40)
    x_in, x_bend, x_merge, x_taper, x_out = 0.0, 5.0, 25.0, 30.0, 60.0
    y_top, y_bot = 0.75, -0.75
    y_merge = wg_width / 2.0
    path_top = [Vec2(x_in, y_top), Vec2(x_bend, y_top)]
    path_bot = [Vec2(x_in, y_bot), Vec2(x_bend, y_bot)]
    bend_resolution = 60
    for i in 1:bend_resolution
        t = i / bend_resolution
        x = x_bend + t * (x_merge - x_bend)
        push!(path_top, Vec2(x, y_merge + (y_top - y_merge) * 0.5 * (1.0 + cos(pi * t))))
        push!(path_bot, Vec2(x, -y_merge + (y_bot - (-y_merge)) * 0.5 * (1.0 + cos(pi * t))))
    end
    path_taper = [Vec2(x_merge, 0.0), Vec2(x_taper, 0.0)]
    path_out = [Vec2(x_taper, 0.0), Vec2(x_out, 0.0)]
    return [
        _poly_f32(generate_waveguide_polygon(path_top, wg_width)),
        _poly_f32(generate_waveguide_polygon(path_bot, wg_width)),
        _poly_f32(generate_tapered_polygon(path_taper, wg_width * 2.0, wg_width)),
        _poly_f32(generate_waveguide_polygon(path_out, wg_width)),
    ]
end

function not_gate_reference_geometry(grid::Grid3D, p::FDTDParams;
        wg_width::Real=0.40, wg_height::Real=0.40,
        x_film_start::Real=40.0e-6, x_film_end::Real=40.008e-6,
        film_disabled::Bool=false, sub_n::Integer=8)
    polys = not_gate_reference_polygons(wg_width=wg_width)
    bboxes = [get_bbox(poly) for poly in polys]
    Nx = length(grid.x.centers)
    Ny = length(grid.y.centers)
    Nz = length(grid.z.centers)
    wg_height_phys = Float64(wg_height) * 1e-6
    xfs, xfe = Float64(x_film_start), Float64(x_film_end)
    if film_disabled
        # Bare waveguide (gold-standard probe normalization): push the dispersive
        # band out of the domain so every waveguide cell is continuous core.
        xfs, xfe = -2.0e-6, -1.0e-6
    end

    # --- 2-D staggered waveguide fills (polygon test in µm) ---
    fill_x_2d = zeros(Float64, Nx, Ny)
    fill_y_2d = zeros(Float64, Nx, Ny)
    fill_z_2d = zeros(Float64, Nx, Ny)
    nsub2 = sub_n * sub_n
    Threads.@threads for j in 1:Ny
        dy_cell = grid.y.edges[j + 1] - grid.y.edges[j]
        sub_dy = dy_cell / sub_n
        @inbounds for i in 1:Nx
            dx_local = grid.x.edges[i + 1] - grid.x.edges[i]
            cx_Ez = grid.x.edges[i]
            cy_Ez = grid.y.centers[j]
            cx_Ex, cy_Ex = grid.x.centers[i], cy_Ez
            cx_Ey, cy_Ey = cx_Ez, cy_Ez + dy_cell / 2
            fx = fy = fz = 0.0
            sub_dx = dx_local / sub_n
            for sy in 1:sub_n, sx in 1:sub_n
                dxs = (sx - 0.5) * sub_dx - dx_local / 2
                dys = (sy - 0.5) * sub_dy - dy_cell / 2
                if is_inside_any((cx_Ez + dxs) * 1e6, (cy_Ez + dys) * 1e6, polys, bboxes); fz += 1.0; end
                if is_inside_any((cx_Ex + dxs) * 1e6, (cy_Ex + dys) * 1e6, polys, bboxes); fx += 1.0; end
                if is_inside_any((cx_Ey + dxs) * 1e6, (cy_Ey + dys) * 1e6, polys, bboxes); fy += 1.0; end
            end
            fill_z_2d[i, j] = fz / nsub2
            fill_x_2d[i, j] = fx / nsub2
            fill_y_2d[i, j] = fy / nsub2
        end
    end

    # --- per-k z fills + combine into staggered ε and per-component act lists ---
    eps_x = fill(Float64(p.epsr_vac), Nx, Ny, Nz)
    eps_y = fill(Float64(p.epsr_vac), Nx, Ny, Nz)
    eps_z = fill(Float64(p.epsr_vac), Nx, Ny, Nz)
    act_idx_x_th = [Int[] for _ in 1:Nz]; act_f_x_th = [Float64[] for _ in 1:Nz]; act_inv_x_th = [Float64[] for _ in 1:Nz]
    act_idx_y_th = [Int[] for _ in 1:Nz]; act_f_y_th = [Float64[] for _ in 1:Nz]; act_inv_y_th = [Float64[] for _ in 1:Nz]
    act_idx_z_th = [Int[] for _ in 1:Nz]; act_f_z_th = [Float64[] for _ in 1:Nz]; act_inv_z_th = [Float64[] for _ in 1:Nz]
    act_vol_th = [Dict{Int,Float64}() for _ in 1:Nz]
    lin_idx = LinearIndices((Nx, Ny, Nz))
    eps0 = Float64(p.eps0)
    er_si = Float64(p.epsr_si3n4)
    er_ox = Float64(p.epsr_sio2)
    er_vac = Float64(p.epsr_vac)

    Threads.@threads for k in 1:Nz
        dz_cell = grid.z.edges[k + 1] - grid.z.edges[k]
        sub_dz = dz_cell / sub_n
        cz_Exy = grid.z.edges[k]
        cz_Ez = grid.z.centers[k]

        fill_z_Exy_cell = 0.0; fill_sub_Exy_cell = 0.0
        for sz in 1:sub_n
            z_val = cz_Exy + (sz - 0.5) * sub_dz - dz_cell / 2
            if 0.0 <= z_val <= wg_height_phys; fill_z_Exy_cell += 1.0; end
            if z_val < 0.0; fill_sub_Exy_cell += 1.0; end
        end
        fill_z_Exy_cell /= sub_n; fill_sub_Exy_cell /= sub_n

        fill_z_Ez_cell = 0.0; fill_sub_Ez_cell = 0.0
        for sz in 1:sub_n
            z_val = cz_Ez + (sz - 0.5) * sub_dz - dz_cell / 2
            if 0.0 <= z_val <= wg_height_phys; fill_z_Ez_cell += 1.0; end
            if z_val < 0.0; fill_sub_Ez_cell += 1.0; end
        end
        fill_z_Ez_cell /= sub_n; fill_sub_Ez_cell /= sub_n

        @inbounds for j in 1:Ny
            for i in 1:Nx
                fill_wg_x = fill_x_2d[i, j] * fill_z_Exy_cell
                fill_wg_y = fill_y_2d[i, j] * fill_z_Exy_cell
                fill_wg_z = fill_z_2d[i, j] * fill_z_Ez_cell

                x_Ex, x_Ey, x_Ez = grid.x.centers[i], grid.x.edges[i], grid.x.edges[i]
                in_disp_x = (xfs <= x_Ex <= xfe)
                in_disp_y = (xfs <= x_Ey <= xfe)
                in_disp_z = (xfs <= x_Ez <= xfe)

                f_val_x = in_disp_x ? fill_wg_x : 0.0; f_si_x = in_disp_x ? 0.0 : fill_wg_x
                f_val_y = in_disp_y ? fill_wg_y : 0.0; f_si_y = in_disp_y ? 0.0 : fill_wg_y
                f_val_z = in_disp_z ? fill_wg_z : 0.0; f_si_z = in_disp_z ? 0.0 : fill_wg_z

                e_x = f_si_x * er_si + fill_sub_Exy_cell * er_ox + (1.0 - f_si_x - fill_sub_Exy_cell) * er_vac
                e_y = f_si_y * er_si + fill_sub_Exy_cell * er_ox + (1.0 - f_si_y - fill_sub_Exy_cell) * er_vac
                e_z = f_si_z * er_si + fill_sub_Ez_cell * er_ox + (1.0 - f_si_z - fill_sub_Ez_cell) * er_vac
                eps_x[i, j, k] = e_x
                eps_y[i, j, k] = e_y
                eps_z[i, j, k] = e_z

                inv_e_x = 1.0 / (eps0 * e_x)
                inv_e_y = 1.0 / (eps0 * e_y)
                inv_e_z = 1.0 / (eps0 * e_z)

                if f_val_x > 0.0
                    push!(act_idx_x_th[k], lin_idx[i, j, k]); push!(act_f_x_th[k], f_val_x); push!(act_inv_x_th[k], inv_e_x)
                end
                if f_val_y > 0.0
                    push!(act_idx_y_th[k], lin_idx[i, j, k]); push!(act_f_y_th[k], f_val_y); push!(act_inv_y_th[k], inv_e_y)
                end
                if f_val_z > 0.0
                    push!(act_idx_z_th[k], lin_idx[i, j, k]); push!(act_f_z_th[k], f_val_z); push!(act_inv_z_th[k], inv_e_z)
                end
                if f_val_x > 0.0 || f_val_y > 0.0 || f_val_z > 0.0
                    dx_local = grid.x.edges[i + 1] - grid.x.edges[i]
                    dy_local = grid.y.edges[j + 1] - grid.y.edges[j]
                    act_vol_th[k][lin_idx[i, j, k]] = dx_local * dy_local * dz_cell
                end
            end
        end
    end

    act_idx_x = reduce(vcat, act_idx_x_th); act_f_x = reduce(vcat, act_f_x_th); act_inv_x = reduce(vcat, act_inv_x_th)
    act_idx_y = reduce(vcat, act_idx_y_th); act_f_y = reduce(vcat, act_f_y_th); act_inv_y = reduce(vcat, act_inv_y_th)
    act_idx_z = reduce(vcat, act_idx_z_th); act_f_z = reduce(vcat, act_f_z_th); act_inv_z = reduce(vcat, act_inv_z_th)

    # Union in the reference's first-occurrence order (all x cells, then new y, then
    # new z) — this also reproduces the reference file's active_linear_index order.
    act_idx_all = unique(vcat(act_idx_x, act_idx_y, act_idx_z))
    vol_map = Dict{Int,Float64}()
    for k in 1:Nz
        merge!(vol_map, act_vol_th[k])
    end
    V_cell_all = Float64[get(vol_map, idx, 1.0) for idx in act_idx_all]

    # All-list max staggered fill (legacy compatibility: writers/diagnostics only;
    # the solver consumes the per-component fills).
    fmax = Dict{Int,Float64}()
    for (idxs, fs) in ((act_idx_x, act_f_x), (act_idx_y, act_f_y), (act_idx_z, act_f_z))
        for (li, f) in zip(idxs, fs)
            fmax[li] = max(get(fmax, li, 0.0), f)
        end
    end
    material_fill = Float64[fmax[li] for li in act_idx_all]

    return (;
        # package Yee convention: 1/ε_r per component (ε0 applied in the E-update)
        epsr=eps_z,
        inv_eps_x=1.0 ./ eps_x,
        inv_eps_y=1.0 ./ eps_y,
        inv_eps_z=1.0 ./ eps_z,
        # collocated all-list (magnetization/thermal/absorption state indexing)
        material_cells=act_idx_all,
        material_fill,
        material_inv_eps=Float64[1.0 / (eps0 * eps_z[li]) for li in act_idx_all],
        n_material=length(act_idx_all),
        # reference per-component dispersive lists (ADE patch); inv = 1/(ε0·ε)
        act_idx_x, act_f_x, act_inv_x,
        act_idx_y, act_f_y, act_inv_y,
        act_idx_z, act_f_z, act_inv_z,
        V_cell_all,
        wg_height_phys,
    )
end

# Reference helpers: position of each `target` entry inside an axis list (0 = absent).
function active_axis_position_map(act_idx_all, act_idx_axis)
    pos = Dict{Int,Int32}()
    for (i, idx) in enumerate(act_idx_axis)
        pos[Int(idx)] = Int32(i)
    end
    return Int32[get(pos, Int(idx), Int32(0)) for idx in act_idx_all]
end
