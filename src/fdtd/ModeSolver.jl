function gaussian_mode_profile(Ny::Integer, Nz::Integer, dy::Real, dz::Real; waist_y=nothing, waist_z=nothing)
    wy = waist_y === nothing ? max(2 * Float64(dy), 0.25 * Ny * Float64(dy)) : Float64(waist_y)
    wz = waist_z === nothing ? max(2 * Float64(dz), 0.25 * Nz * Float64(dz)) : Float64(waist_z)
    cy = 0.5 * (Ny + 1)
    cz = 0.5 * (Nz + 1)
    profile = zeros(Float64, Ny, Nz)
    for j in 1:Ny, k in 1:Nz
        y = (j - cy) * Float64(dy)
        z = (k - cz) * Float64(dz)
        profile[j, k] = exp(-(y / wy)^2 - (z / wz)^2)
    end
    return _normalize_mode(profile, dy, dz)
end

function _normalize_mode(profile, dy::Real, dz::Real)
    norm = sqrt(sum(abs2, profile) * Float64(dy) * Float64(dz))
    norm > 0 || return profile
    return profile ./ norm
end

function _mode_operator(epsr_plane, dy::Real, dz::Real, k0::Real)
    Ny, Nz = size(epsr_plane)
    N = Ny * Nz
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    invdy2 = inv(Float64(dy)^2)
    invdz2 = inv(Float64(dz)^2)
    idx(j, k) = j + (k - 1) * Ny
    for k in 1:Nz, j in 1:Ny
        p = idx(j, k)
        push!(rows, p); push!(cols, p)
        push!(vals, Float64(k0)^2 * Float64(epsr_plane[j, k]) - 2.0 * (invdy2 + invdz2))
        if j > 1
            push!(rows, p); push!(cols, idx(j - 1, k)); push!(vals, invdy2)
        end
        if j < Ny
            push!(rows, p); push!(cols, idx(j + 1, k)); push!(vals, invdy2)
        end
        if k > 1
            push!(rows, p); push!(cols, idx(j, k - 1)); push!(vals, invdz2)
        end
        if k < Nz
            push!(rows, p); push!(cols, idx(j, k + 1)); push!(vals, invdz2)
        end
    end
    return sparse(rows, cols, vals, N, N)
end

function solve_waveguide_mode(epsr_plane, dy::Real, dz::Real, lambda0::Real;
                              neff_guess=nothing, max_iter::Integer=20, tol::Real=1e-9)
    Ny, Nz = size(epsr_plane)
    k0 = 2pi / Float64(lambda0)
    n_min = sqrt(max(minimum(Float64.(epsr_plane)), eps(Float64)))
    n_max = sqrt(maximum(Float64.(epsr_plane)))
    # Seed at the core index: the fundamental guided mode has the highest neff (just
    # below n_core), so a shift just above the spectrum top makes inverse iteration
    # converge to it reliably for any material — no hand-tuned guess needed.
    guess = neff_guess === nothing ? n_max : Float64(neff_guess)
    sigma = (k0 * guess)^2
    A = _mode_operator(epsr_plane, dy, dz, k0)
    Isp = spdiagm(0 => ones(size(A, 1)))
    v = vec(gaussian_mode_profile(Ny, Nz, dy, dz))
    beta2 = sigma
    try
        F = lu(A - sigma * Isp)
        last_beta2 = beta2
        for _ in 1:max(1, Int(max_iter))
            v = F \ v
            v ./= max(norm(v), eps(Float64))
            Av = A * v
            beta2 = real(dot(v, Av) / dot(v, v))
            abs(beta2 - last_beta2) <= Float64(tol) * max(abs(beta2), 1.0) && break
            last_beta2 = beta2
        end
        neff = sqrt(max(beta2, 0.0)) / k0
        profile = reshape(v, Ny, Nz)
        profile = _normalize_mode(profile, dy, dz)
        if !(n_min <= neff <= n_max)
            @warn "mode solve returned neff outside material-index bounds; using Gaussian fallback" neff n_min n_max
            return (profile=gaussian_mode_profile(Ny, Nz, dy, dz), neff=clamp(guess, n_min, n_max))
        end
        return (profile=profile, neff=neff)
    catch err
        @warn "mode solve failed; using Gaussian fallback" exception=(err, catch_backtrace())
        return (profile=gaussian_mode_profile(Ny, Nz, dy, dz), neff=clamp(guess, n_min, n_max))
    end
end
