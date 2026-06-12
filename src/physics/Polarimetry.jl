# Reference convention (_probe_jones_angles_deg): rotation measured from the Ez
# axis (the probe's launch polarization), θ = ½·atan(2·Re(Ey·Ez*), |Ez|²−|Ey|²),
# ellipticity χ = ½·asin(2·Im(Ey·Ez*)/(|Ey|²+|Ez|²)). The previous package form
# measured from Ey (θ' = 90°−θ) with flipped handedness.
function probe_jones_angles_deg(Ey::Complex, Ez::Complex)
    den = abs2(Ey) + abs2(Ez)
    if !isfinite(den) || den <= 0.0
        return (rotation_deg=NaN, ellipticity_deg=NaN)
    end
    theta = 0.5 * atan(2.0 * real(Ey * conj(Ez)), abs2(Ez) - abs2(Ey))
    s3 = clamp(2.0 * imag(Ey * conj(Ez)) / den, -1.0, 1.0)
    chi = 0.5 * asin(s3)
    return (rotation_deg=theta * 180.0 / pi, ellipticity_deg=chi * 180.0 / pi)
end

function energy_balance(; incident::Real, reflected::Real, transmitted::Real)
    inc = Float64(incident)
    R = Float64(reflected) / inc
    T = Float64(transmitted) / inc
    A = 1.0 - R - T
    return (T=T, R=R, A=A)
end
