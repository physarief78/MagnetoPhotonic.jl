function probe_jones_angles_deg(Ey::Complex, Ez::Complex)
    Sy = abs2(Ey)
    Sz = abs2(Ez)
    denom = Sy + Sz
    if denom == 0.0
        return (rotation_deg=NaN, ellipticity_deg=NaN)
    end
    s1 = (Sy - Sz) / denom
    s2 = 2.0 * real(Ey * conj(Ez)) / denom
    s3 = -2.0 * imag(Ey * conj(Ez)) / denom
    rotation = 0.5 * atan(s2, s1)
    ellipticity = 0.5 * asin(clamp(s3, -1.0, 1.0))
    return (rotation_deg=rotation * 180.0 / pi, ellipticity_deg=ellipticity * 180.0 / pi)
end

function energy_balance(; incident::Real, reflected::Real, transmitted::Real)
    inc = Float64(incident)
    R = Float64(reflected) / inc
    T = Float64(transmitted) / inc
    A = 1.0 - R - T
    return (T=T, R=R, A=A)
end
