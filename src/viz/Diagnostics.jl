using Statistics

function compute_spectrum(t, signal)
    n = length(signal)
    n == length(t) || throw(ArgumentError("t and signal must have equal length"))
    n >= 2 || throw(ArgumentError("at least two samples are required"))
    dt = mean(diff(Float64.(t)))
    half = fld(n, 2)
    freqs = collect(0:half) ./ (n * dt)
    spec = ComplexF64[]
    for k in 0:half
        acc = 0.0 + 0.0im
        for j in 0:(n - 1)
            acc += Float64(signal[j + 1]) * exp(-2im * pi * k * j / n)
        end
        push!(spec, acc)
    end
    return (freq=freqs, spectrum=spec, amplitude=abs.(spec))
end

function diagnostic_summary(values)
    v = collect(Float64, values)
    return (min=minimum(v), max=maximum(v), mean=mean(v), rms=sqrt(mean(abs2, v)))
end
