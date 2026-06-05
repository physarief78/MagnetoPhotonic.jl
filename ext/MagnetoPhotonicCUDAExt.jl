module MagnetoPhotonicCUDAExt

using MagnetoPhotonic
import CUDA

MagnetoPhotonic.zeros_backend(::MagnetoPhotonic.CUDABackend, ::Type{T}, dims::Integer...) where {T} =
    CUDA.zeros(T, Int.(dims)...)

MagnetoPhotonic.adapt_backend(::MagnetoPhotonic.CUDABackend, x) = CUDA.cu(x)

end
