module MagnetoPhotonicKernelAbstractionsExt

using MagnetoPhotonic
import KernelAbstractions

kernel_backend(::MagnetoPhotonic.CPUBackend) = KernelAbstractions.CPU()

end
