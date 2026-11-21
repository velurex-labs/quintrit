// Diagnostic kernel: reports which code variant of the module the driver selected for the device.
//
// The module is built for sm_80, sm_86 and sm_89 plus PTX for newer GPUs. `__CUDA_ARCH__` is the architecture the
// loaded code was compiled for (800, 860, 890, or 890 for PTX that was compiled by the driver for a newer GPU).
#include <cstdint>

extern "C" __global__ void quintrit_probe(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef __CUDA_ARCH__
        out[0] = __CUDA_ARCH__;
#else
        out[0] = 0;
#endif
        out[1] = static_cast<int>(gridDim.x * blockDim.x);
    }
}
