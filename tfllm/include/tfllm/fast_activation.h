#pragma once
#include <cstddef>
#include <cstdint>

namespace tfllm::cpu {
// Out-of-place (or exactly aliased), CPU-only inference kernels. FP16 inputs
// compute in FP32 and narrow once; FP32 has no FP16 boundary. Independent of
// the legacy serving activation/LUT rounding conventions.
// Values are the additive Torch ABI IDs; never renumber existing entries.
enum class FastActivation {GeluErf=0, GeluTanh=1, Silu=2, Sigmoid=3, Tanh=4,
                           Mish=5, Softplus=6, Elu=7, Selu=8};
void FastActivate(const uint16_t*,uint16_t*,size_t,FastActivation);
void FastActivate(const float*,float*,size_t,FastActivation);
}
