#include "megavit/vision_runner.hpp"

#include "megavit/rope.hpp"

#include "TFDL2_C_API.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace megavit {
namespace {

uint16_t FloatToBfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even before truncating the low 16 bits.
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float Bfloat16ToFloat(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) |
        ((mantissa + 0x1000u) >> 13));
}

float HalfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    int exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign |
                   (static_cast<uint32_t>(exponent + 127 - 15) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign |
               (static_cast<uint32_t>(exponent + 127 - 15) << 23) |
               (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> CanvasToNchw(const Canvas& canvas) {
    std::vector<float> result(
        static_cast<size_t>(canvas.width) * canvas.height * 3);
    const size_t plane = static_cast<size_t>(canvas.width) * canvas.height;
    for (int y = 0; y < canvas.height; ++y) {
        for (int x = 0; x < canvas.width; ++x) {
            const size_t source =
                (static_cast<size_t>(y) * canvas.width + x) * 3;
            const size_t spatial = static_cast<size_t>(y) * canvas.width + x;
            for (int c = 0; c < 3; ++c) {
                result[static_cast<size_t>(c) * plane + spatial] =
                    static_cast<float>(canvas.rgb[source + c]);
            }
        }
    }
    return result;
}

void CopyFloatValues(
    const std::vector<float>& source,
    TFDL_CAPI::TFTensor destination,
    bool allow_uint8) {
    using namespace TFDL_CAPI;
    if (GetTensorCount(destination, 0) != static_cast<long>(source.size())) {
        throw std::runtime_error("TFDL input shape does not match supplied values");
    }
    void* data = GetTensordata(destination);
    switch (GetTensorType(destination)) {
        case TFCAPI_FLOAT:
            std::memcpy(data, source.data(), source.size() * sizeof(float));
            break;
        case TFCAPI_UINT8: {
            if (!allow_uint8) {
                throw std::runtime_error("RoPE input was unexpectedly quantized to UINT8");
            }
            auto* out = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < source.size(); ++i) {
                out[i] = static_cast<uint8_t>(
                    std::clamp<int>(static_cast<int>(std::lround(source[i])), 0, 255));
            }
            break;
        }
        case TFCAPI_BFLOAT16: {
            auto* out = static_cast<uint16_t*>(data);
            for (size_t i = 0; i < source.size(); ++i) out[i] = FloatToBfloat16(source[i]);
            break;
        }
        case TFCAPI_FLOAT16: {
            auto* out = static_cast<uint16_t*>(data);
            for (size_t i = 0; i < source.size(); ++i) out[i] = FloatToHalf(source[i]);
            break;
        }
        default:
            throw std::runtime_error("unsupported TFDL input tensor type");
    }
}

std::vector<float> ReadFloatOutput(TFDL_CAPI::TFTensor tensor) {
    using namespace TFDL_CAPI;
    const size_t count = static_cast<size_t>(GetTensorCount(tensor, 0));
    std::vector<float> output(count);
    const void* data = GetTensordata(tensor);
    switch (GetTensorType(tensor)) {
        case TFCAPI_FLOAT:
            std::memcpy(output.data(), data, count * sizeof(float));
            break;
        case TFCAPI_BFLOAT16: {
            const auto* input = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; ++i) output[i] = Bfloat16ToFloat(input[i]);
            break;
        }
        case TFCAPI_FLOAT16: {
            const auto* input = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; ++i) output[i] = HalfToFloat(input[i]);
            break;
        }
        default:
            throw std::runtime_error(
                "Mage-ViT output must remain FLOAT/FLOAT16/BFLOAT16 at the LLM boundary");
    }
    return output;
}

}  // namespace

struct VisionRunner::Impl {
    explicit Impl(VisionRunnerConfig input)
        : config(std::move(input)),
          context(LoadContext()),
          executor(TFDL_CAPI::CompileExecutor(
              context, true,
              config.executor_json.empty()
                  ? std::string(
                        "{\"UseHardware\":true,\"FrugalMode\":true,"
                        "\"Core\":[-1],\"cpuLimit\":16,\"useCache\":true,"
                        "\"optimize\":{\"MakeAlign\":true,"
                        "\"AttnSoftmaxImpl\":true}}")
                  : config.executor_json)) {}

    TFDL_CAPI::TFContext LoadContext() {
        if (!config.addon_path.empty()) {
            if (TFDL_CAPI::RegisterCustomOpFromFile(config.addon_path) != 0) {
                throw std::runtime_error(
                    "failed to register ApplyRope addon: " + config.addon_path);
            }
        }
        return TFDL_CAPI::LoadProto(config.model_path);
    }

    VisionRunnerConfig config;
    TFDL_CAPI::TFContext context;
    TFDL_CAPI::TFExecutor executor;
};

VisionRunner::VisionRunner(VisionRunnerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
VisionRunner::~VisionRunner() = default;
VisionRunner::VisionRunner(VisionRunner&&) noexcept = default;
VisionRunner& VisionRunner::operator=(VisionRunner&&) noexcept = default;

VisionEmbeddingBatch VisionRunner::Run(const std::vector<Canvas>& canvases) {
    using namespace TFDL_CAPI;
    VisionEmbeddingBatch batch;
    batch.hidden_size = impl_->config.expected_hidden_size;
    for (const Canvas& canvas : canvases) {
        const auto inputs = GetInputTensors(impl_->executor);
        if (inputs.size() != 3) {
            throw std::runtime_error(
                "mage_vit graph must expose [raw_rgb, rope_sin, rope_cos]");
        }
        const auto raw_nchw = CanvasToNchw(canvas);
        const auto rope = BuildMageVisionRope(canvas.patch_positions);
        CopyFloatValues(raw_nchw, inputs[0], true);
        CopyFloatValues(rope.sin, inputs[1], false);
        CopyFloatValues(rope.cos, inputs[2], false);

        ForwardExecutorAlone(impl_->executor);
        const auto outputs = GetOutputTensors(impl_->executor);
        if (outputs.size() != 1) {
            throw std::runtime_error("mage_vit graph must expose one embedding output");
        }
        auto values = ReadFloatOutput(outputs[0]);
        if (values.size() % static_cast<size_t>(batch.hidden_size) != 0) {
            throw std::runtime_error("Mage-ViT output width is not 2560");
        }
        const int tokens = static_cast<int>(
            values.size() / static_cast<size_t>(batch.hidden_size));
        const int expected = static_cast<int>(canvas.patch_positions.size() / 4);
        if (tokens != expected) {
            throw std::runtime_error(
                "Mage-ViT output token count does not match 2x2 patch merger");
        }
        batch.tokens_per_canvas.push_back(tokens);
        batch.values.insert(batch.values.end(), values.begin(), values.end());
    }
    return batch;
}

void WriteFloatEmbeddings(
    const std::string& path,
    const VisionEmbeddingBatch& embeddings) {
    if (embeddings.values.size() !=
        embeddings.token_count() * static_cast<size_t>(embeddings.hidden_size)) {
        throw std::invalid_argument("embedding data size does not match metadata");
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create " + path);
    out.write(
        reinterpret_cast<const char*>(embeddings.values.data()),
        static_cast<std::streamsize>(embeddings.values.size() * sizeof(float)));
}

}  // namespace megavit
