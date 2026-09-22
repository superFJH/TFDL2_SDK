#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tfllm {
constexpr uint64_t MaxModelParameters = 30000000000ULL;
size_t CheckedMul(size_t a, size_t b);
size_t RoundUp(size_t value, size_t alignment);
uint16_t ToHalf(float value);
float FromHalf(uint16_t value);

struct Config {
    std::string architecture = "qwen3", model_id;
    uint32_t layers = 0, hidden = 0, intermediate = 0;
    uint32_t query_heads = 0, kv_heads = 0, head_dim = 0, vocabulary = 0, context = 0;
    float rms_epsilon = 1e-6f, rope_theta = 1e6f;
    std::string execution = "native", source_gguf, fallback_reason;
    // Optional paired CPU GGUF generated from the same source as this package.
    // model_id continues to identify the original, unrequantized source.
    std::string decode_model_id;
    uint64_t parameter_count = 0;
    bool rope_neox = true, qk_norm = true, gelu = false;
    float embedding_scale = 1, attention_scale = 0;
    float attention_soft_cap = 0, logits_soft_cap = 0;
    float rope_freq_scale = 1, rope_theta_swa = 10000, rope_freq_scale_swa = 1;
    std::vector<uint32_t> sliding_windows; // Empty: dense; otherwise one entry per layer.
    // qwen35: one flag per trunk layer; MTP is exported separately.
    std::vector<uint8_t> recurrent_layers;
    uint32_t rotary_dim=0, linear_key_heads=0, linear_value_heads=0;
    uint32_t linear_key_dim=0, linear_value_dim=0, linear_conv_kernel=0;
    bool IsRecurrent(size_t layer) const {return !recurrent_layers.empty() && recurrent_layers.at(layer)!=0;}
    void Validate() const;
    size_t FirstKey(size_t layer, size_t visible) const;
    float AttentionScore(float dot) const;
};
enum class Encoding : uint8_t { F32, F16, U8RowSymmetric };
// Runtime-only, immutable backend preparation. Never serialized in a package.
struct TensorPreparation { virtual ~TensorPreparation() = default; };
struct Tensor {
    Encoding encoding = Encoding::F32;
    bool dynamic = false; // Execution operand (e.g. KV); preparation identifies an immutable snapshot.
    std::vector<uint32_t> shape;
    const uint8_t* data = nullptr;
    const uint8_t* scales = nullptr;
    size_t bytes = 0, scale_count = 0;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const TensorPreparation> preparation;
    void Validate() const;
    size_t Count() const;
    float At(size_t index) const;
    float Scale(size_t row) const;
    static Tensor Float(std::vector<uint32_t> shape, const std::vector<float>& values);
    static Tensor Half(std::vector<uint32_t> shape, std::vector<uint16_t> values);
};
struct Program;
class Model {
public:
    Config config;
    uint64_t workspace_bytes = 0;
    std::map<std::string, Tensor> tensors;
    std::vector<std::shared_ptr<const Program>> programs;
    const Tensor& Weight(const std::string& name) const;
    void Validate() const;
    static std::shared_ptr<Model> Load(const std::string& prefix);
};
// A streaming writer: payloads can exceed 4 GiB; only metadata stays in RAM.
// A prefix names three files: .tfllm, .weights, .commands. Finish publishes
// metadata last. Existing packages are never silently overwritten.
class PackageWriter {
public:
    PackageWriter(std::string prefix, Config config, uint64_t workspace_bytes = 0);
    ~PackageWriter();
    void AddTensor(const std::string& name, const Tensor& tensor);
    void AddProgram(const Program& program);
    void Finish();
    PackageWriter(const PackageWriter&) = delete;
    PackageWriter& operator=(const PackageWriter&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::shared_ptr<Model> MakeTinyModel();
}
