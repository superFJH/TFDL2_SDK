#pragma once
#include "tfllm/model.h"
#include "tfllm/activation.h"
#include <functional>

namespace tfllm {
// Explicit byte strides support llama.cpp's token-major K and either V layout.
// K contains Q/K-normalized, RoPE-applied values; cache stores KV heads only.
struct KvView {
    uint8_t* data = nullptr;
    size_t bytes = 0, token_stride = 0, head_stride = 0, dim_stride = 2;
};
struct LayerKv { KvView key, value; };
struct DeltaState;
class KvCache {
public:
    KvCache(const Config& config, size_t capacity);
    KvCache(const Config& config, size_t capacity, std::vector<LayerKv> layers,
            std::shared_ptr<void> owner);
    size_t Capacity() const { return capacity_; }
    const Config& Geometry() const { return config_; }
    const std::vector<LayerKv>& Views() const { return layers_; }
    std::vector<std::shared_ptr<DeltaState>>& Recurrent() {return recurrent_;}
    float Read(bool key, size_t layer, size_t token, size_t head, size_t dim) const;
    void Write(bool key, size_t layer, size_t token, size_t head, size_t dim, float value);
    void WriteFp16(size_t layer,size_t past,size_t rows,const Activation& key,const Activation& value);
    Activation HeadFp16(bool key,size_t layer,size_t tokens,size_t head,size_t padded_tokens,bool transpose) const;
private:
    void Validate() const;
    uint8_t* Address(bool key, size_t layer, size_t token, size_t head, size_t dim) const;
    Config config_;
    size_t capacity_;
    std::vector<LayerKv> layers_;
    std::shared_ptr<void> owner_;
    std::vector<std::shared_ptr<DeltaState>> recurrent_;
    void InitializeRecurrent();
};
}
