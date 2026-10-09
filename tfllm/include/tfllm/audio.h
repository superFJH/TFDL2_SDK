#pragma once
#include "tfllm/engine.h"
#include "tfllm/spatial_conv.h"

namespace tfllm {
struct AudioConfig {
    size_t layers=0,hidden=0,intermediate=0,heads=0,output=0,channels=0,mel_bins=128;
    size_t chunk_frames=100,attention_frames=800,conv_chunks=500;
    float epsilon=1e-5f;
    void Validate() const;
    size_t Tokens(size_t frames) const;
};
struct AudioModel {
    AudioConfig config;
    std::string identity;
    std::map<std::string,Tensor> tensors;
    void Validate() const;
};
struct AudioTokens : VisualTokens {
    AudioTokens(){modality=InputModality::Audio;}
};
// Explicit spatial backend; CPU is an FP16 reference, NPU uses SpatialConv.
class AudioConvolution {
public:
    virtual ~AudioConvolution()=default;
    virtual std::shared_ptr<SpatialConvWeight> Prepare(const SpatialConvSpec&,const uint16_t*)=0;
    virtual void Warmup(const std::shared_ptr<SpatialConvWeight>&,size_t /*height*/,size_t /*width*/) {}
    virtual void Run(const std::shared_ptr<SpatialConvWeight>&,const uint16_t*,uint16_t*,
                     size_t batch,size_t height,size_t width)=0;
    // True means the backend executed Conv+bias in accumulator space. False
    // leaves buffers untouched; the caller runs Conv and FP16 bias instead.
    virtual bool RunWithBias(const std::shared_ptr<SpatialConvWeight>&,const uint16_t*,uint16_t*,
                             size_t,size_t,size_t,const uint16_t*) {return false;}
};
std::shared_ptr<AudioConvolution> CreateCpuAudioConvolution();
std::shared_ptr<AudioModel> LoadAudioGguf(const std::string&);
class AudioEncoder {
public:
    AudioEncoder(std::shared_ptr<const AudioModel>,std::shared_ptr<Linear>,std::shared_ptr<AudioConvolution>);
    ~AudioEncoder();
    void Prepare(size_t max_rows=1024);
    // FP16 [mel_bins, frames], with no feature-extractor padding.
    std::shared_ptr<AudioTokens> Encode(const Activation& mel,size_t frames,const std::string& identity) const;
    const AudioConfig& Config() const{return model_->config;}
private:
    std::shared_ptr<const AudioModel> model_;
    std::shared_ptr<Linear> linear_;
    std::shared_ptr<AudioConvolution> convolution_;
    std::vector<std::shared_ptr<SpatialConvWeight>> conv_weights_;
    std::map<std::string,std::vector<float>> vectors_;
    std::map<std::string,Activation> biases_;
};
// Pinned mtmd Mel frontend; does not create an encoder graph or load weights.
class AudioFrontend {
public:
    explicit AudioFrontend(size_t mel_bins);
    ~AudioFrontend();
    Activation Mel(const std::vector<float>& samples) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
