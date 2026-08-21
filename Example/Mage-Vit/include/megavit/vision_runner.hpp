#pragma once

#include "megavit/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace megavit {

struct VisionRunnerConfig {
    std::string model_path;
    std::string addon_path;
    std::string executor_json;
    int expected_hidden_size = 2560;
};

class VisionRunner {
public:
    explicit VisionRunner(VisionRunnerConfig config);
    ~VisionRunner();
    VisionRunner(VisionRunner&&) noexcept;
    VisionRunner& operator=(VisionRunner&&) noexcept;
    VisionRunner(const VisionRunner&) = delete;
    VisionRunner& operator=(const VisionRunner&) = delete;

    VisionEmbeddingBatch Run(const std::vector<Canvas>& canvases);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void WriteFloatEmbeddings(
    const std::string& path,
    const VisionEmbeddingBatch& embeddings);

}  // namespace megavit
