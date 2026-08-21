#include "megavit/vision_runner.hpp"

#include <fstream>
#include <stdexcept>

namespace megavit {

struct VisionRunner::Impl {};

VisionRunner::VisionRunner(VisionRunnerConfig) : impl_(std::make_unique<Impl>()) {
    throw std::runtime_error(
        "Mage-Vit was built without TFDL runtime support; reconfigure with "
        "-DMEGAVIT_WITH_TFDL=ON");
}
VisionRunner::~VisionRunner() = default;
VisionRunner::VisionRunner(VisionRunner&&) noexcept = default;
VisionRunner& VisionRunner::operator=(VisionRunner&&) noexcept = default;

VisionEmbeddingBatch VisionRunner::Run(const std::vector<Canvas>&) {
    throw std::runtime_error("TFDL runtime support is unavailable");
}

void WriteFloatEmbeddings(
    const std::string& path,
    const VisionEmbeddingBatch& embeddings) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create " + path);
    out.write(
        reinterpret_cast<const char*>(embeddings.values.data()),
        static_cast<std::streamsize>(embeddings.values.size() * sizeof(float)));
}

}  // namespace megavit
