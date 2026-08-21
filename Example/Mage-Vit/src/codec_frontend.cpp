#include "megavit/codec_frontend.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace megavit {
namespace {

struct CandidateBlock {
    const DecodedFrame* frame = nullptr;
    int source_x = 0;
    int source_y = 0;
    float score = 0.0f;
};

std::vector<const DecodedFrame*> UniformSample(
    const std::vector<DecodedFrame>& frames,
    int target) {
    if (frames.empty() || target <= 0) {
        return {};
    }
    const int count = std::min<int>(target, frames.size());
    std::vector<const DecodedFrame*> sampled;
    sampled.reserve(static_cast<size_t>(count));
    if (count == 1) {
        sampled.push_back(&frames.front());
        return sampled;
    }
    for (int i = 0; i < count; ++i) {
        const double p = static_cast<double>(i) * (frames.size() - 1) /
                         static_cast<double>(count - 1);
        sampled.push_back(&frames[static_cast<size_t>(std::llround(p))]);
    }
    return sampled;
}

float PixelProxyScore(
    const DecodedFrame& frame,
    const DecodedFrame* previous,
    int x0,
    int y0,
    int block_pixels) {
    double motion = 0.0;
    double edge = 0.0;
    int samples = 0;
    const bool comparable = previous && previous->width == frame.width &&
                            previous->height == frame.height;
    // A 4-pixel stride keeps the fallback inexpensive. This is explicitly a
    // development fallback; production should use patched-FFmpeg bitcost.
    for (int y = y0; y < y0 + block_pixels; y += 4) {
        for (int x = x0; x < x0 + block_pixels; x += 4) {
            const size_t off = (static_cast<size_t>(y) * frame.width + x) * 3;
            const int lum = (77 * frame.rgb[off] + 150 * frame.rgb[off + 1] +
                             29 * frame.rgb[off + 2]) >> 8;
            if (comparable) {
                const int prev_lum =
                    (77 * previous->rgb[off] + 150 * previous->rgb[off + 1] +
                     29 * previous->rgb[off + 2]) >> 8;
                motion += std::abs(lum - prev_lum);
            }
            if (x + 4 < x0 + block_pixels) {
                const size_t nx = off + 12;
                const int next_lum =
                    (77 * frame.rgb[nx] + 150 * frame.rgb[nx + 1] +
                     29 * frame.rgb[nx + 2]) >> 8;
                edge += std::abs(lum - next_lum);
            }
            ++samples;
        }
    }
    if (samples == 0) {
        return 0.0f;
    }
    return static_cast<float>((motion + 0.25 * edge) / samples);
}

float CodecScoreForBlock(
    const ScoreGrid& score,
    int x0,
    int y0,
    int block_pixels) {
    if (!score.valid()) {
        return 0.0f;
    }
    const int gx0 = x0 / score.cell_width;
    const int gy0 = y0 / score.cell_height;
    const int gx1 = (x0 + block_pixels - 1) / score.cell_width;
    const int gy1 = (y0 + block_pixels - 1) / score.cell_height;
    float total = 0.0f;
    int count = 0;
    for (int gy = gy0; gy <= gy1; ++gy) {
        for (int gx = gx0; gx <= gx1; ++gx) {
            total += score.at(gy, gx);
            ++count;
        }
    }
    return count ? total / count : 0.0f;
}

std::vector<CandidateBlock> BuildCandidates(
    const std::vector<const DecodedFrame*>& group,
    const CodecFrontendConfig& config) {
    std::vector<CandidateBlock> result;
    const int block = config.block_pixels();
    const DecodedFrame* previous = nullptr;
    for (const DecodedFrame* frame : group) {
        frame->validate();
        const int rows = frame->height / block;
        const int cols = frame->width / block;
        for (int by = 0; by < rows; ++by) {
            for (int bx = 0; bx < cols; ++bx) {
                const int x = bx * block;
                const int y = by * block;
                float value = CodecScoreForBlock(frame->codec_score, x, y, block);
                if (!frame->codec_score.valid()) {
                    value = PixelProxyScore(*frame, previous, x, y, block);
                }
                if (frame->key_frame) {
                    value += config.keyframe_score_bonus;
                }
                result.push_back({frame, x, y, value});
            }
        }
        previous = frame;
    }
    return result;
}

std::vector<CandidateBlock> SelectCandidates(
    std::vector<CandidateBlock> candidates,
    int requested,
    float per_frame_cap_ratio) {
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const CandidateBlock& a, const CandidateBlock& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.frame->frame_index != b.frame->frame_index) {
                return a.frame->frame_index < b.frame->frame_index;
            }
            if (a.source_y != b.source_y) return a.source_y < b.source_y;
            return a.source_x < b.source_x;
        });

    requested = std::min<int>(requested, candidates.size());
    const int per_frame_cap = std::max(
        1, static_cast<int>(std::ceil(requested * per_frame_cap_ratio)));
    std::unordered_map<int, int> frame_counts;
    std::vector<CandidateBlock> selected;
    std::vector<uint8_t> used(candidates.size(), 0);
    selected.reserve(static_cast<size_t>(requested));

    for (size_t i = 0; i < candidates.size() &&
                       static_cast<int>(selected.size()) < requested; ++i) {
        const int id = candidates[i].frame->frame_index;
        if (frame_counts[id] >= per_frame_cap) continue;
        selected.push_back(candidates[i]);
        used[i] = 1;
        ++frame_counts[id];
    }
    for (size_t i = 0; i < candidates.size() &&
                       static_cast<int>(selected.size()) < requested; ++i) {
        if (!used[i]) selected.push_back(candidates[i]);
    }

    // The causal LLM sees merger outputs in canvas order. Preserve temporal
    // order after top-K selection; 3D RoPE still carries the exact source
    // coordinate into the vision encoder.
    std::stable_sort(
        selected.begin(), selected.end(),
        [](const CandidateBlock& a, const CandidateBlock& b) {
            if (a.frame->frame_index != b.frame->frame_index) {
                return a.frame->frame_index < b.frame->frame_index;
            }
            if (a.source_y != b.source_y) return a.source_y < b.source_y;
            return a.source_x < b.source_x;
        });
    return selected;
}

void CopyBlock(
    const CandidateBlock& block,
    int block_pixels,
    int destination_x,
    int destination_y,
    Canvas* canvas) {
    const DecodedFrame& source = *block.frame;
    for (int row = 0; row < block_pixels; ++row) {
        const size_t src =
            (static_cast<size_t>(block.source_y + row) * source.width +
             block.source_x) * 3;
        const size_t dst =
            (static_cast<size_t>(destination_y + row) * canvas->width +
             destination_x) * 3;
        std::copy_n(
            source.rgb.data() + src,
            static_cast<size_t>(block_pixels) * 3,
            canvas->rgb.data() + dst);
    }
}

void AppendPositions(
    const CandidateBlock& block,
    int patch_size,
    std::vector<PatchPosition>* positions,
    std::vector<double>* timestamps) {
    const int h = block.source_y / patch_size;
    const int w = block.source_x / patch_size;
    // cv-preinfer/src_patch_position.npy uses one-based decoded-frame ids.
    const int t = block.frame->frame_index + 1;
    positions->push_back({t, h, w});
    positions->push_back({t, h, w + 1});
    positions->push_back({t, h + 1, w});
    positions->push_back({t, h + 1, w + 1});
    for (int i = 0; i < 4; ++i) {
        timestamps->push_back(block.frame->timestamp_seconds);
    }
}

double MedianTimestamp(
    const std::vector<CandidateBlock>& selected,
    size_t begin,
    size_t end) {
    std::vector<double> times;
    times.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        times.push_back(selected[i].frame->timestamp_seconds);
    }
    std::sort(times.begin(), times.end());
    if (times.empty()) return 0.0;
    return times[times.size() / 2];
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void WritePpm(const std::filesystem::path& path, const Canvas& canvas) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    out << "P6\n" << canvas.width << ' ' << canvas.height << "\n255\n";
    out.write(
        reinterpret_cast<const char*>(canvas.rgb.data()),
        static_cast<std::streamsize>(canvas.rgb.size()));
}

}  // namespace

void CodecFrontendConfig::validate() const {
    if (patch_size <= 0 || spatial_merge_size <= 0) {
        throw std::invalid_argument("patch and spatial merge sizes must be positive");
    }
    if (canvas_width <= 0 || canvas_height <= 0 ||
        canvas_width % block_pixels() != 0 ||
        canvas_height % block_pixels() != 0) {
        throw std::invalid_argument(
            "canvas dimensions must be positive multiples of patch_size * spatial_merge_size");
    }
    if (group_size <= 0 || images_per_group <= 0 || target_canvases <= 0 ||
        target_canvases % images_per_group != 0) {
        throw std::invalid_argument(
            "group_size/images_per_group/target_canvases are inconsistent");
    }
    if (min_group_frames <= 0 || min_group_frames > group_size) {
        throw std::invalid_argument("min_group_frames must be in [1, group_size]");
    }
    if (!(per_frame_cap_ratio > 0.0f && per_frame_cap_ratio <= 1.0f)) {
        throw std::invalid_argument("per_frame_cap_ratio must be in (0, 1]");
    }
}

CodecFrontend::CodecFrontend(CodecFrontendConfig config)
    : config_(std::move(config)) {
    config_.validate();
}

std::vector<Canvas> CodecFrontend::Build(
    const std::vector<DecodedFrame>& decoded_frames) const {
    if (decoded_frames.empty()) return {};
    for (const auto& frame : decoded_frames) frame.validate();

    const auto sampled = UniformSample(decoded_frames, config_.sampled_frame_budget());
    std::vector<Canvas> canvases;
    for (size_t group_begin = 0;
         group_begin < sampled.size() &&
         static_cast<int>(canvases.size()) < config_.target_canvases;
         group_begin += static_cast<size_t>(config_.group_size)) {
        const size_t group_end = std::min(
            sampled.size(), group_begin + static_cast<size_t>(config_.group_size));
        if (group_end - group_begin < static_cast<size_t>(config_.min_group_frames)) {
            break;
        }
        std::vector<const DecodedFrame*> group(
            sampled.begin() + static_cast<std::ptrdiff_t>(group_begin),
            sampled.begin() + static_cast<std::ptrdiff_t>(group_end));
        auto candidates = BuildCandidates(group, config_);
        const int requested =
            config_.blocks_per_canvas() * config_.images_per_group;
        if (config_.require_full_canvases &&
            static_cast<int>(candidates.size()) < requested) {
            throw std::runtime_error(
                "not enough source blocks to fill fixed Mage canvases; lower canvas size "
                "or disable require_full_canvases for development");
        }
        auto selected = SelectCandidates(
            std::move(candidates), requested, config_.per_frame_cap_ratio);
        const int available_canvases = static_cast<int>(selected.size()) /
                                       config_.blocks_per_canvas();
        const int canvas_count = std::min(config_.images_per_group, available_canvases);
        for (int image = 0; image < canvas_count; ++image) {
            const size_t first = static_cast<size_t>(image * config_.blocks_per_canvas());
            const size_t last = first + static_cast<size_t>(config_.blocks_per_canvas());
            Canvas canvas;
            canvas.width = config_.canvas_width;
            canvas.height = config_.canvas_height;
            canvas.timestamp_seconds = MedianTimestamp(selected, first, last);
            canvas.rgb.resize(static_cast<size_t>(canvas.width) * canvas.height * 3);
            canvas.patch_positions.reserve(
                static_cast<size_t>(config_.canvas_patch_count()));
            canvas.patch_timestamps.reserve(
                static_cast<size_t>(config_.canvas_patch_count()));
            const int block_cols = canvas.width / config_.block_pixels();
            for (int i = 0; i < config_.blocks_per_canvas(); ++i) {
                const CandidateBlock& block = selected[first + static_cast<size_t>(i)];
                const int destination_x = (i % block_cols) * config_.block_pixels();
                const int destination_y = (i / block_cols) * config_.block_pixels();
                CopyBlock(
                    block, config_.block_pixels(), destination_x, destination_y,
                    &canvas);
                AppendPositions(
                    block, config_.patch_size, &canvas.patch_positions,
                    &canvas.patch_timestamps);
            }
            canvases.push_back(std::move(canvas));
            if (static_cast<int>(canvases.size()) >= config_.target_canvases) break;
        }
    }
    return canvases;
}

void WriteFrontendBundle(
    const std::string& output_directory,
    const CodecFrontendConfig& config,
    const std::vector<Canvas>& canvases) {
    namespace fs = std::filesystem;
    const fs::path root(output_directory);
    fs::create_directories(root);

    std::ofstream manifest(root / "manifest.json");
    if (!manifest) {
        throw std::runtime_error("cannot create frontend manifest under " + root.string());
    }
    manifest << "{\n"
             << "  \"format\": \"megavit.frontend.v1\",\n"
             << "  \"patch_size\": " << config.patch_size << ",\n"
             << "  \"spatial_merge_size\": " << config.spatial_merge_size << ",\n"
             << "  \"canvas_width\": " << config.canvas_width << ",\n"
             << "  \"canvas_height\": " << config.canvas_height << ",\n"
             << "  \"canvases\": [\n";
    for (size_t i = 0; i < canvases.size(); ++i) {
        std::ostringstream filename;
        filename << "canvas_" << std::setw(3) << std::setfill('0') << i << ".ppm";
        WritePpm(root / filename.str(), canvases[i]);
        manifest << "    {\"file\": \"" << JsonEscape(filename.str())
                 << "\", \"timestamp_seconds\": " << std::fixed
                 << std::setprecision(6) << canvases[i].timestamp_seconds
                 << ", \"token_count\": "
                 << canvases[i].patch_positions.size() /
                        static_cast<size_t>(config.spatial_merge_size *
                                            config.spatial_merge_size)
                 << ", \"patch_positions\": [";
        for (size_t p = 0; p < canvases[i].patch_positions.size(); ++p) {
            const auto& pos = canvases[i].patch_positions[p];
            if (p) manifest << ',';
            manifest << '[' << pos.t << ',' << pos.h << ',' << pos.w << ']';
        }
        manifest << "], \"patch_timestamps\": [";
        for (size_t p = 0; p < canvases[i].patch_timestamps.size(); ++p) {
            if (p) manifest << ',';
            manifest << std::fixed << std::setprecision(6)
                     << canvases[i].patch_timestamps[p];
        }
        manifest << "]}" << (i + 1 == canvases.size() ? "\n" : ",\n");
    }
    manifest << "  ]\n}\n";
}

}  // namespace megavit
