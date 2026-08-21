#include "megavit/codec_frontend.hpp"
#include "megavit/prompt_assembler.hpp"
#include "megavit/video_decoder.hpp"
#include "megavit/vision_runner.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Arguments {
    std::string video;
    std::string output_directory = "megavit_output";
    std::string model;
    std::string addon;
    std::string executor_config;
    bool synthetic = false;
    bool official_profile = false;
    bool capabilities = false;
    int target_canvases = 0;
};

std::string ReadText(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    return std::string(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

Arguments ParseArguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value after " + key);
            return argv[i];
        };
        if (key == "--video") args.video = value();
        else if (key == "--output-dir") args.output_directory = value();
        else if (key == "--model") args.model = value();
        else if (key == "--addon") args.addon = value();
        else if (key == "--executor-config") args.executor_config = value();
        else if (key == "--target-canvases") args.target_canvases = std::stoi(value());
        else if (key == "--synthetic") args.synthetic = true;
        else if (key == "--official-profile") args.official_profile = true;
        else if (key == "--capabilities") args.capabilities = true;
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: megavit_frontend (--video FILE | --synthetic) [options]\n"
                << "  --output-dir DIR       bundle and embeddings output\n"
                << "  --model FILE           optional mage_vit.quant.fb\n"
                << "  --addon FILE           libTFDLAddOn.so for ApplyRope\n"
                << "  --executor-config FILE TFDL executor JSON\n"
                << "  --target-canvases N    override output canvas count\n"
                << "  --capabilities         print compiled backend support as JSON\n"
                << "  --official-profile     use 288x512/32-canvas profile for synthetic input\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    const bool has_video = !args.video.empty();
    if (!args.capabilities && args.synthetic == has_video) {
        throw std::invalid_argument("choose exactly one of --video or --synthetic");
    }
    return args;
}

std::vector<megavit::DecodedFrame> MakeSyntheticFrames(
    const megavit::CodecFrontendConfig& config) {
    const int count = config.sampled_frame_budget();
    const int width = std::max(128, config.canvas_width);
    const int height = std::max(128, config.canvas_height);
    std::vector<megavit::DecodedFrame> frames;
    frames.reserve(static_cast<size_t>(count));
    for (int t = 0; t < count; ++t) {
        megavit::DecodedFrame frame;
        frame.frame_index = t;
        frame.timestamp_seconds = t / 4.0;
        frame.width = width;
        frame.height = height;
        frame.key_frame = (t % config.group_size) == 0;
        frame.rgb.resize(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t at = (static_cast<size_t>(y) * width + x) * 3;
                frame.rgb[at] = static_cast<uint8_t>((x + t * 3) & 255);
                frame.rgb[at + 1] = static_cast<uint8_t>((y + t * 5) & 255);
                frame.rgb[at + 2] = static_cast<uint8_t>((x + y + t * 7) & 255);
            }
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        using Clock = std::chrono::steady_clock;
        const Arguments args = ParseArguments(argc, argv);
        if (args.capabilities) {
            std::cout << "{\"ffmpeg\":"
                      << (MEGAVIT_HAS_FFMPEG ? "true" : "false")
                      << ",\"tfdl\":"
                      << (MEGAVIT_HAS_TFDL ? "true" : "false")
                      << ",\"patched_bitcost\":"
                      << (megavit::HasPatchedFfmpegBitcost() ? "true" : "false")
                      << "}\n";
            return 0;
        }
        megavit::CodecFrontendConfig frontend_config;
        if (args.synthetic && !args.official_profile) {
            frontend_config.canvas_width = 64;
            frontend_config.canvas_height = 64;
            frontend_config.group_size = 4;
            frontend_config.images_per_group = 1;
            frontend_config.target_canvases = 2;
            frontend_config.min_group_frames = 4;
        }
        if (args.target_canvases > 0) {
            frontend_config.target_canvases = args.target_canvases;
        }
        frontend_config.validate();

        const auto decode_begin = Clock::now();
        std::vector<megavit::DecodedFrame> frames;
        if (args.synthetic) {
            frames = MakeSyntheticFrames(frontend_config);
        } else {
            auto decoder = megavit::CreateFfmpegVideoDecoder();
            if (!decoder) {
                throw std::runtime_error(
                    "this build has no FFmpeg development libraries; install them and "
                    "reconfigure with -DMEGAVIT_WITH_FFMPEG=ON");
            }
            frames = decoder->Decode(args.video, frontend_config.sampled_frame_budget());
        }
        const auto decode_end = Clock::now();

        megavit::CodecFrontend frontend(frontend_config);
        auto canvases = frontend.Build(frames);
        const auto frontend_end = Clock::now();
        if (canvases.empty()) {
            throw std::runtime_error("codec frontend produced no full canvas");
        }
        megavit::WriteFrontendBundle(
            args.output_directory, frontend_config, canvases);

        const auto prompt = megavit::BuildPromptPlan(
            canvases, frontend_config.spatial_merge_size);
        std::filesystem::create_directories(args.output_directory);
        std::ofstream prompt_file(
            std::filesystem::path(args.output_directory) / "vision_content.txt");
        prompt_file << prompt.vision_content;

        std::cout << "decoded_frames=" << frames.size()
                  << " canvases=" << canvases.size()
                  << " visual_tokens=" << prompt.total_visual_tokens << '\n';
        double vision_ms = 0.0;
        if (!args.model.empty()) {
            megavit::VisionRunnerConfig runner_config;
            runner_config.model_path = args.model;
            runner_config.addon_path = args.addon;
            runner_config.executor_json = ReadText(args.executor_config);
            megavit::VisionRunner runner(std::move(runner_config));
            const auto vision_begin = Clock::now();
            auto embeddings = runner.Run(canvases);
            const auto vision_end = Clock::now();
            vision_ms = std::chrono::duration<double, std::milli>(
                vision_end - vision_begin).count();
            const auto output =
                std::filesystem::path(args.output_directory) /
                "visual_embeddings.f32";
            megavit::WriteFloatEmbeddings(output.string(), embeddings);
            std::cout << "embedding_shape=[" << embeddings.token_count() << ','
                      << embeddings.hidden_size << "] file=" << output << '\n';
        } else {
            std::cout
                << "TFDL stage skipped (pass --model mage_vit.quant.fb to run it)\n";
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(
            decode_end - decode_begin).count();
        const double frontend_ms = std::chrono::duration<double, std::milli>(
            frontend_end - decode_end).count();
        std::ofstream metrics(
            std::filesystem::path(args.output_directory) / "metrics.json");
        metrics << "{\n"
                << "  \"decoded_frames\": " << frames.size() << ",\n"
                << "  \"canvases\": " << canvases.size() << ",\n"
                << "  \"visual_tokens\": " << prompt.total_visual_tokens << ",\n"
                << "  \"decode_ms\": " << decode_ms << ",\n"
                << "  \"canvas_select_pack_ms\": " << frontend_ms << ",\n"
                << "  \"tfdl_vision_ms\": " << vision_ms << "\n"
                << "}\n";
        std::cout << "decode_ms=" << decode_ms
                  << " canvas_select_pack_ms=" << frontend_ms
                  << " tfdl_vision_ms=" << vision_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "megavit_frontend: " << error.what() << '\n';
        return 2;
    }
}
