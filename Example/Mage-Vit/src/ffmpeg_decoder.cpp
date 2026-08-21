#include "megavit/video_decoder.hpp"

#include "megavit/patched_ffmpeg_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libswscale/swscale.h>
}

#if defined(__GNUC__)
extern "C" int mage_ffmpeg_get_bitcost(
    const AVFrame*, MageFfmpegBitcostView*) __attribute__((weak));
#endif

namespace megavit {
namespace {

std::string AvError(int code) {
    char message[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, message, sizeof(message));
    return message;
}

void ThrowAv(const std::string& operation, int code) {
    throw std::runtime_error(operation + ": " + AvError(code));
}

ScoreGrid CopyPatchedBitcost(const AVFrame* frame) {
    ScoreGrid score;
#if defined(__GNUC__)
    if (!mage_ffmpeg_get_bitcost) return score;
#endif
    MageFfmpegBitcostView view{};
    if (mage_ffmpeg_get_bitcost(frame, &view) != 1 || !view.values ||
        view.grid_width <= 0 || view.grid_height <= 0 ||
        view.cell_width <= 0 || view.cell_height <= 0) {
        return score;
    }
    score.width = view.grid_width;
    score.height = view.grid_height;
    score.cell_width = view.cell_width;
    score.cell_height = view.cell_height;
    score.values.assign(
        view.values,
        view.values + static_cast<size_t>(score.width * score.height));
    return score;
}

ScoreGrid MotionVectorScore(const AVFrame* frame) {
    constexpr int kCell = 16;
    ScoreGrid score;
    score.width = (frame->width + kCell - 1) / kCell;
    score.height = (frame->height + kCell - 1) / kCell;
    score.cell_width = kCell;
    score.cell_height = kCell;
    score.values.assign(static_cast<size_t>(score.width * score.height), 0.0f);

    const AVFrameSideData* side =
        av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!side || side->size < sizeof(AVMotionVector)) return score;
    const auto* vectors = reinterpret_cast<const AVMotionVector*>(side->data);
    const size_t count = side->size / sizeof(AVMotionVector);
    for (size_t i = 0; i < count; ++i) {
        const AVMotionVector& mv = vectors[i];
        const int x = std::clamp<int>(mv.dst_x / kCell, 0, score.width - 1);
        const int y = std::clamp<int>(mv.dst_y / kCell, 0, score.height - 1);
        const float scale = mv.motion_scale ? static_cast<float>(mv.motion_scale) : 1.0f;
        const float magnitude =
            std::sqrt(static_cast<float>(mv.motion_x) * mv.motion_x +
                      static_cast<float>(mv.motion_y) * mv.motion_y) /
            scale;
        score.values[static_cast<size_t>(y * score.width + x)] += magnitude;
    }
    return score;
}

class FfmpegVideoDecoder final : public VideoDecoder {
public:
    std::vector<DecodedFrame> Decode(
        const std::string& path,
        int max_frames) override {
        AVFormatContext* raw_format = nullptr;
        int rc = avformat_open_input(&raw_format, path.c_str(), nullptr, nullptr);
        if (rc < 0) ThrowAv("avformat_open_input", rc);
        std::unique_ptr<AVFormatContext, void (*)(AVFormatContext*)> format(
            raw_format, [](AVFormatContext* value) {
                avformat_close_input(&value);
            });
        rc = avformat_find_stream_info(format.get(), nullptr);
        if (rc < 0) ThrowAv("avformat_find_stream_info", rc);

        const int stream_index = av_find_best_stream(
            format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index < 0) ThrowAv("av_find_best_stream", stream_index);
        AVStream* stream = format->streams[stream_index];
        const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec) throw std::runtime_error("FFmpeg video decoder not found");

        AVCodecContext* raw_codec = avcodec_alloc_context3(codec);
        if (!raw_codec) throw std::bad_alloc();
        std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)> codec_context(
            raw_codec, [](AVCodecContext* value) { avcodec_free_context(&value); });
        rc = avcodec_parameters_to_context(codec_context.get(), stream->codecpar);
        if (rc < 0) ThrowAv("avcodec_parameters_to_context", rc);
        codec_context->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
        AVDictionary* options = nullptr;
        av_dict_set(&options, "threads", "auto", 0);
        av_dict_set(&options, "thread_type", "slice", 0);
        rc = avcodec_open2(codec_context.get(), codec, &options);
        av_dict_free(&options);
        if (rc < 0) ThrowAv("avcodec_open2", rc);

        std::unique_ptr<AVPacket, void (*)(AVPacket*)> packet(
            av_packet_alloc(), [](AVPacket* value) { av_packet_free(&value); });
        std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame(
            av_frame_alloc(), [](AVFrame* value) { av_frame_free(&value); });
        if (!packet || !frame) throw std::bad_alloc();

        SwsContext* sws = nullptr;
        std::vector<DecodedFrame> decoded;
        int decoded_index = 0;
        uint64_t reservoir_state = 0x9e3779b97f4a7c15ULL;
        std::vector<int64_t> target_indices;
        if (max_frames > 0 && stream->nb_frames > 0) {
            const int keep = std::min<int64_t>(max_frames, stream->nb_frames);
            target_indices.reserve(static_cast<size_t>(keep));
            for (int i = 0; i < keep; ++i) {
                const double position = keep == 1
                    ? 0.0
                    : static_cast<double>(i) * (stream->nb_frames - 1) /
                          static_cast<double>(keep - 1);
                target_indices.push_back(static_cast<int64_t>(std::llround(position)));
            }
        }
        auto receive = [&]() {
            while (true) {
                const int receive_rc = avcodec_receive_frame(codec_context.get(), frame.get());
                if (receive_rc == AVERROR(EAGAIN) || receive_rc == AVERROR_EOF) return;
                if (receive_rc < 0) ThrowAv("avcodec_receive_frame", receive_rc);

                const int current_index = decoded_index++;
                bool keep_frame = max_frames <= 0;
                if (max_frames > 0 && !target_indices.empty()) {
                    keep_frame = std::binary_search(
                        target_indices.begin(), target_indices.end(), current_index);
                } else if (max_frames > 0 &&
                           static_cast<int>(decoded.size()) < max_frames) {
                    keep_frame = true;
                } else if (max_frames > 0) {
                    // Unknown total-frame count: deterministic reservoir sample.
                    reservoir_state =
                        reservoir_state * 6364136223846793005ULL + 1ULL;
                    const uint64_t slot = reservoir_state %
                        static_cast<uint64_t>(current_index + 1);
                    keep_frame = slot < static_cast<uint64_t>(max_frames);
                    if (keep_frame) {
                        decoded.erase(decoded.begin() + static_cast<std::ptrdiff_t>(slot));
                    }
                }
                if (!keep_frame) {
                    av_frame_unref(frame.get());
                    continue;
                }

                DecodedFrame out;
                out.frame_index = current_index;
                out.width = frame->width;
                out.height = frame->height;
                // AV_FRAME_FLAG_KEY was added after the older FFmpeg
                // releases still shipped by several ARM distributions.
                // AVFrame::key_frame is the equivalent legacy field.
#if defined(AV_FRAME_FLAG_KEY)
                out.key_frame =
                    (frame->flags & AV_FRAME_FLAG_KEY) != 0 ||
                    frame->pict_type == AV_PICTURE_TYPE_I;
#else
                out.key_frame = frame->key_frame != 0 ||
                                frame->pict_type == AV_PICTURE_TYPE_I;
#endif
                int64_t pts = frame->best_effort_timestamp;
                if (pts != AV_NOPTS_VALUE) {
                    out.timestamp_seconds = pts * av_q2d(stream->time_base);
                } else {
                    const AVRational fps = av_guess_frame_rate(format.get(), stream, frame.get());
                    out.timestamp_seconds = fps.num > 0
                        ? static_cast<double>(out.frame_index) * fps.den / fps.num
                        : static_cast<double>(out.frame_index);
                }

                out.codec_score = CopyPatchedBitcost(frame.get());
                if (!out.codec_score.valid()) {
                    out.codec_score = MotionVectorScore(frame.get());
                }

                sws = sws_getCachedContext(
                    sws, frame->width, frame->height,
                    static_cast<AVPixelFormat>(frame->format),
                    frame->width, frame->height, AV_PIX_FMT_RGB24,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws) throw std::runtime_error("sws_getCachedContext failed");
                out.rgb.resize(static_cast<size_t>(out.width) * out.height * 3);
                uint8_t* destination[] = {out.rgb.data(), nullptr, nullptr, nullptr};
                int destination_stride[] = {out.width * 3, 0, 0, 0};
                sws_scale(
                    sws, frame->data, frame->linesize, 0, frame->height,
                    destination, destination_stride);
                decoded.push_back(std::move(out));
                av_frame_unref(frame.get());
            }
        };

        while (av_read_frame(format.get(), packet.get()) >= 0) {
            if (packet->stream_index == stream_index) {
                rc = avcodec_send_packet(codec_context.get(), packet.get());
                if (rc < 0 && rc != AVERROR(EAGAIN)) ThrowAv("avcodec_send_packet", rc);
                receive();
            }
            av_packet_unref(packet.get());
        }
        rc = avcodec_send_packet(codec_context.get(), nullptr);
        if (rc >= 0 || rc == AVERROR_EOF) receive();
        sws_freeContext(sws);
        std::sort(
            decoded.begin(), decoded.end(),
            [](const DecodedFrame& a, const DecodedFrame& b) {
                return a.frame_index < b.frame_index;
            });
        return decoded;
    }
};

}  // namespace

bool HasPatchedFfmpegBitcost() {
    return mage_ffmpeg_get_bitcost != nullptr;
}

std::unique_ptr<VideoDecoder> CreateFfmpegVideoDecoder() {
    return std::make_unique<FfmpegVideoDecoder>();
}

}  // namespace megavit
