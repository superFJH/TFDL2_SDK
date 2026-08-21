#include "megavit/video_decoder.hpp"

namespace megavit {

std::unique_ptr<VideoDecoder> CreateFfmpegVideoDecoder() {
    return nullptr;
}

bool HasPatchedFfmpegBitcost() {
    return false;
}

}  // namespace megavit
