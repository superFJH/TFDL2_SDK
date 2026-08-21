#pragma once

#include "megavit/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace megavit {

class VideoDecoder {
public:
    virtual ~VideoDecoder() = default;
    // max_frames > 0 retains an approximately uniform sample across the full
    // stream. The decoder still scans every packet so codec dependencies and
    // timestamps remain valid.
    virtual std::vector<DecodedFrame> Decode(
        const std::string& path,
        int max_frames = 0) = 0;
};

// Returns nullptr when the project was built without FFmpeg development
// headers/libraries. The executable remains usable with --synthetic so the
// canvas/TFDL interface can still be evaluated on the SDK image.
std::unique_ptr<VideoDecoder> CreateFfmpegVideoDecoder();

// True when the optional Mage codec-bitcost shim is linked. Upstream FFmpeg
// remains usable through the motion-vector/pixel-residual fallback.
bool HasPatchedFfmpegBitcost();

}  // namespace megavit
