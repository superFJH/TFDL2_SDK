#pragma once

// Stable shim between this example and a bitcost-instrumented FFmpeg fork.
// The Mage codec-video-prep patch is not part of upstream FFmpeg and its
// internal AVFrame storage is version-dependent. Keep that dependency behind
// one weak C symbol instead of leaking patched FFmpeg structs into the rest of
// the application.

#include <cstdint>

struct AVFrame;

extern "C" {

struct MageFfmpegBitcostView {
    int32_t grid_width;
    int32_t grid_height;
    int32_t cell_width;
    int32_t cell_height;
    const float* values;
};

// Supply this symbol in a small shim linked against the selected patched
// FFmpeg build. Return 1 and a valid view when bitcost is present; return 0
// otherwise. The decoder copies the view before the AVFrame is released.
int mage_ffmpeg_get_bitcost(
    const AVFrame* frame,
    MageFfmpegBitcostView* view);

}
