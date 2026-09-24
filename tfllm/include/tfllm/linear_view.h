#pragma once
#include "tfllm/activation.h"
#include <utility>

namespace tfllm {
// Borrowed, synchronous views. Strides are in elements. Optional row indices
// select rows of a larger tensor (e.g. window attention); output rows must be
// distinct. The caller retains storage until the call, including failed jobs,
// has drained. Input and output storage must not overlap.
struct Fp16InputView {
    const uint16_t* data=nullptr;
    size_t rows=0,columns=0,stride=0;
    const size_t* indices=nullptr;
    const uint16_t* Row(size_t r) const {return data+(indices ? indices[r]:r)*stride;}
};
struct Fp16OutputView {
    uint16_t* data=nullptr;
    size_t rows=0,columns=0,stride=0;
    const size_t* indices=nullptr;
    uint16_t* Row(size_t r) const {return data+(indices ? indices[r]:r)*stride;}
};
// A producer writes row-symmetric UINT8 directly into the execution workspace.
// It writes only logical columns and positive finite row scales. The backend
// owns K/M padding (zeroPoint=128) and calls disjoint row ranges concurrently.
// Producers may run CPU kernels, but must not submit/wait for NPU operations.
struct QuantizedInput {
    size_t rows=0,columns=0;
    std::function<void(size_t,size_t,uint8_t*,size_t,float*)> write;
    // Optional identity of an IMMUTABLE snapshot. Borrowed/mutable producers
    // leave this empty. Backends may use it to identify shared CPU preparation;
    // the NPU runtime slot still receives a copy on every execution.
    std::shared_ptr<const void> identity={};
};
// One KV head, with all its query/GQA rows expressed as indexed views. The
// callback returns [first, end) visible keys for a logical query row. A backend
// may consume INT32 scores directly; FP16 score/probability rounding remains
// part of the contract. Returning false must leave output untouched.
// K/V may contain physical zero padding. A shorter query.columns is padded
// with zeros by the backend; only output.columns are written. The mask and
// scale describe logical keys/head dimensions, never their padded sizes.
struct AttentionMask {
    float scale=1,soft_cap=0;
    std::function<std::pair<size_t,size_t>(size_t)> interval;
};
// Snapshot shared by projections with the same input. Does not own NPU memory.
QuantizedInput QuantizeInput(Fp16InputView);
void ValidateLinearViews(const Tensor&,Fp16InputView,Fp16OutputView);
}
