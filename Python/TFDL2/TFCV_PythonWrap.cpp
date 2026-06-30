// TFCV_PythonWrap.cpp — _tfcv pybind 模块
//
// 高性能流式推理: C++ 内部绑定 {source(TFCV) + Executor + worker 线程},
// Python 只 poll 输出张量(numpy) + 对应原始帧(numpy), 在 Python 里做后处理/事件判断.
//
// 关键: worker 线程纯 C++ (不持 GIL); poll() 在等队列时释放 GIL.
// 加载: 运行前需先懒加载 libTFCV (见 Python 侧 TFCV.py 的 _ensure_loaded).
//
// 链接: libTFCV (+ tfdec/tfenc/tfgs/mk_api, 均 NEEDED) + libTFDL2_LITE_C_API (TFDL::tlog 等),
//       仅来自 lib/CV_NPU40T + lib/, 不引外部库.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "TFCV/TFCV.h"
#include "TFDL2_C_API.h"

namespace py = pybind11;
using namespace TFDL_CAPI;

// 这些访问器在头文件里被注释了, 但 libTFDL2_LITE_C_API.so 实际有导出.
namespace TFDL_CAPI {
    extern const std::vector<float> &GetQuantizationScale(Quantization quantization);
    extern const std::vector<int>   &GetQuantizationZeroPoint(Quantization quantization);
}

namespace {

// 把一个 TFTensor 转成 numpy (按其 shape/dtype), 返回 numpy + (qscale,qzp) 量化信息.
// uint8/float32/float16/int32/int64 直接拷; 量化反量化交给 Python (按需).
py::object tensor_to_numpy(const TFTensor &t, py::object &quant_out) {
    std::vector<int> shape = GetTensorShape(t);
    TFCAPI_DATATYPE dt = GetTensorType(t);
    void *data = GetTensordata(t);
    size_t n = 1;
    for (int s : shape) n *= (size_t)(s > 0 ? s : 1);
    std::vector<py::ssize_t> pyshape(shape.begin(), shape.end());

    Quantization q = GetTensorQuantize(t);
    const auto &qs = GetQuantizationScale(q);
    const auto &qz = GetQuantizationZeroPoint(q);
    py::list slist, zlist;
    for (float v : qs) slist.append(v);
    for (int v : qz) zlist.append(v);
    quant_out = py::make_tuple(slist, zlist);

    switch (dt) {
        case TFCAPI_UINT8: {
            py::array_t<uint8_t> a(pyshape);
            if (data && n) std::memcpy(a.request().ptr, data, n * sizeof(uint8_t));
            return std::move(a);
        }
        case TFCAPI_FLOAT: {
            py::array_t<float> a(pyshape);
            if (data && n) std::memcpy(a.request().ptr, data, n * sizeof(float));
            return std::move(a);
        }
        case TFCAPI_FLOAT16: {
            py::array_t<uint16_t> a(pyshape);  // 保留 raw half; Python 端可用 numpy float16 解释
            if (data && n) std::memcpy(a.request().ptr, data, n * sizeof(uint16_t));
            return std::move(a);
        }
        case TFCAPI_INT32: {
            py::array_t<int32_t> a(pyshape);
            if (data && n) std::memcpy(a.request().ptr, data, n * sizeof(int32_t));
            return std::move(a);
        }
        case TFCAPI_INT64: {
            py::array_t<int64_t> a(pyshape);
            if (data && n) std::memcpy(a.request().ptr, data, n * sizeof(int64_t));
            return std::move(a);
        }
        default: {
            py::array_t<uint8_t> a(pyshape);  // 兜底: 按字节
            return std::move(a);
        }
    }
}

struct Result {
    int status = 1;                 // 1=ok, 0=EOS, -1=error
    int64_t frameIndex = 0;
    std::vector<uint8_t> frame;     // 原始分辨率 HxWxC (keep_frame 时填)
    int h = 0, w = 0, c = 0;
    std::vector<TFTensor> outputs;
    std::string errMsg;
    // 空间预处理元信息 (供 Python 把模型坐标反映射回原图):
    //   centercrop: crop = [left, top, cw, ch] (在源帧上的中心裁剪区域)
    //   letterbox : lbox = [ratio, pad_left, pad_top]
    std::vector<int> crop;
    std::vector<double> lbox;
};

// 把当前帧 dump 成 numpy HxWxC uint8 (解码段释放 GIL, 构造 numpy 时重新持有).
// 注意: GetHeight/GetWidth 返回的是源帧原始尺寸, 不受 scale/crop 影响;
//       DumpImgData 的输出 buffer 已按 "先 scale 再 crop" 处理, 故输出尺寸要自己算.
py::array vision_to_numpy(TFVision &v, TFCV::DECODER_FLAGS flags, float scale,
                          TFCV::CropSize crop) {
    std::vector<uint8_t> dd;
    {
        py::gil_scoped_release gil;   // 解码是纯 C++, 放 GIL 让其它 Python 线程跑
        TFCV::DumpImgData(v, dd, flags, scale, crop);
    }
    int H0 = TFCV::GetHeight(v);
    int W0 = TFCV::GetWidth(v);
    int c = (flags == TFCV::TFCV_Gray) ? 1 : 3;
    if (H0 <= 0 || W0 <= 0 || dd.empty()) throw std::runtime_error("TFCV dump empty frame");

    // 输出尺寸: 先 scale (等比) 再 crop (取 crop 区域). GetHeight/Width 不随其变化, 故自行推算.
    long Hs = std::lround(H0 * (double)scale);   // scale 后的高
    long Ws = std::lround(W0 * (double)scale);   // scale 后的宽
    int cl = std::get<0>(crop), ct = std::get<1>(crop);
    int cw = std::get<2>(crop), ch = std::get<3>(crop);
    long out_h = Hs, out_w = Ws;
    if (cw > 0 && ch > 0) { out_h = ch; out_w = cw; }   // 有 crop: 输出 = crop 区域 (scale 之后)

    // 容错: 若推算尺寸与实际 buffer 不符 (语义差异), 用 buffer 大小 + scale 后宽高比兜底.
    if ((long)dd.size() != out_h * out_w * c) {
        long total = (long)dd.size() / c;
        if (Hs > 0) {
            out_w = (cw > 0) ? cw : (long)std::lround(std::sqrt((double)total * Ws / Hs));
            out_h = (out_w > 0) ? total / out_w : Hs;
        }
    }
    py::array_t<uint8_t> arr({(py::ssize_t)out_h, (py::ssize_t)out_w, (py::ssize_t)c});
    std::memcpy(arr.request().ptr, dd.data(), dd.size());
    return arr;
}

TFCV::MediaTYPE mediaTypeFromStr(const std::string &s) {
    if (s == "png") return TFCV::tPNG;
    if (s == "bmp") return TFCV::tBMP;
    if (s == "hevc" || s == "h265") return TFCV::tHEVC;
    return TFCV::tH264;
}

// ===================== NEON 加速: resize + HWC->NCHW 转置 =====================
#if defined(__aarch64__) || defined(__ARM_NEON)
#define TFCV_HAVE_NEON 1
#include <arm_neon.h>
#endif

// 双线性缩放 (半像素约定, 与 cv2.INTER_LINEAR 一致), HWC uint8, 3 通道.
// NEON: 每个输出像素的 3 通道并行算 (uint16x8 用 3 lane). 权重定点化 (Q12).
inline void bilinear_resize_hwc(const uint8_t *src, int sH, int sW,
                                uint8_t *dst, int dH, int dW) {
    if (sH <= 0 || sW <= 0 || dH <= 0 || dW <= 0) return;
    const double sy = (double)sH / dH, sx = (double)sW / dW;
    for (int y = 0; y < dH; y++) {
        double fy = (y + 0.5) * sy - 0.5;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        double wy = fy - y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y1 > sH - 1) y1 = sH - 1;
        for (int x = 0; x < dW; x++) {
            double fx = (x + 0.5) * sx - 0.5;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            double wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x1 > sW - 1) x1 = sW - 1;
            const uint8_t *p00 = &src[(y0 * sW + x0) * 3];
            const uint8_t *p01 = &src[(y0 * sW + x1) * 3];
            const uint8_t *p10 = &src[(y1 * sW + x0) * 3];
            const uint8_t *p11 = &src[(y1 * sW + x1) * 3];
            uint8_t *d = &dst[(y * dW + x) * 3];
#if defined(TFCV_HAVE_NEON)
            // 权重 Q8 定点 (uqrshrn 移位范围 1..8; Q8 下 acc<=255*256=65280 不溢出 uint16)
            int w00 = (int)((1 - wx) * (1 - wy) * 256 + 0.5);
            int w01 = (int)(wx * (1 - wy) * 256 + 0.5);
            int w10 = (int)((1 - wx) * wy * 256 + 0.5);
            int w11 = (int)(wx * wy * 256 + 0.5);
            uint16x8_t a00 = vmovl_u8(vld1_u8(p00));
            uint16x8_t a01 = vmovl_u8(vld1_u8(p01));
            uint16x8_t a10 = vmovl_u8(vld1_u8(p10));
            uint16x8_t a11 = vmovl_u8(vld1_u8(p11));
            uint16x8_t acc = vmulq_n_u16(a00, (uint16_t)w00);
            acc = vmlaq_n_u16(acc, a01, (uint16_t)w01);
            acc = vmlaq_n_u16(acc, a10, (uint16_t)w10);
            acc = vmlaq_n_u16(acc, a11, (uint16_t)w11);
            uint8x8_t res = vqrshrn_n_u16(acc, 8);   // 四舍五入窄回 uint8
            d[0] = vget_lane_u8(res, 0);
            d[1] = vget_lane_u8(res, 1);
            d[2] = vget_lane_u8(res, 2);
#else
            for (int c = 0; c < 3; c++) {
                double val = p00[c] * (1 - wx) * (1 - wy) + p01[c] * wx * (1 - wy)
                           + p10[c] * (1 - wx) * wy + p11[c] * wx * wy;
                d[c] = (uint8_t)(val + 0.5);
            }
#endif
        }
    }
}

// HWC(uint8, HxWx3) -> NCHW(uint8, 3xHxW) 转置. NEON: vld3q 一次解交错 16 像素.
inline void hwc_to_nchw(const uint8_t *hwc, int H, int W, uint8_t *nchw) {
    const size_t plane = (size_t)H * W;
    uint8_t *rP = nchw;
    uint8_t *gP = nchw + plane;
    uint8_t *bP = nchw + plane * 2;
    for (int y = 0; y < H; y++) {
        const uint8_t *row = hwc + (size_t)y * W * 3;
        uint8_t *rRow = rP + (size_t)y * W;
        uint8_t *gRow = gP + (size_t)y * W;
        uint8_t *bRow = bP + (size_t)y * W;
        int x = 0;
#if defined(TFCV_HAVE_NEON)
        for (; x + 16 <= W; x += 16) {
            uint8x16x3_t v = vld3q_u8(row + (size_t)x * 3);  // 解交错 16 像素 RGB
            vst1q_u8(rRow + x, v.val[0]);
            vst1q_u8(gRow + x, v.val[1]);
            vst1q_u8(bRow + x, v.val[2]);
        }
#endif
        for (; x < W; x++) {
            rRow[x] = row[x * 3 + 0];
            gRow[x] = row[x * 3 + 1];
            bRow[x] = row[x * 3 + 2];
        }
    }
}

// 解码当前帧 + 空间预处理 (stretch/centercrop/letterbox) + 写进 C++ TFTensor.
// 这是 StreamInfer.inferOne 与 ImgReader/VideoCapture.into_tensor 的**唯一共享实现**,
// 二者传同一个 TFTensor 进来 => 逐字节完全一致.
// letterbox 自行控制输出尺寸 (与 ultralytics LetterBox 对齐: nw=round(sW*r) 等),
// resize + HWC->NCHW 均用 NEON. 纯 C++, 调用方负责 GIL 释放.
void apply_preprocess(TFVision &v, TFCV::DECODER_FLAGS flags, TFTensor &tensor,
                      const std::string &preprocess, int padValue, Result &r) {
    const std::vector<int> &tsh = GetTensorShape(tensor);
    int tH = tsh.size() > 2 ? tsh[2] : 0;
    int tW = tsh.size() > 3 ? tsh[3] : 0;
    int sH = TFCV::GetHeight(v);
    int sW = TFCV::GetWidth(v);

    if (preprocess == "centercrop" && sH > 0 && sW > 0 && tH > 0 && tW > 0) {
        int cw = std::min(sW, (int)std::round(sH * (double)tW / tH));
        int ch = (int)std::round(cw * (double)tH / tW);
        int left = (sW - cw) / 2, top = (sH - ch) / 2;
        TFCV::DumpImgData(v, tensor, 0, flags, TFCV::CropSize(left, top, cw, ch));
        r.crop = {left, top, cw, ch};
    } else if (preprocess == "letterbox" && sH > 0 && sW > 0 && tH > 0 && tW > 0) {
        // 与 ultralytics LetterBox 对齐: r=min(tW/sW,tH/sH); nw=round(sW*r), nh=round(sH*r);
        // left=round((tW-nw)/2 - 0.1), top=round((tH-nh)/2 - 0.1).
        double ratio = std::min((double)tW / sW, (double)tH / sH);
        int nw = (int)std::round(sW * ratio);
        int nh = (int)std::round(sH * ratio);
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        int padL = (int)std::round((tW - nw) / 2.0 - 0.1);
        int padT = (int)std::round((tH - nh) / 2.0 - 0.1);
        if (padL < 0) padL = 0;
        if (padT < 0) padT = 0;

        // 1) 解码原始帧 (HWC)
        std::vector<uint8_t> src;
        TFCV::DumpImgData(v, src, flags);
        // 2) NEON 双线性缩放到精确的 nh x nw (自行控制尺寸, 无歧义)
        std::vector<uint8_t> scaled((size_t)nh * nw * 3);
        bilinear_resize_hwc(src.data(), sH, sW, scaled.data(), nh, nw);
        // 3) 居中放进 tH x tW 画布, pad 区填 padValue
        std::vector<uint8_t> out((size_t)tH * tW * 3, (uint8_t)padValue);
        int copyH = std::min(nh, tH - padT);
        int copyW = std::min(nw, tW - padL);
        for (int y = 0; y < copyH; y++)
            std::memcpy(&out[((size_t)(y + padT)) * tW * 3 + (size_t)padL * 3],
                        &scaled[(size_t)y * nw * 3], (size_t)copyW * 3);
        // 4) NEON HWC -> NCHW 写进 tensor
        uint8_t *td = (uint8_t *)GetTensordata(tensor);
        if (td) hwc_to_nchw(out.data(), tH, tW, td);
        r.lbox = {ratio, (double)padL, (double)padT};
    } else {
        TFCV::DumpImgData(v, tensor, 0, flags);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 独立对象: ImgReader / VideoCapture — 只负责解码, 返回 numpy 帧,
// 由 Python 自行做预处理后再喂进 executor (executor.GetInputs()[0].fromNumpy(...)).
// ---------------------------------------------------------------------------
class PyImgReader {
public:
    PyImgReader() { v_ = TFCV::NewImgReader(); }
    void open(const std::string &path, int flags) {
        flags_ = (TFCV::DECODER_FLAGS)flags;
        if (!TFCV::OpenURL(v_, path))
            throw std::runtime_error("ImgReader OpenURL failed: " + path);
        // 图片: OpenURL 后已自动解码, frame() 直接 dump 当前图.
    }
    py::array frame(float scale, TFCV::CropSize crop) {
        return vision_to_numpy(v_, flags_, scale, crop);
    }
    // 解码 + 空间预处理 (stretch/centercrop/letterbox) + 写进 tensor (与 StreamInfer 同实现).
    // 返回 meta dict (centercrop->crop / letterbox->letterbox, 供坐标反映射).
    py::dict into_tensor(py::object tensor, const std::string &preprocess, int padValue) {
        uintptr_t h = py::cast<uintptr_t>(tensor.attr("_tftensor_handle")());
        TFTensor *tPtr = reinterpret_cast<TFTensor *>(h);
        Result r;
        {
            py::gil_scoped_release gil;   // 解码+写 tensor 纯 C++
            apply_preprocess(v_, flags_, *tPtr, preprocess, padValue, r);
        }
        py::dict meta;
        if (!r.crop.empty()) meta["crop"] = r.crop;
        if (!r.lbox.empty()) meta["letterbox"] = r.lbox;
        return meta;
    }
    int width() { return TFCV::GetWidth(v_); }   // 源帧原始宽
    int height() { return TFCV::GetHeight(v_); }  // 源帧原始高
private:
    TFVision v_;
    TFCV::DECODER_FLAGS flags_ = TFCV::TFCV_BGR;
};

class PyVideoCapture {
public:
    PyVideoCapture(std::string mediaType, std::string decoderPath)
        : mediaType_(mediaTypeFromStr(mediaType)), decoderPath_(std::move(decoderPath)) {}
    void open(const std::string &url, int flags) {
        flags_ = (TFCV::DECODER_FLAGS)flags;
        v_ = TFCV::NewVideoReader(mediaType_, decoderPath_);
        if (!TFCV::OpenURL(v_, url))
            throw std::runtime_error("VideoCapture OpenURL failed: " + url);
        opened_ = true;
    }
    void set_fps(int fps) { if (opened_ && fps > 0) TFCV::SetFps(v_, fps); }
    // ReadFrame: 1=成功 0=流结束 -1=错误 (释放 GIL)
    int read() { return TFCV::ReadFrame(v_); }
    py::array frame(float scale, TFCV::CropSize crop) {
        return vision_to_numpy(v_, flags_, scale, crop);
    }
    // 解码 + 空间预处理 (stretch/centercrop/letterbox) + 写进 tensor. 返回 meta dict.
    py::dict into_tensor(py::object tensor, const std::string &preprocess, int padValue) {
        uintptr_t h = py::cast<uintptr_t>(tensor.attr("_tftensor_handle")());
        TFTensor *tPtr = reinterpret_cast<TFTensor *>(h);
        Result r;
        {
            py::gil_scoped_release gil;
            apply_preprocess(v_, flags_, *tPtr, preprocess, padValue, r);
        }
        py::dict meta;
        if (!r.crop.empty()) meta["crop"] = r.crop;
        if (!r.lbox.empty()) meta["letterbox"] = r.lbox;
        return meta;
    }
    int width() { return TFCV::GetWidth(v_); }
    int height() { return TFCV::GetHeight(v_); }
    int fps() { return TFCV::GetFps(v_); }
    void close() { if (opened_) { TFCV::Close(v_); opened_ = false; } }
    ~PyVideoCapture() { if (opened_) TFCV::Close(v_); }
private:
    TFCV::MediaTYPE mediaType_ = TFCV::tH264;
    std::string decoderPath_;
    TFCV::DECODER_FLAGS flags_ = TFCV::TFCV_BGR;
    TFVision v_;
    bool opened_ = false;
};



class PyStreamInfer {
public:
    PyStreamInfer(std::string modelFb, std::string config,
                  std::string sourceKind, py::object source,
                  std::string mediaType, std::string decoderPath, int fps, bool loop,
                  int flags, std::vector<float> scale, std::vector<float> means,
                  std::string inputName, int queueSize, bool keepFrame,
                  std::string preprocess, int padValue)
        : modelFb_(std::move(modelFb)), config_(std::move(config)),
          sourceKind_(sourceKind == "image" ? 1 : 0),
          mediaType_(mediaType), decoderPath_(decoderPath), fps_(fps), loop_(loop),
          flags_(flags), scale_(std::move(scale)), means_(std::move(means)),
          inputName_(inputName), queueSize_(queueSize > 0 ? queueSize : 8),
          keepFrame_(keepFrame), preprocess_(preprocess), padValue_(padValue) {
        if (sourceKind_ == 0) {
            videoUrl_ = py::cast<std::string>(source);
        } else {
            imagePaths_ = py::cast<std::vector<std::string>>(source);
        }
        // 1) 加载模型 + 编译 executor
        ctx_ = LoadProto(modelFb_);
        exe_ = CompileExecutor(ctx_, true, config_);
        // 3) 取输入张量 (帧将解码进这里)
        auto inputs = GetInputTensors(exe_);
        if (inputs.empty()) throw std::runtime_error("model has no input tensor");
        inTensor_ = inputs[0];
        if (inputName_.empty()) inputName_ = GetTensorName(inTensor_);  // 自动取输入名
        // 2) 可选: executor 内归一化 (scale/mean) — 放到 inTensor_ 取到之后, 用真实输入名
        if (!scale_.empty() || !means_.empty()) {
            SetPreprocess(exe_, inputName_, scale_, means_);
        }
    }

    ~PyStreamInfer() { stop(); }

    void start() {
        if (running_.exchange(true)) return;  // 已在跑
        stopReq_ = false;
        produced_ = consumed_ = 0;
        // 打开源
        if (sourceKind_ == 0) {
            src_ = TFCV::NewVideoReader(toMediaType(mediaType_), decoderPath_);
            if (!TFCV::OpenURL(src_, videoUrl_))
                throw std::runtime_error("TFCV OpenURL failed: " + videoUrl_);
            if (fps_ > 0) TFCV::SetFps(src_, fps_);
        } else {
            src_ = TFCV::NewImgReader();
        }
        th_ = std::thread(&PyStreamInfer::worker, this);
    }

    void stop() {
        if (!running_.load()) return;
        stopReq_ = true;
        {
            std::lock_guard<std::mutex> lk(m_);
            // 清空队列唤醒可能阻塞在 push 的 worker
            std::queue<Result> empty;
            std::swap(q_, empty);
        }
        cvPush_.notify_all();
        cvPop_.notify_all();
        if (th_.joinable()) th_.join();
        TFCV::Close(src_);
        running_ = false;
    }

    bool is_running() const { return running_.load() && !stopReq_.load(); }

    // 阻塞/超时取一个结果; 超时空队列返回 None.
    py::object poll(int timeoutMs) {
        Result r;
        {
            std::unique_lock<std::mutex> lk(m_);
            // 释放 GIL 等队列, 让其它 Python 线程能跑
            py::gil_scoped_release gil;
            auto ok = cvPop_.wait_for(lk, std::chrono::milliseconds(timeoutMs > 0 ? timeoutMs : 0),
                                      [this] { return !q_.empty() || stopReq_.load(); });
            (void)ok;
            if (q_.empty()) return py::none();   // GIL 重新持有后返回
            r = std::move(q_.front());
            q_.pop();
        }
        cvPush_.notify_one();                    // 通知 worker 有空位
        consumed_++;

        py::dict d;
        d["status"] = r.status;
        d["frame_index"] = r.frameIndex;
        if (r.status == -1) d["error"] = r.errMsg;

        // 原始帧 -> numpy
        if (keepFrame_ && !r.frame.empty() && r.h > 0 && r.w > 0) {
            std::vector<py::ssize_t> fsh = {r.h, r.w, r.c};
            py::array_t<uint8_t> fr(fsh);
            std::memcpy(fr.request().ptr, r.frame.data(), r.frame.size());
            d["frame"] = fr;
        } else {
            d["frame"] = py::none();
        }

        // 空间预处理元信息 (把模型坐标反映射回原图用)
        if (!r.crop.empty()) d["crop"] = r.crop;            // centercrop: [left,top,cw,ch]
        if (!r.lbox.empty()) d["letterbox"] = r.lbox;       // letterbox:  [ratio,pad_left,pad_top]

        // 输出张量 -> list[numpy] + list[(qscale,qzp)]
        py::list outs, quants;
        for (auto &t : r.outputs) {
            py::object qinfo;
            outs.append(tensor_to_numpy(t, qinfo));
            quants.append(qinfo);
        }
        d["outputs"] = outs;
        d["quant"] = quants;
        return std::move(d);
    }

    py::dict stats() {
        std::lock_guard<std::mutex> lk(m_);
        py::dict d;
        d["produced"] = (int64_t)produced_.load();
        d["consumed"] = (int64_t)consumed_.load();
        d["queue_len"] = (int)q_.size();
        d["queue_cap"] = queueSize_;
        d["running"] = is_running();
        return d;
    }

private:
    static TFCV::MediaTYPE toMediaType(const std::string &s) {
        if (s == "png") return TFCV::tPNG;
        if (s == "bmp") return TFCV::tBMP;
        if (s == "hevc" || s == "h265") return TFCV::tHEVC;
        return TFCV::tH264;  // 默认 h264
    }

    void pushWait(Result &r) {
        std::unique_lock<std::mutex> lk(m_);
        cvPush_.wait(lk, [this] { return (int)q_.size() < queueSize_ || stopReq_.load(); });
        if (stopReq_.load()) return;
        q_.push(std::move(r));
        cvPop_.notify_one();
    }

    bool inferOne(Result &r) {
        // 解码 + 空间预处理 + 写进输入张量 (与 into_tensor 共享同一实现 => 逐字节一致)
        apply_preprocess(src_, (TFCV::DECODER_FLAGS)flags_, inTensor_,
                         preprocess_, padValue_, r);
        // 原始分辨率帧 (供 Python 画框)
        if (keepFrame_) {
            TFCV::DumpImgData(src_, r.frame, (TFCV::DECODER_FLAGS)flags_);
            r.h = TFCV::GetHeight(src_);
            r.w = TFCV::GetWidth(src_);
            r.c = (flags_ == TFCV::TFCV_Gray) ? 1 : 3;
        }
        ForwardExecutorAlone(exe_);
        r.outputs = GetOutputTensors(exe_);
        produced_++;
        return true;
    }

    void worker() {
        int64_t idx = 0;
        try {
            if (sourceKind_ == 0) {
                // 视频流
                while (!stopReq_.load()) {
                    int ret = TFCV::ReadFrame(src_);
                    if (ret == 0) {  // EOS: 文件视频结束
                        if (!loop_) break;
                        TFCV::Close(src_);
                        src_ = TFCV::NewVideoReader(toMediaType(mediaType_), decoderPath_);
                        TFCV::OpenURL(src_, videoUrl_);
                        if (fps_ > 0) TFCV::SetFps(src_, fps_);
                        continue;
                    }
                    if (ret < 0) {
                        Result r; r.frameIndex = idx++; r.status = -1; r.errMsg = "read error";
                        pushWait(r);
                        continue;
                    }
                    Result r; r.frameIndex = idx++; r.status = 1;
                    inferOne(r);
                    pushWait(r);
                }
            } else {
                // 图片流
                while (!stopReq_.load()) {
                    for (const auto &p : imagePaths_) {
                        if (stopReq_.load()) break;
                        if (!TFCV::OpenURL(src_, p)) {
                            Result r; r.frameIndex = idx++; r.status = -1;
                            r.errMsg = "image open failed: " + p;
                            pushWait(r);
                            continue;
                        }
                        Result r; r.frameIndex = idx++; r.status = 1;
                        inferOne(r);
                        pushWait(r);
                    }
                    if (!loop_ || imagePaths_.empty()) break;
                }
            }
        } catch (const std::exception &e) {
            Result r; r.status = -1; r.errMsg = std::string("worker exception: ") + e.what();
            pushWait(r);
        }
        // 自然结束: 直接推一个 EOS 让消费者干净退出 (不受 stopReq_ 拦截, 否则 pushWait 会跳过).
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!stopReq_.load()) {  // 用户主动 stop() 时不推
                Result eos; eos.status = 0; eos.frameIndex = idx;
                q_.push(std::move(eos));
            }
            stopReq_ = true;
            cvPop_.notify_all();  // 唤醒可能在等结果的 poll
        }
    }

    // 配置
    std::string modelFb_, config_;
    int sourceKind_;
    std::string videoUrl_, mediaType_, decoderPath_;
    int fps_;
    bool loop_;
    int flags_;
    std::vector<float> scale_, means_;
    std::string inputName_;
    int queueSize_;
    bool keepFrame_;
    std::string preprocess_;   // "stretch" / "centercrop" / "letterbox"
    int padValue_;             // letterbox 填充值 (默认 114)
    std::vector<std::string> imagePaths_;

    // 运行时
    TFContext ctx_;
    TFExecutor exe_;
    TFVision src_;
    TFTensor inTensor_;
    std::thread th_;
    std::mutex m_;
    std::condition_variable cvPush_, cvPop_;
    std::queue<Result> q_;
    std::atomic<bool> running_{false}, stopReq_{false};
    std::atomic<int64_t> produced_{0}, consumed_{0};
};

PYBIND11_MODULE(_tfcv, m) {
    m.doc() = "TFCV streaming inference (NPU40T): source + executor bound in a C++ worker thread";
    // 注意: 不在此绑定 TFCV::DECODER_FLAGS 枚举 —— 主 TFDL2 模块已绑定该 C++ 类型,
    // pybind 不允许跨模块重复注册同一 C++ enum. flags 用 int 传入即可 (BGR=0/RGB=1/Gray=2).

    py::class_<PyStreamInfer>(m, "StreamInfer")
        .def(py::init<std::string, std::string, std::string, py::object,
                      std::string, std::string, int, bool, int,
                      std::vector<float>, std::vector<float>, std::string, int, bool,
                      std::string, int>(),
             py::arg("model"), py::arg("config"),
             py::arg("source_kind"), py::arg("source"),
             py::arg("media_type") = "h264", py::arg("decoder_path") = "",
             py::arg("fps") = 0, py::arg("loop") = true,
             py::arg("flags") = (int)TFCV::TFCV_RGB,
             py::arg("scale") = std::vector<float>(),
             py::arg("means") = std::vector<float>(),
             py::arg("input_name") = "", py::arg("queue_size") = 8,
             py::arg("keep_frame") = true,
             py::arg("preprocess") = "stretch",
             py::arg("pad_value") = 114)
        .def("start", &PyStreamInfer::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &PyStreamInfer::stop, py::call_guard<py::gil_scoped_release>())
        .def("poll", &PyStreamInfer::poll, py::arg("timeout_ms") = 1000)
        .def("is_running", &PyStreamInfer::is_running)
        .def("stats", &PyStreamInfer::stats);

    // ---- 独立解码对象: 只返回 numpy 帧, 预处理交给 Python ----
    py::class_<PyImgReader>(m, "ImgReader",
        "图片解码器. open(path) 后 frame() 返回 HxWxC uint8 numpy. "
        "预处理 (resize/归一化/量化) 在 Python 做, 再 executor.GetInputs()[0].fromNumpy(...).")
        .def(py::init<>())
        .def("open", &PyImgReader::open, py::arg("path"), py::arg("flags") = (int)TFCV::TFCV_BGR,
             py::call_guard<py::gil_scoped_release>())
        .def("frame", &PyImgReader::frame,
             py::arg("scale") = 1.0f,
             py::arg("crop") = std::make_tuple(-1, -1, -1, -1))
        .def("into_tensor", &PyImgReader::into_tensor,
             py::arg("tensor"), py::arg("preprocess") = "stretch", py::arg("pad_value") = 114,
             "解码当前图 + 空间预处理(stretch/centercrop/letterbox) 写进 tensor, 返回 meta dict")
        .def("width", &PyImgReader::width)
        .def("height", &PyImgReader::height);

    py::class_<PyVideoCapture>(m, "VideoCapture",
        "视频流解码器 (RTSP/文件, 可硬件解码). read()->1/0/-1, frame()->numpy. "
        "flags=BGR/RGB/Gray; decoder_path='/dev/mv500' 硬解, ''=软解.")
        .def(py::init<std::string, std::string>(),
             py::arg("media_type") = "h264", py::arg("decoder_path") = "")
        .def("open", &PyVideoCapture::open, py::arg("url"), py::arg("flags") = (int)TFCV::TFCV_BGR,
             py::call_guard<py::gil_scoped_release>())
        .def("set_fps", &PyVideoCapture::set_fps, py::arg("fps"),
             py::call_guard<py::gil_scoped_release>())
        .def("read", &PyVideoCapture::read, py::call_guard<py::gil_scoped_release>(),
             "返回 1=成功 0=流结束 -1=错误")
        .def("frame", &PyVideoCapture::frame,
             py::arg("scale") = 1.0f,
             py::arg("crop") = std::make_tuple(-1, -1, -1, -1))
        .def("into_tensor", &PyVideoCapture::into_tensor,
             py::arg("tensor"), py::arg("preprocess") = "stretch", py::arg("pad_value") = 114,
             "解码当前帧 + 空间预处理(stretch/centercrop/letterbox) 写进 tensor, 返回 meta dict")
        .def("width", &PyVideoCapture::width)
        .def("height", &PyVideoCapture::height)
        .def("fps", &PyVideoCapture::fps)
        .def("close", &PyVideoCapture::close, py::call_guard<py::gil_scoped_release>());
}
