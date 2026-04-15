//
// Created by test on 2020/3/23.
//

#ifndef NPU40T_TFMODULE_H
#define NPU40T_TFMODULE_H

#include "TFDL2_C_API.h"
#include <assert.h>
#include <TFCV/TFCV.h>
// Net Define function
using std::pair;
namespace TFDL_CAPI {
    namespace TFFunc {
        typedef std::pair<DataUnion,TFCAPI_DATATYPE> ScalarType;
        class TFDL2_CAPI_EXPORT TeleExecutor {
        public:
            TeleExecutor(TFContext tfContext, bool shareWeight, string config);

            TeleExecutor(const TeleExecutor &) = delete;

            TeleExecutor(TeleExecutor &&);

        private:
        };
        //BinaryOp
        TFDL2_CAPI_EXPORT  string Add(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Sub(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Mul(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Div(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Min(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Max(TFContext &, string inputA, string inputB);

        TFDL2_CAPI_EXPORT  string Add(TFContext &, string inputA, DataUnion data, TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string Sub(TFContext &, string inputA, DataUnion data, TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string Mul(TFContext &, string inputA, DataUnion data, TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string Div(TFContext &, string inputA, DataUnion data, TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string Add(TFContext &tfContext_, DataUnion data, TFCAPI_DATATYPE dtype, string inputA);

        TFDL2_CAPI_EXPORT  string Sub(TFContext &tfContext_, DataUnion data, TFCAPI_DATATYPE dtype, string inputA);

        TFDL2_CAPI_EXPORT  string Mul(TFContext &tfContext_, DataUnion data, TFCAPI_DATATYPE dtype, string inputA);

        TFDL2_CAPI_EXPORT  string Div(TFContext &tfContext_, DataUnion data, TFCAPI_DATATYPE dtype, string inputA);

        TFDL2_CAPI_EXPORT  string BoolGT(TFContext &, string input0,string input1);

        TFDL2_CAPI_EXPORT  string BoolGE(TFContext &, string input0,string input1);

        TFDL2_CAPI_EXPORT  string BoolLT(TFContext &, string input0,string input1);

        TFDL2_CAPI_EXPORT  string BoolLE(TFContext &, string input0,string input1);

        TFDL2_CAPI_EXPORT  string BoolEQ(TFContext &, string input0,string input1);

        TFDL2_CAPI_EXPORT  string BoolGT(TFContext &, string input0,DataUnion input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolGE(TFContext &, string input0,DataUnion input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolLT(TFContext &, string input0,DataUnion input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolLE(TFContext &, string input0,DataUnion input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolEQ(TFContext &, string input0,DataUnion input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolGT(TFContext &, DataUnion input0,string input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolGE(TFContext &, DataUnion input0,string input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolLT(TFContext &, DataUnion input0,string input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolLE(TFContext &, DataUnion input0,string input1,TFCAPI_DATATYPE dtype);

        TFDL2_CAPI_EXPORT  string BoolEQ(TFContext &, DataUnion input0,string input1,TFCAPI_DATATYPE dtype);

        //nn OP
        TFDL2_CAPI_EXPORT  string LSTM(TFContext &, string input,int hidden_size,string direction);

        TFDL2_CAPI_EXPORT  string
        Convolution(TFContext &, string input, vector<int> kernel, vector<int> pad, vector<int> stride, int dilation,
                    int outChannel, int group, bool hasBias);

        TFDL2_CAPI_EXPORT  string
        TensorflowConvolution(TFContext &, string bias, vector<int> kernel, bool Same, vector<int> stride, int dilation,
                              int outChannel, int group, bool hasBias);

        TFDL2_CAPI_EXPORT  string
        DeConvolution(TFContext &, string input, vector<int> kernel, vector<int> pad, vector<int> stride, int dilation,
                      int outChannel, int group, bool hasBias);

        TFDL2_CAPI_EXPORT  string Scale(TFContext &, string input, bool hasbias);

        TFDL2_CAPI_EXPORT  string LSTM(TFContext &, string input, string W, string R, string B,int hidden_size,string direction);

        TFDL2_CAPI_EXPORT  string
        Convolution(TFContext &, string input,string weight,string bias, vector<int> kernel, vector<int> pad, vector<int> stride, int dilation,
                    int outChannel, int group);

        TFDL2_CAPI_EXPORT  string
        DeConvolution(TFContext &, string input,string weight,string bias, vector<int> kernel, vector<int> pad, vector<int> stride, int dilation,
                      int outChannel, int group);

        TFDL2_CAPI_EXPORT  string Scale(TFContext &, string input, string weight,string bias);

        TFDL2_CAPI_EXPORT  string
        MatMul(TFContext &, string inputA, string inputB, bool transA = false, bool transB = false,bool bias=false,bool bias2row=false);

        TFDL2_CAPI_EXPORT  string
        MatMul(TFContext &, string inputA, string inputB, string addbias, bool transA = false, bool transB = false,bool bias2row=false);

        TFDL2_CAPI_EXPORT  string
        MatMul(TFContext &tfContext_, string inputA,int dimm,int dimk, bool transA, bool transB,bool hasbias,bool bias2row=false);

        TFDL2_CAPI_EXPORT  string InnerProduct(TFContext &, string input, bool hasbias, int outchannels);

        TFDL2_CAPI_EXPORT  string InnerProduct(TFContext &, string input, string weight, string bias, int outchannels);

        TFDL2_CAPI_EXPORT  string GlobalAvePooling(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string
        TensorflowAvePooling(TFContext &, string input, vector<int> kernel, vector<int> stride, bool same);

        TFDL2_CAPI_EXPORT  string
        TensorflowMaxPooling(TFContext &, string input, vector<int> kernel, vector<int> stride, bool same);

        TFDL2_CAPI_EXPORT  string
        AvePooling(TFContext &, string input, vector<int> kernel, vector<int> pad, vector<int> stride,
                   bool ceilMode = true);

        TFDL2_CAPI_EXPORT  string
        MaxPooling(TFContext &, string input, vector<int> kernel, vector<int> pad, vector<int> stride,
                   bool ceilMode = true);

        //Activation Op
        TFDL2_CAPI_EXPORT  string ReLU(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string LeakyReLU(TFContext &, string input, float negativeslop);

        TFDL2_CAPI_EXPORT  string PReLU(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string ReLUX(TFContext &, string input, float threshold);

        TFDL2_CAPI_EXPORT  string Swish(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string HardSwish(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Tanh(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string GeLU(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Exp(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Sigmoid(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string HardSigmoid(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string HardSigmoid(TFContext &, string input,float alpha,float beta);

        TFDL2_CAPI_EXPORT  string Mish(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Softmax(TFContext &, string input, int axis);

        TFDL2_CAPI_EXPORT  string ELU(TFContext &, string input, float alpha);

        TFDL2_CAPI_EXPORT  string LayerNorm(TFContext &, string input,int axis);

        TFDL2_CAPI_EXPORT  string LayerNorm(TFContext &, string input,int axis,string scale,string bias);

        // shape change op
        TFDL2_CAPI_EXPORT  string Reshape(TFContext &, string input, vector<int> shape);

        TFDL2_CAPI_EXPORT  string Reshape(TFContext &, string input, string shape);

        TFDL2_CAPI_EXPORT  string Squeeze(TFContext &, string input, vector<int> dims);

        TFDL2_CAPI_EXPORT  string BroadCast(TFContext &, string input, vector<int> shape);

        TFDL2_CAPI_EXPORT  string BroadCast(TFContext &, string input, string shape);

        TFDL2_CAPI_EXPORT  string Expand(TFContext &, string input, string shape);

        TFDL2_CAPI_EXPORT  string expand_dims(TFContext &, string input, vector<int> dims);

        TFDL2_CAPI_EXPORT  string Flatten(TFContext &, string input, int startdim, int enddim);

        TFDL2_CAPI_EXPORT  string Flatten2Matrix(TFContext &tfContext_, string input, int axis);

        TFDL2_CAPI_EXPORT  string Pad(TFContext &, string input, vector<int> pad);

        TFDL2_CAPI_EXPORT  string Pad(TFContext &, string input, string pads);

        TFDL2_CAPI_EXPORT  string ZeroMask(TFContext &, string input, vector<int> mask);
        //reduce op
        TFDL2_CAPI_EXPORT  string ReduceMean(TFContext &, string input, vector<int> dims,bool keep_dims);

        TFDL2_CAPI_EXPORT  string ReduceSum(TFContext &, string input, vector<int> dims,bool keep_dims);

        TFDL2_CAPI_EXPORT  string ReduceMin(TFContext &, string input, vector<int> dims,bool keep_dims);

        TFDL2_CAPI_EXPORT  string ReduceMax(TFContext &, string input, vector<int> dims,bool keep_dims);

        TFDL2_CAPI_EXPORT  string MeanOp(TFContext &, string input, vector<int> dims,bool keep_dims);

        //Tensor Op
        TFDL2_CAPI_EXPORT  string Transpose(TFContext &, string input, vector<int> order);

        TFDL2_CAPI_EXPORT  string Concat(TFContext &, vector<string> inputs, int axis);

        TFDL2_CAPI_EXPORT  string Gather(TFContext &, string input, string indices="", int axis=0);

        TFDL2_CAPI_EXPORT  string Flip(TFContext &, string input, vector<int> dims);

        TFDL2_CAPI_EXPORT  vector<string> Slice(TFContext &, string input, int axis, vector<int> split);

        TFDL2_CAPI_EXPORT  vector<string> Split(TFContext &tfContext_, string input, int axis, int split);

        TFDL2_CAPI_EXPORT  string Quantize(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string DeQuantize(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Pow(TFContext &, string input,int power);

        TFDL2_CAPI_EXPORT  string Sqrt(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Ceil(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Floor(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Log2(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Shape(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string ConstantOfShape(TFContext &, string input,TFCAPI_DATATYPE dtype,DataUnion value);

        TFDL2_CAPI_EXPORT  string Where(TFContext &, string conditionMat,string data0,string data1);

        TFDL2_CAPI_EXPORT  string Where(TFContext &, string conditionMat,string data0,ScalarType data1);

        TFDL2_CAPI_EXPORT  string Where(TFContext &, string conditionMat,ScalarType data0,string data1);

        TFDL2_CAPI_EXPORT  string Where(TFContext &, string conditionMat,ScalarType data0,ScalarType data1);

        TFDL2_CAPI_EXPORT  string Cast(TFContext &, string input,TFCAPI_DATATYPE dst_dtype);

        TFDL2_CAPI_EXPORT  string Requantize(TFContext &, string input,vector<uint8_t>);

        TFDL2_CAPI_EXPORT  string ArgMax(TFContext &, string input,int dim,bool keep_dims);

        TFDL2_CAPI_EXPORT  string ArgMin(TFContext &, string input,int dim,bool keep_dims);

        // cv Op
        /*
        align_corners : 
            false: half_pixel, //x_original = [(x_resized + 0.5) / scale - 0.5]
            true :align_corners, //x_original = [x_resized * (length_original - 1) / (length_resized - 1)]
        */
        TFDL2_CAPI_EXPORT  string BilnearReSize(TFContext &, string input, int outheight, int outwidth,bool align_corners);

        TFDL2_CAPI_EXPORT  string NearestReSize(TFContext &, string input, int outheight, int outwidth);
        /*
        align_corners : 
            false: half_pixel, //x_original = [(x_resized + 0.5) / scale - 0.5]
            true :align_corners, //x_original = [x_resized * (length_original - 1) / (length_resized - 1)]
        */
        TFDL2_CAPI_EXPORT  string BilnearReSize(TFContext &, string input, float scale,bool align_corners);
        /*
        align_corners : 
            false: half_pixel, //x_original = [(x_resized + 0.5) / scale - 0.5]
            true :align_corners, //x_original = [x_resized * (length_original - 1) / (length_resized - 1)]
        */
        TFDL2_CAPI_EXPORT  string NearestReSize(TFContext &, string input, float scale);
        /*
        align_corners : 
            false: half_pixel, //x_original = [(x_resized + 0.5) / scale - 0.5]
            true :align_corners, //x_original = [x_resized * (length_original - 1) / (length_resized - 1)]
        */
        TFDL2_CAPI_EXPORT  string BilnearReSize(TFContext &, string input, string shape,bool align_corners);

        TFDL2_CAPI_EXPORT  string NearestReSize(TFContext &, string input, string shape);

        TFDL2_CAPI_EXPORT  string Upsample(TFContext &, string input, int coeff);

        TFDL2_CAPI_EXPORT  string CropAndResize(TFContext &, string input,string alignmask,int outheight,int outwidth);

        TFDL2_CAPI_EXPORT  string RoiPooling(TFContext &, string input,string Rois,int outheight,int outwidth,float spatial_scale);

        TFDL2_CAPI_EXPORT  string Range(TFContext &, string start,string limit,string delta);

        TFDL2_CAPI_EXPORT  string ReArrange(TFContext &, string input,int mode,vector<int> oldshape = {});

        TFDL2_CAPI_EXPORT  string MaskSoftmax(TFContext &, string input,int axis,bool upper,int diagonals,int mod);

        TFDL2_CAPI_EXPORT  string Crop(TFContext &, string img,string Crop);

        TFDL2_CAPI_EXPORT  string Crop(TFContext &, string img,string start,string end,string axis,string step);

        TFDL2_CAPI_EXPORT  string Crop(TFContext &, string img,vector<std::tuple<int,int,int>>crops,int startAxis=2);

        TFDL2_CAPI_EXPORT  string WarpWithOpFlow(TFContext &, string input,string flow);

        TFDL2_CAPI_EXPORT  string WarpWithAffine(TFContext &, string input,string transMat,int outheight,int outwidth,bool padMethod);

        TFDL2_CAPI_EXPORT  string WarpWithPerspect(TFContext &, string input,string transMat,int outheight,int outwidth,bool padMethod);

        TFDL2_CAPI_EXPORT  string WarpWithGridFace(TFContext &tfContext_, string input,string transMat,int gridSize,int outheight,int outwidth,bool padMethod);

        TFDL2_CAPI_EXPORT  string GreedyCTC(TFContext &, string input);
        TFDL2_CAPI_EXPORT  string NextFrame(TFContext &, string input);

        TFDL2_CAPI_EXPORT  string Unfold(TFContext &, string input,vector<int> kernel, vector<int> pad, vector<int> stride, int dilation,bool reverse=false);

        TFDL2_CAPI_EXPORT  string Correlation(TFContext &, string input1,string input2,int kernel_size,int pad_size,int max_displacement,int stride1,int stride2);

        TFDL2_CAPI_EXPORT  string Einsum(TFContext &, string operation,vector<string> inputs);

        TFDL2_CAPI_EXPORT  string
        ReadImg(TFContext &tfContext_, int imgLen, vector<float> scale, vector<float> mean, vector<int> shape,
                TFCAPI_DATATYPE outDatatype, TFCV::DECODER_FLAGS decoderFlags);

        TFDL2_CAPI_EXPORT  string
        Placeholder(TFContext &tfContext_, int batch, vector<float> scale, vector<float> mean, vector<int> shape,
                    TFCAPI_DATATYPE outDatatype);

        TFDL2_CAPI_EXPORT  string
        Placeholder(TFContext &tfContext_, vector<float> scale, vector<float> mean, vector<int> shape,
                    TFCAPI_DATATYPE outDatatype);

        TFDL2_CAPI_EXPORT  string
        Variable(TFContext &tfContext_, vector<int> shape,
                    TFCAPI_DATATYPE outDatatype);

        //Custom
        TFDL2_CAPI_EXPORT  void
        Custom(TFContext &, vector<string> inputs, vector<string> outputs, string OpName, string &jsonconfig);

        TFDL2_CAPI_EXPORT  void
        Assign(TFContext &, string from,string to);
        // Function
        TFDL2_CAPI_EXPORT  void Flatten(TFTensor inputs);

        TFDL2_CAPI_EXPORT  void Reshape(TFTensor inputs);

        TFDL2_CAPI_EXPORT  void Squeeze(TFTensor inputs);

    }
}



#endif //NPU40T_TFMODULE_H
