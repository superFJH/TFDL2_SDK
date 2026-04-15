//
// Created by test on 2019/10/30.
//

#ifndef NPU40T_TFDL2_C_API_H
#define NPU40T_TFDL2_C_API_H
#include <string>
#include <vector>
#include <map>
#include <set>
#include "TFDL2_Common.h"

namespace TFDL_CAPI {
/*!
 *  @brief: get version of SDK
 */
    TFDL2_CAPI_EXPORT extern string TFDL2_Version();
/*!
 *  @brief: Set SDK logger with user-define TLOG
 */
    TFDL2_CAPI_EXPORT extern void TFDL2_INITLOG(TLOG *NewLog);
/*!
 * @brief:  Build empty TFExecutor, without net struct and weight.
 *          User can use ModifyTFExecutor to build net and random weight in the TFExecutor
 * @param:  contextName with user define
 */
    TFDL2_CAPI_EXPORT extern TFContext NewTFContext(string contextName);
/*!
 * @brief:  Load net struct and weight data from proto file in Mem
 *          Be able to duplicate one more Nets
 *          But this func won't allocate Memory for Internal Data in the Net
 *          and won't build Running Cmds. So forwarding or set/get Tensor mem is forbidden.
 *          Only TFExecutor can support these.
 *          On the Other hand, LoadProto will cover the original TFExecutor
 * @param:  ptr is the mem start addr
 * @param:  len of mem size
 */
    TFDL2_CAPI_EXPORT extern TFContext LoadProto(string protopath);
/*!
 * @brief:  Load net struct and weight data from proto file in Mem
 *          Be able to duplicate one more Nets
 *          But this func won't allocate Memory for Internal Data in the Net
 *          and won't build Running Cmds. So forwarding or set/get Tensor mem is forbidden.
 *          Only TFExecutor can support these.
 *          On the Other hand, LoadProto will cover the original TFExecutor
 * @param:  ptr is the mem start addr
 * @param:  len of mem size
 */
    TFDL2_CAPI_EXPORT extern TFContext LoadProtoFromMem(uint8_t *ptr, size_t len);
/*!
 * @brief:  This function allow user use json-defined net to modify loaded Context struct
 *          the json like this:
 *          {
 *               "AddOnPass": [],
 *               "DeleteLayer": [],// the name of delete Node
 *               "Layer": [
 *                      // and the json-define Node which user want to change
 *               ]
 *           }
 * @param:  target context
 * @param:  string of json
 */
    TFDL2_CAPI_EXPORT extern void ModifyTFContext(TFContext, string netjson);
// TFContext Handle function
/*!
 *  @brief: Get The tensor from the context
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByName(TFContext tfContext, string dataname);
/*!
 *  @brief: Get The tensor from the context
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern string GetNodeAttrJsonByName(TFContext tfContext, string dataname);
/*!
 *  @brief: replace one Node with one or more nodes. but they should share in-out name
 *  @param: target context
 *  @param: Node Name
 *  @param: function of builder
 */
    TFDL2_CAPI_EXPORT extern void ReplaceNodeByName(TFContext tfContext, string nodename,
                                                    std::function<vector<string>(TFContext &, vector<string>)> builder);
/*!
 *  @brief: remove a node from context, the removed node can't have multi-input and one output.
 *  @param: target context
 *  @param: target node name
 */
    TFDL2_CAPI_EXPORT extern void RemoveNodeByName(TFContext tfContext, string nodename);
/*!
 *  @brief: get the context name
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern string GetContextName(TFContext tfContext);
/*!
 *  @brief: get the output nodes' name
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern vector<string> GetContextOutputs(TFContext tfContext);
/*!
 *  @brief: get the output nodes' name
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern vector<string> GetContextInputs(TFContext tfContext);
/*!
 *  @brief: set the output nodes's name
 *  @param: target context
 *  @param: output nodes' name
 */
    TFDL2_CAPI_EXPORT extern void SetContextOutputs(TFContext tfContext,vector<string>OutputNodes);
/*!
 *  @brief: get the tensor's quantinfo
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantizeInfo(TFContext tfContext, string dataname);
/*!
 *  @brief: Only open the Context, and return true.
 *          User can Append Op or Param.
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern bool OpenContext(TFContext);
/*!
 *  @brief: this function should use with OpenContext.
 *          tell the SDK, the Modifying operation is finished
 *  @param: the context opened
 */
    TFDL2_CAPI_EXPORT extern bool CloseContext(TFContext);
/*!
 *  @brief: remove a Registered parameter from Context
 *  @param: target context
 *  @param: parameter's name
 */
    TFDL2_CAPI_EXPORT extern bool RemoveParam(TFContext, string paramname);
/*!
 *  @brief: register a Tensor Parameter into the Context.
 *          Attention min_max must be 1 or equal to shape[0]
 *          if fail, return false
 *  @param: target context
 *  @param: parameter's name
 *  @param: shape of parameter
 *  @param: data type of parameter
 *  @param: ptr of data
 *  @param: size of mem
 *  @param: quantization info(min-max) size=(1 or shape[0])
 */
    TFDL2_CAPI_EXPORT extern bool
    RegisterParamTensor(TFContext, string paramname, vector<int> shape, TFCAPI_DATATYPE datatype,const void *ptr, size_t len,
                        vector<std::pair<float, float>> min_max = {});
/*!
 *  @brief: Get Registered Param from Context,
 *          if fail, will get TFTensor(nullptr)
 *  @param: target context
 *  @param: parameter's name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetParam(TFContext, string paramname);
/*!
 *  @brief: Register a Tensor's Quantization, the tensor only support 1 q-info
 *          if fail, return false
 *  @param: target context
 *  @param: tensor's name
 *  @param: qmax
 *  @param: qmin
 */
    TFDL2_CAPI_EXPORT extern bool RegisterQuantInfo(TFContext, string tensorName, float max, float min);
/*!
 *  @brief: Register a Tensor's Quantization, the tensor only support tensor q-info
 *          if fail, return false
 *  @param: target context
 *  @param: tensor's name
 *  @param: scale
 *  @param: zeropoint
 */
    TFDL2_CAPI_EXPORT extern bool RegisterQuantInfo(TFContext, string tensorName, float scale, int zeropoint);
/*!
 *  @brief: Dump context into file as protocol: TFDL1.0(json+bin)
 *  @param: target context
 *  @param: outfile name
 */
    TFDL2_CAPI_EXPORT extern void DumpContextToTFDL1(TFContext, string outname);
/*!
 *  @brief: Dump context into file as protocol: TFDL2.0(flatbuffer)
 *  @param: target context
 *  @param: outfile name
 */
    TFDL2_CAPI_EXPORT extern void DumpContextToFile(TFContext, string outname);
/*!
 *  @brief: Dump context into mem as protocol: TFDL2.0(flatbuffer)
 *  @param: target context
 *  @param: set context name
 *  @param: target stringstream
 */
    TFDL2_CAPI_EXPORT extern void DumpContext(TFContext, string outname, std::stringstream &stream);
/*!
 *  @brief: set print-level to executor(mean if print internal info while forwarding)
 *  @param: target executor
 *  @param: bool
 */
    TFDL2_CAPI_EXPORT extern void SetPrintInfo(TFExecutor tfExecutor, bool info);
/*!
 *  @brief: Set Preprocess info(scale,mean for image)
 *  @param: target executor
 *  @param: name of which input
 *  @param: scales
 *  @param: means
 */
    TFDL2_CAPI_EXPORT extern void SetPreprocess(TFExecutor tfExecutor, string inputname,
            vector<float> scale, vector<float> means);
/*!
 *  @brief: Register the CustomOp. this will register global CustomOp.
 *          if define same name CustomOp will cover the old one;
 *  @param: customOp name
 */
    TFDL2_CAPI_EXPORT extern TFCAPIOP &RegisterCustomOp(string customOptype);

/*!
 *  @brief: Register the CustomOp from lib file.
 *          This function will overwrite the same name customop
 *          User need make sure the feeding file, won't have Ops with same name
 *          if fail return -1, else return 0
 *  @param: custom op lib file
 */
    TFDL2_CAPI_EXPORT extern int RegisterCustomOpFromFile(string filepath);
/*!
 *  @brief: Using CustomOp to replace the Original Op.
 *          The customOpname will be used in dict, so please define unique customOpname.
 *          if the customOpname has not been registered, will throw error while running.
 *  @param: target executor
 *  @param: target Node name
 *  @param: custom op name
 */
    TFDL2_CAPI_EXPORT extern void ReplaceOpwithcustomOp(TFExecutor tfExecutor, string nodeName, string customOpname);
/*!
 *  @brief: Optimize executor , allocate memory, build Cmds Queue(Software Cmds and Hardware Cmds)
 *          Every Executor can only run this once then the TFExecutor can forward
 *          {
 *          "Layer":[
 *              {
 *                  "input": [],
 *                   "layerName": "data",
 *                   "layerType": "SetInput",
 *                   "output": ["data"],
 *                   "outputDataType" : "TFDtypeUint8"          \\ this will allow Mix-precision in one Executor
 *                   "ActivationType": "ReLU"
 *                   "param": {
 *                       "tfDataType": "TFDtypeUint8",
 *                       "mean": [128,128,128],
 *                       "scale": [0.008712500334,0.008712500334,0.008712500334],
 *                       "shape": [1,3,300,300]},
 *                   "weight": []
 *               }
 * @param shareWeight: Share Weight with Given TFContext. If true, User can't Free TFContext until all Executor, which share Weight of itself.
 * @param config : string of Json11, which set some options for Freezing
 */

    TFDL2_CAPI_EXPORT extern TFExecutor
    CompileExecutor(TFContext tfContext, bool shareWeight = true, string config = "{\"UseHardware\":false,\n"
                                                                                  "    \"FrugalMode\":true,\n"
                                                                                  "    \"optimize\":{\n"
                                                                                  "        \"DoAlign\":false,\n"
                                                                                  "        \"TryReverse\":false\n"
                                                                                  "    },\n"
                                                                                  "    \"InputShape\":[\n"
                                                                                  "    ]}");
/*!
 * @brief Update the RNN Hidden-Var with zero
 */
    TFDL2_CAPI_EXPORT extern void ResetExecutorHiddenVar(TFExecutor tfExecutor);
/*!
 *  @brief: Forward the Executor independently.
 *          This mean if User Run Multi-thread. These Executor will forward under the Scheduler's Control
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern void ForwardExecutor(TFExecutor tfExecutor);
/*!
 *  @brief: executors run in multi-thread, which will share cpu and npu,
 *          and window size need given to task-scheduler
 *  @param: window size
 */
    TFDL2_CAPI_EXPORT extern void SetForwardWindowSize(int);
/*!
 *  @brief: Forward the Executor independently.This mean if User Run Multi-thread.
 *          These Executors won't forward under the Scheduler's Control
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern void ForwardExecutorAlone(TFExecutor tfExecutor);
/*!
 * Get the output tensors from Executor
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetOutputTensors(TFExecutor tfExecutor);
/*!
 * Get the input tensors from Executor
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetInputTensors(TFExecutor tfExecutor);
/*!
 *  @brief: Get internal Tensor from Executor, But will Get NULL when
 *          User open FrugalMode(Memory Reused by Each Node), or Hardware Compiler merge
 *          the tensor. So please use this function carefully.
 *  @param: target executor
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByName(TFExecutor tfExecutor, string);

/*!
 *  @brief: Get Numbers of Tensor from Executor
 *          if Fruglemode, this will return -1;
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern int GetTensorNumbers(TFExecutor tfExecutor);

/*!
 *  @brief: Get internal Tensor from Executor, use index.
 *          this index can't large than Numbers of Tensors
 *          if Fruglemode , this will return nullptr
 *  @param: tfExcutor executor
 *  @param: index
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByIndex(TFExecutor tfExecutor, int index);

/*!
 *  @brief: Get output data`s float pointer
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern vector<float *> GetFloatOutputData(TFExecutor tfExecutor);
/*!
 *  @brief: New Quantize Info
 *  @param: qmax
 *  @param: qmin
 *  @param: quantize bit
 */
    TFDL2_CAPI_EXPORT extern Quantization NewQuantizeInfo(vector<float> max, vector<float> min, int bit = 8);
/*!
 *  @brief: Get Tensor's quant-info from executor
 *  @param: target executor
 *  @param: Tensor name
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantizeInfo(TFExecutor tfExecutor, string dataname);

// handle of TFServer
/*!  (deprecated)
 *  @brief: build TFServer, which will try run one context with more than one NPU
 *
 */
    TFDL2_CAPI_EXPORT extern TFServer
    CompileServer(TFContext tfContext, bool shareWeight = true, string config = "{\"UseHardware\":true,\n"
                                                                                "    \"FrugalMode\":true,\n"
                                                                                "    \"optimize\":{\n"
                                                                                "        \"DoAlign\":true,\n"
                                                                                "        \"TryReverse\":true\n"
                                                                                "    },\n"
                                                                                "    \"InputShape\":[\n"
                                                                                "    ]}");
/*!  (deprecated)
 *  @brief: Get the output tensors from TFServer
 *  @param: target server
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetOutputTensors(TFServer tfServer);
/*!  (deprecated)
 *  @brief: Get the input tensors from TFServer
 *  @param: target server
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetInputTensors(TFServer tfServer);
/*!  (deprecated)
 *  @brief: Get the tensors from TFServer
 *  @param: target server
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByName(TFServer tfServer, string);

    TFDL2_CAPI_EXPORT extern bool StartServer(TFServer tfServer, int threadnum);

    TFDL2_CAPI_EXPORT extern bool StopServer(TFServer tfServer);

    TFDL2_CAPI_EXPORT extern bool FeedServer(TFServer tfServer, std::function<void(void *)> callback, void *arg);

    TFDL2_CAPI_EXPORT extern bool PopServer(TFServer tfServer, std::function<void(void *)> callback, void *arg);

//TFMath Handle function
/*!
 *  @brief: dequantize the tensor data into float
 *  @param: dst ptr
 *  @param: src ptr
 *  @param: data len
 *  @param: quantize-info
 *  @param: quantize in which channel for multi-channel quantization
 */
    TFDL2_CAPI_EXPORT extern bool
    DeQuantizeTensorData(float *dst, uint8_t *src, long len, Quantization quantizeInfo, int channel = 0);
/*!
 *  @brief: quantize the tensor data
 *  @param: dst ptr
 *  @param: src ptr
 *  @param: data len
 *  @param: quantize-info
 *  @param: quantize in which channel for multi-channel quantization
 */
    TFDL2_CAPI_EXPORT extern bool QuantizeTensorData(uint8_t *dst, float *src, long len, Quantization quantizeInfo,int channel=0);

    TFDL2_CAPI_EXPORT extern bool SimpleMapdata(uint8_t *dst, uint8_t *src, uint8_t *maptable, long len);
    // transpose the tensor to dst which allocate outside
    TFDL2_CAPI_EXPORT extern void TransPosedata(void* dst, TFTensor src,vector<int>axis);

// TFTensor Handle function
/*!
 *  @brief: get brief info of tensor
 */
    TFDL2_CAPI_EXPORT extern string ToString(TFTensor tfTensor);
/*!
 * @brief allocate new tensor from TFContext, and name will be auto-assigned
 * @return TFTensor
 */
    TFDL2_CAPI_EXPORT extern TFTensor NewTensor(TFContext);
/*!
 * @brief copy and resize tensor from another one
 * @param from
 * @param to
 */
    TFDL2_CAPI_EXPORT extern void TensorCopy(TFTensor from, TFTensor to);
/*!
 *  @brief: get name info of tensor
 */
    TFDL2_CAPI_EXPORT extern string GetTensorName(TFTensor tfTensor);
/*!
 *  @brief: Only used in the CustomOp Reshape Op.
 *          resize the tensor in logic, won't allocate
 *  @param: tfContext: Context contain the tensor
 *  @param: tfTensor: target tensor
 *  @param: shape: target shape
 */
    TFDL2_CAPI_EXPORT extern bool ReSizeTensor(TFTensor tfTensor, vector<int> shape);
/*!
 *  @brief: allocate mem for tensor
 *  @param: tfContext: Context contain the tensor
 *  @param: tfTensor: target tensor
 */
    TFDL2_CAPI_EXPORT extern bool AllocateCpuMem(TFTensor tfTensor);
/*!
 *  @brief: set the tensor datatype in logic
 *  @param: target context contain the tensor
 *  @param: target datatype
 */
    TFDL2_CAPI_EXPORT extern bool SetTensorType(TFTensor tfTensor, TFCAPI_DATATYPE tfcapiDatatype);

/*!
 *  @brief: get shape info of tensor
 */
    TFDL2_CAPI_EXPORT extern vector<int> GetTensorShape(TFTensor tfTensor);
/*!
 *  @brief: Get Quantization info from tensor.
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantize(TFTensor tfTensor);
/*!
 *  @brief: get dtype info of tensor
 */
    TFDL2_CAPI_EXPORT extern TFCAPI_DATATYPE GetTensorType(TFTensor tfTensor);
/*!
 *  @brief: if Tensor is not scalar, can use this function
 */
    TFDL2_CAPI_EXPORT extern void *GetTensordata(TFTensor tfTensor);
/*!
 *  @brief: if the Tensor is scalar, must use this function
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, int);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, float);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, double);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, int64_t);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, uint8_t);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, uint16_t);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, string);

    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, char const *);
/*!
 *  @brief: get Count info of tensor
 */
    TFDL2_CAPI_EXPORT extern long GetTensorCount(TFTensor tfTensor, int st, int ed);
/*!
 *  @brief: get Count info of tensor
 */
    TFDL2_CAPI_EXPORT extern long GetTensorCount(TFTensor tfTensor, int st);
/*!
 *  @brief: get DataSize info of tensor
 */
    TFDL2_CAPI_EXPORT extern size_t GetTensorDataSize(TFTensor tfTensor);
/*!
 *  @brief: get dim info of tensor
 */
    TFDL2_CAPI_EXPORT extern int GetTensorDim(TFTensor tfTensor, int dim);
//TFNode Handle function
/*!
 *  @brief: return the node information from the TFNode
 */
    TFDL2_CAPI_EXPORT extern NodeInfo GetNodeInfo(TFNode tfNode);
/*!
 *  @brief: User can design their own Struct and pass it to Node
 *          Then user can run Eval or reshape with these structs,
 *          By GetCustomParam. But TFDL2 SDK won't free these memory
 *          User need free these memory in Free
 */
    TFDL2_CAPI_EXPORT extern string GetNodeCustomJsonStr(TFNode tfNode);
/*!
 * @brief Get the CustomParam ptr from Node
 * @param tfNode
 * @return User defined struct's ptr
 */
    TFDL2_CAPI_EXPORT extern void *GetNodeCustomParam(TFNode tfNode);
/*!
 * @brief Init the CustomParam into Node with User defined Init function
 * @param tfNode
 * @param Init
 */
    TFDL2_CAPI_EXPORT extern void NewNodeCustomParam(TFNode tfNode, std::function<void*()>Init);
/*!
 * @brief Free The CustomParam with User defined Free function, if custom's ptr were NULL, this function won't run
 * @param tfNode
 * @param Free
 */
    TFDL2_CAPI_EXPORT extern void FreeNodeCustomParam(TFNode tfNode, std::function<void(void*)>Free);

/*
 * Quantize handler
 */
    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationMax(Quantization quantization);

    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationMin(Quantization quantization);

    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationScale(Quantization quantization);

    TFDL2_CAPI_EXPORT extern const vector<int> &GetQuantizationZeroPoint(Quantization quantization);

    TFDL2_CAPI_EXPORT extern const int GetQuantizationbits(Quantization quantization);

// Get Some Device Info
    TFDL2_CAPI_EXPORT extern const int GetDeviceNum();

    TFDL2_CAPI_EXPORT extern const Device GetDevicetype();
// TFCalibration handle function
/*!
 * these function can help user convert their own net from(caffe(prototxt,caffemodel), tensorflow(pb), onnx(onnx), tflite(tflite), FakeNet(json))
 * to Our struct(TFDL). Then user can calibration the net (if it is float) to uint8 or dump the json to debug net struct
 */

    TFDL2_CAPI_EXPORT extern TFCalibration NewCalibration(TFCalibrationMode calibrationMode);

    TFDL2_CAPI_EXPORT extern bool
    Convert(TFCalibration tfCalibration, TFConvertType tfConvertType, string protopath, string modelpath = "");

    TFDL2_CAPI_EXPORT extern TFCalibration CompileCalibration(TFCalibrationMode calibrationMode, TFContext context);
// build Calibration from netjson(define by User)
    TFDL2_CAPI_EXPORT extern void ModifyCalibration(TFCalibration tfCalibration, string netjson);

    TFDL2_CAPI_EXPORT extern bool InitCalibration(TFCalibration tfCalibration, string config);

    TFDL2_CAPI_EXPORT extern void
    SetPreprocess(TFCalibration tfCalibration, string inputname, vector<float> scale, vector<float> means);
//TFDL2_CAPI_EXPORT extern TFTensor* GetTensor(TFCalibration* tfCalibration,string TensorName);
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetOutputTensors(TFCalibration tfCalibration);

    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetInputTensors(TFCalibration tfCalibration);

    TFDL2_CAPI_EXPORT extern void Calibration(TFCalibration tfCalibration);

    TFDL2_CAPI_EXPORT extern void Quantize(TFCalibration tfCalibration, map<string, TFCAPI_DATATYPE> input_type,
                                           const std::set<string> &avoidnodes = {},const std::set<string> &stopnodes = {}, bool mergeeltwise = false,
                                           bool mergeconcat = true,bool perchannel=true,bool ignore_activation_finetune = false);

    TFDL2_CAPI_EXPORT extern void Save2TFDL(TFCalibration tfCalibration, string outname);

    TFDL2_CAPI_EXPORT extern void Save2TFDL1(TFCalibration tfCalibration, string outname);

    TFDL2_CAPI_EXPORT extern void Save2TFDL(TFCalibration tfCalibration, string outname, std::stringstream &stream);

    TFDL2_CAPI_EXPORT extern void DumptoJson(TFCalibration tfCalibration, string &outjson);

    TFDL2_CAPI_EXPORT extern void DumpSimulator(TFExecutor tfExecutor, string outname);
}
#endif //NPU40T_TFDL2_C_API_H
