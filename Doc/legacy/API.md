### TFDL2 C API

尽管TFDL2内部有着复杂的逻辑和调用方式，我们提供了简单易用的C语言接口，以方便使用者可以尽可能快地上手使用。（注：目前TFDL2处于快速迭代阶段，部分接口可能不稳定）

####核心函数接口

核心函数接口是直接与TFDL2网络运行相关的函数，了解这些函数的使用方法就可以控制一个网络对象的生命周期：包括创建、固化、推理和回收。

####关于C_API中的一些关键类的概述
1. **TFContext**
网络上下文管理类，可以从中获取网络的节点，张量等信息。之后的执行体和转化器的初始化都需要先得到一个上下文。
2. **TFNode**
网络节点类，存储这节点信息，同时具有用户自定义指针，用户可以自由管理这个指针，框架不会对这个指针执行任何操作。
3. **TFTensor**
网络张量类，存储张量的内存，形状，数据类型等。
4. **TFExecutor**
执行体类，作为网络推理的直接接口，可以执行网络的推理和与外界进行IO交互。
5. **TFCalibration**
转换器类，负责网络的转化和量化，可以直接读取caffe，tensorflow，onnx等模型文件，并转化成TFDL模型。同时也负责模型的量化任务。
6. **Quantization**
存储量化信息
7. **TFVision**
视频图片读写抽象类，内部会根据SDK环境使用TFG或者Opencv

###一般的执行流程
```c++
    //读取网络结构与参数，生成网路描述上下文（TFContext）
    auto context = LoadProto(protopath);
    //新版SDK保持使用json修改网络的功能，通过以下接口使用json文件修改网络
    ModifyTFContext(context,configstring2);
    //新版SDK将网络结构与运行时分开，用户可以使用上下文生成执行体，每个执行体可以共享同一个上下文的权值内存，并使用
    auto tfExcutor = CompileExecutor(context,true,configstring);
    //设置打印信息
    SetPrintInfo(tfExcutor,printinfo);
    //获取执行体的输入张量
    auto inputtensors = GetInputTensors(tfExcutor);
    //新建一个图片读取类
    TFVision tfVision = TFCV::NewImgReader();
    //从内存中读取图片jpeg流，也可以使用OpenURL从文件中读取
    TFCV::OpenSource(tfVision,imgss.data(),imgss.size());
    //将Vision中的解码数据写到张量中
    assert(TFCV::DumpImgData(tfVision,inputtensors[0])==true);
    //开始模型推理
    ForwardExecutorAlone(tfExcutor);
    //获取执行的输出张量
    auto outputtensors = GetOutputTensors(tfExcutor);
    //根据张量的数据类型，获取内存指针。
    for(auto tensor : outputtensors){
        switch(GetTensorType(tensor)){
            case TFCAPI_UINT8:{
                uint8_t* ptr = (uint8_t*)GetTensordata(tensor);
                break;
            }
            case TFCAPI_FLOAT:{
                float* ptr = (float*)GetTensordata(tensor);
                break;
            }
            default:exit(-1);
        }
    }
```

```c++
 /*!
 *  @brief: 获取SDK版本
 */
    TFDL2_CAPI_EXPORT extern string TFDL2_Version();
/*!
 *  @brief: 使用用户自定义的Log
 */
    TFDL2_CAPI_EXPORT extern void TFDL2_INITLOG(TLOG *NewLog);
/*!
 * @brief:  生成空的上下文对象
 * @param:  contextName 上下文名称
 */
    TFDL2_CAPI_EXPORT extern TFContext NewTFContext(string contextName);
/*!
 * @brief:  从已有的网络描述文件读取网络结构与参数生成上下文对象
 * @param:  描述文件路径
 */
    TFDL2_CAPI_EXPORT extern TFContext LoadProto(string protopath);
/*!
 * @brief:  使用一段内存从已有的网络描述文件读取网络结构与参数生成上下文对象
 * @param:  内存指针
 * @param:  内存段长度
 */
    TFDL2_CAPI_EXPORT extern TFContext LoadProtoFromMem(uint8_t *ptr, size_t len);
/*!
 * @brief:  在新版SDK中我们依旧保持使用Json文件修改网络结构的能力，用户可在一个新的json文件中采用
 *          如下结构书写进而修改上下文:
 *          {
 *               "AddOnPass": [],可选项支持选择优化器（选择是否开启优化器中的哪些选项：并层，对齐等）
 *               "DeleteLayer": [],// 要删除的层的名字列表
 *               "Layer": [
 *                      // 使用json定义的层信息修改列表大致结构如下：
 *                      // 1 修改层量化信息时
 *                      {
 *                          "layerName": "Softmax", // ----- 必须写
 *                          "OutDataMin": [0],// -- 可选，对于Data多通道量化时这个列表长度等于通道数
 *                          "OutDataMax": [1.0]
 *                      }，
 *                      // 2 修改层前后链接信息时
 *                      {
 *                          "input": [
 *                                     "efficientnet-b0/model/blocks_0/tpu_batch_normalization_1/batchnorm/add_1" // --- 修改输入节点信息
 *                                    ],
 *                          "layerName": "Quantize0", // ---- 必须写
 *                          "output": [
 *                                      "Quantize0"  // -- 修改输出节点信息
 *                                     ],
 *                      },
 *                      // 3 修改层输出数据类型
 *                      {
 *                          "layerName": "efficientnet-b0/model/head/dense/MatMul",
 *                          "outputDataType":  "TFDtypeFp32"
 *                      }
 *                      // 4 完整的单层描述体：
 *                      {
 *                          "input": [],
 *                          "layerName": "data",
 *                          "layerType": "SetInput",
 *                          "input": [],
 *                          "output": ["data"],
 *                          "outputDataType" : "TFDtypeUint8"          \\ this will allow Mix-precision in one Executor
 *                          "ActivationType": "ReLU"
 *                          "param": {
 *                              "tfDataType": "TFDtypeUint8",
 *                              "mean": [128,128,128],
 *                              "scale": [0.008712500334,0.008712500334,0.008712500334],
 *                              "shape": [1,3,300,300]
 *                          }
 *               }
 *               ]
 *           }
 * @param:  target context
 * @param:  string of json
 */
    TFDL2_CAPI_EXPORT extern void ModifyTFContext(TFContext, string netjson);
// TFContext Handle function
/*!
 *  @brief: 通过名字从上下文中获取张量对象
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByName(TFContext tfContext, string dataname);
/*!
 *  @brief: 通过名称从上下文中获取对应节点的参数信息
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern string GetNodeAttrJsonByName(TFContext tfContext, string dataname);
/*!
 *  @brief: 传递一个函数用来替换某对应名字的节点（为Python接口准备的）
 *  @param: target context
 *  @param: Node Name
 *  @param: function of builder
 */
    TFDL2_CAPI_EXPORT extern void ReplaceNodeByName(TFContext tfContext, string nodename,
                                                    std::function<vector<string>(TFContext &, vector<string>)> builder);
/*!
 *  @brief: 删除上下文中制定名字的节点
 *  @param: target context
 *  @param: target node name
 */
    TFDL2_CAPI_EXPORT extern void RemoveNodeByName(TFContext tfContext, string nodename);
/*!
 *  @brief: 获取上下文对象的名字，一般转模型时未制定名字的，这里会为空。
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern string GetContextName(TFContext tfContext);
/*!
 *  @brief: 获取一个上下文的所有输出节点名称
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern vector<string> GetContextOutputs(TFContext tfContext);
/*!
 *  @brief: 手动指定上下文的输出节点位置（这种指定也会影响之后运行时编译器的结果）
 *  @param: target context
 *  @param: output nodes' name
 */
    TFDL2_CAPI_EXPORT extern void SetContextOutputs(TFContext tfContext,vector<string>OutputNodes);
/*!
 *  @brief: 获取张量的量化信息
 *  @param: target context
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantizeInfo(TFContext tfContext, string dataname);
/*!
 *  @brief: 修改上下文时的开关接口（python接口使用）
 *  @param: target context
 */
    TFDL2_CAPI_EXPORT extern bool OpenContext(TFContext);
/*!
 *  @brief: 修改上下文时的开关接口（python接口使用）
 *  @param: the context opened
 */
    TFDL2_CAPI_EXPORT extern bool CloseContext(TFContext);
/*!
 *  @brief: 删除上下文指定某参数张量
 *  @param: target context
 *  @param: parameter's name
 */
    TFDL2_CAPI_EXPORT extern bool RemoveParam(TFContext, string paramname);
/*!
 *  @brief: 注册一个权值张量
 *  @param: target context
 *  @param: parameter's name
 *  @param: shape of parameter
 *  @param: data type of parameter
 *  @param: ptr of data
 *  @param: size of mem
 *  @param: quantization info(min-max) size=(1 or shape[0])
 */
    TFDL2_CAPI_EXPORT extern bool
    RegisterParamTensor(TFContext, string paramname, vector<int> shape, TFCAPI_DATATYPE datatype, void *ptr, size_t len,
                        vector<std::pair<float, float>> min_max = {});
/*!
 *  @brief: 从上下文获取权值张量
 *  @param: target context
 *  @param: parameter's name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetParam(TFContext, string paramname);
/*!
 *  @brief: 注册一个张量的量化信息
 *  @param: target context
 *  @param: tensor's name
 *  @param: qmax
 *  @param: qmin
 */
    TFDL2_CAPI_EXPORT extern bool RegisterQuantInfo(TFContext, string tensorName, float max, float min);
/*!
 *  @brief: 将上下文dump成一个老版TFDL的模型文件
 *  @param: target context
 *  @param: outfile name
 */
    TFDL2_CAPI_EXPORT extern void DumpContextToTFDL1(TFContext, string outname);
/*!
 *  @brief: 将上下文Dump成新版TFDL的模型描述文件
 *  @param: target context
 *  @param: outfile name
 */
    TFDL2_CAPI_EXPORT extern void DumpContextToFile(TFContext, string outname);
/*!
 *  @brief: 将上下文Dump成新版TFDL的模型描述文件到内存
 *  @param: target context
 *  @param: set context name
 *  @param: target stringstream
 */
    TFDL2_CAPI_EXPORT extern void DumpContext(TFContext, string outname, std::stringstream &stream);
/*!
 *  @brief: 设置运行时的打印信息
 *  @param: target executor
 *  @param: bool
 */
    TFDL2_CAPI_EXPORT extern void SetPrintInfo(TFExecutor tfExecutor, bool info);
/*!
 *  @brief: 设置运行时的预处理
 *  @param: target executor
 *  @param: name of which input
 *  @param: scales
 *  @param: means
 */
    TFDL2_CAPI_EXPORT extern void SetPreprocess(TFExecutor tfExecutor, string inputname,
            vector<float> scale, vector<float> means);
/*!
 *  @brief: 注册自定义算子接口
 *  @param: customOp name
 */
    TFDL2_CAPI_EXPORT extern TFCAPIOP &RegisterCustomOp(string customOptype);

/*!
 *  @brief: 从lib文件中注册自定义算子，关于自定义算子的写法与编译在Addon文件夹
 *  @param: custom op lib file
 */
    TFDL2_CAPI_EXPORT extern int RegisterCustomOpFromFile(string filepath);
/*!
 *  @brief: 将某节点替换成对应自定义算子
 *  @param: target executor
 *  @param: target Node name
 *  @param: custom op name
 */
    TFDL2_CAPI_EXPORT extern void ReplaceOpwithcustomOp(TFExecutor tfExecutor, string nodeName, string customOpname);
/*!
 *  @brief: 使用上下文，按照用户指定的设置信息，编译生成运行时执行体。
 *  @param shareWeight: Share Weight with Given TFContext. If true, User can't Free TFContext until all Executor, which share Weight of itself.
 *  @param config : string of Json11, which set some options for Compiling
 *          {
 *              "UseHardware":true, -- 是否使用硬件
 *              "FrugalMode": true, -- 开启节省内存模式
 *              "Core": [], -- 该执行体将使用哪几个硬件核心(NPU10T只能支持单执行体单硬件，NPU40T支持单执行体多硬件)
 *              "optimize":{  -- 模型优化选项，在编译网络时这些将被考虑
 *                  "DoAlign":false,
 *                  "TryReverse":false,
 *                  "HighAccuracy": true,
 *                  "SplitDeconv": true
 *              },
 *              "InputShape":[{"NodeName":"TFDL_Placeholder_0","Shape":[1,3,384,384]}] --- 设置输入节点的形状，之后编译器将按照这一形状生成命令字
 *          }  
 */

    TFDL2_CAPI_EXPORT extern TFExecutor
    CompileExecutor(TFContext tfContext, bool shareWeight = true, string config = "{\"UseHardware\":true,\n"
                                                                                  "    \"FrugalMode\":true,\n"
                                                                                  "    \"optimize\":{\n"
                                                                                  "        \"DoAlign\":false,\n"
                                                                                  "        \"TryReverse\":false\n"
                                                                                  "    },\n"
                                                                                  "    \"InputShape\":[\n"
                                                                                  "    ]}");
/*!
 *  @brief: 执行体开始执行推理（非独立，静态调度器会根据同时运行的执行体执行情况替换执行顺序）
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern void ForwardExecutor(TFExecutor tfExecutor);
/*!
 *  @brief: 设置运行时队列窗长度，针对于非独立运行时的调度设置的
 *  @param: window size
 */
    TFDL2_CAPI_EXPORT extern void SetForwardWindowSize(int);
/*!
 *  @brief: 执行体开始执行推理（独立，不参与协调）
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern void ForwardExecutorAlone(TFExecutor tfExecutor);
/*!
 *  @brief: 获取执行体的输出张量（不同于上下文的张量，只有执行体的张量拥有内存，上下文的张量只是一个描述头，没有内存分配）
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetOutputTensors(TFExecutor tfExecutor);
/*!
 *  @brief: 获取执行体的输入张量（不同于上下文的张量，只有执行体的张量拥有内存，上下文的张量只是一个描述头，没有内存分配）
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetInputTensors(TFExecutor tfExecutor);
/*!
 *  @brief: 获取执行体的张量（不同于上下文的张量，只有执行体的张量拥有内存，上下文的张量只是一个描述头，没有内存分配）
 *  @param: target executor
 *  @param: tensor name
 */
    TFDL2_CAPI_EXPORT extern TFTensor GetTensorByName(TFExecutor tfExecutor, string);
/*!
 *  @brief: 获取执行体输出数据的float指针
 *  @param: target executor
 */
    TFDL2_CAPI_EXPORT extern vector<float *> GetFloatOutputData(TFExecutor tfExecutor);
/*!
 *  @brief: 生成量化信息
 *  @param: qmax
 *  @param: qmin
 *  @param: quantize bit
 */
    TFDL2_CAPI_EXPORT extern Quantization NewQuantizeInfo(vector<float> max, vector<float> min, int bit = 8);
/*!
 *  @brief: 获取执行体的张量的量化信息，这个接口与上下文的接口基本相同
 *  @param: target executor
 *  @param: Tensor name
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantizeInfo(TFExecutor tfExecutor, string dataname);

//TFMath Handle function
/*!
 *  @brief: 使用量化信息对一段int8内存进行反量化
 *  @param: dst ptr
 *  @param: src ptr
 *  @param: data len
 *  @param: quantize-info
 *  @param: quantize in which channel for multi-channel quantization
 */
    TFDL2_CAPI_EXPORT extern bool
    DeQuantizeTensorData(float *dst, uint8_t *src, int len, Quantization quantizeInfo, int channel = 0);
/*!
 *  @brief: 使用量化信息对一段float内存进行量化
 *  @param: dst ptr
 *  @param: src ptr
 *  @param: data len
 *  @param: quantize-info
 *  @param: quantize in which channel for multi-channel quantization
 */
    TFDL2_CAPI_EXPORT extern bool QuantizeTensorData(uint8_t *dst, float *src, int len, Quantization quantizeInfo,int channel=0);
/*!
 *  @brief: 使用查找表对一段int8进行转译
 *  @param: dst ptr
 *  @param: src ptr
 *  @param: maptable
 *  @param: data len
 */
    TFDL2_CAPI_EXPORT extern bool SimpleMapdata(uint8_t *dst, uint8_t *src, uint8_t *maptable, int len);
/*!
 *  @brief: 将目标张量进行转置，并将转置后的结果写到目标地址
 *  @param: dst ptr
 *  @param: src Tensor
 *  @param: permute axis
 */
    TFDL2_CAPI_EXPORT extern void TransPosedata(void* dst, TFTensor src,vector<int>axis);

// TFTensor Handle function
/*!
 *  @brief: 快速打印张量信息
 */
    TFDL2_CAPI_EXPORT extern string ToString(TFTensor tfTensor);
/*!
 * @brief 在上下文中新生成一个张量，名字将由系统指定
 * @return TFTensor
 */
    TFDL2_CAPI_EXPORT extern TFTensor NewTensor(TFContext);
/*!
 * @brief 将一个张量的信息全部拷贝到另一个张量上
 * @param from
 * @param to
 */
    TFDL2_CAPI_EXPORT extern void TensorCopy(TFTensor from, TFTensor to);
/*!
 *  @brief: 获取张量的名字
 */
    TFDL2_CAPI_EXPORT extern string GetTensorName(TFTensor tfTensor);
/*!
 *  @brief: 改变张量的形状描述信息，但是不会改变内存
 *  @param: tfContext: Context contain the tensor
 *  @param: tfTensor: target tensor
 *  @param: shape: target shape
 */
    TFDL2_CAPI_EXPORT extern bool ReSizeTensor(TFTensor tfTensor, vector<int> shape);
/*!
 *  @brief: 根据形状描述分配cpu内存
 *  @param: tfContext: Context contain the tensor
 *  @param: tfTensor: target tensor
 */
    TFDL2_CAPI_EXPORT extern bool AllocateCpuMem(TFTensor tfTensor);
/*!
 *  @brief: 设置张量的数据类型
 *  @param: target context contain the tensor
 *  @param: target datatype
 */
    TFDL2_CAPI_EXPORT extern bool SetTensorType(TFTensor tfTensor, TFCAPI_DATATYPE tfcapiDatatype);

/*!
 *  @brief: 获取张量形状
 */
    TFDL2_CAPI_EXPORT extern vector<int> GetTensorShape(TFTensor tfTensor);
/*!
 *  @brief: 获取张量量化信息
 */
    TFDL2_CAPI_EXPORT extern Quantization GetTensorQuantize(TFTensor tfTensor);
/*!
 *  @brief: 获取张量数据类型
 */
    TFDL2_CAPI_EXPORT extern TFCAPI_DATATYPE GetTensorType(TFTensor tfTensor);
/*!
 *  @brief: 获取张量内存的头指针
 */
    TFDL2_CAPI_EXPORT extern void *GetTensordata(TFTensor tfTensor);
/*!
 *  @brief: 将张量变成int标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, int);
/*!
 *  @brief: 将张量变成float标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, float);
/*!
 *  @brief: 将张量变成double标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, double);
/*!
 *  @brief: 将张量变成int64标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, int64_t);
/*!
 *  @brief: 将张量变成uint8标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, uint8_t);
/*!
 *  @brief: 将张量变成uint16标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, uint16_t);
/*!
 *  @brief: 将张量变成string标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, string);
/*!
 *  @brief: 将张量变成string标量
 */
    TFDL2_CAPI_EXPORT extern bool SetScalardata(TFTensor tfTensor, char const *);
/*!
 *  @brief: 统计对应维度大小
 */
    TFDL2_CAPI_EXPORT extern int GetTensorCount(TFTensor tfTensor, int st, int ed);
/*!
 *  @brief: 统计对应维度大小
 */
    TFDL2_CAPI_EXPORT extern int GetTensorCount(TFTensor tfTensor, int st);
/*!
 *  @brief: 获取张量内存的大小
 */
    TFDL2_CAPI_EXPORT extern size_t GetTensorDataSize(TFTensor tfTensor);
/*!
 *  @brief: 获取对应维度大小
 */
    TFDL2_CAPI_EXPORT extern int GetTensorDim(TFTensor tfTensor, int dim);
//TFNode Handle function
/*!
 *  @brief: 获取节点信息
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
 * @brief 获取用户自定地址指针
 * @param tfNode
 * @return User defined struct's ptr
 */
    TFDL2_CAPI_EXPORT extern void *GetNodeCustomParam(TFNode tfNode);
/*!
 * @brief 传递一个用户自定函数，为节点设置用户自定地址指针
 * @param tfNode
 * @param Init
 */
    TFDL2_CAPI_EXPORT extern void NewNodeCustomParam(TFNode tfNode, std::function<void*()>Init);
/*!
 * @brief 传递一个用户自定函数，释放之前节点设置的用户自定地址指针
 * @param tfNode
 * @param Free
 */
    TFDL2_CAPI_EXPORT extern void FreeNodeCustomParam(TFNode tfNode, std::function<void(void*)>Free);

/*
 * Quantize handler
 */
   /*!
 *  @brief: 量化信息中的最大值信息
 */ 
    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationMax(Quantization quantization);
/*!
 *  @brief: 量化信息中的最小值信息
 */
    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationMin(Quantization quantization);
/*!
 *  @brief: 量化信息中的scale信息
 */
    TFDL2_CAPI_EXPORT extern const vector<float> &GetQuantizationScale(Quantization quantization);
/*!
 *  @brief: 量化信息中的zeropoint信息
 */
    TFDL2_CAPI_EXPORT extern const vector<int> &GetQuantizationZeroPoint(Quantization quantization);
/*!
 *  @brief: 量化信息的量化bit
 */
    TFDL2_CAPI_EXPORT extern const int GetQuantizationbits(Quantization quantization);

/*!
 *  @brief: 获取芯片中硬件个数
 */
    TFDL2_CAPI_EXPORT extern const int GetDeviceNum();
/*!
 *  @brief: 获取芯片的硬件版本
 */
    TFDL2_CAPI_EXPORT extern const Device GetDevicetype();


// 模型转换与量化
/*!
 * 本SDK可以直接进行模型转换 从(caffe(prototxt,caffemodel), tensorflow(pb), onnx(onnx), tflite(tflite) 转换为TFDL2 的模型描述文件
 */
/*!
 *  @brief: 生成对应量化版本的转化器
 */
    TFDL2_CAPI_EXPORT extern TFCalibration NewCalibration(TFCalibrationMode calibrationMode);
/*!
 *  @brief: 开始模型转化
 *  @param 转换器
 *  @param 转换类型
 *  @param 源文件
 *  @param 源模型（只针对caffe）
 */
    TFDL2_CAPI_EXPORT extern bool
    Convert(TFCalibration tfCalibration, TFConvertType tfConvertType, string protopath, string modelpath = "");
/*!
 *  @brief: 直接由上下文生成量化器
 */
    TFDL2_CAPI_EXPORT extern TFCalibration CompileCalibration(TFCalibrationMode calibrationMode, TFContext context);
/*!
 *  @brief: 修改转化器
 */
    TFDL2_CAPI_EXPORT extern void ModifyCalibration(TFCalibration tfCalibration, string netjson);
/*!
 *  @brief: 根据设置初始化转化器
 */
    TFDL2_CAPI_EXPORT extern bool InitCalibration(TFCalibration tfCalibration, string config);
/*!
 *  @brief: 设置转化器预处理
 */
    TFDL2_CAPI_EXPORT extern void
    SetPreprocess(TFCalibration tfCalibration, string inputname, vector<float> scale, vector<float> means);
/*!
 *  @brief: 获取转化器输出张量
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetOutputTensors(TFCalibration tfCalibration);
/*!
 *  @brief: 获取转化器输入张量
 */
    TFDL2_CAPI_EXPORT extern vector<TFTensor> GetInputTensors(TFCalibration tfCalibration);
/*!
 *  @brief: 开启量化推理
 */
    TFDL2_CAPI_EXPORT extern void Calibration(TFCalibration tfCalibration);
/*!
 *  @brief: 完成量化推理后，执行模型完整量化
 *  @param 转换器
 *  @param 量化后模型所有输入的数据类型
 *  @param 量化中忽略的节点
 *  @param 量化中合并eltwise量化信息
 *  @param 量化中合并concat量化信息
 *  @param 量化中对weight进行多通道量化
 */
    TFDL2_CAPI_EXPORT extern void Quantize(TFCalibration tfCalibration, map<string, TFCAPI_DATATYPE> input_type,
                                           const std::set<string> &avoidnodes = {}, bool mergeeltwise = false,
                                           bool mergeconcat = true,bool perchannel=true);
/*!
 *  @brief: 转化好后的模型，存储为新SDK模型
 */
    TFDL2_CAPI_EXPORT extern void Save2TFDL(TFCalibration tfCalibration, string outname);
/*!
 *  @brief: 转化好后的模型，存储为旧SDK模型
 */
    TFDL2_CAPI_EXPORT extern void Save2TFDL1(TFCalibration tfCalibration, string outname);
/*!
 *  @brief: 转化好后的模型，存储为新SDK模型
 */
    TFDL2_CAPI_EXPORT extern void Save2TFDL(TFCalibration tfCalibration, string outname, std::stringstream &stream);
/*!
 *  @brief: 转化好后的模型，将其打印为json
 */
    TFDL2_CAPI_EXPORT extern void DumptoJson(TFCalibration tfCalibration, string &outjson);
}
```
