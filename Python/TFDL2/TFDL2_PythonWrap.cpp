//
// Created by test on 2020/4/27.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <TFDL2_C_API.h>
#include <TFDL2_Common.h>
#include <TFModule.h>
//#include "TFCV.h"
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace TFDL_CAPI;
namespace PythonInter {
    class PyLog : public TLOG{
    public:
        void TFLOG(std::string err) {
            py::print(err);
        }
        void Warning(std::string err) {
            py::print("Warning:"+err);
        }
    };

    static PyLog pylog;
#define TPlog(expr, log)\
    if(!(expr))popError(log);\


    class TFSymbol {
    public:
        TFSymbol() {
            tfContext = nullptr;
            name = "";
            weight = false;
        }

        TFSymbol(TFContext tfContext_, string name_ = "", bool ifweight = false) : tfContext(tfContext_), name(name_),
                                                                                   weight(ifweight) {

        }

        string getName() {
            return name;
        }

        TFContext GetContext() {
            return tfContext;
        }

        bool ifParam() {
            return weight;
        }

        ~TFSymbol() {
        }

    private:
        string name;
        bool weight = false;
        TFContext tfContext;
    };

    bool RegisterQuantInfo(TFSymbol& self,float min,float max){
        return RegisterQuantInfo(self.GetContext(),self.getName(),max,min);
    }
    class pyTFTensor{
    public:
        pyTFTensor(){
        }
        pyTFTensor(TFTensor&& tensor,Quantization&& q){
            self = std::move(tensor);
            quantization = std::move(q);
        }
        pyTFTensor(TFTensor&& tensor){
            self = std::move(tensor);
        }
        pyTFTensor(const pyTFTensor& other){
            self = other.self;
            quantization = other.quantization;
        }
        pyTFTensor(pyTFTensor&& other) noexcept {
            self = std::move(other.self);
            quantization = std::move(other.quantization);
        }
        ~pyTFTensor(){
        }
        TFTensor self;
        Quantization quantization;
    };
    class pyTFContext{
    public:
        pyTFContext(py::str name){
            self = std::move(NewTFContext(name));
        }
        pyTFContext(py::tuple reads){
            string read = py::cast<string>(reads[0]);
            self = std::move(LoadProto(read));
        }
        pyTFContext(const pyTFContext& other){
            self = other.self;
        }
        pyTFContext(pyTFContext&& other) noexcept {
            self = std::move(other.self);
        }
        ~pyTFContext(){
        }
        TFContext self;
    };
    class pyTFExecutor{
    public:
        pyTFExecutor(pyTFContext& context,py::str jsonconfig){
            string cjsonconfig = jsonconfig;
            self = std::move(CompileExecutor(context.self,true,cjsonconfig));
        }
        pyTFExecutor(pyTFExecutor&& other){
            self = std::move(other.self);
        }
        pyTFExecutor(const pyTFExecutor& other){
            self = other.self;
        }
        ~pyTFExecutor(){
        }
        TFExecutor self;
    };
    /*
    class pyImgReader{
    public:
        pyImgReader(){
            self = std::move(TFCV::NewImgReader());
        }
        pyImgReader(pyImgReader&& other){
            self = std::move(other.self);
        }
        pyImgReader(const pyImgReader& other){
            self = other.self;
        }
        ~pyImgReader(){
        }

        TFVision self;
        TFCV::DECODER_FLAGS flags;
    };
    class pyVideoCapture{
    public:
        pyVideoCapture(){
            self = std::move(TFCV::NewVideoReader());
        }
        pyVideoCapture(pyVideoCapture&& other){
            self = std::move(other.self);
        }
        pyVideoCapture(const pyVideoCapture& other){
            self = other.self;
        }
        ~pyVideoCapture(){
        }
        void Open(string path,TFCV::DECODER_FLAGS flags){
            this->flags = flags;
            TFCV::OpenURL(self,path);
        }
        void ReadFrame(pyTFTensor& tensor,int batch){
            if(TFCV::ReadFrame(self)>0) {
                TFCV::DumpImgData(self, tensor.self, batch, flags);
            }
        }
        py::array ToNumpy(){
            vector<uint8_t >dd;
            TFCV::DumpImgData(self,dd,flags);
            vector<int> dims;
            vector<int> shape = {TFCV::GetHeight(self),TFCV::GetWidth(self),flags==TFCV::TFCV_Gray?1:3};
            if(shape.empty()){
                py::tuple newshape;
                return py::array();
            }
            size_t datasize = dd.size();
            for(size_t i=0;i<shape.size();i++){
                size_t onedim = 1;
                for(size_t j=i+1;j<shape.size();j++){
                    onedim *= shape[j];
                }
                dims.push_back(onedim);
            }
            auto type = TFCAPI_UINT8;//GetTensorType(self.self);

            switch(type){
                case(TFCAPI_FLOAT):{
                    for(auto& i : dims){
                        i *= sizeof(float);
                    }
                    auto array = py::array_t<float>(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                case(TFCAPI_FLOAT64):{
                    for(auto& i : dims){
                        i *= sizeof(double);
                    }
                    auto array = py::array_t<double >(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                case(TFCAPI_INT32):{
                    for(auto& i : dims){
                        i *= sizeof(int);
                    }
                    auto array = py::array_t<int>(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                case(TFCAPI_INT64):{
                    for(auto& i : dims){
                        i *= sizeof(int64_t);
                    }
                    auto array = py::array_t<int64_t >(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                case(TFCAPI_UINT8):{
                    auto array = py::array_t<uint8_t >(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                case(TFCAPI_UINT16):{
                    for(auto& i : dims){
                        i *= sizeof(uint16_t);
                    }
                    auto array = py::array_t<uint16_t >(shape,dims);
                    auto buf = array.request();
                    memcpy(buf.ptr,dd.data(),datasize);
                    return array;
                }
                default:{
                    popError("dtype error");
                }
            }
        }
        TFVision self;
        TFCV::DECODER_FLAGS flags;
    };
    template<typename T>
    void Open(T& reader,string path,TFCV::DECODER_FLAGS flags){
        reader.flags = flags;
        TFCV::OpenURL(reader.self,path);
    }
    template<typename T>
    void ReadFrame(T& reader){
        TFCV::ReadFrame(reader.self);
    }
    template<typename T>
    void ReadFrame2Tensor(T& reader,pyTFTensor& tensor,int batch){
        TFCV::DumpImgData(reader.self,tensor.self,batch,reader.flags);
    }
    template<typename T>
    py::array ToNumpy(T& reader){
        vector<uint8_t >dd;
        TFCV::DumpImgData(reader.self,dd,reader.flags);
        vector<int> dims;
        vector<int> shape = {TFCV::GetHeight(reader.self),TFCV::GetWidth(reader.self),reader.flags==TFCV::TFCV_Gray?1:3};
        if(shape.empty()){
            py::tuple newshape;
            return py::array();
        }
        size_t datasize = dd.size();
        for(size_t i=0;i<shape.size();i++){
            size_t onedim = 1;
            for(size_t j=i+1;j<shape.size();j++){
                onedim *= shape[j];
            }
            dims.push_back(onedim);
        }
        auto type = TFCAPI_UINT8;//GetTensorType(self.self);

        switch(type){
            case(TFCAPI_FLOAT):{
                for(auto& i : dims){
                    i *= sizeof(float);
                }
                auto array = py::array_t<float>(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            case(TFCAPI_FLOAT64):{
                for(auto& i : dims){
                    i *= sizeof(double);
                }
                auto array = py::array_t<double >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            case(TFCAPI_INT32):{
                for(auto& i : dims){
                    i *= sizeof(int);
                }
                auto array = py::array_t<int>(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            case(TFCAPI_INT64):{
                for(auto& i : dims){
                    i *= sizeof(int64_t);
                }
                auto array = py::array_t<int64_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            case(TFCAPI_UINT8):{
                auto array = py::array_t<uint8_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            case(TFCAPI_UINT16):{
                for(auto& i : dims){
                    i *= sizeof(uint16_t);
                }
                auto array = py::array_t<uint16_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,dd.data(),datasize);
                return array;
            }
            default:{
                popError("dtype error");
            }
        }
    }
     void TensorReadImg(pyTFTensor& self,py::list imglist,TFCV::DECODER_FLAGS decoder){
        vector<string> imglist_;
        TFVision tfVision = TFCV::NewImgReader();
        for(int i=0;i<py::len(imglist);i++){
            TFCV::OpenURL(tfVision,py::cast<string>(imglist[i]));
            TFCV::DumpImgData(tfVision,self.self,i,decoder);
        }
    }
*/


    class pyTFCalibration{
    public:
        pyTFCalibration(pyTFContext& context,TFCalibrationMode mode,py::str jsonconfig){
            string cjsonconfig = jsonconfig;
            self = std::move(CompileCalibration(mode,context.self));
            InitCalibration(self,cjsonconfig);
        }
        pyTFCalibration(pyTFCalibration&& other){
            self = std::move(other.self);
        }
        pyTFCalibration(const pyTFCalibration& other){
            self = other.self;
        }
        ~pyTFCalibration(){
        }
        TFCalibration self;
    };

    py::array toNumpy(pyTFTensor& self){
        vector<int> dims;
        vector<int> shape = GetTensorShape(self.self);
        if(shape.empty()){
            py::tuple newshape;
            return py::array();
        }
        for(size_t i=0;i<shape.size();i++){
            dims.push_back(GetTensorCount(self.self,i+1));
        }
        auto type = GetTensorType(self.self);

        switch(type){
            case(TFCAPI_FLOAT):{
                for(auto& i : dims){
                    i *= sizeof(float);
                }
                auto array = py::array_t<float>(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_FLOAT64):{
                for(auto& i : dims){
                    i *= sizeof(double);
                }
                auto array = py::array_t<double >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_INT32):{
                for(auto& i : dims){
                    i *= sizeof(int);
                }
                auto array = py::array_t<int>(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_INT64):{
                for(auto& i : dims){
                    i *= sizeof(int64_t);
                }
                auto array = py::array_t<int64_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_UINT8):{
                auto array = py::array_t<uint8_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_UINT16):{
                for(auto& i : dims){
                    i *= sizeof(uint16_t);
                }
                auto array = py::array_t<uint16_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_FLOAT16):{
                for(auto& i : dims){
                    i *= sizeof(short);
                }
                auto array = py::array_t<short >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            case(TFCAPI_INT16):{
                for(auto& i : dims){
                    i *= sizeof(int16_t);
                }
                auto array = py::array_t<int16_t >(shape,dims);
                auto buf = array.request();
                memcpy(buf.ptr,GetTensordata(self.self),GetTensorDataSize(self.self));
                return array;
            }
            default:{
                popError("dtype error");
            }
        }
    }

    void FromNumpy(pyTFTensor& self,py::array inputData){
        int nd = inputData.ndim();
        vector<int> shape(nd, 0);
        size_t dsize = inputData.dtype().itemsize();
        size_t len = dsize;
        for (int shape_i = 0; shape_i < nd; shape_i++) {
            shape[shape_i] = inputData.shape(shape_i);
            len *= shape[shape_i];
        }
        TFCAPI_DATATYPE dtype;
        string dtypename = py::cast<string>(inputData.dtype().attr("name"));
        if (dtypename == "float32") {
            dtype = TFCAPI_FLOAT;
        } else if (dtypename == "uint8") {
            dtype = TFCAPI_UINT8;
        } else if (dtypename == "uint16") {
            dtype = TFCAPI_UINT16;
        } else if (dtypename == "float64") {
            dtype = TFCAPI_FLOAT64;
        } else if (dtypename == "int32") {
            dtype = TFCAPI_INT32;
        } else if (dtypename == "int64") {
            dtype = TFCAPI_INT64;
        } else if (dtypename == "float16") {
            dtype = TFCAPI_FLOAT16;
        } else if (dtypename == "int16") {
            dtype = TFCAPI_INT16;
        } else {
            string err = "TFDL2 Can't support this dtype";
            PyErr_SetString(PyExc_RuntimeError, err.c_str());
            PyErr_Print();
        }
        if(!(GetTensorShape(self.self) == shape)){
            string err = "shape is not match";
            PyErr_SetString(PyExc_RuntimeError, err.c_str());
            PyErr_Print();
            return;
        }
        if(GetTensorType(self.self) != dtype){
            string err = "dtype is not match";
            PyErr_SetString(PyExc_RuntimeError, err.c_str());
            PyErr_Print();
            return;
        }
        auto buf = inputData.request();
        memcpy(GetTensordata(self.self),buf.ptr,GetTensorDataSize(self.self));

    }



    string printsymbol(TFSymbol& self){
        return self.getName();
    }

    py::list Shape(pyTFTensor& self){
        py::list shape;
        for(auto i : GetTensorShape(self.self)){
            shape.append(i);
        }
        return shape;
    }
    TFCAPI_DATATYPE Type(pyTFTensor& self){
        return (TFCAPI_DATATYPE)GetTensorType(self.self);
    }
    vector<float> tensorMaxs(pyTFTensor& self){
        vector<float> maxs;
        if(self.quantization.IsValid()){
            maxs = GetQuantizationMax(self.quantization);
        }
        return maxs;
        

    }
    vector<float> tensorMins(pyTFTensor& self){
        vector<float> mins;
        if(self.quantization.IsValid()){
            mins = GetQuantizationMin(self.quantization);
        }
        return mins;
        
    }
    vector<float> tensorScales(pyTFTensor& self){
        vector<float> scales;
        if(self.quantization.IsValid()){
            scales = GetQuantizationScale(self.quantization);
        }
        return scales;
        
    }
    vector<int> tensorZeroPoints(pyTFTensor& self){
        vector<int> zeros;
        if(self.quantization.IsValid()){
            zeros = GetQuantizationZeroPoint(self.quantization);
        }
        return zeros;
        
    }
    string tensorName(pyTFTensor& self){
        return GetTensorName(self.self);
    }

    string Tostring(pyTFTensor& self){
        return ToString(self.self);
    }
    void SetScalar(pyTFTensor& self,double data,TFCAPI_DATATYPE dtype){
        switch(dtype){

            case(TFCAPI_FLOAT):{
                SetScalardata(self.self, static_cast<float>(data));
                break;
            }
            case(TFCAPI_FLOAT64):{
                SetScalardata(self.self,static_cast<double>(data));
                break;
            }
            case(TFCAPI_INT32):{
                SetScalardata(self.self,static_cast<int>(data));
                break;
            }
            case(TFCAPI_INT64):{
                SetScalardata(self.self,static_cast<int64_t>(data));
                break;
            }
            case(TFCAPI_UINT8):{
                SetScalardata(self.self,static_cast<uint8_t >(data));
                break;
            }
            case(TFCAPI_UINT16):{
                SetScalardata(self.self,static_cast<uint16_t >(data));
                break;
            }

            default:{
                popError("dtype error");
            }
        }
    }
    void SetScalar2(pyTFTensor& self,py::str data){
        string cdata = string(data);
        SetScalardata(self.self,cdata);
    }


    void DumpContext(pyTFContext& self,py::str outname){
        string coutname = string(outname);
        DumpContextToFile(self.self,coutname);
    }

    void RegisterParamToContext(pyTFContext &self,py::dict inputd) {

        for(auto ob : inputd) {
            string key = py::cast<string>(ob.first);
            py::array inputData = py::cast<py::array>(ob.second);
            int nd = inputData.ndim();
            vector<int> shape;
            size_t dsize = inputData.dtype().itemsize();
            size_t len = dsize;
            for (int shape_i = 0; shape_i < nd; shape_i++) {
                shape.push_back(inputData.shape(shape_i));
                len *= shape[shape_i];
            }
            TFCAPI_DATATYPE dtype;
            if (inputData.dtype().equal(py::dtype::of<float>())) {
                dtype = TFCAPI_FLOAT;
            } else if (inputData.dtype().equal(py::dtype::of<uint8_t >())) {
                dtype = TFCAPI_UINT8;
            } else if (inputData.dtype().equal(py::dtype::of<uint16_t >())) {
                dtype = TFCAPI_UINT16;
            } else if (inputData.dtype().equal(py::dtype::of<double >())) {
                dtype = TFCAPI_FLOAT64;
            } else if (inputData.dtype().equal(py::dtype::of<int>())) {
                dtype = TFCAPI_INT32;
            } else if (inputData.dtype().equal(py::dtype::of<int64_t >())) {
                dtype = TFCAPI_INT64;
            } else {
                string err = "TFDL2 Can't support this dtype";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }


            if (!RegisterParamTensor(self.self, key, shape, dtype, inputData.data(), len)) {
                string err = "TFDL2 register param fail, name is used";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }

    }

    pyTFTensor GetParam(pyTFContext& self,py::str paramName){
        string cparamName = string(paramName);
        auto param = GetParam(self.self,cparamName);
        return pyTFTensor(std::move(param),std::move(GetTensorQuantizeInfo(self.self,cparamName)));
    }

    bool RegistorInt8config(pyTFContext& self,py::str paramName,float max, float min){
        return RegisterQuantInfo(self.self,paramName,max,min);
    }

    string GetAttr(pyTFContext& self,py::str paramName) {
        string cparamName = string(paramName);
        return GetNodeAttrJsonByName(self.self, cparamName);
    }
    TFSymbol GetParamSymbol(pyTFContext& self,py::str paramName){
        string cparamName = string(paramName);
        auto param = GetParam(self.self,cparamName);
        vector<TFSymbol> outputs;
        if(param.IsValid()){
            return TFSymbol(self.self,cparamName,true);
        }else{
            return TFSymbol();
        }
    }
    bool pyOpenContext(pyTFContext& self){
        return OpenContext(self.self);
    }

    vector<TFSymbol> pyGetOutSymbolsContext(pyTFContext& self){
        auto outs = GetContextOutputs(self.self);
        vector<TFSymbol> outputs;
        for(auto name: outs){
            outputs.push_back(std::move(TFSymbol(self.self,name,false)));
        }
        return outputs;
    }

    vector<TFSymbol> pyGetInputSymbolsContext(pyTFContext& self){
        auto ins = GetContextInputs(self.self);
        vector<TFSymbol> inputs;
        for(auto name: ins){
            inputs.push_back(std::move(TFSymbol(self.self,name,false)));
        }
        return inputs;
    }

    bool pySetOutContext(pyTFContext& self,vector<string> outnames){
        SetContextOutputs(self.self,outnames);
        return true;
    }
    void pyModifyContext(pyTFContext& self,string json){
        ModifyTFContext(self.self,json);
    }
    void pyAssignContext(pyTFContext& self,string from,string to){
        TFFunc::Assign(self.self,from,to);
    }
    void pycloseContext(pyTFContext& self){
        CloseContext(self.self);
    }


    void pySetPrintInfo(pyTFExecutor& self,bool print){
        SetPrintInfo(self.self,print);
    }

    void pyDumpSimulator(pyTFExecutor& self,string outname){
        DumpSimulator(self.self,outname);
    }

    py::list Forward(pyTFExecutor& self,bool alone = true){
        if(alone)ForwardExecutorAlone(self.self);
        else ForwardExecutor(self.self);
        py::list outlist;
        auto inputs = GetOutputTensors(self.self);
        for(auto& input : inputs){
            auto q = GetTensorQuantizeInfo(self.self,GetTensorName(input));
            outlist.append(pyTFTensor(std::move(input),std::move(q)));
        }
        return outlist;
    }

    py::list GetInputTensor(pyTFExecutor& self){
        py::list outlist;
        auto inputs = GetInputTensors(self.self);
        for(auto& input : inputs){
            auto q = GetTensorQuantizeInfo(self.self,GetTensorName(input));
            outlist.append(pyTFTensor(std::move(input),std::move(q)));
        }

        return outlist;
    }

    pyTFTensor PYGetTensorByName(pyTFExecutor& self,py::str name){
        string cname = string(name);
        auto q = GetTensorQuantizeInfo(self.self,name);
        return pyTFTensor(std::move(GetTensorByName(self.self,cname)),std::move(q));
    }



    //BinaryOp
    TFSymbol Add(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Add(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol Sub(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Sub(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol rSub(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Sub(tfContext,inputB.getName(),inputA.getName()));
    }
    TFSymbol Mul(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Mul(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol Div(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Div(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol rDiv(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Div(tfContext,inputB.getName(),inputA.getName()));
    }
    TFSymbol GT(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::BoolGT(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol GE(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::BoolGE(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol LT(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::BoolLT(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol LE(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::BoolLE(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol EQ(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        auto tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::BoolEQ(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol GT1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolGT(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol GE1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolGE(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol LT1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolLT(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol LE1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolLE(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol EQ1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolEQ(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol GT2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolGT(tfContext,dd,inputB.getName(),dtype));
    }
    TFSymbol GE2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolGE(tfContext,dd,inputB.getName(),dtype));
    }
    TFSymbol LT2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolLT(tfContext,dd,inputB.getName(),dtype));
    }
    TFSymbol LE2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolLE(tfContext,dd,inputB.getName(),dtype));
    }
    TFSymbol EQ2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::BoolEQ(tfContext,dd,inputB.getName(),dtype));
    }
    TFSymbol Add1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Add(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol Sub1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Sub(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol Mul1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Mul(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol Div1(TFSymbol inputA, double data, TFCAPI_DATATYPE dtype){
        auto tfContext = inputA.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Div(tfContext,inputA.getName(),dd,dtype));
    }
    TFSymbol Add2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Add(tfContext,dd,dtype,inputB.getName()));
    }
    TFSymbol Sub2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Sub(tfContext,dd,dtype,inputB.getName()));
    }
    TFSymbol Mul2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Mul(tfContext,dd,dtype,inputB.getName()));
    }
    TFSymbol Div2(double data, TFCAPI_DATATYPE dtype, TFSymbol inputB){
        auto tfContext = inputB.GetContext();
        DataUnion dd;
        switch (dtype){
            case(TFCAPI_FLOAT64):{
                dd.d = data;
                break;
            }
            case(TFCAPI_FLOAT):{
                dd.f = static_cast<float>(data);
                break;
            }
            case(TFCAPI_INT32):{
                dd.i32 = static_cast<int>(data);
                break;
            }
            case(TFCAPI_INT64):{
                dd.i64 = static_cast<int64_t>(data);
                break;
            }
            default:{
                string err = "TFDL2 Scalar dtype Wrong";
                PyErr_SetString(PyExc_RuntimeError, err.c_str());
                PyErr_Print();
            }
        }
        return TFSymbol(tfContext,TFFunc::Div(tfContext,dd,dtype,inputB.getName()));
    }
    TFSymbol Addf(TFSymbol inputA, float inputB){
        return Add1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol Subf(TFSymbol inputA, float inputB){
        return Sub1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol Mulf(TFSymbol inputA, float inputB){
        return Mul1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol Divf(TFSymbol inputA, float inputB){
        return Div1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol GTf(TFSymbol inputA, float inputB){
        return GT1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol GEf(TFSymbol inputA, float inputB){
        return GE1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol LTf(TFSymbol inputA, float inputB){
        return LT1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol LEf(TFSymbol inputA, float inputB){
        return LE1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol EQf(TFSymbol inputA, float inputB){
        return EQ1(inputA,inputB,TFCAPI_FLOAT);
    }
    TFSymbol GTi(TFSymbol inputA, int inputB){
        return GT1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol GEi(TFSymbol inputA, int inputB){
        return GE1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol LTi(TFSymbol inputA, int inputB){
        return LT1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol LEi(TFSymbol inputA, int inputB){
        return LE1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol EQi(TFSymbol inputA, int inputB){
        return EQ1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol rAddf(TFSymbol inputA, float inputB){
        return Add2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rSubf(TFSymbol inputA, float inputB){
        return Sub2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rMulf(TFSymbol inputA, float inputB){
        return Mul2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rDivf(TFSymbol inputA, float inputB){
        return Div2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol Addi(TFSymbol inputA, int inputB){
        return Add1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol Subi(TFSymbol inputA, int inputB){
        return Sub1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol Muli(TFSymbol inputA, int inputB){
        return Mul1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol Divi(TFSymbol inputA, int inputB){
        return Div1(inputA,inputB,TFCAPI_INT32);
    }
    TFSymbol rAddi(TFSymbol inputA, int inputB){
        return Add2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rSubi(TFSymbol inputA, int inputB){
        return Sub2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rMuli(TFSymbol inputA, int inputB){
        return Mul2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol rDivi(TFSymbol inputA, int inputB){
        return Div2(inputB,TFCAPI_FLOAT,inputA);
    }
    TFSymbol Min(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Min(tfContext,inputA.getName(),inputB.getName()));
    }
    TFSymbol Max(TFSymbol inputA, TFSymbol inputB){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::Max(tfContext,inputA.getName(),inputB.getName()));
    }

    //nn OP
    TFSymbol Convolution(TFSymbol input,py::tuple kernel,py::tuple pad,py::tuple stride,int dilation,int outChannel,int group,bool hasBias){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();

        return TFSymbol(tfContext,TFFunc::Convolution(tfContext,input.getName(),kernel_,pad_,stride_,dilation,outChannel,group,hasBias));
    }
    TFSymbol Convolution2(TFSymbol input,TFSymbol weight,TFSymbol bias,py::tuple kernel,py::tuple pad,py::tuple stride,int dilation,int outChannel,int group){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();

        return TFSymbol(tfContext,TFFunc::Convolution(tfContext,input.getName(),weight.getName(),bias.getName(),kernel_,pad_,stride_,dilation,outChannel,group));
    }
    TFSymbol Unfold(TFSymbol input,py::tuple kernel,py::tuple pad,py::tuple stride,int dilation,bool reverse){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();

        return TFSymbol(tfContext,TFFunc::Unfold(tfContext,input.getName(),kernel_,pad_,stride_,dilation,reverse));
    }
    TFSymbol TFConvolution(TFSymbol input,py::tuple kernel,bool Same,py::tuple stride,int dilation,int outChannel,int group,bool hasBias){
        vector<int>kernel_,stride_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }

        TFContext tfContext = input.GetContext();

        return TFSymbol(tfContext,TFFunc::TensorflowConvolution(tfContext,input.getName(),kernel_,Same,stride_,dilation,outChannel,group,hasBias));
    }
    TFSymbol DeConvolution(TFSymbol input,py::tuple kernel,py::tuple pad,py::tuple stride,int dilation,int outChannel,int group,bool hasBias, int outPadH, int outPadW){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::DeConvolution(tfContext,input.getName(),kernel_,pad_,stride_,dilation,outChannel,group,hasBias,outPadH,outPadW));
    }
    TFSymbol DeConvolution2(TFSymbol input,TFSymbol weight,TFSymbol bias,py::tuple kernel,py::tuple pad,py::tuple stride,int dilation,int outChannel,int group,int outPadH, int outPadW){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::DeConvolution(tfContext,input.getName(),weight.getName(),bias.getName(),kernel_,pad_,stride_,dilation,outChannel,group,outPadH,outPadW));
    }
    TFSymbol MatMul(TFSymbol inputA, TFSymbol inputB,bool transA=false,bool transB=false,bool hasBias=false,bool bias2row=false){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::MatMul(tfContext,inputA.getName(),inputB.getName(),transA,transB,hasBias,bias2row));
    }
    TFSymbol MatMul2(TFSymbol inputA, TFSymbol inputB,TFSymbol bias,bool transA=false,bool transB=false,bool bias2row=false){
        TPlog(inputA.GetContext() == inputB.GetContext(),"inputs' Context is different")
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::MatMul(tfContext,inputA.getName(),inputB.getName(),bias.getName(),transA,transB,bias2row));
    }
    TFSymbol MatMulWeight(TFSymbol inputA,int dimm, int dimk,bool transA=false,bool transB=false,bool hasBias=false,bool bias2row=false){
        TFContext tfContext = inputA.GetContext();
        return TFSymbol(tfContext,TFFunc::MatMul(tfContext,inputA.getName(),dimm,dimk,transA,transB,hasBias,bias2row));
    }
    TFSymbol InnerProduct(TFSymbol input,bool hasBias,int outchannels){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::InnerProduct(tfContext,input.getName(),hasBias,outchannels));
    }
    TFSymbol InnerProduct2(TFSymbol input,TFSymbol weight,TFSymbol bias,int outchannels){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::InnerProduct(tfContext,input.getName(),weight.getName(),bias.getName(),outchannels));
    }
    TFSymbol Scale(TFSymbol input,bool hasBias){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Scale(tfContext,input.getName(),hasBias));
    }
    TFSymbol Scale2(TFSymbol input,TFSymbol weight,TFSymbol bias){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Scale(tfContext,input.getName(),weight.getName(),bias.getName()));
    }
    TFSymbol LayerNorm(TFSymbol input,int axis){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::LayerNorm(tfContext,input.getName(),axis));
    }
    TFSymbol LayerNorm2(TFSymbol input,int axis,TFSymbol scale,TFSymbol bias){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::LayerNorm(tfContext,input.getName(),axis,scale.getName(),bias.getName()));
    }
    TFSymbol ConstantOfShape(TFSymbol input,TFCAPI_DATATYPE dtype,py::array value){
        TFContext tfContext = input.GetContext();
        DataUnion ivalue;
        string dtypename = py::cast<string>(value.dtype().attr("name"));
        auto buf = value.request();
        if (dtypename == "float32") {
            //dtype = TFCAPI_FLOAT;
            ivalue.f = reinterpret_cast<float*>(buf.ptr)[0];
        } else if (dtypename == "uint8") {
            //dtype = TFCAPI_UINT8;
            ivalue.u8 = reinterpret_cast<uint8_t*>(buf.ptr)[0];
        } else if (dtypename == "uint16") {
            //dtype = TFCAPI_UINT16;
            ivalue.u16 = reinterpret_cast<uint16_t*>(buf.ptr)[0];
        } else if (dtypename == "float64") {
            //dtype = TFCAPI_FLOAT64;
            ivalue.d = reinterpret_cast<double*>(buf.ptr)[0];
        } else if (dtypename == "int32") {
            //dtype = TFCAPI_INT32;
            ivalue.i32 = reinterpret_cast<int*>(buf.ptr)[0];
        } else if (dtypename == "int64") {
            //dtype = TFCAPI_INT64;
            ivalue.i64 = reinterpret_cast<int64_t*>(buf.ptr)[0];
        } else if (dtypename == "int16") {
            //dtype = TFCAPI_INT16;
            ivalue.u16 = reinterpret_cast<int16_t*>(buf.ptr)[0];
        } else {
            string err = "TFDL2 Can't support this dtype";
            PyErr_SetString(PyExc_RuntimeError, err.c_str());
            PyErr_Print();
        }

        return TFSymbol(tfContext,TFFunc::ConstantOfShape(tfContext,input.getName(),dtype,ivalue));
    }
    TFSymbol Range(TFSymbol arg1,TFSymbol arg2,TFSymbol arg3){
        TFContext tfContext = arg1.GetContext();
        return TFSymbol(tfContext,TFFunc::Range(tfContext,arg1.getName(),arg2.getName(),arg3.getName()));
    }
    TFSymbol MaskSoftmax(TFSymbol input,int axis,bool upper,int diagonals,int mod){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::MaskSoftmax(tfContext,input.getName(),axis,upper,diagonals,mod));
    }
    TFSymbol GlobalAvePooling(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::GlobalAvePooling(tfContext,input.getName()));
    }
    TFSymbol AvePooling(TFSymbol input,py::tuple kernel,py::tuple pad,py::tuple stride,bool ceilmode){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::AvePooling(tfContext,input.getName(),kernel_,pad_,stride_,ceilmode));
    }
    TFSymbol MaxPooling(TFSymbol input,py::tuple kernel,py::tuple pad,py::tuple stride,bool ceilmode){
        vector<int>kernel_,stride_,pad_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::MaxPooling(tfContext,input.getName(),kernel_,pad_,stride_,ceilmode));
    }
    TFSymbol TFAvePooling(TFSymbol input,py::tuple kernel,bool same,py::tuple stride){
        vector<int>kernel_,stride_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::TensorflowAvePooling(tfContext,input.getName(),kernel_,stride_,same));
    }
    TFSymbol TFMaxPooling(TFSymbol input,py::tuple kernel,bool same,py::tuple stride){
        vector<int>kernel_,stride_;
        for(int i = 0;i<py::len(kernel);i++){
            kernel_.push_back(py::cast<int>(kernel[i]));
        }
        for(int i = 0;i<py::len(stride);i++){
            stride_.push_back(py::cast<int>(stride[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::TensorflowMaxPooling(tfContext,input.getName(),kernel_,stride_,same));
    }

    //Activation Op
    TFSymbol ReLU(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReLU(tfContext,input.getName()));
    }
    TFSymbol Requantize(TFSymbol input,vector<int>maptable){
        TFContext tfContext = input.GetContext();
        TPlog(maptable.size() == 256,"requantize's maptable must be 256 size");
        vector<uint8_t>mapt(256);
        for(int i=0;i<256;i++){
            mapt[i] = maptable[i];
        }
        return TFSymbol(tfContext,TFFunc::Requantize(tfContext,input.getName(),mapt));
    }
    TFSymbol LeakyReLU(TFSymbol input,float negativeslop){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::LeakyReLU(tfContext,input.getName(),negativeslop));
    }
    TFSymbol PReLU(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::PReLU(tfContext,input.getName()));
    }
    TFSymbol ReLUX(TFSymbol input,float threshold){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReLUX(tfContext,input.getName(),threshold));
    }
    TFSymbol ELU(TFSymbol input,float alpha){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ELU(tfContext,input.getName(),alpha));
    }
    TFSymbol Swish(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Swish(tfContext,input.getName()));
    }
    TFSymbol HardSwish(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::HardSwish(tfContext,input.getName()));
    }
    TFSymbol Tanh(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Tanh(tfContext,input.getName()));
    }
    TFSymbol GeLU(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::GeLU(tfContext,input.getName()));
    }
    TFSymbol Exp(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Exp(tfContext,input.getName()));
    }
    TFSymbol Sigmoid(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Sigmoid(tfContext,input.getName()));
    }
    TFSymbol Mish(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Mish(tfContext,input.getName()));
    }
    TFSymbol HardSigmoid(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::HardSigmoid(tfContext,input.getName()));
    }
    TFSymbol HardSigmoid2(TFSymbol input,float alpha,float beta){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::HardSigmoid(tfContext,input.getName(),alpha,beta));
    }
    TFSymbol Softmax(TFSymbol input,int axis){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Softmax(tfContext,input.getName(),axis));
    }

    // shape change op
    TFSymbol Reshape(TFSymbol input,py::tuple shape){
        vector<int> shape_;
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Reshape(tfContext,input.getName(),shape_));
    }
    TFSymbol Reshape2(TFSymbol input,TFSymbol shape){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Reshape(tfContext,input.getName(),shape.getName()));
    }
    TFSymbol BroadCast(TFSymbol input,py::tuple shape){
        vector<int> shape_;
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::BroadCast(tfContext,input.getName(),shape_));
    }
    TFSymbol BroadCast2(TFSymbol input,TFSymbol shape){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::BroadCast(tfContext,input.getName(),shape.getName()));
    }
    TFSymbol Squeeze(TFSymbol input,py::tuple dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Squeeze(tfContext,input.getName(),shape_));
    }
    TFSymbol Pad(TFSymbol input,py::tuple pad){
        vector<int> pad_;
        for(int i = 0;i<py::len(pad);i++){
            pad_.push_back(py::cast<int>(pad[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Pad(tfContext,input.getName(),pad_));
    }
    TFSymbol ZeroMask(TFSymbol input,py::tuple mask){
        vector<int> pad_;
        for(int i = 0;i<py::len(mask);i++){
            pad_.push_back(py::cast<int>(mask[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ZeroMask(tfContext,input.getName(),pad_));
    }
    TFSymbol Pad2(TFSymbol input,TFSymbol pad){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Pad(tfContext,input.getName(),pad.getName()));
    }
    TFSymbol Flatten(TFSymbol input,int startdim,int enddim){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Flatten(tfContext,input.getName(),startdim,enddim));
    }
    TFSymbol Flatten2Matrix(TFSymbol input,int axis){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Flatten2Matrix(tfContext,input.getName(),axis));
    }
    //reduce op
    TFSymbol ReduceMean(TFSymbol input,py::tuple dims,bool keep_dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReduceMean(tfContext,input.getName(),shape_,keep_dims));
    }
    TFSymbol MeanOp(TFSymbol input,py::tuple dims,bool keep_dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::MeanOp(tfContext,input.getName(),shape_,keep_dims));
    }
    TFSymbol ReduceSum(TFSymbol input,py::tuple dims,bool keep_dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReduceSum(tfContext,input.getName(),shape_,keep_dims));
    }
    TFSymbol ArgMax(TFSymbol input,int dim,bool keep_dims){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ArgMax(tfContext,input.getName(),dim,keep_dims));
    }
    TFSymbol ArgMin(TFSymbol input,int dim,bool keep_dims){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ArgMin(tfContext,input.getName(),dim,keep_dims));
    }
    TFSymbol ReduceMin(TFSymbol input,py::tuple dims,bool keep_dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReduceMin(tfContext,input.getName(),shape_,keep_dims));
    }
    TFSymbol ReduceMax(TFSymbol input,py::tuple dims,bool keep_dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReduceMax(tfContext,input.getName(),shape_,keep_dims));
    }

    //Tensor Op
    TFSymbol Transpose(TFSymbol input,py::tuple order){
        vector<int> shape_;
        for(int i = 0;i<py::len(order);i++){
            shape_.push_back(py::cast<int>(order[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Transpose(tfContext,input.getName(),shape_));
    }
    TFSymbol Concat(py::tuple inputs,int axis){
        vector<string> input_;
        TFContext tfContext = nullptr;
        for(int i = 0;i<py::len(inputs);i++){
            TFSymbol oss = py::cast<TFSymbol>(inputs[i]);
            if(tfContext== nullptr)tfContext = oss.GetContext();
            TPlog(tfContext == oss.GetContext(),"inputs' Context is different")
            input_.push_back(oss.getName());
        }
        return TFSymbol(tfContext,TFFunc::Concat(tfContext,input_,axis));
    }
    TFSymbol Gather(TFSymbol input,TFSymbol indices,int axis){
        TFContext tfContext = input.GetContext();
        TPlog(input.GetContext() == indices.GetContext(),"inputs' Context is different")
        return TFSymbol(tfContext,TFFunc::Gather(tfContext,input.getName(),indices.getName(),axis));
    }
    TFSymbol Gather2(TFSymbol input,py::tuple indices,int axis){
        std::vector<int> newitem = py::cast<std::vector<int>>(indices);
        auto gather = Gather(input,TFSymbol(input.GetContext()),axis);
        RegisterParamTensor(gather.GetContext(),gather.getName()+":0",{1,(int)newitem.size()},TFCAPI_INT32,newitem.data(),newitem.size()*sizeof(int));
        return gather;
    }
    TFSymbol Correlation(TFSymbol input1,TFSymbol input2,int kernel_size,int pad_size,int max_displacement,int stride1,int stride2){
        TFContext tfContext = input1.GetContext();
        TPlog(input1.GetContext() == input2.GetContext(),"inputs' Context is different")
        return TFSymbol(tfContext,TFFunc::Correlation(tfContext,input1.getName(),input2.getName(),kernel_size,pad_size,max_displacement,stride1,stride2));
    }
    TFSymbol Flip(TFSymbol input,py::tuple dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Flip(tfContext,input.getName(),shape_));
    }
    py::list Slice(TFSymbol input,int axis,py::tuple slicePoint){
        vector<int> split_;
        for(int i = 0;i<py::len(slicePoint);i++){
            split_.push_back(py::cast<int>(slicePoint[i]));
        }
        py::list outputs;
        TFContext tfContext = input.GetContext();
        auto output = TFFunc::Slice(tfContext,input.getName(),axis,split_);
        for(auto out : output){
            outputs.append(TFSymbol(tfContext,out));
        }
        return outputs;
    }
    TFSymbol ReArrange(TFSymbol input,int mode){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::ReArrange(tfContext,input.getName(),mode));
    }
    TFSymbol deReArrange(TFSymbol input,int mode,py::tuple oldshape){
        TFContext tfContext = input.GetContext();
        vector<int> oldshape_;
        for(int i = 0;i<py::len(oldshape);i++){
            oldshape_.push_back(py::cast<int>(oldshape[i]));
        }
        return TFSymbol(tfContext,TFFunc::ReArrange(tfContext,input.getName(),mode,oldshape_));
    }
    py::list Split(TFSymbol input,int axis,int split){
        py::list outputs;
        TFContext tfContext = input.GetContext();
        auto output = TFFunc::Split(tfContext,input.getName(),axis,split);
        for(auto out : output){
            outputs.append(TFSymbol(tfContext,out));
        }
        return outputs;
    }
    TFSymbol Quantize(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Quantize(tfContext,input.getName()));
    }
    TFSymbol DeQuantize(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::DeQuantize(tfContext,input.getName()));
    }
    TFSymbol GreedyCTC(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::GreedyCTC(tfContext,input.getName()));
    }
    TFSymbol Einsum(string eq,py::tuple inputs){
        vector<string> input_;
        TFContext tfContext = nullptr;
        for(int i = 0;i<py::len(inputs);i++){
            TFSymbol oss = py::cast<TFSymbol>(inputs[i]);
            if(tfContext== nullptr)tfContext = oss.GetContext();
            TPlog(tfContext == oss.GetContext(),"inputs' Context is different")
            input_.push_back(oss.getName());
        }
        return TFSymbol(tfContext,TFFunc::Einsum(tfContext,eq,input_));
    }
    // img Op
    TFSymbol BilnearReSize1(TFSymbol input,int outheight,int outwidth,bool align_corners){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::BilnearReSize(tfContext,input.getName(),outheight,outwidth,align_corners));
    }
    TFSymbol NearestReSize1(TFSymbol input,int outheight,int outwidth){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::NearestReSize(tfContext,input.getName(),outheight,outwidth));
    }
    TFSymbol BilnearReSize2(TFSymbol input,float scale,bool align_corners){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::BilnearReSize(tfContext,input.getName(),scale,align_corners));
    }
    TFSymbol NearestReSize2(TFSymbol input,float scale){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::NearestReSize(tfContext,input.getName(),scale));
    }
    TFSymbol BilnearReSize3(TFSymbol input,TFSymbol shape,bool align_corners){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::BilnearReSize(tfContext,input.getName(),shape.getName(),align_corners));
    }
    TFSymbol NearestReSize3(TFSymbol input,TFSymbol shape){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::NearestReSize(tfContext,input.getName(),shape.getName()));
    }
    TFSymbol Upsample(TFSymbol input,int coeff){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Upsample(tfContext,input.getName(),coeff));
    }
    TFSymbol CropAndResize(TFSymbol input,TFSymbol alignmask,int outheight,int outwidth){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::CropAndResize(tfContext,input.getName(),alignmask.getName(),outheight,outwidth));
    }
    TFSymbol RoiPool(TFSymbol input,TFSymbol Rois,int outheight,int outwidth,float spatial_scale){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::RoiPooling(tfContext,input.getName(),Rois.getName(),outheight,outwidth,spatial_scale));
    }
    TFSymbol WarpWithOpFlow(TFSymbol input,TFSymbol flow){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::WarpWithOpFlow(tfContext,input.getName(),flow.getName()));
    }
    TFSymbol WarpWithOpAffine(TFSymbol input,TFSymbol transMat,int outheight,int outwidth,bool padMethod){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::WarpWithAffine(tfContext,input.getName(),transMat.getName(),outheight,outwidth,padMethod));
    }
    TFSymbol WarpWithOpPerspect(TFSymbol input,TFSymbol transMat,int outheight,int outwidth,bool padMethod){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::WarpWithPerspect(tfContext,input.getName(),transMat.getName(),outheight,outwidth,padMethod));
    }
    TFSymbol WarpWithOpGridFace(TFSymbol input,TFSymbol transMat,int gridSize,int outheight,int outwidth,bool padMethod){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::WarpWithGridFace(tfContext,input.getName(),transMat.getName(),gridSize,outheight,outwidth,padMethod));
    }
    TFSymbol Where(TFSymbol condition,TFSymbol data0,TFSymbol data1){
        TFContext tfContext = condition.GetContext();
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),data0.getName(),data1.getName()));
    }
    TFSymbol Where1(TFSymbol condition,TFSymbol data0,float data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.f = data1;
        scalar.second = TFCAPI_FLOAT;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),data0.getName(),scalar));
    }
    TFSymbol Where2(TFSymbol condition,TFSymbol data0,int data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.i32 = data1;
        scalar.second = TFCAPI_INT32;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),data0.getName(),scalar));
    }
    TFSymbol Where3(TFSymbol condition,float data0,TFSymbol data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.f = data0;
        scalar.second = TFCAPI_FLOAT;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),scalar,data1.getName()));
    }
    TFSymbol Where4(TFSymbol condition,int data0,TFSymbol data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.i32 = data0;
        scalar.second = TFCAPI_INT32;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),scalar,data1.getName()));
    }
    TFSymbol Where5(TFSymbol condition,float data0,float data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.f = data0;
        scalar.second = TFCAPI_FLOAT;
        TFFunc::ScalarType scalar2;
        scalar2.first.f = data1;
        scalar2.second = TFCAPI_FLOAT;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),scalar,scalar2));
    }
    TFSymbol Where6(TFSymbol condition,int data0,int data1){
        TFContext tfContext = condition.GetContext();
        TFFunc::ScalarType scalar;
        scalar.first.i32 = data0;
        scalar.second = TFCAPI_INT32;
        TFFunc::ScalarType scalar2;
        scalar2.first.f = data1;
        scalar2.second = TFCAPI_INT32;
        return TFSymbol(tfContext,TFFunc::Where(tfContext,condition.getName(),scalar,scalar2));
    }
    TFSymbol Expand(TFSymbol input,TFSymbol shape){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::Expand(tfContext,input.getName(),shape.getName()));
    }
    TFSymbol Crop(TFSymbol img,TFSymbol Crop){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Crop(tfContext,img.getName(),Crop.getName()));
    }
    TFSymbol Crop2(TFSymbol img,TFSymbol start,TFSymbol end,TFSymbol axis,TFSymbol step){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Crop(tfContext,img.getName(),start.getName(),end.getName(),axis.getName(),step.getName()));
    }
    TFSymbol Crop3(TFSymbol img,py::tuple crops,int startAxis){
        TFContext tfContext = img.GetContext();
        vector<std::tuple<int,int,int>> crops_;
        for(int i = 0;i<py::len(crops);i++){
            auto one = py::cast<py::tuple>(crops[i]);
            crops_.emplace_back(std::make_tuple(py::cast<int>(one[0]),py::cast<int>(one[1]),py::cast<int>(one[2])));
        }
        return TFSymbol(tfContext,TFFunc::Crop(tfContext,img.getName(),crops_,startAxis));
    }
    TFSymbol Cast(TFSymbol img,TFCAPI_DATATYPE dst_type){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Cast(tfContext,img.getName(),dst_type));
    }
    TFSymbol Ceil(TFSymbol img){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Ceil(tfContext,img.getName()));
    }
    TFSymbol Pow(TFSymbol img,int power){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Pow(tfContext,img.getName(),power));
    }
    TFSymbol Sqrt(TFSymbol img){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Sqrt(tfContext,img.getName()));
    }
    TFSymbol Floor(TFSymbol img){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Floor(tfContext,img.getName()));
    }
    TFSymbol Log2(TFSymbol img){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Log2(tfContext,img.getName()));
    }
    TFSymbol ShapeOp(TFSymbol img){
        TFContext tfContext = img.GetContext();
        return TFSymbol(tfContext,TFFunc::Shape(tfContext,img.getName()));
    }

    TFSymbol LSTM(TFSymbol x,int hidden_size,string dir){
        TFContext tfContext = x.GetContext();
        return TFSymbol(tfContext,TFFunc::LSTM(tfContext,x.getName(),hidden_size,dir));
    }
    TFSymbol LSTM2(TFSymbol x,TFSymbol w,TFSymbol r,TFSymbol b,int hidden_size,string dir){
        TFContext tfContext = x.GetContext();
        return TFSymbol(tfContext,TFFunc::LSTM(tfContext,x.getName(),w.getName(),r.getName(),b.getName(),hidden_size,dir));
    }
    TFSymbol Expand_dims(TFSymbol input,py::tuple dims){
        vector<int> shape_;
        for(int i = 0;i<py::len(dims);i++){
            shape_.push_back(py::cast<int>(dims[i]));
        }
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::expand_dims(tfContext,input.getName(),shape_));
    }

    TFSymbol ReadImg(pyTFContext tfContext, int imgLen, py::tuple scale, py::tuple mean,py::tuple shape, TFCAPI_DATATYPE outDatatype,TFCV::DECODER_FLAGS decoderFlags){
        vector<float>scale_,mean_;
        vector<int>shape_;
        for(int i = 0;i<py::len(scale);i++){
            scale_.push_back(py::cast<float>(scale[i]));
        }
        for(int i = 0;i<py::len(mean);i++){
            mean_.push_back(py::cast<float>(mean[i]));
        }
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }

        return TFSymbol(tfContext.self,TFFunc::ReadImg(tfContext.self,imgLen,scale_,mean_,shape_,outDatatype,decoderFlags));
    }

    TFSymbol Placeholder(pyTFContext tfContext, int batch, py::tuple scale, py::tuple mean,py::tuple shape, TFCAPI_DATATYPE outDatatype){
        vector<float>scale_,mean_;
        vector<int>shape_;
        for(int i = 0;i<py::len(scale);i++){
            scale_.push_back(py::cast<float>(scale[i]));
        }
        for(int i = 0;i<py::len(mean);i++){
            mean_.push_back(py::cast<float>(mean[i]));
        }
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }

        return TFSymbol(tfContext.self,TFFunc::Placeholder(tfContext.self,batch,scale_,mean_,shape_,outDatatype));
    }

    TFSymbol Placeholder2(pyTFContext tfContext, py::tuple scale, py::tuple mean,py::tuple shape, TFCAPI_DATATYPE outDatatype){
        vector<float>scale_,mean_;
        vector<int>shape_;
        for(int i = 0;i<py::len(scale);i++){
            scale_.push_back(py::cast<float>(scale[i]));
        }
        for(int i = 0;i<py::len(mean);i++){
            mean_.push_back(py::cast<float>(mean[i]));
        }
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }

        return TFSymbol(tfContext.self,TFFunc::Placeholder(tfContext.self,scale_,mean_,shape_,outDatatype));
    }

    TFSymbol Variable(pyTFContext tfContext, py::tuple shape, TFCAPI_DATATYPE outDatatype){
        vector<int>shape_;
        for(int i = 0;i<py::len(shape);i++){
            shape_.push_back(py::cast<int>(shape[i]));
        }

        return TFSymbol(tfContext.self,TFFunc::Variable(tfContext.self,shape_,outDatatype));
    }

    TFSymbol NextFrame(TFSymbol input){
        TFContext tfContext = input.GetContext();
        return TFSymbol(tfContext,TFFunc::NextFrame(tfContext,input.getName()));
    }

    py::list Custom(py::tuple inputs,py::tuple outputNames,py::str Opname,py::str jsonconfig){
        vector<string> input_,outputs_;
        TFContext tfContext = nullptr;
        for(int i = 0;i<py::len(inputs);i++){
            TFSymbol oss = py::cast<TFSymbol>(inputs[i]);
            if(tfContext== nullptr)tfContext = oss.GetContext();
            TPlog(tfContext == oss.GetContext(),"inputs' Context is different")
            input_.push_back(oss.getName());
        }
        for(int i = 0;i<py::len(outputNames);i++){
            string oss = py::cast<string>(outputNames[i]);
            outputs_.push_back(oss);
        }
        string copname = string(Opname);
        string cjsonconfig = string(jsonconfig);
        TFFunc::Custom(tfContext,input_,outputs_,copname,cjsonconfig);
        py::list outputs;
        for(auto outname : outputs_){
            outputs.append(TFSymbol(tfContext,outname));
        }
        return outputs;
    }


    void pyCalibration(pyTFCalibration& self){
        Calibration(self.self);
    }

    py::list GetInputTensorCalibration(pyTFCalibration& self){
        py::list outlist;
        auto inputs = GetInputTensors(self.self);
        for(auto& input : inputs){
            outlist.append(std::move(pyTFTensor(std::move(input))));
        }
        return outlist;
    }

    void QuantizeCalibration(pyTFCalibration& self,py::dict inputtype,py::tuple avoidnodes,py::tuple stopquantnodes,bool mergeeltwise,bool mergeconcat,bool perchannel){

        map<string,TFCAPI_DATATYPE> inputtype_;
        std::set<string> avoidnodes_;
        std::set<string> stopquantnodes_;
        for(auto ob: inputtype){
            string key = py::cast<string>(ob.first);
            TFCAPI_DATATYPE dtype = py::cast<TFCAPI_DATATYPE>(ob.second);
            inputtype_[key] = dtype;
        }
        for(int i=0;i<py::len(avoidnodes);i++){
            avoidnodes_.insert(py::cast<string>(avoidnodes[i]));
        }
        for(int i=0;i<py::len(stopquantnodes);i++){
            stopquantnodes_.insert(py::cast<string>(stopquantnodes[i]));
        }
        Quantize(self.self,inputtype_,avoidnodes_,stopquantnodes_,mergeeltwise,mergeconcat,perchannel);
    }

    void LoadCustomOp(py::str path){
        string cpath = path;
        RegisterCustomOpFromFile(cpath);
    }

    void pyRegisterCustomOp(py::str OpName,py::function Reshape,py::function Eval){
        RegisterCustomOp(string(OpName)).Set([](TFContext context,TFNode node){
        },[Reshape](TFContext context,TFNode node){
            auto info = GetNodeInfo(node);
            py::tuple inputs(info.InputNames.size());
            int index = 0;
            for(auto name: info.InputNames){
                inputs[index] = py::cast(pyTFTensor(std::move(GetTensorByName(context,name)),std::move(GetTensorQuantizeInfo(context,name))));
                index++;
            }

            py::dict kwargs;
            kwargs["config"] = GetNodeCustomJsonStr(node);
            auto outputs = Reshape(*inputs,**kwargs);
            if(py::isinstance<py::tuple>(outputs)){
                auto output = py::cast<py::tuple>(outputs);
                index =0;
                for(auto out : output){
                    auto ob = py::cast<py::dict>(out);

                    auto shape = py::cast<vector<int>>(ob["shape"]);
                    auto dtype = py::cast<TFCAPI_DATATYPE>(ob["dtype"]);
                    auto data = GetTensorByName(context,info.OutputNames[index]);
                    ReSizeTensor(data,shape);
                    SetTensorType(data,dtype);
                    index++;
                }
            }else if(py::isinstance<py::dict>(outputs)){
                auto ob = py::cast<py::dict>(outputs);
                auto shape = py::cast<vector<int>>(ob["shape"]);
                auto dtype = py::cast<TFCAPI_DATATYPE>(ob["dtype"]);
                auto data = GetTensorByName(context,info.OutputNames[0]);
                ReSizeTensor(data,shape);
                SetTensorType(data,dtype);


            }
        },[Eval](TFContext context,TFNode node){
            auto info = GetNodeInfo(node);
            py::tuple inputs(info.InputNames.size());
            int index = 0;
            for(auto name: info.InputNames){
                inputs[index] = py::cast(pyTFTensor(std::move(GetTensorByName(context,name)),std::move(GetTensorQuantizeInfo(context,name))));
                index++;
            }
            vector<Quantization> outconfig;
            for(auto name: info.OutputNames){
                outconfig.push_back(std::move(GetTensorQuantizeInfo(context,name)));
            }
            py::dict kwargs;
            kwargs["config"] = GetNodeCustomJsonStr(node);
            vector<py::dict> allconfig;
            for(auto& config:outconfig){
                if(config.IsValid()){
                    py::dict oneconfig;
                    oneconfig["max"] = GetQuantizationMax(config);
                    oneconfig["min"] = GetQuantizationMin(config);
                    oneconfig["scale"] = GetQuantizationScale(config);
                    oneconfig["zeropoint"] = GetQuantizationZeroPoint(config);
                    allconfig.push_back(oneconfig);
                }
            }
            if(!allconfig.empty())kwargs["outconfig"] = allconfig;
            auto outputs = Eval(*inputs,**kwargs);
            if(py::isinstance<py::tuple>(outputs)){
                auto output = py::cast<py::tuple>(outputs);
                index =0;
                for(auto out : output){
                    auto ob = py::cast<py::array>(out);
                    auto data = pyTFTensor(std::move(GetTensorByName(context,info.OutputNames[index])));
                    FromNumpy(data,ob);
                    index++;
                }
            }else if(py::isinstance<py::array>(outputs)){
                auto ob = py::cast<py::array>(outputs);
                auto data = pyTFTensor(std::move(GetTensorByName(context,info.OutputNames[0])));
                FromNumpy(data,ob);

            }else{
                popError("unknown output type:must be(tuple,numpy)");
            }
        },[](TFContext context,TFNode node){

        });
    }

    void ReplaceOp(pyTFContext& self,py::str NodeName,py::function builder){
        string nodename  = NodeName;
        ReplaceNodeByName(self.self,nodename,[&builder](TFContext& context,vector<string>inputs)->vector<string>{
            py::tuple pyinputs(inputs.size());
            int index= 0;
            for(auto name : inputs){
                pyinputs[index] = std::move(TFSymbol(context,name));
                index++;
            }
            auto outputs = builder(*pyinputs);
            vector<string> outnames;
            if(py::isinstance<py::tuple>(outputs)) {
                for (auto output : outputs) {
                    TFSymbol symbol = py::cast<TFSymbol>(output);
                    outnames.push_back(symbol.getName());
                }
            }else if(py::isinstance<TFSymbol>(outputs)){
                TFSymbol symbol = py::cast<TFSymbol>(outputs);
                outnames.push_back(symbol.getName());
            }else{
                popError("Not allowed return type(only TFSymbol or tuple(TFSymbol))");
            }
            return outnames;
        });
    }
    void RemoveOp(pyTFContext& self,py::str NodeName){
        string nodename  = NodeName;
        RemoveNodeByName(self.self,nodename);
    }

    TFSymbol symbol_getitem(TFSymbol &self,int index){
        auto symbol = Gather(self,TFSymbol(self.GetContext()),0);
        RegisterParamTensor(self.GetContext(),symbol.getName()+":0",{1},TFCAPI_INT32,&index,sizeof(int));
        return symbol;
    }
    TFSymbol symbol_getitem_help(TFSymbol &self,py::tuple index,int axis){

    }

    TFSymbol symbol_getitem1(TFSymbol &self,py::tuple index){

        TFSymbol intern = self;
        vector<int> Crops;
        bool ifhascrop = false;

        //parse slice
        for(auto item : index){
            if(py::isinstance<py::slice>(item)){
                py::slice indices = py::cast<py::slice>(item);
                size_t start, stop, step, slicelength;

                if (!indices.compute(INT64_MAX, &start, &stop, &step, &slicelength))
                    throw py::error_already_set();
                if(stop == INT64_MAX)stop = -1;

                Crops.push_back((int)start);
                Crops.push_back((int)stop);
                Crops.push_back((int)step);
                if(!(start ==0 && stop == -1 && step ==1)){
                    ifhascrop = true;
                }

            }else{
                Crops.push_back((int)0);
                Crops.push_back((int)-1);
                Crops.push_back((int)1);
            }
        }

        if(ifhascrop){
            intern = Crop(intern,TFSymbol(intern.GetContext()));

            RegisterParamTensor(intern.GetContext(), intern.getName() + ":0", {(int)Crops.size()/3, 3}, TFCAPI_INT32,
                                Crops.data(), Crops.size() * sizeof(int));
        }


        //parse int or ints
        int axis = 0;
        for(auto item : index){

            if(py::isinstance<py::int_>(item)){
                intern = Gather(intern,TFSymbol(intern.GetContext()),axis);
                int dd = py::cast<int>(item);
                RegisterParamTensor(intern.GetContext(),intern.getName()+":0",{1},TFCAPI_INT32,&dd,sizeof(int));
            }else if(py::isinstance<py::tuple>(item)){
                std::vector<int> newitem = py::cast<std::vector<int>>(item);
                intern = Gather(intern,TFSymbol(intern.GetContext()),axis);
                RegisterParamTensor(intern.GetContext(),intern.getName()+":0",{(int)newitem.size()},TFCAPI_INT32,newitem.data(),newitem.size()*sizeof(int));
            }
            axis++;
        }

        return intern;
    }
    TFSymbol symbol_getitem2(TFSymbol &self,py::slice indices){
        size_t start, stop, step, slicelength;

        if (!indices.compute(INT64_MAX, &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        TFSymbol symbol;
        if(stop == INT64_MAX && step == 1 && start == 0){
            symbol = self;
        }else {
            symbol = Crop(self,TFSymbol(self.GetContext()));

            vector<int> param = {(int)start,(int)stop,(int)step};

            RegisterParamTensor(self.GetContext(), symbol.getName() + ":0", {1, 3}, TFCAPI_INT32,
                                param.data(), param.size() * sizeof(int));
        }
        return symbol;
    }
}



void initModule(){
    TFDL2_INITLOG(&PythonInter::pylog);
}
PYBIND11_MODULE(TFDL2,m) {

    initModule();
    py::enum_<TFCAPI_DATATYPE>(m, "_TFDataType")
            .value("FLOAT", TFCAPI_DATATYPE::TFCAPI_FLOAT)
            .value("UINT8", TFCAPI_DATATYPE::TFCAPI_UINT8)
            .value("INT32", TFCAPI_DATATYPE::TFCAPI_INT32)
            .value("FLOAT64", TFCAPI_DATATYPE::TFCAPI_FLOAT64)
            .value("INT64", TFCAPI_DATATYPE::TFCAPI_INT64)
            .value("STRING", TFCAPI_DATATYPE::TFCAPI_STRING)
            .value("UINT16", TFCAPI_DATATYPE::TFCAPI_UINT16)
            .value("FLOAT16", TFCAPI_DATATYPE::TFCAPI_FLOAT16)
            .value("BFLOAT16", TFCAPI_DATATYPE::TFCAPI_BFLOAT16)
            .value("INT16", TFCAPI_DATATYPE::TFCAPI_INT16);

    py::enum_<TFConvertType>(m, "_TFConvertType")
            .value("CAFFE2TFDL", TFConvertType::CAFFE2TFDL)
            .value("TENSORFLOW2TFDL", TFConvertType::TENSORFLOW2TFDL)
            .value("ONNX2TFDL", TFConvertType::ONNX2TFDL)
            .value("TFLITE2TFDL", TFConvertType::TFLITE2TFDL);

    py::enum_<TFCalibrationMode>(m, "_CalibrationMode")
            .value("Naive", TFCalibrationMode::TFCali_Naive)
            .value("MEAN", TFCalibrationMode::TFCali_MEAN)
            .value("COVERAGE", TFCalibrationMode::TFCali_COVERAGE)
            .value("KLD", TFCalibrationMode::TFCali_KLD);

    py::enum_<TFCV::DECODER_FLAGS>(m, "_DECODER_FLAGS")
            .value("BGR", TFCV::DECODER_FLAGS::TFCV_BGR)
            .value("RGB", TFCV::DECODER_FLAGS::TFCV_RGB)
            .value("Gray", TFCV::DECODER_FLAGS::TFCV_Gray);


#define defop(op)\
    m.def(#op,PythonInter::op,py::return_value_policy::move);\

    defop(Add)
    defop(Mul)
    defop(Div)
    defop(Sub)
    defop(Min)
    defop(Max)
    defop(Add1)
    defop(Sub1)
    defop(Mul1)
    defop(Div1)
    defop(Add2)
    defop(Sub2)
    defop(Mul2)
    defop(Div2)
    defop(Einsum)

    defop(GT)
    defop(GE)
    defop(LT)
    defop(LE)
    defop(EQ)
    defop(GT1)
    defop(GE1)
    defop(LT1)
    defop(LE1)
    defop(EQ1)
    defop(GT2)
    defop(GE2)
    defop(LT2)
    defop(LE2)
    defop(EQ2)
    defop(LayerNorm)
    defop(LayerNorm2)
    defop(ConstantOfShape)
    defop(Range)
    defop(ReArrange)
    defop(deReArrange)
    defop(MaskSoftmax)

    defop(Convolution)
    defop(Convolution2)
    defop(Requantize)
    defop(Scale)
    defop(Scale2)
    defop(InnerProduct)
    defop(InnerProduct2)
    defop(DeConvolution)
    defop(DeConvolution2)
    defop(MaxPooling)
    defop(TFMaxPooling)
    defop(TFAvePooling)
    defop(TFConvolution)
    defop(GlobalAvePooling)
    defop(MatMul)
    defop(MatMul2)
    defop(AvePooling)
    defop(ReLU)
    defop(ELU)
    defop(ReLUX)
    defop(LeakyReLU)
    defop(PReLU)
    defop(Swish)
    defop(Mish)
    defop(HardSwish)
    defop(Tanh)
    defop(GeLU)
    defop(Sigmoid)
    defop(HardSigmoid)
    defop(HardSigmoid2)
    defop(Expand)
    defop(Exp)
    defop(Softmax)
    defop(Reshape)
    defop(Reshape2)
    defop(BroadCast)
    defop(BroadCast2)
    defop(Squeeze)
    defop(Pad)
    defop(ZeroMask)
    defop(Pad2)
    defop(Flatten)
    defop(Flatten2Matrix)
    defop(ReduceMax)
    defop(ReduceMean)
    defop(MeanOp)
    defop(ReduceSum)
    defop(ReduceMin)
    defop(Transpose)
    defop(Concat)
    defop(Gather)
    defop(Gather2)
    defop(Flip)
    defop(Split)
    defop(Slice)
    defop(Quantize)
    defop(DeQuantize)
    defop(ReadImg)
    defop(Placeholder)
    defop(Placeholder2)
    defop(BilnearReSize1)
    defop(BilnearReSize2)
    defop(BilnearReSize3)
    defop(NearestReSize1)
    defop(NearestReSize2)
    defop(NearestReSize3)
    defop(Upsample)
    defop(CropAndResize)
    defop(WarpWithOpFlow)
    defop(WarpWithOpAffine)
    defop(WarpWithOpPerspect)
    defop(WarpWithOpGridFace)
    defop(RoiPool)
    defop(Crop)
    defop(Crop2)
    defop(Crop3)
    defop(Where1)
    defop(Where2)
    defop(Where3)
    defop(Where4)
    defop(Where5)
    defop(Where6)
    defop(Where)
    defop(Cast)
    defop(Ceil)
    defop(Floor)
    defop(Pow)
    defop(Sqrt)
    defop(Log2)
    defop(ShapeOp)
    defop(Expand_dims)
    defop(Custom)
    defop(Correlation)
    defop(Variable)
    defop(Unfold)
    defop(GreedyCTC)
    defop(ArgMin)
    defop(ArgMax)
    defop(NextFrame)
    defop(MatMulWeight)
    defop(LSTM)
    defop(LSTM2)

    m.def("LoadCustomOp", PythonInter::LoadCustomOp, py::return_value_policy::move);

    m.def("AddInt8Config", PythonInter::RegisterQuantInfo, py::return_value_policy::move);

    m.def("RegisterCustomOp", PythonInter::pyRegisterCustomOp);


    py::class_<PythonInter::pyTFTensor>(m, "_TFTensor", "Tensor of TFDL for python")
            .def(py::init<PythonInter::pyTFTensor&>())
            //.def("_ReadImg", PythonInter::TensorReadImg)
            .def("_shape", PythonInter::Shape)
            .def("_tonumpy", PythonInter::toNumpy)
            .def("_fromNumpy", PythonInter::FromNumpy)
            .def("_SetScalar", PythonInter::SetScalar)
            .def("_SetScalar2", PythonInter::SetScalar2)
            .def("_dtype", PythonInter::Type)
            .def("_maxs", PythonInter::tensorMaxs)
            .def("_mins", PythonInter::tensorMins)
            .def("_scales", PythonInter::tensorScales)
            .def("_zeropoints", PythonInter::tensorZeroPoints)
            .def("_name", PythonInter::tensorName)
            .def("__str__", PythonInter::Tostring);


    py::class_<PythonInter::TFSymbol>(m, "TFSymbol", "TFSymbol of TFDL for python")
            .def(py::init())
            .def("__str__", PythonInter::printsymbol)
            .def("__add__", PythonInter::Add)
            .def("__add__", PythonInter::Addf)
            .def("__add__", PythonInter::Addi)
            .def("__radd__", PythonInter::Add)
            .def("__radd__", PythonInter::rAddf)
            .def("__radd__", PythonInter::rAddi)
            .def("__sub__", PythonInter::Sub)
            .def("__sub__", PythonInter::Subi)
            .def("__sub__", PythonInter::Subf)
            .def("__rsub__", PythonInter::rSub)
            .def("__rsub__", PythonInter::rSubi)
            .def("__rsub__", PythonInter::rSubf)
            .def("__mul__", PythonInter::Mul)
            .def("__mul__", PythonInter::Muli)
            .def("__mul__", PythonInter::Mulf)
            .def("__rmul__", PythonInter::Mul)
            .def("__rmul__", PythonInter::rMuli)
            .def("__rmul__", PythonInter::rMulf)
            .def("__truediv__", PythonInter::Div)
            .def("__truediv__", PythonInter::Divi)
            .def("__truediv__", PythonInter::Divf)
            .def("__rtruediv__", PythonInter::rDiv)
            .def("__rtruediv__", PythonInter::rDivi)
            .def("__rtruediv__", PythonInter::rDivf)
            .def("__lt__", PythonInter::LT)
            .def("__lt__", PythonInter::LTi)
            .def("__lt__", PythonInter::LTf)
            .def("__gt__", PythonInter::GT)
            .def("__gt__", PythonInter::GTi)
            .def("__gt__", PythonInter::GTf)
            .def("__le__", PythonInter::LE)
            .def("__le__", PythonInter::LEi)
            .def("__le__", PythonInter::LEf)
            .def("__ge__", PythonInter::GE)
            .def("__ge__", PythonInter::GEi)
            .def("__ge__", PythonInter::GEf)
            .def("__eq__", PythonInter::EQ)
            .def("__eq__", PythonInter::EQi)
            .def("__eq__", PythonInter::EQf)
            .def("__getitem__",PythonInter::symbol_getitem)
            .def("__getitem__",PythonInter::symbol_getitem1)
            .def("__getitem__",PythonInter::symbol_getitem2);

    py::class_<PythonInter::pyTFContext>(m, "_TFContext", "this is the Context of tfdl　for python")
            .def(py::init<py::tuple>())
            .def(py::init<py::str>())
            .def("_RegisterParamToContext", PythonInter::RegisterParamToContext)
            .def("_GetParamSymbol", PythonInter::GetParamSymbol)
            .def("_GetParam", PythonInter::GetParam)
            .def("_RegistorInt8config", PythonInter::RegistorInt8config)
            .def("_GetAttr", PythonInter::GetAttr)
            .def("_DumpToFile", PythonInter::DumpContext)
            .def("_Close", PythonInter::pycloseContext)
            .def("_ReplaceOp", PythonInter::ReplaceOp)
            .def("_RemoveOp", PythonInter::RemoveOp)
            .def("_GetOutSymbols", PythonInter::pyGetOutSymbolsContext)
            .def("_GetInputSymbols", PythonInter::pyGetInputSymbolsContext)
            .def("_SetOut", PythonInter::pySetOutContext)
            .def("_Modify", PythonInter::pyModifyContext)
            .def("_Assign", PythonInter::pyAssignContext)
            .def("_open", PythonInter::pyOpenContext);

    py::class_<PythonInter::pyTFExecutor>(m, "_TFExecutor", "this is the wrape of tfdl　for python")
            .def(py::init<PythonInter::pyTFContext &, py::str>())
            .def("_Forward", PythonInter::Forward)
            .def("_Getinputs", PythonInter::GetInputTensor)
            .def("_SetPrintInfo", PythonInter::pySetPrintInfo)
            .def("_DumpSim", PythonInter::pyDumpSimulator)
            .def("_GetTensorByName", PythonInter::PYGetTensorByName);

    py::class_<PythonInter::pyTFCalibration>(m, "_TFCalibration",
                                             "this is the wrape of tfdl calibration　for python")
            .def(py::init<PythonInter::pyTFContext &, TFCalibrationMode,py::str>())
            .def("_Calibration", PythonInter::pyCalibration)
            .def("_Getinputs", PythonInter::GetInputTensorCalibration)
            .def("_Quantize", PythonInter::QuantizeCalibration);
/*
    py::class_<PythonInter::pyImgReader>(m, "_TFImgReader",
                                             "this is the wrape of tfdl calibration　for python")
            .def(py::init<>())
            .def("_Open", PythonInter::Open<PythonInter::pyImgReader>)
            .def("_ReadFrame", PythonInter::ReadFrame<PythonInter::pyImgReader>)
            .def("_ReadFrame2Tensor", PythonInter::ReadFrame2Tensor<PythonInter::pyImgReader>)
            .def("_tonumpy", PythonInter::ToNumpy<PythonInter::pyImgReader>);

    py::class_<PythonInter::pyVideoCapture>(m, "_TFVideoCapture",
                                         "this is the wrape of tfdl calibration　for python")
            .def(py::init<>())
            .def("_Open", PythonInter::Open<PythonInter::pyVideoCapture>)
            .def("_ReadFrame", PythonInter::ReadFrame<PythonInter::pyVideoCapture>)
            .def("_ReadFrame2Tensor", PythonInter::ReadFrame2Tensor<PythonInter::pyVideoCapture>)
            .def("_tonumpy", PythonInter::ToNumpy<PythonInter::pyVideoCapture>);
            */
}