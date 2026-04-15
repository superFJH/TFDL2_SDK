//
// Created by test on 2021/2/19.
//

#include <iostream>
#include "TFDL2_C_API.h"
#include <unistd.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <assert.h>
#include <cstring>
#include "json11.hpp"
#include <cmath>
#include "TFCV/TFCV.h"
#include<mutex>
using namespace TFDL_CAPI;

static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};
void SetRandom(TFTensor data) {
    if(GetTensorType(data) == TFDL_CAPI::TFCAPI_FLOAT) {
        float *ptr = (float*)GetTensordata(data);
        long len = GetTensorCount(data,0);
        for (long i = 0; i < len; i++) {
            ptr[i] = std::rand() % 2 == 0 ? 10.f / ((std::rand() % 1000) + 1.f) : -10.f / ((std::rand() % 1000) + 1.f);
        }
    }else if(GetTensorType(data) == TFDL_CAPI::TFCAPI_INT32){
        int *ptr = (int *)GetTensordata(data);
        long len = GetTensorCount(data,0);
        for (long i = 0; i < len; i++) {
            ptr[i] = std::rand() % 255;
        }
    }else if(GetTensorType(data) == TFDL_CAPI::TFCAPI_INT64){
        long *ptr = (long *)GetTensordata(data);
        long len = GetTensorCount(data,0);
        for (long i = 0; i < len; i++) {
            ptr[i] = std::rand() % 255;
        }
    }else{
        uint8_t *ptr = (uint8_t *)GetTensordata(data);
        long len = GetTensorCount(data,0);
        for (long i = 0; i < len; i++) {
            ptr[i] = std::rand() % 255;
        }
    }
}
int Num = 0;
std::atomic_int cnt = {0};
std::mutex standlck;
std::map<string,vector<uint8_t>> standdatas;
int main(int argc, char** argv) {
    srand(time(NULL));
    std::string protopath = argv[1];
    int thread_num = atoi(argv[2]);
    Num = atoi(argv[3]);
    int debug = atoi(argv[4]);
    string config = argv[5];

    std::fstream file(config);
    string configstring((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    string err;



    auto tfContext = LoadProto(protopath);
    auto reader = TFCV::NewImgReader();
    TFCV::OpenURL(reader,argv[6]);
    bool ischeck =  atoi(argv[7]);
    if(debug){
        {

            json11::Json configjson = json11::Json::parse(configstring,err);
            json11::Json::object& cb = (json11::Json::object&)configjson.object_items();
            cb["UseHardware"] = false;
            cb["FrugalMode"] = false;
            cb["cpuLimit"] = 1;
            auto tfExecutor = CompileExecutor(tfContext, false, configstring);
            auto tfExecutor2 = CompileExecutor(tfContext, false,configjson.dump());
            //printf("compile proto\n");
            //auto dd = config.object_items();
            //dd["UseHardware"] = false;
            //auto tfExecutor2 = CompileExecutor(tfContext, true, json11::Json(dd).dump());


            //SetPrintInfo(tfExecutor, true);
            auto inputs = GetInputTensors(tfExecutor);
            auto inputs2 = GetInputTensors(tfExecutor2);
            auto outputs = GetOutputTensors(tfExecutor);
            auto outputs2 = GetOutputTensors(tfExecutor2);
            int debugCount = debug;
            int wrongCount = 0;
            while(debugCount) {
                int index = 0;
                for (auto &input: inputs) {
                    //memset(GetTensordata(input),0,GetTensorDataSize(input));
                    //memset(GetTensordata(inputs2[index]),1,GetTensorDataSize(inputs2[index]));
                    //SetRandom(input);
                    TFCV::DumpImgData(reader,input,0,TFDL_CAPI::TFCV::TFCV_BGR);
                    memcpy(GetTensordata(inputs2[index]), GetTensordata(input), GetTensorDataSize(input));
                    index++;
                }
                index = 0;
                for (auto &output: outputs) {
                    memset(GetTensordata(output),0,GetTensorDataSize(output));
                    //memset(GetTensordata(inputs2[index]),1,GetTensorDataSize(inputs2[index]));
                    index++;
                }
                /*auto inputs2 = GetInputTensors(tfExecutor2);
                for(auto input : inputs2){
                    memset(GetTensordata(input),0,GetTensorDataSize(input));
                }*/
                //printf("start forward\n");
                //SetPrintInfo(tfExecutor, false);
                //ForwardExecutorAlone(tfExecutor);
                if (debug == 1)SetPrintInfo(tfExecutor, true);

                ForwardExecutorAlone(tfExecutor);

                ForwardExecutorAlone(tfExecutor2);

                bool isWrong = false;
                //for(int i=0;i<GetTensorNumbers(tfExecutor2);i++){
                for (int i = 0; i < outputs.size(); i++) {
                    /*auto testtensor = GetTensorByIndex(tfExecutor2,i);
                    auto testtensor2 = GetTensorByName(tfExecutor,GetTensorName(testtensor));
                    auto size = GetTensorCount(testtensor,0);
                    printf("%s %s %d\n", GetTensorName(testtensor2).c_str(),GetTensorName(testtensor).c_str(),size);
                    uint8_t *o1_t = (uint8_t *)GetTensordata(testtensor2);
                    vector<uint8_t> tmpdata(size);
                    if (false && GetTensorCount(testtensor2,2)%128!=0) {
                        size_t bytesize = sizeof(uint8_t);
                        int outer = GetTensorCount(testtensor2,0,2);
                        int alignspatial = ((GetTensorCount(testtensor2,2)-1)/128+1)*128;
                        int inner = GetTensorCount(testtensor2,2);
                        printf("make align %d %d %d\n",outer,alignspatial,inner);
                        uint8_t *tmpptr = tmpdata.data();
                        for (int i = 0; i < outer; i++) {
                            memcpy(tmpptr, o1_t + i * alignspatial * bytesize, inner * bytesize);
                            tmpptr += inner * bytesize;
                        }
                    } else {
                        memcpy(tmpdata.data(), o1_t, size);
                    }
                    uint8_t *o2 = (uint8_t *)GetTensordata(testtensor);
                    uint8_t * o1 = tmpdata.data();*/
                    int ooo = 0;
                    auto size = GetTensorCount(outputs[i], 0);
                    if (debug == 1) {
                        printf("%s %s %d\n", GetTensorName(outputs[i]).c_str(), GetTensorName(outputs2[i]).c_str(),
                               size);
                    }
                    if (GetTensorType(outputs[i]) == TFDL_CAPI::TFCAPI_FLOAT) {
                        float *o1 = (float *) GetTensordata(outputs[i]);
                        float *o2 = (float *) GetTensordata(outputs2[i]);
                        for (size_t j = 0; j < size; j++) {
                            if (fabs(*o1 - *o2) > 0.01) {
                                ooo++;
                                if (ooo <= 10) {
                                    if (debug == 1)printf("Check Error float %d %f %f\n", j, *o1, *o2);
                                }
                            }
                            o1++;
                            o2++;
                        }
                    }
                    else if (GetTensorType(outputs[i]) == TFDL_CAPI::TFCAPI_INT64) {
                        int64_t *o1 = (int64_t *) GetTensordata(outputs[i]);
                        int64_t *o2 = (int64_t *) GetTensordata(outputs2[i]);
                        for (size_t j = 0; j < size; j++) {
                            if (o1[j] != o2[j]) {
                                ooo++;
                                if (ooo > 10)continue;
                                if (debug == 1)printf("Check Error long  %d %ld %ld\n", j, o1[j], o2[j]);
                            }
                        }
                    }else {
                        uint8_t *o1 = (uint8_t *) GetTensordata(outputs[i]);
                        uint8_t *o2 = (uint8_t *) GetTensordata(outputs2[i]);
                        for (size_t j = 0; j < size; j++) {
                            if (o1[j] != o2[j]) {
                                ooo++;
                                if (ooo > 10)continue;
                                if (debug == 1)printf("Check Error int8 %d %d %d\n", j, o1[j], o2[j]);
                            }
                        }
                    }
                    if(ooo > 0)isWrong = true;
                    if (debug == 1)printf("Error size %d\n", ooo);
                    ooo = 0;
                }
                if(isWrong)wrongCount++;
                debugCount--;
            }
            if(wrongCount == 0){
                printf("Check Success\n");
            }else{
                printf("Check Fail %d\n",wrongCount);
            }
        }
        return 0;
    }

    //prepare template{
    if(ischeck){
        json11::Json configjson2 = json11::Json::parse(configstring, err);
        json11::Json::object &cb2 = (json11::Json::object &) configjson2.object_items();
        cb2["UseHardware"] = false;
        cb2["FrugalMode"] = false;
        cb2["cpuLimit"] = 1;
        auto standtfExecutor2 = CompileExecutor(tfContext, false, configjson2.dump());
        auto inputs = GetInputTensors(standtfExecutor2);
        int index = 0;
        for (auto &input: inputs) {
            //memset(GetTensordata(input),0,GetTensorDataSize(input));
            //memset(GetTensordata(inputs2[index]),1,GetTensorDataSize(inputs2[index]));
            //SetRandom(input);
            TFCV::DumpImgData(reader, input, 0, TFDL_CAPI::TFCV::TFCV_BGR);
            index++;
        }
        ForwardExecutorAlone(standtfExecutor2);
        auto outputs = GetOutputTensors(standtfExecutor2);
        for(auto& oo : outputs){
            auto size = GetTensorDataSize(oo);
            auto name = GetTensorName(oo);
            standdatas[name].resize(size);
            memcpy((void*)standdatas[name].data(), GetTensordata(oo),size);

        }
    }
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(pthread_self(),SCHED_FIFO,&param);
    json11::Json configjson = json11::Json::parse(configstring,err);
    json11::Json::object cb = configjson.object_items();
    vector<TFExecutor> tfexcutors;
    vector<std::thread*> thread_pool;
    vector<TFVision> imgs;
    for(int i=0;i<thread_num;i++){
        if(i % 2 == 0){
            cb["Core"] = json11::Json::array{-1};
            configstring = json11::Json(cb).dump();
        }else{
            cb["Core"] = json11::Json::array{-2};
            configstring = json11::Json(cb).dump();
        }
        auto tfExecutor = CompileExecutor(tfContext,true,configstring);
        SetPrintInfo(tfExecutor,false);
        ForwardExecutor(tfExecutor);
        auto reader2 = TFCV::NewImgReader();
        TFCV::OpenURL(reader2,argv[6]);
        imgs.emplace_back(reader2);
        tfexcutors.push_back(std::move(tfExecutor));
    }
    auto st = std::chrono::system_clock::now();
    for(int i=0;i<thread_num;i++){
        if(ischeck) {
            auto thread = new std::thread([](TFExecutor tfExecutor1, TFVision img) {
                struct sched_param param;
                param.sched_priority = sched_get_priority_max(SCHED_FIFO);
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
                while (cnt < Num) {
                    auto inputs_ = GetInputTensors(tfExecutor1);
                    for (auto &input: inputs_) {
                        TFCV::DumpImgData(img, input, 0, TFDL_CAPI::TFCV::TFCV_BGR);
                    }
                    ForwardExecutorAlone(tfExecutor1);
                    standlck.lock();
                    auto outputs = GetOutputTensors(tfExecutor1);
                    for (auto &oo: outputs) {
                        auto size = GetTensorDataSize(oo);
                        auto name = GetTensorName(oo);
                        auto &standdata = standdatas[name];
                        vector<uint8_t> thisdata(size);
                        memcpy((void *) thisdata.data(), GetTensordata(oo), size);
                        if (thisdata != standdata) {
                            printf("Check Fail \n");
                        }

                    }
                    standlck.unlock();
                    cnt++;
                }
            }, tfexcutors[i], imgs[i]);
            thread_pool.push_back(thread);
        }else{
            auto thread = new std::thread([](TFExecutor tfExecutor1) {
                struct sched_param param;
                param.sched_priority = sched_get_priority_max(SCHED_FIFO);
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
                while (cnt < Num) {
                    ForwardExecutorAlone(tfExecutor1);
                    cnt++;
                }
            }, tfexcutors[i]);
            thread_pool.push_back(thread);
        }
    }
    string bar = "";
    const char *lable = "|/-\\";



    int last = 0;
    while (cnt < Num) {
        int cur = cnt;
        if (cur * 100 / Num > last) {
            last = cur * 100 / Num;
            if ((last + 1) % 4 == 0) {
                bar += "#";
            }

            printf("[%-25s][%d%%][%c]\r", bar.c_str(), last + 1, lable[(last + 1) % 4]);
            if (last < 100 - 1) {
                fflush(stdout);
            } else {
                printf("\n");
            }
        }

        usleep(1000);
    }

    auto end = std::chrono::system_clock::now();
    float spend = GetSpan(st, end);
    printf("total time = %f s.\n", spend);
    printf("single image spend %f s\n", spend / (float) (cnt));
    printf("fps = %f\n", (float) (cnt) / spend);
    for(auto thread : thread_pool){
        if(thread->joinable())thread->join();
    }
    printf("all thread over\n");
    return 0;
}