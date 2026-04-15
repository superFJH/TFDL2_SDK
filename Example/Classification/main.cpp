#include <iostream>
#include "TFDL2_C_API.h"
#include "TFCV/TFCV.h"
#include <unistd.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <assert.h>
#include <cmath>
#include <algorithm>
using namespace TFDL_CAPI;
static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};
int main(int argc, char** argv) {
    std::string imageFile = argv[1];
    std::string labelFile = argv[2];
    string config = argv[3];
    std::fstream file(config);
    string configstring((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    std::string protopath = argv[4];
    
    /* Load labels. */
    std::vector <string> labels;
    std::ifstream inputLabel(labelFile.c_str());
    string line;
    while (std::getline(inputLabel, line))
        labels.emplace_back(line);

    
    auto reader = TFCV::NewImgReader();

    auto context = LoadProto(protopath);
    
    auto tfExecutor =CompileExecutor(context,true,configstring);
    //warm up
    ForwardExecutorAlone(tfExecutor);

    auto st = std::chrono::system_clock::now();

    auto images = GetInputTensors(tfExecutor)[0];

    //read img
    TFCV::OpenURL(reader,imageFile);

    auto st1 = std::chrono::system_clock::now();
    //set img data into tensor (include resize and RGB->BGR)
    TFCV::DumpImgData(reader,images,0,TFDL_CAPI::TFCV::TFCV_BGR);

    auto st2 = std::chrono::system_clock::now();

    ForwardExecutorAlone(tfExecutor);

    auto st3 = std::chrono::system_clock::now();

    auto outputtensor = GetOutputTensors(tfExecutor)[0];

    //开始top5 排序检索。
    std::vector <std::pair <float , string> > ans;
    std::vector <std::pair <float , string> > tmp;
    std::cout<<"top5:"<<std::endl;
    if(GetTensorType(outputtensor) == TFCAPI_FLOAT) {
        std::vector<float *> outputs;
        outputs.push_back((float*)GetTensordata(outputtensor));

        for (float *output : outputs) {
            ans.clear();
            for (int i = 0; i < labels.size(); i++) {
                ans.emplace_back(make_pair(output[i], labels[i]));
            }
            sort(ans.begin(), ans.end());
            reverse(ans.begin(), ans.end());
            for (int i = 0; i < 5; i++) {
                auto point = trunc(ans[i].first * 100000) / 100000;
                if (tmp.size() < 5) {
                    tmp.emplace_back(std::pair<float, string>(point, ans[i].second));
                    std::cout << std::left << std::setw(9) << std::setfill('0') << std::showpoint << point << " - "
                              << ans[i].second << std::endl;

                } else {
                    if (tmp[i].first != point || tmp[i].second != ans[i].second) {
                        std::cout << i << ": " << std::left << std::setw(9) << std::setfill('0') << std::showpoint
                                  << point << " - " << ans[i].second << std::endl;
                    }
                }
            }
        }
    }else{
        std::vector<uint8_t *> outputs;
        outputs.push_back((uint8_t *)GetTensordata(outputtensor));

        for (uint8_t *output : outputs) {
            ans.clear();
            for (int i = 0; i < labels.size(); i++) {
                ans.emplace_back(make_pair(output[i], labels[i]));
            }
            sort(ans.begin(), ans.end());
            reverse(ans.begin(), ans.end());
            for (int i = 0; i < 5; i++) {
                auto point = trunc(ans[i].first * 100000) / 100000;
                if (tmp.size() < 5) {
                    tmp.emplace_back(std::pair<float, string>(point, ans[i].second));
                    std::cout << std::left << std::setw(9) << std::setfill('0') << std::showpoint << point << " - "
                              << ans[i].second << std::endl;

                } else {
                    if (tmp[i].first != point || tmp[i].second != ans[i].second) {
                        std::cout << i << ": " << std::left << std::setw(9) << std::setfill('0') << std::showpoint
                                  << point << " - " << ans[i].second << std::endl;
                    }
                }
            }
        }
    }
    auto ed = std::chrono::system_clock::now();
    std::cout<<std::endl;
    std::cout<<"Spend total time "<<GetSpan(st,ed)<<std::endl;
    std::cout<<"  Read img spend time "<<GetSpan(st,st1)<<std::endl<<"  set img spend time "<<GetSpan(st1,st2)<<std::endl<<"  forward spend time "<<GetSpan(st2,st3)<<std::endl<<"  PostProcess spend time "<<GetSpan(st3,ed)<<std::endl;
    
    

    
    std::cout<<std::endl;
    std::cout<<std::endl;
    return 0;
}