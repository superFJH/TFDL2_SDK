#include <opencv2/opencv.hpp>
#include "TFDL2_C_API.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <assert.h>
#include <cstring>
#include <cmath>
#include "TFCV/TFCV.h"

using namespace TFDL_CAPI;
static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

// Color map for visualization (same as Python version)
const std::vector<std::vector<uint8_t>> mvColorMap = {
    {128, 64, 128},      //  0. road
    {244, 35, 232},      //  1. sidewalk
    {70, 70, 70},        //  2. building
    {102, 102, 156},     //  3. wall
    {190, 153, 153},     //  4. fence
    {153, 153, 153},     //  5. pole
    {250, 170, 30},      //  6. traffic light
    {220, 220, 0},       //  7. traffic sign
    {107, 142, 35},      //  8. vegetation
    {152, 251, 152},     //  9. terrain
    {70, 130, 180},      // 10. sky
    {220, 20, 60},       // 11. person
    {255, 0, 0},         // 12. rider
    {0, 0, 142},         // 13. car
    {0, 0, 70},          // 14. truck
    {0, 60, 100},        // 15. bus
    {0, 80, 100},        // 16. train
    {0, 0, 230},         // 17. motorcycle
    {119, 11, 32}        // 18. bicycle
};

// Preprocess function
cv::Mat preprocess(const cv::Mat& img, const cv::Size& target_size) {
    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    // Resize image
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, target_size);
    
    // Convert HWC to CHW and add batch dimension
    // In C++ OpenCV, image is stored as HWC (Height x Width x Channels)
    std::vector<cv::Mat> channels;
    cv::split(resized_img, channels);
    
    // Normalize if needed (commented out as in Python code)
    // for (auto& channel : channels) {
    //     channel.convertTo(channel, CV_32F, 1.0 / 255.0);
    // }
    
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F);
    
    // Create a 4D blob (batch, channel, height, width)
    cv::Mat blob;
    cv::dnn::blobFromImage(float_img, blob, 1.0, target_size, cv::Scalar(), true, false, CV_32F);
    
    return blob;
}

// Visualization function
cv::Mat visualize_segmentation_mask(const cv::Mat& mask, int num_classes, 
                                   const std::vector<std::vector<uint8_t>>& colormap = mvColorMap) {
    cv::Mat segmentation_map(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        cv::Mat mask_class = (mask == class_idx);
        segmentation_map.setTo(cv::Scalar(colormap[class_idx][0], 
                                         colormap[class_idx][1], 
                                         colormap[class_idx][2]), 
                                         mask_class);
    }
    
    return segmentation_map;
}

int num_classes = 19;

int main(int argc,char** argv) {

    srand(time(NULL));
    std::string protopath = argv[1];
    string config = argv[2];

    std::fstream file(config);
    string configstring((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    string err;



    auto tfContext = LoadProto(protopath);
    auto reader = TFCV::NewImgReader();
    


    auto tfExecutor = CompileExecutor(tfContext, true, configstring);
    //warm up
    ForwardExecutorAlone(tfExecutor);

    auto st = std::chrono::system_clock::now();

    auto images = GetInputTensors(tfExecutor)[0];

    //read img
    TFCV::OpenURL(reader,argv[3]);

    auto st1 = std::chrono::system_clock::now();
    //set img data into tensor (include resize and RGB->BGR)
    TFCV::DumpImgData(reader,images,0,TFDL_CAPI::TFCV::TFCV_RGB);

    auto st2 = std::chrono::system_clock::now();

    ForwardExecutorAlone(tfExecutor);

    auto st3 = std::chrono::system_clock::now();

    auto output = GetOutputTensors(tfExecutor)[0];

    
    
    // Process output
    float* output_data = (float*)GetTensordata(output);
    auto output_shape = GetTensorShape(output);
    
    // Assuming output shape is [1, H, W]
    int batch = output_shape[0];
    int height = output_shape[1];
    int width = output_shape[2];

    std::cout<<"output shape is "<<batch<<" "<<height<<" "<<width<<" "<<GetTensorType(output)<<std::endl;
    
    // Create mask by taking argmax across classes
    cv::Mat mask(height, width, CV_32SC1);
    

    memcpy(mask.data, output_data, height * width * sizeof(int));




    //如果没有argmax函数，可以手动实现，我这里网络输出是经过argmax的，所以直接使用
    /*
    // Assuming output shape is [1, num_classes, H, W]
    int num_classes = output_shape[1];
    int height = output_shape[2];
    int width = output_shape[3];
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int max_class = 0;
            float max_val = -FLT_MAX;
            for (int c = 0; c < num_classes; ++c) {
                float val = output_data[c * height * width + h * width + w];
                if (val > max_val) {
                    max_val = val;
                    max_class = c;
                }
            }
            mask.at<uint8_t>(h, w) = max_class;
        }
    }*/
    
    // Visualize the segmentation mask
    cv::Mat segmentation_map = visualize_segmentation_mask(mask, num_classes);

    auto ed = std::chrono::system_clock::now();
    
    // Display or save the result
    cv::imwrite("output.png", segmentation_map);

    std::cout<<"Spend total time "<<GetSpan(st,ed)<<std::endl;
    std::cout<<"  Read img spend time "<<GetSpan(st,st1)<<std::endl<<"  set img spend time "<<GetSpan(st1,st2)<<std::endl<<"  forward spend time "<<GetSpan(st2,st3)<<std::endl<<"  PostProcess spend time "<<GetSpan(st3,ed)<<std::endl;
    
    

    
    std::cout<<std::endl;
    std::cout<<std::endl;
    
    return 0;
}
