#include <iostream>
#include "TFDL2_C_API.h"
#include <unistd.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <assert.h>
#include <cstring>
#include <cmath>
#include "TFCV/TFCV.h"
#include<mutex>
#include <algorithm>
using namespace TFDL_CAPI;
struct Rect2f{
    float x,y,width,height;
    Rect2f(float x,float y,float w,float h):x(x),y(y),width(w),height(h){
        
    }
    Rect2f(){

    }
};
typedef struct BoxInfo {
        Rect2f bbox;
        float score;
        int label;
        string labelname;
        vector<string> attributes;
        int Id = -1;//tracking id
        BoxInfo() {

        }

        BoxInfo(float score, int label, Rect2f bbox) : score(score), label(label), bbox(bbox) {

        }
    } BoxInfo;

static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};
inline void applyNms(std::vector<BoxInfo> &input_boxes, float nms_threshold,bool agnostic=true){
        if(!agnostic) {
            std::map<int, int> catmap;
            std::vector<std::vector<BoxInfo>> catBoxes;
            for (auto &bb: input_boxes) {
                if (catmap.find(bb.label) == catmap.end()) {
                    catmap[bb.label] = catBoxes.size();
                    catBoxes.emplace_back();
                }
                catBoxes[catmap[bb.label]].emplace_back(bb);
            }
            std::vector<BoxInfo> newboxes;
            for(auto& boxes : catBoxes){
                applyNms(boxes,nms_threshold,true);
                newboxes.insert(newboxes.end(),boxes.begin(),boxes.end());
            }
            input_boxes.swap(newboxes);


        }else {
            std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
            std::vector<float> vArea(input_boxes.size());
            for (int i = 0; i < int(input_boxes.size()); ++i) {
                vArea[i] = (input_boxes.at(i).bbox.width)
                           * (input_boxes.at(i).bbox.height);
            }
            for (int i = 0; i < int(input_boxes.size()); ++i) {
                for (int j = i + 1; j < int(input_boxes.size());) {
                    float xx1 = (std::max)(input_boxes[i].bbox.x, input_boxes[j].bbox.x);
                    float yy1 = (std::max)(input_boxes[i].bbox.y, input_boxes[j].bbox.y);
                    float xx2 = (std::min)(input_boxes[i].bbox.x + input_boxes[i].bbox.width - 1,
                                           input_boxes[j].bbox.x + input_boxes[j].bbox.width - 1);
                    float yy2 = (std::min)(input_boxes[i].bbox.y + input_boxes[i].bbox.height - 1,
                                           input_boxes[j].bbox.y + input_boxes[j].bbox.height - 1);
                    float w = (std::max)(float(0), xx2 - xx1 + 1);
                    float h = (std::max)(float(0), yy2 - yy1 + 1);
                    float inter = w * h;
                    float ovr = inter / (vArea[i] + vArea[j] - inter);
                    if (ovr >= nms_threshold) {
                        input_boxes.erase(input_boxes.begin() + j);
                        vArea.erase(vArea.begin() + j);
                    } else {
                        j++;
                    }
                }
            }
        }
    }
vector<BoxInfo> yolov5Detect(TFTensor &output,int netHeight,int netWidth,float score_threshold,float nms_threshold,int class_num,bool class_agnostic=true) {
            vector <BoxInfo > boxes;
            
                int boxsize = GetTensorDim(output,2);
                int results = GetTensorDim(output,1);
            assert(results == 4+class_num);
                float* ptr = (float*)GetTensordata(output);
                for(int idx=0;idx<boxsize;idx++) {
                    float maxValue = 0.0;
                    float x = ptr[(0)*boxsize + idx];
                    float y = ptr[(1)*boxsize + idx];
                    float ww = ptr[(2)*boxsize + idx];
                    float hh = ptr[(3)*boxsize + idx];
                    int cls = 0;
                    for (int i = 0; i < class_num; i++) {
                        float dd = ptr[(i+4)*boxsize + idx];
                        if(dd > maxValue){
                            maxValue = dd;
                            cls = i;
                        }
                    }

                    if (maxValue >= score_threshold) {
                            boxes.emplace_back(BoxInfo(maxValue, cls, Rect2f((x - ww / 2), (y - hh / 2), ww, hh)));
                    }
                    
                }
            vector<BoxInfo> outcome;
            applyNms(boxes, nms_threshold,class_agnostic);
            //LOG(INFO)<<this->threadname<<" detect "<<boxes.size();
            for(auto& box: boxes) {
                outcome.emplace_back(box.score,box.label,Rect2f((box.bbox.x)/netWidth,(box.bbox.y)/netHeight,box.bbox.width/netWidth,box.bbox.height/netHeight));
            
            }

            return outcome;
            //cout<<"detect :"<<this->outcome.size()<<endl;
        }

vector<string> label_name =  {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",  "fire hydrant",  "stop sign",  "parking meter",  "bench",  "bird",  "cat",  "dog",  "horse",  "sheep",  "cow",  "elephant",  "bear",  "zebra",  "giraffe",  "backpack",  "umbrella",  "handbag",  "tie",  "suitcase",  "frisbee",  "skis",  "snowboard",  "sports ball",  "kite",  "baseball bat",  "baseball glove",  "skateboard",  "surfboard",  "tennis racket",  "bottle",  "wine glass",  "cup",  "fork",  "knife",  "spoon",  "bowl",  "banana",  "apple",  "sandwich",  "orange",  "broccoli",  "carrot",  "hot dog",  "pizza",  "donut",  "cake",  "chair",  "couch",  "potted plant",  "bed",  "dining table",  "toilet",  "tv",  "laptop",  "mouse",  "remote",  "keyboard",  "cell phone",  "microwave",  "oven",  "toaster",  "sink",  "refrigerator",  "book",  "clock",  "vase",  "scissors",  "teddy bear",  "hair drier",  "toothbrush"};
int main(int args,char** argv){

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

    int height = GetTensorDim(images,2);

    int width = GetTensorDim(images,3);

    auto boxes = yolov5Detect(output,height,width,0.45,0.2,label_name.size());

    auto ed = std::chrono::system_clock::now();

    std::cout<<"Spend total time "<<GetSpan(st,ed)<<std::endl;
    std::cout<<"  Read img spend time "<<GetSpan(st,st1)<<std::endl<<"  set img spend time "<<GetSpan(st1,st2)<<std::endl<<"  forward spend time "<<GetSpan(st2,st3)<<std::endl<<"  PostProcess spend time "<<GetSpan(st3,ed)<<std::endl;
    std::cout<<"Total detect "<<boxes.size()<<" targets:"<<std::endl;
    for(auto& box : boxes){
        std::cout<<"  "<<label_name[box.label]<<"  "<<box.score<<"  "<<box.bbox.x<<"  "<<box.bbox.y<<"  "<<box.bbox.width<<"  "<<box.bbox.height<<std::endl;
    }
    std::cout<<std::endl;
    

    
    std::cout<<std::endl;
    std::cout<<std::endl;






    return 0;
}
