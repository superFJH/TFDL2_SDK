#include "httplib.h"
#include "TFDL2_C_API.h"
#include "TFCV/TFCV.h"
#include <mutex>
#include <list>
#include <vector>
#include <string>
#include <unistd.h>
#include <iostream>
#define BUFSIZE 16
using namespace std;
using namespace TFDL_CAPI;
int main(int argc, char **argv)
{
    string rtspaddr = argv[1]; //输入从簇可以访问的rtsp的地址
    int port = atoi(argv[2]);
    // 使用 RTSP_IMPL_FFMPEG FFMPEG内核的RTSP Server
    auto video = TFCV::NewVideoReader(0,"/dev/mv500");
    assert(TFCV::OpenURL(video,rtspaddr)==true);
    httplib::Server svr;
    char buf[800];
    getcwd(buf, 800);
    string basepath = buf;
    std::cout << basepath << std::endl;
    svr.set_base_dir(basepath, "/");
    svr.Get("/Img", [&](const httplib::Request &req, httplib::Response &res)
            {
                if (TFCV::ReadFrame(video) == 1)
                {
                    vector<uint8_t> buf;
                    TFCV::Compress(video,".jpg",buf);//目前只支持jpg压缩

                    string result(reinterpret_cast<char*>(buf.data()),buf.size());
                    
                    result = "data:image/jpeg;base64," + httplib::detail::base64_encode(result);
                    res.set_content(result, "text/plain");
                }
                else
                {
                    res.set_content("No", "text/plain");
                }
            });

    if (!svr.listen("0.0.0.0", port))
    {
        std::cout << "Can't build server" << std::endl;
    }
    return 0;
}