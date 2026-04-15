//
// Created by test on 2019/8/8.
//

#ifndef NPU40T_TFLOG_H
#define NPU40T_TFLOG_H
#include <string>
    class TLOG {
    public:
        TLOG() {};
        void Setlevel(int l){
            level = l;
        }
        int Getlevel() const{
            return level;
        }
        virtual void TFLOG(std::string err) {
            if(level<2)return;
            struct tm *ptr;
            time_t lt;
            lt = time(NULL);
            ptr = localtime(&lt);
            std::string tim = asctime(ptr);
            tim[tim.size() - 1] = ' ';
            printf("<Log>[%s]: %s\n", tim.c_str(), err.c_str());
        }
        virtual void Warning(std::string err) {
            if(level<1)return;
            struct tm *ptr;
            time_t lt;
            lt = time(NULL);
            ptr = localtime(&lt);
            std::string tim = asctime(ptr);
            tim[tim.size() - 1] = ' ';
            printf("<Warning>[%s]: %s\n", tim.c_str(), err.c_str());
        }
        virtual void Fatal(std::string err) {
            if(level<0)return;
            struct tm *ptr;
            time_t lt;
            lt = time(NULL);
            ptr = localtime(&lt);
            std::string tim = asctime(ptr);
            tim[tim.size() - 1] = ' ';
            printf("<Fatal>[%s]: %s\n", tim.c_str(), err.c_str());
        }

    protected:
        int level = 2;
    };
#endif //NPU40T_TFLOG_H
