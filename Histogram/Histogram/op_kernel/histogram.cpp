#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T> class KernelHistogram {
public:
    __aicore__ inline KernelHistogram() {}
    __aicore__ inline void Init(GM_ADDR x,  GM_ADDR y, uint64_t totalSize,
                                uint64_t bins, float minNum, float maxNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalSize = totalSize;
        this->bins = bins;
        this->minNum = (T)minNum;
        this->maxNum = (T)maxNum;

        Gm_x.SetGlobalBuffer((__gm__ T*)x, totalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, bins);
    }

    __aicore__ inline void Process()
    {
        // 先给y初始化为0
        float ling = 0.0;
        for(uint64_t i=0; i<bins; i++){
            Gm_y.SetValue(i, (T)(ling));
        }

        uint64_t flag = 0;
        if((float)this->minNum == 0 && (float)this->maxNum == 0){
            flag = 1;

            if constexpr (std::is_same_v<T, half>){
                this->minNum = 4096.0;
                this->maxNum = -4096.0;
            }else if constexpr (std::is_same_v<T, float>){
                this->minNum = 1125899900.0;
                this->maxNum = -1125899900.0;
            }else{
                this->minNum = 2147483647;
                this->maxNum = -2147483647;
            }
            for(uint64_t i=0; i < totalSize; i++){
                if((float)Gm_x.GetValue(i) > (float)this->maxNum){
                    this->maxNum = Gm_x.GetValue(i);
                }
                if((float)Gm_x.GetValue(i) < (float)this->minNum){
                    this->minNum = Gm_x.GetValue(i);
                }
            }
        }

        float one = 1.0;
        for(uint64_t i=0; i<totalSize; i++){
            if((float)Gm_x.GetValue(i) >= (float)minNum && (float)Gm_x.GetValue(i) < (float)maxNum){
                Gm_y.SetValue((int)(((float)Gm_x.GetValue(i)-(float)minNum) / (((float)maxNum-(float)minNum)/bins)),
                (T)((float)Gm_y.GetValue((int)(((float)Gm_x.GetValue(i)-(float)minNum) / (((float)maxNum-(float)minNum)/bins)))+one));
            }else if((float)Gm_x.GetValue(i) == (float)maxNum){
                Gm_y.SetValue(bins - 1, (T)((float)Gm_y.GetValue(bins - 1)+one));
            }
        }

    }

private:
    GlobalTensor<T> Gm_x, Gm_y;
    uint64_t totalSize;
    uint64_t bins;
    T minNum, maxNum;
};

extern "C" __global__ __aicore__ void histogram(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHistogram<DTYPE_X> op;
    op.Init(x, y, tiling_data.totalSize, tiling_data.bins, tiling_data.minNum, tiling_data.maxNum);
    op.Process();
}
