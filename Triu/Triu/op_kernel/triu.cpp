#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr float zerof = 0.0;
constexpr float onef = 1.0;
constexpr float tmpf = 5.0;

template<typename T> class KernelTriu {  // 模板类
public:
    __aicore__ inline KernelTriu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalSize,
                                uint64_t batchSize,  uint64_t diagonal,
                                uint64_t dimLast, uint64_t dimFirst) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->batchSize = batchSize;
        this->totalSize = totalSize;
        this->diagonal = diagonal;
        this->squareSize = totalSize / batchSize;
        this->dimLast = dimLast;
        this->dimFirst = dimFirst;

        Gm_x.SetGlobalBuffer((__gm__ T*)x, totalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, totalSize);

        float ling = 0.0;
        for(uint64_t i=0; i<totalSize; i++){
            Gm_y.SetValue(i, (T)(Gm_x.GetValue(i)));
        }
    }
    __aicore__ inline void Process() {
        for(uint64_t j=0; j<batchSize; j++){
            uint64_t oneint = 1, zeroint = 0;
            float onetmp = 1.0, ling=0.0;;
            int64_t length = dimLast - diagonal;
            for(uint64_t i=0; i<dimFirst; i++){
                if(length < dimLast && length >= 0){
                    for(uint64_t k=0; k<dimLast-length; k++){
                        Gm_y.SetValue(j*squareSize + i*dimLast + k, (T)(ling));
                    }
                }else if(length < 0){
                    for(uint64_t k=0; k<dimLast; k++){
                        Gm_y.SetValue(j*squareSize + i*dimLast + k, (T)(ling));
                    }
                }
                length--;
            }
        }

    }


private:
    TPipe pipe;
    GlobalTensor<T> Gm_x, Gm_y;
    uint64_t batchSize, totalSize, diagonal, squareSize, dimLast, dimFirst;
};

template<typename T> class KernelTriuFast {  // 模板类
public:
    __aicore__ inline KernelTriuFast() {}
    // blockSize是tiling所能处理的最大元素个数，已经256对齐了
    // totalSize是给这个核上要处理的元素数量，因为就以一个核，就等于totalSize输入

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalSize,
                                uint64_t batchSize, uint64_t diagonal,
                                uint64_t dimLast, uint64_t dimFirst,
                                uint64_t ALIGN_NUM, uint64_t block_size) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->batchSize = batchSize;
        this->squareSize = totalSize / batchSize;
        this->totalSize = totalSize;
        this->diagonal = diagonal;
        this->dimLast = dimLast;
        this->dimFirst = dimFirst;
        this->tileLength = block_size;
        // 当前核所需要的tile数量(不能被整除tile数就+1)  (dimFirst*batchSize) / (this->tileLength/dimLast)=
        this->tileNum = this->totalSize / this->tileLength + (this->totalSize % this->tileLength > 0);

        Gm_x.SetGlobalBuffer((__gm__ T*)x, this->totalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, this->totalSize);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(T));      // Q_x队列，buffer=2, 每个buffer长度是tileLength*sizeof(T)
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);  // 第几个tile,tile的元素数
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        // 这时候的tileLength能被squareSize整除，length时还剩下的square数量，也一定可以被256整除
        auto length = this->totalSize - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = Q_x.AllocTensor<T>();
        DataCopy(xLocal, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = Q_x.DeQue<T>();
        LocalTensor<T> yLocal = Q_y.AllocTensor<T>();

        int onelength;
        for (int j = 0; j < length/dimLast; ++j) {     // packNumber中每个H*W
            onelength = diagonal+((progress*tileLength/dimLast+j)%dimFirst);
            if(onelength < dimLast && onelength > 0){
                Muls(yLocal[j*dimLast], xLocal[j*dimLast], (T)onef, dimLast);
                Muls(yLocal[j*dimLast], xLocal[j*dimLast], (T)zerof, onelength);
            }else if (onelength >= dimLast){
                Muls(yLocal[j*dimLast], xLocal[j*dimLast], (T)zerof, dimLast);
            }else if(onelength <= 0){
                Muls(yLocal[j*dimLast], xLocal[j*dimLast], (T)onef, dimLast);
            }
        }

        Q_x.FreeTensor(xLocal);
        Q_y.EnQue<T>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<T> yLocal = Q_y.DeQue<T>();
        DataCopy(Gm_y[progress * this->tileLength], yLocal, length);
        Q_y.FreeTensor(yLocal);
    }


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    GlobalTensor<T> Gm_x, Gm_y;
    uint64_t totalSize, diagonal, squareSize, dimLast, dimFirst, batchSize, tileLength, tileNum;
};


extern "C" __global__ __aicore__ void triu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(tiling_data.dimLast * sizeof(DTYPE_X) % 256 !=0 || tiling_data.dimFirst > tiling_data.block_size){
        KernelTriu<DTYPE_X> op;
        op.Init(x, y, tiling_data.totalSize, tiling_data.batchSize,
            tiling_data.diagonal, tiling_data.dimLast, tiling_data.dimFirst);
        op.Process();
    }else{
        KernelTriuFast<DTYPE_X> op;
        op.Init(x, y, tiling_data.totalSize, tiling_data.batchSize,
            tiling_data.diagonal, tiling_data.dimLast, tiling_data.dimFirst,
            tiling_data.ALIGN_NUM, tiling_data.block_size);
        op.Process();
    }
}
