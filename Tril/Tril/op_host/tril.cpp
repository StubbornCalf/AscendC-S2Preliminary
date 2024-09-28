
#include "tril_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    TrilTilingData tiling;
    uint64_t totalSize = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); // 获取Shape中所有dim的累乘结果
    const gert::StorageShape* shape = context->GetInputShape(0);                // 获取第1个input的shape
    int last = shape->GetStorageShape().GetDimNum() - 1;
    uint64_t dimLast = shape->GetStorageShape().GetDim(last);
    uint64_t dimFirst = shape->GetStorageShape().GetDim(last-1);
    uint64_t batchSize = totalSize / (shape->GetStorageShape().GetDim(last) * shape->GetStorageShape().GetDim(last-1));


    auto ptr = context->GetAttrs()->GetInt(0);
    tiling.set_diagonal(*ptr);
    tiling.set_totalSize(totalSize);  // 设置totalSize V*C*N*N
    tiling.set_batchSize(batchSize);  // 设置batchsize [V*C]
    tiling.set_dimLast(dimLast);        // 设置矩阵dim
    tiling.set_dimFirst(dimFirst);

    const uint64_t BLOCKSIZE = 32;  // 32字节对齐
    uint64_t sizeofdatatype;          // 输入数据类型所占的字节数
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {  // 估计这道题支持的数据类型也是bf16 fp16 fp32
        sizeofdatatype = 2;
    }
    else {                           // 如果是fp32则sizeofdatatype=4
        sizeofdatatype = 4;
    }
    uint64_t ALIGN_NUM = BLOCKSIZE / sizeofdatatype;
    tiling.set_ALIGN_NUM(ALIGN_NUM);

    // 因为instance norm是求WH均值，把WH看作最小的，将(N*C)分配给core
    constexpr int32_t NUM = 8;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // 获取硬件平台信息
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);   // 硬件平台存储空间的内存大小->ub_size
    uint32_t block_size = ((ub_size) / sizeofdatatype / 2) / 4; //每个tiling_size32位对齐，能分成多少个block

    // block_size已经被256整除了，squareNumber也被256整除，
    if(block_size > totalSize){   // dimLast*sizeof一定是256对齐的
        block_size = totalSize;
    }else if(block_size > dimLast){
        // block_size的设置要能够被packNumber整除，也就是向下取整squareNumber
        block_size = block_size / dimLast * dimLast;
    }

    tiling.set_block_size(block_size);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Tril : public OpDef {
public:
    explicit Tril(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Tril);
}
