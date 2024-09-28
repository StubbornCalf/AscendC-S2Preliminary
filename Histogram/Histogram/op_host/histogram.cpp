
#include "histogram_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    HistogramTilingData tiling;
    uint64_t totalSize = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); // 获取Shape中所有dim的累乘结果

    uint32_t sizeofdatatype;  //输入的类型
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
    }else{
        sizeofdatatype = 4;
    }
    
    tiling.set_totalSize(totalSize);  // 设置totalSize 三个向量中最大的N*C*H*W
    auto bins = context->GetAttrs()->GetInt(0);
    tiling.set_bins(*bins);
    auto minNum = context->GetAttrs()->GetFloat(1);
    tiling.set_minNum(*minNum);
    auto maxNum = context->GetAttrs()->GetFloat(2);
    tiling.set_maxNum(*maxNum);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto ptr = context->GetAttrs()->GetInt(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->AppendDim(*ptr);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Histogram : public OpDef {
public:
    explicit Histogram(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("bins").AttrType(OPTIONAL).Int(100);
        this->Attr("min").AttrType(OPTIONAL).Float(0.0);
        this->Attr("max").AttrType(OPTIONAL).Float(0.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(Histogram);
}
