
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HistogramTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint64_t, bins); // 份数
  TILING_DATA_FIELD_DEF(float, minNum); // 小
  TILING_DATA_FIELD_DEF(float, maxNum); // 大
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Histogram, HistogramTilingData)
}
