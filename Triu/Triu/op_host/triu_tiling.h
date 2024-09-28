
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TriuTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint64_t, batchSize);
  TILING_DATA_FIELD_DEF(uint64_t, diagonal);
  TILING_DATA_FIELD_DEF(uint64_t, dimLast);
  TILING_DATA_FIELD_DEF(uint64_t, dimFirst);
  TILING_DATA_FIELD_DEF(uint64_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint64_t, block_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Triu, TriuTilingData)
}
