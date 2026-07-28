// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace webgpu {

// Per-workgroup output tiling for one MatMul problem: the tile shape and split-K
// factor chosen by a vendor-specific policy. The subgroup-matrix shape itself is
// separate from this selection.
//
// This lives in its own header (rather than in subgroup_matrix_matmul.h) so that
// consumers which only need the tiling type -- notably the offline autotuner --
// do not have to pull in the WGSL shader/program machinery and its generated
// template headers.
struct SubgroupMatrixTiling {
  uint32_t tile_m;   // output rows per workgroup
  uint32_t tile_n;   // output cols per workgroup
  uint32_t split_k;  // subgroups cooperating along K (1 = no split)
};

}  // namespace webgpu
}  // namespace onnxruntime
