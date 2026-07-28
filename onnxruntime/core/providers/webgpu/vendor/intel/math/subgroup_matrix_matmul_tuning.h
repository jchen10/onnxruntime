// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Deliberately only the tiling type, not subgroup_matrix_matmul.h: this header is
// included by test code that must not depend on the generated WGSL template headers.
#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Hooks used by the offline autotuner (WebGpuSgMatMulTuning) to drive the Intel
// subgroup-matrix MatMul selector from the outside. They are process-global and
// meant to be set from a single thread while no other inference is running; the
// steady-state cost in a normal (untuned) process is one relaxed atomic load per
// MatMul dispatch.

// Forces every subsequent subgroup-matrix MatMul to use `tiling`, bypassing the
// pretuned table and the heuristic. The tiling is still validated against the
// problem's K; an invalid one falls through to the normal selection. Pass
// std::nullopt to restore normal selection.
void SetSgMatMulTilingOverride(std::optional<SubgroupMatrixTiling> tiling);

// When disabled, the Intel selector declines every problem so MatMul falls back
// to the generic WebGPU path. Used to measure the non-subgroup-matrix baseline.
void SetSgMatMulDisabled(bool disabled);

// The device architecture string (e.g. "xe-3lpg") captured the first time the
// selector ran. Empty until at least one MatMul has been dispatched.
std::string GetSgMatMulDeviceArch();

// All tile + split-K configurations the selector could legally produce for this
// problem, i.e. the search space the autotuner benchmarks. Empty when the
// problem is not eligible for the subgroup-matrix kernel at all (e.g. K is not a
// multiple of the subgroup-matrix K). Tiles larger than the dimension they cover
// are skipped: they produce the same tile grid as the next smaller candidate
// while wasting lanes, and the selector never picks them either.
std::vector<SubgroupMatrixTiling> EnumerateSubgroupMatrixTilings(uint32_t M, uint32_t N, uint32_t K);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
