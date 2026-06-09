// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

constexpr std::string_view ComponentTypeName[] = {"unknown", "f32", "f16", "u32", "i32"};
template <std::size_t N>
constexpr bool ValidateComponentTypeName(const std::array<wgpu::SubgroupMatrixComponentType, N>& component_type) {
  bool matched = true;
  for (auto type : component_type) {
    switch (type) {
      case wgpu::SubgroupMatrixComponentType::F32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F32)] == "f32";
        break;
      case wgpu::SubgroupMatrixComponentType::F16:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F16)] == "f16";
        break;
      case wgpu::SubgroupMatrixComponentType::U32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::U32)] == "u32";
        break;
      case wgpu::SubgroupMatrixComponentType::I32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::I32)] == "i32";
        break;
      default:
        return false;
    }

    if (!matched) {
      return matched;
    }
  }

  return matched;
}
static_assert(ValidateComponentTypeName<4>({wgpu::SubgroupMatrixComponentType::F32,
                                            wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::U32,
                                            wgpu::SubgroupMatrixComponentType::I32}),
              "The elements' sequence of ComponentTypeName array do not match wgpu::SubgroupMatrixComponentType");

// Vendor-agnostic subgroup matrix config: {componentType, resultComponentType, M, N, K, subgroupMinSize, subgroupMaxSize, needsPrepack}
// Any GPU reporting a matching config from wgpu::AdapterPropertiesSubgroupMatrixConfigs is supported.
struct SupportedSubgroupMatrixConfig {
  wgpu::SubgroupMatrixComponentType componentType;
  wgpu::SubgroupMatrixComponentType resultComponentType;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t subgroupMinSize;
  uint32_t subgroupMaxSize;
  bool needsPrepack;  // Whether input A needs layout optimization for subgroupMatrixLoad
};

static const SupportedSubgroupMatrixConfig supported_subgroup_matrix_configs[] = {
    // 16x16x16 config with 128x128 tiles (NVIDIA Blackwell, subgroup size 32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 16, 16, 16, 32, 32, true},
    // 8x16x16 config (Intel Xe2/Xe3, subgroup size 16-32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 16, 16, 16, 32, true},
    // 8x8x8 config (Apple M-series, etc.)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 8, 8, 32, 32, false},
    {wgpu::SubgroupMatrixComponentType::F32, wgpu::SubgroupMatrixComponentType::F32, 8, 8, 8, 32, 32, false},
};

bool IsSubgroupMatrixConfigSupported(onnxruntime::webgpu::ComputeContext& context, bool is_fp16, int32_t& config_index) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();
  int32_t index = 0;
  for (const auto& supported_config : supported_subgroup_matrix_configs) {
    // F16 configs require FP16 output; skip them when output is F32.
    // F32 configs require FP32 output; skip them when output is FP16.
    if ((supported_config.componentType == wgpu::SubgroupMatrixComponentType::F16 && !is_fp16) ||
        (supported_config.componentType == wgpu::SubgroupMatrixComponentType::F32 && is_fp16)) {
      index++;
      continue;
    }
    for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
      const auto& device_config = subgroup_matrix_configs.configs[i];
      if (device_config.componentType == supported_config.componentType &&
          device_config.resultComponentType == supported_config.resultComponentType &&
          device_config.M == supported_config.M &&
          device_config.N == supported_config.N &&
          device_config.K == supported_config.K &&
          adapter_info.subgroupMinSize == supported_config.subgroupMinSize &&
          adapter_info.subgroupMaxSize == supported_config.subgroupMaxSize) {
        config_index = index;
        return true;
      }
    }
    index++;
  }
  return false;
}

// Sentinel config_index used to select the Intel int8 DPAS path.
constexpr int32_t kI8ConfigIndex = -2;

// Re-quantization block size in the K dimension for the int8 DPAS path.
constexpr uint32_t kI8BlockSizeK = 128;

// Checks whether the adapter reports an int8 subgroup matrix config usable by the
// 8x16x16 i8 DPAS kernel (componentType I8, resultComponentType I32, 8x16x32 DPAS shape).
bool IsSubgroupMatrixI8ConfigSupported(onnxruntime::webgpu::ComputeContext& context) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();
  for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
    const auto& device_config = subgroup_matrix_configs.configs[i];
    if (device_config.componentType == wgpu::SubgroupMatrixComponentType::I8 &&
        device_config.resultComponentType == wgpu::SubgroupMatrixComponentType::I32 &&
        device_config.M == 8 && device_config.N == 16 && device_config.K == 32 &&
        adapter_info.subgroupMinSize == 16 && adapter_info.subgroupMaxSize == 32) {
      return true;
    }
  }
  return false;
}

// Quantize fp16 A to i8 and prepack into 32x32 tiles for the i8 DPAS path (SIMD16).
// full_k selects the whole-K variant (one scale per row spanning all of K, two-pass).
class QuantizePrepackI8Program final : public Program<QuantizePrepackI8Program> {
 public:
  explicit QuantizePrepackI8Program(bool full_k) : Program{"QuantizePrepackI8"}, full_k_(full_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

 private:
  bool full_k_;
};

Status QuantizePrepackI8Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& output_a = shader.AddOutput("output_a", ShaderUsage::UseUniform);
  const auto& scales_a = shader.AddOutput("scales_a", ShaderUsage::UseUniform);
  if (full_k_) {
    return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_quantize_prepack_i8_fullk.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(input_a, a),
                               WGSL_TEMPLATE_VARIABLE(output_a, output_a),
                               WGSL_TEMPLATE_VARIABLE(scales_a, scales_a));
  }
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_quantize_prepack_i8.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(input_a, a),
                             WGSL_TEMPLATE_VARIABLE(output_a, output_a),
                             WGSL_TEMPLATE_VARIABLE(scales_a, scales_a));
}

// Re-quantize 4-bit weights to i8 and prepack into 32x32 tiles for the i8 DPAS path (SIMD16).
// full_k selects the whole-K variant (one scale per row spanning all of K, two-pass).
class RequantPrepackI8Program final : public Program<RequantPrepackI8Program> {
 public:
  RequantPrepackI8Program(bool has_zero_points, bool has_weight_idx, bool has_weight_idx_indirect, bool full_k)
      : Program{"RequantPrepackI8"},
        has_zero_points_(has_zero_points),
        has_weight_idx_(has_weight_idx),
        has_weight_idx_indirect_(has_weight_idx_indirect),
        full_k_(full_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"orig_block_size", ProgramUniformVariableDataType::Uint32},
      {"orig_blocks_per_col", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_zero_points_;
  bool has_weight_idx_;
  bool has_weight_idx_indirect_;
  bool full_k_;
};

Status RequantPrepackI8Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform);
  const auto& sb = shader.AddInput("scales_b", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& out_b = shader.AddOutput("output_b", ShaderUsage::UseUniform);
  const auto& out_sb = shader.AddOutput("prepacked_scales_b", ShaderUsage::UseUniform);
  if (full_k_) {
    return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_requant_prepack_i8_fullk.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                               WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                               WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                               WGSL_TEMPLATE_PARAMETER(n_bits, 4),
                               WGSL_TEMPLATE_PARAMETER(output_type_i32, true),
                               WGSL_TEMPLATE_VARIABLE(input_b, b),
                               WGSL_TEMPLATE_VARIABLE(output_b, out_b),
                               WGSL_TEMPLATE_VARIABLE(prepacked_scales_b, out_sb),
                               WGSL_TEMPLATE_VARIABLE(scales_b, sb));
  }
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_requant_prepack_i8.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, 4),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, true),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output_b, out_b),
                             WGSL_TEMPLATE_VARIABLE(prepacked_scales_b, out_sb),
                             WGSL_TEMPLATE_VARIABLE(scales_b, sb));
}

Status SubgroupMatrixMatMulNBitsI8Program::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  const auto& scales_a = shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  const auto& scales_b = shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  if (full_k_) {
    // Whole-K matmul (64x32 output tile): no block_size_k param (single scale block spans all of K).
    return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x16x16_i8_fullk_64x32.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                               WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(scales_a, scales_a),
                               WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
  }
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x16x16_i8.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(block_size_k, block_size_k_),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_a, scales_a),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

// This program optimizes the layout of input matrix A(MxK) for SubgroupMatrixLoad, so that all elements of each
// subgroup matrix(mxk) are arranged continuously in memory.
// Take "M = 4, K = 4, m = 2, k = 2" as an example, the input matrix A is arranged in row-major order as follows:
// d00, d01, | d02, d03,
// d10, d11, | d12, d13,
// ---------------------
// d20, d21, | d22, d23,
// d30, d31, | d32, d33,
//
// The prepack program rearranges the input matrix A to be in the following order:
// d00, d01,
// d10, d11,
// ---------
// d02, d03,
// d12, d13,
// ---------
// d20, d21,
// d30, d31,
// ---------
// d22, d23,
// d32, d33,
class PrepackProgram final : public Program<PrepackProgram> {
 public:
  PrepackProgram(uint32_t m, uint32_t k) : Program{"SubgroupMatrixMatMulLayout"},
                                           m_(m),
                                           k_(k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t m_;
  uint32_t k_;
};

Status PrepackProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddOutput("output_a", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, k_),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, m_));
}

Status GenerateShaderCode16x16x16(ShaderHelper& shader,
                                  const ShaderVariableHelper& b,
                                  const ShaderVariableHelper& scales_b,
                                  const ShaderVariableHelper& output,
                                  uint32_t nbits, int32_t config_index, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  const auto& config = supported_subgroup_matrix_configs[config_index];
  // Use 128x128 tile shader for 16x16x16 config (index 0)
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status GenerateShaderCode8x16x16(ShaderHelper& shader,
                                 const ShaderVariableHelper& b,
                                 const ShaderVariableHelper& scales_b,
                                 const ShaderVariableHelper& output,
                                 uint32_t nbits, int32_t config_index, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  const auto& config = supported_subgroup_matrix_configs[config_index];
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status GenerateShaderCode8x8x8(ShaderHelper& shader, const ShaderVariableHelper& a, const ShaderVariableHelper& b,
                               const ShaderVariableHelper& scales_b,
                               const ShaderVariableHelper& output, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x8x8.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status SubgroupMatrixMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform);
  const auto& scales_b = shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.M == 8 && config.N == 8 && config.K == 8) {
    return GenerateShaderCode8x8x8(shader, a, b, scales_b, output, nbits_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else if (config.M == 8 && config.N == 16 && config.K == 16) {
    return GenerateShaderCode8x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else if (config.M == 16 && config.N == 16 && config.K == 16) {
    return GenerateShaderCode16x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                  "Unsupported subgroup matrix config dimensions.");
  }
}

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      const Tensor* zero_points, const Tensor* bias,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t nbits,
                                      uint32_t zero_blocks_per_col,
                                      int32_t config_index,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y,
                                      const uint32_t weight_index,
                                      const Tensor* weight_index_indirect) {
  // Intel int8 DPAS path (config_index == kI8ConfigIndex). A is dynamically quantized to
  // int8 and B is re-quantized from 4-bit to int8, both re-quantized with block_size_k=128
  // and prepacked into 32x32 tiles, then multiplied via hardware i8 DPAS.
  if (config_index == kI8ConfigIndex) {
    constexpr uint32_t kTileSize = 32;
    constexpr uint32_t kVec4Components = 4;
    constexpr uint32_t kU32Components = 4;
    constexpr uint32_t kWorkgroupSize = 32;
    // Whole-K mode (USE_SG_MAT_INT8_FULLK=1) collapses the re-quantization scale block to
    // the entire K: one scale per row, maximal scale/SLM-flush amortization, lower precision.
    // It runs the 64x32 output-tile matmul (consumes the 32x32 whole-K prepack buffers).
    const bool full_k = Env::Default().GetEnvironmentVar("USE_SG_MAT_INT8_FULLK") == "1";
    const uint32_t kBlockSizeK = full_k ? K : 128u;  // re-quantization block size in K dimension
    constexpr uint32_t kPrepackWgSize = 16;   // SIMD16 for b128 (all lanes active)
    constexpr uint32_t kOrigBlockSize = 32;   // CanApply enforces 4-bit block_size == 32

    const uint32_t num_tiles_k = K / kTileSize;
    const uint32_t num_blocks_k = K / kBlockSizeK;
    const uint32_t orig_blocks_per_col = K / kOrigBlockSize;
    const bool has_zero_points = zero_points != nullptr;
    const bool has_bias = bias != nullptr;
    const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
    const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;

    // Step 1: Quantize A from fp16 to int8 (symmetric, block kBlockSizeK) and prepack into 32x32 tiles.
    // The whole-K 64x32 matmul tile spans two adjacent 32-row prepack tiles, so pad A to a multiple
    // of 64 (an even number of 32-row tiles); padded rows carry scale 0 and contribute nothing.
    const uint32_t kMPadTile = full_k ? 64u : kTileSize;
    const uint32_t M_padded = (M + kMPadTile - 1) / kMPadTile * kMPadTile;
    const uint32_t num_tiles_m = M_padded / kTileSize;

    QuantizePrepackI8Program quantize_prepack_program{full_k};
    quantize_prepack_program.SetWorkgroupSize(kPrepackWgSize);
    quantize_prepack_program.SetDispatchGroupSize(num_tiles_m, num_blocks_k, 1);

    // Prepacked A: tiles of 32x32 i8 stored contiguously as u32 (256 u32 per tile).
    TensorShape a_quant_shape{static_cast<int64_t>(num_tiles_m * num_tiles_k * 256)};
    Tensor a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
    // Prepacked scales_a: one f16 per row per block = 32 per block.
    TensorShape a_scale_shape{static_cast<int64_t>(num_tiles_m * num_blocks_k * kTileSize)};
    Tensor a_scale = context.CreateGPUTensor(a->DataType(), a_scale_shape);

    quantize_prepack_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
        .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, a_quant.Shape(), 1},
                     {&a_scale, ProgramTensorMetadataDependency::Rank, a_scale.Shape(), 1}})
        .AddUniformVariables({{M}, {K}});
    ORT_RETURN_IF_ERROR(context.RunProgram(quantize_prepack_program));

    // Step 2: Dequantize B from 4-bit to i8, re-quantize with kBlockSizeK, and prepack into 32x32 tiles.
    const uint32_t N_padded = (N + kTileSize - 1) / kTileSize * kTileSize;
    const uint32_t num_tiles_n = N_padded / kTileSize;

    RequantPrepackI8Program requant_prepack_program{has_zero_points, has_weight_idx, has_weight_idx_indirect, full_k};
    requant_prepack_program.SetWorkgroupSize(kPrepackWgSize);
    requant_prepack_program.SetDispatchGroupSize(num_tiles_n, num_blocks_k, 1);

    // Prepacked B: tiles of 32x32 i8 stored contiguously as i32 (256 i32 per tile).
    TensorShape b_quant_shape{static_cast<int64_t>(num_tiles_n * num_tiles_k * 256)};
    Tensor b_prepacked = context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), b_quant_shape);
    // Prepacked scales_b: one scale per N-row per block = 32 per block.
    TensorShape b_scale_shape{static_cast<int64_t>(num_tiles_n * num_blocks_k * kTileSize)};
    Tensor b_scale = context.CreateGPUTensor(scales->DataType(), b_scale_shape);

    requant_prepack_program.AddInputs({{b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kU32Components)},
                                       {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}});
    if (has_zero_points) {
      requant_prepack_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
    }
    if (has_weight_idx_indirect) {
      requant_prepack_program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    requant_prepack_program
        .AddOutputs({{&b_prepacked, ProgramTensorMetadataDependency::Rank, b_prepacked.Shape(), 1},
                     {&b_scale, ProgramTensorMetadataDependency::Rank, b_scale.Shape(), 1}})
        .AddUniformVariables({{N}, {K}, {zero_blocks_per_col}, {weight_index}, {kOrigBlockSize}, {orig_blocks_per_col}})
        .CacheHint("requant_prepack_i8", has_zero_points, has_weight_idx, has_weight_idx_indirect);
    ORT_RETURN_IF_ERROR(context.RunProgram(requant_prepack_program));

    // Step 3: Run i8 SubgroupMatrix matmul with both A and B prepacked.
    Tensor a_quant_i32(DataTypeImpl::GetType<int32_t>(), a_quant.Shape(),
                       a_quant.MutableDataRaw(), a_quant.Location());
    TensorShape y_shape{1, M, N};

    SubgroupMatrixMatMulNBitsI8Program mul_program{has_bias, has_weight_idx, has_weight_idx_indirect, kBlockSizeK, full_k};
    mul_program.SetWorkgroupSize(kWorkgroupSize);
    // Cap dispatch Y to target hardware occupancy. Each workgroup processes a
    // contiguous block of M-tiles sequentially, amortizing dispatch overhead for large M.
    uint32_t hw_subgroups = 0;
    if (context.AdapterInfo().architecture == std::string_view{"xe-3lpg"}) {
      hw_subgroups = 384;  // 12 XeCore x 32 subgroups
    } else if (context.AdapterInfo().architecture == std::string_view{"xe-2lpg"}) {
      hw_subgroups = 256;  // 8 XeCore x 32 subgroups
    }
    constexpr uint32_t kOccupancyFactor = 2;
    const uint32_t target_workgroups = hw_subgroups * kOccupancyFactor;
    // Matmul output tile is kTileSize-wide (32). M-tile height is 32 for the block-K default
    // kernel, or 64 for the whole-K 64x32 kernel.
    const uint32_t matmul_tile_n = kTileSize;
    const uint32_t matmul_tile_m = full_k ? 64u : kTileSize;
    const uint32_t num_m_tiles_matmul = (M + matmul_tile_m - 1) / matmul_tile_m;
    const uint32_t dispatch_x = (N + matmul_tile_n - 1) / matmul_tile_n;
    // Unknown architecture (hw_subgroups == 0): don't cap -- dispatch one workgroup per M-tile.
    const uint32_t dispatch_y = hw_subgroups == 0
                                    ? num_m_tiles_matmul
                                    : std::min(num_m_tiles_matmul, std::max(1u, target_workgroups / dispatch_x));
    const uint32_t tiles_per_wg = (num_m_tiles_matmul + dispatch_y - 1) / dispatch_y;
    mul_program.SetDispatchGroupSize(dispatch_x, dispatch_y, 1);
    // Inputs: input_a (prepacked i32), scales_a (prepacked), input_b (prepacked i32), scales_b (prepacked).
    mul_program.AddInputs({{&a_quant_i32, ProgramTensorMetadataDependency::TypeAndRank, 1},
                           {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, 1},
                           {&b_prepacked, ProgramTensorMetadataDependency::TypeAndRank, 1},
                           {&b_scale, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddUniformVariables({{M}, {N}, {K}, {weight_index}, {tiles_per_wg}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, 1})
        .CacheHint("i8", has_bias, has_weight_idx, has_weight_idx_indirect, full_k);
    if (has_bias) {
      mul_program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      mul_program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    return context.RunProgram(mul_program);
  }

  // Determine tile sizes first (needed for prepack padding).
  const auto& config = supported_subgroup_matrix_configs[config_index];
  uint32_t tile_size_a = 32;
  uint32_t tile_size_b = 64;
  uint32_t work_group_size = 128;
  if (config.M == 8 && config.N == 16 && config.K == 16) {
    // 8x16x16 config: 8 subgroups, 256 threads, 64x64 tiles
    tile_size_a = 64;
    work_group_size = 256;
  } else if (config.M == 16 && config.N == 16 && config.K == 16) {
    // 16x16x16 config: 4 subgroups, 128 threads, 128x128 tiles
    tile_size_a = 128;
    tile_size_b = 128;
    work_group_size = 128;
  }

  // If applicable, layout optimization of input matrix A(MxK) can be used for SubgroupMatrixLoad.
  Tensor a_prepack;
  if (config.needsPrepack) {
    const auto m = config.M;
    const auto k = config.K;

    // Optimize the layout of input matrix A(MxK) for SubgroupMatrixLoad.
    PrepackProgram prepack_program{m, k};
    constexpr uint32_t kSubgroupSize = 32;
    prepack_program.SetWorkgroupSize(kSubgroupSize);

    // Pad M to workgroup tile size so all subgroups read valid prepacked data.
    const uint32_t padded_M = ((M + tile_size_a - 1) / tile_size_a) * tile_size_a;
    const auto dispatch_group_size_x = padded_M / m;
    ORT_ENFORCE(K % k == 0, "K must be a multiple of ", k);
    const auto dispatch_group_size_y = K / k;
    // Each workgroup will process one subgroup matrix of size m x k.
    prepack_program.SetDispatchGroupSize(dispatch_group_size_x, dispatch_group_size_y, 1);

    TensorShape a_prepack_shape{padded_M, K};
    a_prepack = context.CreateGPUTensor(a->DataType(), a_prepack_shape);
    prepack_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutputs({{&a_prepack, ProgramTensorMetadataDependency::Rank, a_prepack.Shape(), 1}})
        .AddUniformVariables({{M}, {K}})
        .CacheHint(m, k);
    ORT_RETURN_IF_ERROR(context.RunProgram(prepack_program));
    a = &a_prepack;
  }

  constexpr uint32_t kU32Components = 4;
  TensorShape y_shape{1, M, N};
  const bool has_zero_points = zero_points != nullptr;
  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  SubgroupMatrixMatMulNBitsProgram mul_program{nbits, config_index, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect};
  mul_program.SetWorkgroupSize(work_group_size);
  uint32_t dispatch_x = (N + tile_size_b - 1) / tile_size_b;
  uint32_t num_m_tiles = (M + tile_size_a - 1) / tile_size_a;
  uint32_t dispatch_y = num_m_tiles;
  // For large M on Intel Xe, cap dispatch_y so each workgroup processes multiple
  // M-tiles sequentially, reducing scheduling overhead.
  if (M > 2048 && context.AdapterInfo().vendor == std::string_view{"intel"}) {
    // Each XeCore has 4 XVE x 8 SIMD-32 hardware threads = 32 subgroups.
    uint32_t hw_subgroups = 0;
    if (context.AdapterInfo().architecture == std::string_view{"xe-3lpg"}) {
      hw_subgroups = 384;  // 12 XeCore x 32
    } else if (context.AdapterInfo().architecture == std::string_view{"xe-2lpg"}) {
      hw_subgroups = 256;  // 8 XeCore x 32
    }
    if (hw_subgroups > 0) {
      constexpr uint32_t kOccupancyFactor = 16;  // empirically tuned on Xe2/Xe3 devices
      uint32_t target_wgs = hw_subgroups * kOccupancyFactor / (work_group_size / 32);
      dispatch_y = std::min(dispatch_y, (target_wgs + dispatch_x - 1) / dispatch_x);
    }
  }
  uint32_t m_tiles_per_wg = (num_m_tiles + dispatch_y - 1) / dispatch_y;
  mul_program.SetDispatchGroupSize(dispatch_x, dispatch_y, 1);
  mul_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(nbits == 4 ? kU32Components : 2 * kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{M}, {N}, {K}, {zero_blocks_per_col}, {weight_index}, {m_tiles_per_wg}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, 1})
      .CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect);
  if (has_zero_points) {
    mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (bias) {
    mul_program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    mul_program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(mul_program);
}

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       uint32_t nbits,
                                       bool is_fp16,
                                       int32_t& config_index,
                                       uint32_t M,
                                       bool has_weight_idx_indirect) {
  // Subgroup matrix kernels only support 4-bit/8-bit quantization.
  if (nbits != 4 && nbits != 8) {
    return false;
  }

  // Dispatch precondition: the subgroup-matrix kernel is reserved for the
  // tile-optimized M range without indirect weight indexing.
  if (!(M >= kMinMForTileOptimization && !has_weight_idx_indirect)) {
    return false;
  }

  bool has_subgroup_matrix = context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  if (has_subgroup_matrix) {
    // Check if the adapter reports a subgroup matrix config we support.
    has_subgroup_matrix = IsSubgroupMatrixConfigSupported(context, is_fp16, config_index);
    if (has_subgroup_matrix) {
      if (context.AdapterInfo().vendor == std::string_view{"apple"}) {
        // For now SubgroupMatrixMatMulNBits is only supported for accuracy level 4, because with Fp16 there are
        // some precision issues with subgroupMatrixMultiplyAccumulate. It is possible to support higher accuracy
        // by setting compute_precision to Fp32, but that will be slower. For 1K token prefill FP16 Phi 3.5 is around 5s,
        // FP32 is around 7s.
        has_subgroup_matrix = accuracy_level == 4;
      } else if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
        // Optionally switch to the int8 DPAS path (8x16x16 i8) on Intel. This dynamically
        // quantizes A to int8 and re-quantizes the 4-bit weights to int8.
        // USE_SG_MAT_INT8=1 uses a 128-wide re-quantization block; USE_SG_MAT_INT8_FULLK=1 uses one
        // scale block spanning the whole K (64x32 output tile). Either env var enables the i8 path.
        const bool want_i8 = Env::Default().GetEnvironmentVar("USE_SG_MAT_INT8") == "1" ||
                             Env::Default().GetEnvironmentVar("USE_SG_MAT_INT8_FULLK") == "1";
        if (want_i8 &&
            is_fp16 && accuracy_level == 4 && nbits == 4 &&
            block_size == 32 && K % kI8BlockSizeK == 0 &&
            IsSubgroupMatrixI8ConfigSupported(context)) {
          config_index = kI8ConfigIndex;
        }
      }
    }
  }

  // Int8 DPAS path has its own dispatch constraints (re-quantized to block_size_k=128).
  if (config_index == kI8ConfigIndex) {
    return batch_count == 1 &&
           K % kI8BlockSizeK == 0 &&
           N % 32 == 0;
  }

  return has_subgroup_matrix &&
         block_size == 32 &&
         batch_count == 1 &&
         K % 32 == 0 &&
         N % 64 == 0;
}
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

#endif
