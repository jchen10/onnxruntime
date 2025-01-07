// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/gatherbase.h"

namespace onnxruntime {
namespace webgpu {

class GatherProgram final : public Program<GatherProgram> {
 public:
  GatherProgram(const uint32_t axis) : Program{"Gather"}, axis_{axis} {
    // Overide the uniform variables of ProgramMetadata in the base class.
    auto& uniforms = UniformVariables();
    uniforms = {uniform_overrides.data(), uniform_overrides.size()};
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  void SetUniformOverrideDataType(size_t index, ProgramUniformVariableDataType data_type) {
    uniform_overrides[index].data_type = data_type;
  }

  static constexpr ProgramUniformVariableDefinition kDataSizeUniform = {"data_size",
                                                                        // The type may be unknown at this point.
                                                                        ProgramUniformVariableDataType::Float32};

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(kDataSizeUniform);

 private:
  uint32_t axis_;
  std::array<ProgramUniformVariableDefinition, 1> uniform_overrides{kDataSizeUniform};
};

class Gather final : public WebGpuKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : WebGpuKernel(info), GatherBase(info) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
