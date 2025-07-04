// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <filesystem>

#include "core/common/inlined_containers.h"
#include "core/graph/graph.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {
class DataTransferManager;
struct KernelCreateInfo;

class OptimizerExecutionFrame final : public IExecutionFrame {
 public:
  class Info {
   public:
    Info(const std::vector<const Node*>& nodes,
         const InitializedTensorSet& initialized_tensor_set,
         const std::filesystem::path& model_path,
         const IExecutionProvider& execution_provider,
         const std::function<bool(const std::string&)>& is_sparse_initializer_func,
         const logging::Logger& logger);

    Info(const std::vector<const Node*>& nodes,
         const std::unordered_map<std::string, OrtValue>& initialized_tensor_set,
         const std::filesystem::path& model_path,
         const IExecutionProvider& execution_provider,
         const std::function<bool(const std::string&)>& is_sparse_initializer_func,
         const logging::Logger& logger);

    ~Info() = default;

    const AllocatorPtr& GetAllocator() const {
      return allocator_ptr_;
    }

    const OrtValueNameIdxMap& GetMLValueNameIdxMap() const noexcept { return ort_value_name_idx_map_; }
    const std::unordered_map<int, const NodeArg*>& GetMLValueIdxNodeArgMap() const noexcept {
      return ort_value_idx_nodearg_map_;
    }
    const std::unordered_map<int, OrtValue>& GetInitializers() const noexcept { return initializers_; }
    const NodeIndexInfo& GetNodeIndexInfo() const { return *node_index_info_; }
    int GetMLValueIndex(const std::string& name) const {
      int index = -1;
      if (ort_value_name_idx_map_.GetIdx(name, index) == Status::OK()) {
        return index;
      }
      return -1;
    }

    std::unique_ptr<const OpKernel> CreateKernel(const Node* node, const ConfigOptions& config_options) const;

    // Check if an kernel create info can be found in the registry.
    Status TryFindKernel(const Node* node, const KernelCreateInfo** out) const;

    const DataTransferManager& GetDataTransferManager() const { return data_transfer_mgr_; }

    const std::function<bool(const std::string&)>& GetSparseInitializerLookupFunc() const {
      return is_sparse_initializer_func_;
    }

   private:
    AllocatorPtr allocator_ptr_;
    DataTransferManager data_transfer_mgr_;
    // MLValues for optimizer
    OrtValueNameIdxMap ort_value_name_idx_map_;
    std::unordered_map<int, const NodeArg*> ort_value_idx_nodearg_map_;
    std::unordered_map<int, OrtValue> initializers_;
    std::unique_ptr<NodeIndexInfo> node_index_info_;
    const IExecutionProvider& execution_provider_;
    const std::function<bool(const std::string&)>& is_sparse_initializer_func_;
    const logging::Logger& logger_;

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Info);
  };

  OptimizerExecutionFrame(const Info& info,
                          const std::vector<int>& fetch_mlvalue_idxs,
                          const std::vector<OrtValue>& fetches = {});

  ~OptimizerExecutionFrame() override = default;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerExecutionFrame);

  AllocatorPtr GetAllocatorImpl(const OrtDevice& info) const override;

  Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) override;

  Status CopyTensor(const Tensor& src, Tensor& dest) const override;

  const DataTransferManager& GetDataTransferManager() const override;

  const Info& info_;
};

}  // namespace onnxruntime
