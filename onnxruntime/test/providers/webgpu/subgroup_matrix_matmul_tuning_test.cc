// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Offline autotuner for the Intel subgroup-matrix MatMul kernel.
//
// This is NOT a correctness test. It sweeps a grid of (batch, M, N, K) problems
// and, for each problem, benchmarks every valid tile + split-K config (enumerated
// by EnumerateSubgroupMatrixTilings) on the real device, emitting one CSV row per
// config plus the generic-MatMul baseline. A companion
// Python script (tools/python/gen_sgmm_tree.py) trains a cost model on those rows
// and distills it into the selector's lookup table / decision tree.
//
// Measurement protocol
//   * One session per problem, not per config: the tiling override is consulted
//     on every Run and the pipeline cache is keyed by the tile params, so configs
//     are switched between Runs. This pays the (large) weight upload once.
//   * Inputs are bound through IOBinding so they are uploaded once and reused.
//   * The MatMul output stays on the GPU; a 1-element Slice of it is fetched to
//     the host each Run, which drains the queue (bounding submission backlog and
//     making wall time meaningful) without downloading up to hundreds of MB.
//   * GPU time comes from the profiler's timestamp events, filtered to the
//     SubgroupMatrixMatMul program so the probe Slice is excluded.
//   * Per config: a warmup absorbs the one-time pipeline compilation, a short
//     probe estimates per-run cost, then the config is timed with a run count
//     sized to ~kTargetTimedUs of work so fast configs get more runs (tighter
//     medians) and slow ones fewer (bounded wall time).
//   * Config order is shuffled per problem so slow thermal drift does not
//     systematically favour whichever config is measured first.
//
// Sizing limits (why the sweep is budgeted rather than a dense grid): the kernel
// indexes A/B/Y with u32 flat offsets, and a tensor larger than
// maxStorageBufferBindingSize gets segmented into multiple bindings, which the
// subgroupMatrixLoad path cannot consume. Problems are therefore filtered by a
// per-tensor cap and a total per-problem GPU-memory budget.
//
// It only runs when the ORT_WEBGPU_SGMM_TUNE environment variable is set, since
// it is long-running and only produces useful data on Intel Xe2/Xe3 hardware
// that exposes the 8x16x16 F16 subgroup-matrix config.
//
// Environment:
//   ORT_WEBGPU_SGMM_TUNE            1 to run; 2 to also log every candidate.
//   ORT_WEBGPU_SGMM_TUNE_OUT        output CSV path (default sgmm_tuning.csv).
//   ORT_WEBGPU_SGMM_TUNE_BUDGET_MB  total A+W+Y budget per problem (default 1024).
//   ORT_WEBGPU_SGMM_TUNE_TENSOR_MB  per-tensor cap; keep <= the device's
//                                   maxStorageBufferBindingSize (default 512).
//   ORT_WEBGPU_SGMM_TUNE_SHARD      "i/n" to run only shard i of n.
//   ORT_WEBGPU_SGMM_TUNE_RESUME     1 to append to an existing CSV and skip the
//                                   problems it already contains.
//
// Example:
//   set ORT_WEBGPU_SGMM_TUNE=1
//   set ORT_WEBGPU_SGMM_TUNE_OUT=D:\tmp\sgmm_tuning.csv
//   onnxruntime_provider_test.exe --gtest_filter=WebGpuSgMatMulTuning.*

#if defined(__wasm__)
// The Intel subgroup-matrix kernel and its tuning hooks are not built for wasm.
#else

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gsl/gsl>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/platform/env_var.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/vendor/intel/intel_device_info.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_matmul_tuning.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"

#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
namespace {

namespace wgpu = onnxruntime::webgpu;
namespace wi = onnxruntime::webgpu::intel;

// --- Sweep axes --------------------------------------------------------------
//
// Log2 spines over the target envelope [batch, M, N, K] <= [1K, 4K, 64K, 64K].
// The full cross product is 18590 points; the memory/index filter below removes
// the (large) majority that cannot be allocated or indexed.
constexpr uint32_t kBatchGrid[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
constexpr uint32_t kMGrid[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
constexpr uint32_t kNGrid[] = {16, 32, 64, 128, 256, 512, 1024, 2048,
                               4096, 8192, 16384, 32768, 65536};
constexpr uint32_t kKGrid[] = {16, 32, 64, 128, 256, 512, 1024, 2048,
                               4096, 8192, 16384, 32768, 65536};

// --- Budgets -----------------------------------------------------------------

// The shader addresses A/B/Y with u32 flat element offsets.
constexpr uint64_t kMaxTensorElems = 0xFFFFFFFFull;
constexpr uint64_t kDefaultBudgetMiB = 256;  // A + W + Y per problem
constexpr uint64_t kDefaultTensorMiB = 256;   // single tensor; <= maxStorageBufferBindingSize

// --- Benchmark knobs ---------------------------------------------------------

constexpr double kTargetTimedUs = 60000.0;  // ~60ms of timed work per config
constexpr int kWarmupRuns = 2;              // absorb one-time pipeline compilation
constexpr int kProbeRuns = 3;               // estimate per-run cost to size the timed batch
constexpr int kMinTimedRuns = 3;
constexpr int kMaxTimedRuns = 25;
constexpr int kBaselineWarmupRuns = 2;
constexpr int kBaselineTimedRuns = 5;

// --- Small helpers -----------------------------------------------------------

uint64_t GetEnvU64(const char* name, uint64_t fallback) {
  const std::string value = onnxruntime::detail::GetEnvironmentVar(name);
  if (value.empty()) {
    return fallback;
  }
  char* end = nullptr;
  const uint64_t parsed = std::strtoull(value.c_str(), &end, 10);
  return (end == value.c_str()) ? fallback : parsed;
}

double Median(std::vector<double> v) {
  if (v.empty()) {
    return -1.0;
  }
  std::sort(v.begin(), v.end());
  const size_t mid = v.size() / 2;
  return (v.size() % 2 == 0) ? 0.5 * (v[mid - 1] + v[mid]) : v[mid];
}

double MedianOfRange(const std::vector<double>& v, size_t first, size_t count) {
  if (count == 0 || first >= v.size() || first + count > v.size()) {
    return -1.0;
  }
  return Median(std::vector<double>(v.begin() + first, v.begin() + first + count));
}

struct Problem {
  uint32_t batch = 1;
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;

  // batch == 1 models the shared 2D weight case (B is [K, N]); everything else is
  // a true bmm where each slice has its own weight ([batch, K, N]).
  bool IsSharedWeight() const { return batch == 1; }
  uint64_t AElems() const { return uint64_t{batch} * M * K; }
  uint64_t WElems() const { return IsSharedWeight() ? uint64_t{K} * N : uint64_t{batch} * K * N; }
  uint64_t YElems() const { return uint64_t{batch} * M * N; }
};

// Rejects problems the kernel cannot run (u32 flat indexing, single-binding size)
// or that would not fit the caller's GPU memory budget. f16 => 2 bytes/element.
bool IsProblemFeasible(const Problem& p, uint64_t budget_bytes, uint64_t tensor_bytes) {
  const uint64_t a = p.AElems();
  const uint64_t w = p.WElems();
  const uint64_t y = p.YElems();
  if (a == 0 || w == 0 || y == 0) {
    return false;
  }
  if (a > kMaxTensorElems || w > kMaxTensorElems || y > kMaxTensorElems) {
    return false;
  }
  if (2 * a > tensor_bytes || 2 * w > tensor_bytes || 2 * y > tensor_bytes) {
    return false;
  }
  return 2 * (a + w + y) <= budget_bytes;
}

OrtValue MakeF16Input(const std::vector<int64_t>& dims, uint32_t seed) {
  // A short repeating pattern of small magnitudes: cheap to fill for the hundreds
  // of millions of elements involved, and keeps the accumulator well inside f16
  // range so timings are not perturbed by inf/NaN handling.
  constexpr size_t kPatternLen = 251;
  std::vector<MLFloat16> pattern(kPatternLen);
  for (size_t i = 0; i < kPatternLen; ++i) {
    pattern[i] = MLFloat16(0.05f * static_cast<float>((i + seed) % 11) - 0.25f);
  }

  AllocatorPtr allocator = std::make_shared<CPUAllocator>();
  OrtValue value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), TensorShape(dims), allocator, value);
  auto span = value.GetMutable<Tensor>()->MutableDataAsSpan<MLFloat16>();
  for (size_t i = 0; i < span.size(); ++i) {
    span[i] = pattern[i % kPatternLen];
  }
  return value;
}

// Builds a serialized f16 model:  Y = A @ W ; probe = Slice(Y, [0..], [1..]).
// A and W are graph inputs (not initializers): a constant weight would have to be
// embedded in the ModelProto, which both doubles host memory and hits protobuf's
// 2GB message limit well inside the target size envelope. The kernel only requires
// a constant B for odd N, and the sweep grid keeps N even. The tiny Slice output is
// what forces a host sync per Run.
std::string BuildMatMulModel(const Problem& p, std::string& y_name, std::string& probe_name) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 14;

  Model model("SgMatMulTuner", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  const int64_t batch = static_cast<int64_t>(p.batch);
  const int64_t m = static_cast<int64_t>(p.M);
  const int64_t n = static_cast<int64_t>(p.N);
  const int64_t k = static_cast<int64_t>(p.K);

  const std::optional<std::vector<int64_t>> a_dims =
      p.IsSharedWeight() ? std::vector<int64_t>{m, k} : std::vector<int64_t>{batch, m, k};
  const std::optional<std::vector<int64_t>> w_dims =
      p.IsSharedWeight() ? std::vector<int64_t>{k, n} : std::vector<int64_t>{batch, k, n};

  auto* a = builder.MakeInput<MLFloat16>(a_dims, std::string("A"));
  auto* w = builder.MakeInput<MLFloat16>(w_dims, std::string("W"));
  auto* y = builder.MakeOutput();
  builder.AddNode("MatMul", {a, w}, {y});

  const size_t rank = a_dims->size();
  const std::vector<int64_t> starts(rank, 0);
  const std::vector<int64_t> ends(rank, 1);
  std::vector<int64_t> axes(rank);
  std::iota(axes.begin(), axes.end(), int64_t{0});
  auto* probe = builder.MakeOutput();
  builder.AddNode("Slice",
                  {y,
                   builder.Make1DInitializer<int64_t>(starts),
                   builder.Make1DInitializer<int64_t>(ends),
                   builder.Make1DInitializer<int64_t>(axes)},
                  {probe});

  builder.SetGraphOutputs();
  EXPECT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(builder.output_names_.size(), size_t{2});
  y_name = builder.output_names_[0];
  probe_name = builder.output_names_[1];

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// Per-kernel GPU durations (profiler "Api" events) in submission order. `include`
// and `exclude`, when non-empty, filter on the event name (which is formatted as
// "node&type&program").
std::vector<double> ParseKernelDurations(const std::string& profile_path,
                                         std::string_view include,
                                         std::string_view exclude) {
  std::vector<double> durations;
  std::ifstream stream(profile_path);
  if (!stream.is_open()) {
    return durations;
  }
  nlohmann::json events;
  try {
    stream >> events;
  } catch (const std::exception&) {
    return durations;
  }
  if (!events.is_array()) {
    return durations;
  }
  for (const auto& e : events) {
    if (!e.is_object() || !e.contains("cat") || !e.contains("dur") || !e.contains("name")) {
      continue;
    }
    if (e["cat"].get<std::string>() != "Api") {
      continue;
    }
    const std::string name = e["name"].get<std::string>();
    if (!include.empty() && name.find(include) == std::string::npos) {
      continue;
    }
    if (!exclude.empty() && name.find(exclude) != std::string::npos) {
      continue;
    }
    durations.push_back(static_cast<double>(e["dur"].get<long long>()));
  }
  return durations;
}

// Collapses per-kernel durations into one value per Run. Returns empty when the
// event count is not a whole multiple of the run count, i.e. the mapping from
// events to runs is ambiguous and GPU time cannot be trusted.
std::vector<double> GroupPerRun(const std::vector<double>& durations, size_t total_runs) {
  if (total_runs == 0 || durations.empty() || durations.size() % total_runs != 0) {
    return {};
  }
  const size_t per_run = durations.size() / total_runs;
  std::vector<double> grouped(total_runs, 0.0);
  for (size_t i = 0; i < total_runs; ++i) {
    for (size_t j = 0; j < per_run; ++j) {
      grouped[i] += durations[i * per_run + j];
    }
  }
  return grouped;
}

// One session bound to one problem. Configs are switched between Run() calls via
// the tiling override, so the model build, weight upload and output allocation are
// paid once per problem instead of once per config.
class ProblemSession {
 public:
  bool Init(const Problem& p) {
    profile_prefix_ = std::filesystem::temp_directory_path() / ORT_TSTR("sgmm_tune");
    session_options_.enable_profiling = true;
    session_options_.profile_file_prefix = profile_prefix_.native();

    session_ = std::make_unique<InferenceSessionWrapper>(session_options_, GetEnvironment());
    auto ep = DefaultWebGpuExecutionProvider();
    if (ep == nullptr || !session_->RegisterExecutionProvider(std::move(ep)).IsOK()) {
      return false;
    }

    const std::string model_data = BuildMatMulModel(p, y_name_, probe_name_);
    if (model_data.empty() ||
        !session_->Load(model_data.data(), static_cast<int>(model_data.size())).IsOK() ||
        !session_->Initialize().IsOK() ||
        !session_->NewIOBinding(&binding_).IsOK()) {
      return false;
    }

    // Scoped so the host-side staging copies are released as soon as BindInput has
    // moved the data onto the device.
    {
      const std::vector<int64_t> a_dims =
          p.IsSharedWeight() ? std::vector<int64_t>{p.M, p.K}
                             : std::vector<int64_t>{p.batch, p.M, p.K};
      const std::vector<int64_t> w_dims =
          p.IsSharedWeight() ? std::vector<int64_t>{p.K, p.N}
                             : std::vector<int64_t>{p.batch, p.K, p.N};
      if (!binding_->BindInput("A", MakeF16Input(a_dims, 11)).IsOK() ||
          !binding_->BindInput("W", MakeF16Input(w_dims, 29)).IsOK() ||
          !binding_->SynchronizeInputs().IsOK()) {
        return false;
      }
    }

    // Y stays resident on the GPU; only the 1-element probe comes back, which is
    // enough to drain the queue without paying a full output download per Run.
    return binding_->BindOutput(y_name_, wgpu::WebGpuDevice).IsOK() &&
           binding_->BindOutput(probe_name_).IsOK();
  }

  // Executes `count` runs. `median_wall_us`, when non-null, receives the median
  // wall-clock time per run.
  bool Run(int count, double* median_wall_us) {
    std::vector<double> wall;
    wall.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      if (!session_->Run(run_options_, *binding_).IsOK()) {
        return false;
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      ++runs_done_;
    }
    if (median_wall_us != nullptr) {
      *median_wall_us = Median(std::move(wall));
    }
    return true;
  }

  size_t runs_done() const { return runs_done_; }

  // Ends profiling and returns the per-Run GPU durations for the kernels matching
  // the filter, or an empty vector if the device produced no usable timestamps.
  std::vector<double> FinishAndCollectGpuTimes(std::string_view include, std::string_view exclude) {
    const std::string profile_path = session_->EndProfiling();
    auto cleanup = gsl::finally([&profile_path] {
      std::error_code ec;
      std::filesystem::remove(profile_path, ec);
    });
    return GroupPerRun(ParseKernelDurations(profile_path, include, exclude), runs_done_);
  }

 private:
  SessionOptions session_options_;
  std::filesystem::path profile_prefix_;
  std::unique_ptr<InferenceSessionWrapper> session_;
  std::unique_ptr<IOBinding> binding_;
  RunOptions run_options_;
  std::string y_name_;
  std::string probe_name_;
  size_t runs_done_ = 0;
};

struct Measurement {
  wgpu::SubgroupMatrixTiling tiling{};  // tile + split-K config
  double gpu_us = -1.0;                 // median GPU kernel time (us); -1 if timestamps absent
  double wall_us = -1.0;                // median wall-clock per Run() (us)
  int runs = 0;
  size_t first_run = 0;  // index of this measurement's first timed run in the session
};

double MetricOf(const Measurement& m) { return (m.gpu_us > 0.0) ? m.gpu_us : m.wall_us; }

using ProblemKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;  // batch, M, N, K

// Problem keys already present in an existing CSV, so a resumed sweep can skip them.
std::set<ProblemKey> ReadCompletedProblems(const std::string& path) {
  std::set<ProblemKey> done;
  std::ifstream in(path);
  if (!in.is_open()) {
    return done;
  }
  auto split = [](const std::string& s) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
      parts.push_back(item);
    }
    return parts;
  };

  std::string line;
  if (!std::getline(in, line)) {
    return done;
  }
  const std::vector<std::string> header = split(line);
  auto index_of = [&header](std::string_view name) -> int {
    for (size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  const int mi = index_of("M");
  const int ni = index_of("N");
  const int ki = index_of("K");
  const int bi = index_of("batch");
  if (mi < 0 || ni < 0 || ki < 0 || bi < 0) {
    return done;
  }
  const size_t needed = static_cast<size_t>(std::max({mi, ni, ki, bi})) + 1;
  while (std::getline(in, line)) {
    const std::vector<std::string> parts = split(line);
    if (parts.size() < needed) {
      continue;
    }
    done.emplace(static_cast<uint32_t>(std::strtoul(parts[bi].c_str(), nullptr, 10)),
                 static_cast<uint32_t>(std::strtoul(parts[mi].c_str(), nullptr, 10)),
                 static_cast<uint32_t>(std::strtoul(parts[ni].c_str(), nullptr, 10)),
                 static_cast<uint32_t>(std::strtoul(parts[ki].c_str(), nullptr, 10)));
  }
  return done;
}

}  // namespace

// Sweeps the (batch, M, N, K) grid, benchmarks every valid config per problem
// plus the generic (non-subgroup-matrix) baseline, and writes one CSV row per
// (problem, config) measurement.
TEST(WebGpuSgMatMulTuning, SweepAndEmitCsv) {
  const std::string enable = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE");
  if (enable.empty()) {
    GTEST_SKIP() << "Set ORT_WEBGPU_SGMM_TUNE=1 to run the subgroup-matrix MatMul autotuner.";
  }
  // ORT_WEBGPU_SGMM_TUNE=2 also logs every candidate config (very chatty).
  const bool verbose = enable == "2";

  std::string out_path = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE_OUT");
  if (out_path.empty()) {
    out_path = "sgmm_tuning.csv";
  }

  const uint64_t budget_bytes = GetEnvU64("ORT_WEBGPU_SGMM_TUNE_BUDGET_MB", kDefaultBudgetMiB) << 20;
  const uint64_t tensor_bytes = GetEnvU64("ORT_WEBGPU_SGMM_TUNE_TENSOR_MB", kDefaultTensorMiB) << 20;

  // Optional "i/n" sharding so the sweep can be split across processes/machines.
  uint64_t shard_index = 0;
  uint64_t shard_count = 1;
  {
    const std::string shard = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE_SHARD");
    const size_t slash = shard.find('/');
    if (slash != std::string::npos) {
      shard_index = std::strtoull(shard.substr(0, slash).c_str(), nullptr, 10);
      shard_count = std::max<uint64_t>(1, std::strtoull(shard.substr(slash + 1).c_str(), nullptr, 10));
      ASSERT_LT(shard_index, shard_count) << "ORT_WEBGPU_SGMM_TUNE_SHARD must be i/n with i < n.";
    }
  }

  const bool resume = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE_RESUME") == "1";
  const std::set<ProblemKey> already_done = resume ? ReadCompletedProblems(out_path) : std::set<ProblemKey>{};

  // Build the work list up front: enumerating configs is cheap, and knowing the
  // total makes the progress output (and shard balancing) meaningful.
  std::vector<Problem> problems;
  uint64_t candidate_index = 0;
  for (uint32_t batch : kBatchGrid) {
    for (uint32_t m : kMGrid) {
      for (uint32_t n : kNGrid) {
        for (uint32_t k : kKGrid) {
          const Problem p{batch, m, n, k};
          if (!IsProblemFeasible(p, budget_bytes, tensor_bytes)) {
            continue;
          }
          if (wi::EnumerateSubgroupMatrixTilings(m, n, k).empty()) {
            continue;
          }
          const uint64_t index = candidate_index++;
          if (index % shard_count != shard_index) {
            continue;
          }
          if (already_done.count(ProblemKey{batch, m, n, k}) != 0) {
            continue;
          }
          problems.push_back(p);
        }
      }
    }
  }

  std::error_code size_ec;
  const auto existing_size = std::filesystem::file_size(out_path, size_ec);
  std::ofstream csv(out_path, resume ? std::ios::app : std::ios::trunc);
  ASSERT_TRUE(csv.is_open()) << "Cannot open output CSV: " << out_path;
  if (!resume || size_ec || existing_size == 0) {
    csv << "arch,hw_subgroups,M,N,K,batch,tile_m,tile_n,split_k,time_ms,gpu_us,wall_us,runs,"
        << "baseline_gpu_us,baseline_wall_us\n";
  }

  std::cout << "[sgmm-tune] " << problems.size() << " problems (of " << candidate_index
            << " feasible; shard " << shard_index << "/" << shard_count
            << "); budget " << (budget_bytes >> 20) << "MiB, per-tensor cap "
            << (tensor_bytes >> 20) << "MiB; output -> " << out_path << std::endl;

  const auto sweep_start = std::chrono::steady_clock::now();
  // Device architecture (e.g. "xe-2lpg"), captured after the first MatMul runs;
  // each tuned table is keyed by this so distinct GPU arches get distinct tables.
  std::string arch;
  uint32_t hw_subgroups = 0;
  // Best-config speedup over the generic baseline per problem, for the summary
  // distribution printed at the end.
  std::vector<double> speedups;
  size_t problem_index = 0;

  for (const Problem& p : problems) {
    ++problem_index;
    const std::vector<wgpu::SubgroupMatrixTiling> enumerated =
        wi::EnumerateSubgroupMatrixTilings(p.M, p.N, p.K);
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sweep_start).count();
    std::cout << "[sgmm-tune] (" << problem_index << "/" << problems.size() << ", "
              << static_cast<int>(100.0 * problem_index / problems.size()) << "%, "
              << static_cast<int>(elapsed_s) << "s) batch=" << p.batch << " M=" << p.M
              << " N=" << p.N << " K=" << p.K << " : " << enumerated.size() << " configs..."
              << std::endl;

    // Idle briefly so the GPU settles between problems (avoids thermal/clock
    // carry-over from the previous problem skewing the first measurements).
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Baseline: the generic WebGPU MatMul path, in its own session so its programs
    // never appear in the tuned session's profile.
    double baseline_gpu_us = -1.0;
    double baseline_wall_us = -1.0;
    {
      wi::SetSgMatMulDisabled(true);
      auto reenable = gsl::finally([] { wi::SetSgMatMulDisabled(false); });

      ProblemSession baseline_session;
      if (baseline_session.Init(p) &&
          baseline_session.Run(kBaselineWarmupRuns, nullptr)) {
        const size_t first = baseline_session.runs_done();
        if (baseline_session.Run(kBaselineTimedRuns, &baseline_wall_us)) {
          // Any kernel other than the probe Slice is part of the baseline MatMul.
          const std::vector<double> gpu = baseline_session.FinishAndCollectGpuTimes("", "Slice");
          baseline_gpu_us = MedianOfRange(gpu, first, kBaselineTimedRuns);
        }
      }
    }
    const double baseline_metric = (baseline_gpu_us > 0.0) ? baseline_gpu_us : baseline_wall_us;

    // The arch is known once at least one MatMul has been dispatched.
    if (arch.empty()) {
      arch = wi::GetSgMatMulDeviceArch();
      ASSERT_FALSE(arch.empty()) << "No WebGPU MatMul reached the subgroup-matrix selector; "
                                    "the tuner cannot identify the device.";
      hw_subgroups = wgpu::intel::HwSubgroups(arch);
      std::cout << "[sgmm-tune] device architecture: " << arch
                << " (hw_subgroups=" << hw_subgroups << ")" << std::endl;
    }

    if (verbose) {
      std::cout << "[sgmm-tune]     baseline (generic) -> gpu=" << baseline_gpu_us
                << "us wall=" << baseline_wall_us << "us"
                << (baseline_metric <= 0.0 ? " (FAILED)" : "") << std::endl;
    }

    ProblemSession session;
    if (!session.Init(p)) {
      std::cout << "[sgmm-tune]   -> session setup failed (skipped)" << std::endl;
      continue;
    }
    auto clear_override = gsl::finally([] { wi::SetSgMatMulTilingOverride(std::nullopt); });

    // Fixed-seed shuffle: measurement order is decorrelated from config order, so
    // slow drift (clocks, thermals) does not systematically favour any tile shape,
    // while runs stay reproducible.
    std::vector<wgpu::SubgroupMatrixTiling> tilings = enumerated;
    std::shuffle(tilings.begin(), tilings.end(), std::mt19937{12345u});

    // Benchmark every config: warmup (absorbs the one-time pipeline compilation for
    // this tile shape), a short probe to estimate per-run cost, then a timed batch
    // sized to ~kTargetTimedUs of work. Wall time is the only signal available here
    // because GPU timestamps are readable only after EndProfiling (which would end
    // the session); GPU time is back-filled for every config below.
    std::vector<Measurement> measurements;
    measurements.reserve(tilings.size());
    for (const auto& tiling : tilings) {
      wi::SetSgMatMulTilingOverride(tiling);
      if (!session.Run(kWarmupRuns, nullptr)) {
        continue;
      }
      double probe_wall = 0.0;
      if (!session.Run(kProbeRuns, &probe_wall)) {
        continue;
      }
      const int runs = static_cast<int>(std::clamp<long>(
          std::lround(kTargetTimedUs / std::max(probe_wall, 1.0)), kMinTimedRuns, kMaxTimedRuns));
      Measurement m;
      m.tiling = tiling;
      m.runs = runs;
      m.first_run = session.runs_done();  // after warmup+probe, so the timed batch is clean
      if (!session.Run(runs, &m.wall_us)) {
        continue;
      }
      measurements.push_back(m);
    }

    wi::SetSgMatMulTilingOverride(std::nullopt);

    // Back-fill GPU time for every measurement from the single per-session profile.
    const std::vector<double> gpu_per_run =
        session.FinishAndCollectGpuTimes("SubgroupMatrixMatMul", "");
    for (Measurement& m : measurements) {
      m.gpu_us = MedianOfRange(gpu_per_run, m.first_run, static_cast<size_t>(m.runs));
      if (verbose) {
        std::cout << "[sgmm-tune]     tile " << m.tiling.tile_m << "x" << m.tiling.tile_n
                  << " split_k=" << m.tiling.split_k << " -> gpu=" << m.gpu_us
                  << "us wall=" << m.wall_us << "us" << std::endl;
      }
    }

    const Measurement* best = nullptr;
    for (const Measurement& m : measurements) {
      const double metric = MetricOf(m);
      if (metric <= 0.0) {
        continue;
      }
      csv << arch << "," << hw_subgroups << "," << p.M << "," << p.N << "," << p.K << "," << p.batch
          << "," << m.tiling.tile_m << "," << m.tiling.tile_n << "," << m.tiling.split_k << ","
          << (metric / 1000.0) << "," << m.gpu_us << "," << m.wall_us << "," << m.runs << ","
          << baseline_gpu_us << "," << baseline_wall_us << "\n";
      if (best == nullptr || metric < MetricOf(*best)) {
        best = &m;
      }
    }
    csv.flush();

    if (best == nullptr) {
      std::cout << "[sgmm-tune]   -> no config ran successfully (skipped)" << std::endl;
      continue;
    }

    const double best_metric = MetricOf(*best);
    const double speedup =
        (baseline_metric > 0.0 && best_metric > 0.0) ? baseline_metric / best_metric : -1.0;
    if (speedup > 0.0) {
      speedups.push_back(speedup);
    }
    std::cout << "[sgmm-tune]   -> best tile " << best->tiling.tile_m << "x" << best->tiling.tile_n
              << " split_k=" << best->tiling.split_k << " (gpu=" << best->gpu_us
              << "us wall=" << best->wall_us << "us) vs baseline gpu=" << baseline_gpu_us
              << "us wall=" << baseline_wall_us << "us";
    if (speedup > 0.0) {
      std::cout << " => " << speedup << "x";
    }
    std::cout << std::endl;
  }

  csv.close();
  const double total_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - sweep_start).count();
  std::cout << "[sgmm-tune] done: " << problem_index << " problems in "
            << static_cast<int>(total_s) << "s -> " << out_path << std::endl;

  // Speedup distribution: how many problems fall into each best-vs-baseline
  // speedup bucket. Buckets are half-open [lo, hi); the last is [10x, inf).
  if (!speedups.empty()) {
    constexpr double kEdges[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0};
    constexpr size_t kNumEdges = std::size(kEdges);
    size_t counts[kNumEdges + 1] = {0};  // one extra bucket for >= last edge
    for (double s : speedups) {
      size_t bucket = kNumEdges;  // default: >= last edge
      for (size_t i = 0; i < kNumEdges; ++i) {
        if (s < kEdges[i]) {
          bucket = i;
          break;
        }
      }
      ++counts[bucket];
    }

    const size_t total = speedups.size();
    std::cout << "[sgmm-tune] speedup distribution over " << total << " problems:" << std::endl;
    for (size_t i = 0; i <= kNumEdges; ++i) {
      std::ostringstream label;
      if (i == 0) {
        label << "      < " << kEdges[0] << "x";
      } else if (i == kNumEdges) {
        label << "    >= " << kEdges[kNumEdges - 1] << "x";
      } else {
        label << "[" << kEdges[i - 1] << "x, " << kEdges[i] << "x)";
      }
      const double pct = 100.0 * static_cast<double>(counts[i]) / static_cast<double>(total);
      std::cout << "[sgmm-tune]   " << label.str() << " : " << counts[i]
                << " (" << static_cast<int>(pct + 0.5) << "%)" << std::endl;
    }
  }

  GTEST_LOG_(INFO) << "Subgroup-matrix MatMul tuning written to " << out_path;
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
