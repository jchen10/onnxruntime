#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Distill a subgroup-matrix MatMul tile/split-K selector into dependency-free C++.

Pipeline (offline): load sweep timings -> train a GBM cost model -> label a dense
problem grid by argmin over valid configs -> distill into one depth-capped decision
tree -> emit it as an if/else `PredictTilingTree` -> report regret
(chosen_time / oracle_best_time - 1) for the tree and the cost model.

The emitted function mirrors `HeuristicTiling` in
`subgroup_matrix_tiling_selector.cc`: a performance oracle only, so the caller must
still run `IsTilingValid`.

Sweep CSV, as emitted by `WebGpuSgMatMulTuning.SweepAndEmitCsv`:

    arch,hw_subgroups,M,N,K,batch,tile_m,tile_n,split_k,time_ms,gpu_us,wall_us,
    runs,baseline_gpu_us,baseline_wall_us

All but the timing columns are required; `hw_subgroups`, still emitted by the sweep,
is ignored. `gpu_us` is the timing used -- the kernel's own timestamp span, excluding
the host-sync probe; rows lacking it (-1) are dropped rather than mixed with wall
time, which carries a large constant probe offset. The sweep emits one row per
(problem, config) measured, so a pair appears at most once per sweep; when sweeps are
concatenated, the row with more `runs` wins.

`arch` comes from the file, not the command line, and a mixed-arch CSV is rejected:
one model and one .inc per arch.

Tuning depth vs. teacher density: the run prints regret for both `distilled_tree`
and `cost_model`. The tree is distilled from the cost model, so it cannot beat it --
the cost_model row is a lower bound. Tune against the gap between the two:
  * distilled_tree regret >> cost_model regret -> the student is under-resolved.
    Raise --max-depth / --max-leaves FIRST; a depth-8 tree over 4 features resolves
    only a few thresholds per axis, so it is usually depth, not density, that binds.
  * distilled_tree regret ~= cost_model regret -> distillation is faithful; the
    residual is model error, and more depth or density is wasted work.
Only densify TEACHER_*_AXIS (below) once a deeper tree stops closing the gap and the
max/p99 rows are still driven by boundary misplacement. The axes are geometric
(~2 points/octave) because every feature is log2; keep them log-uniform, and keep
each axis capped at its swept range so labels are never extrapolated past the data.

Usage:

    python tools/python/gen_sgmm_tree.py --input sweep.csv

Requires: numpy, scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:  # pragma: no cover - dependency hint
    print(
        "This tool needs scikit-learn and numpy. Install with:\n"
        "    python -m pip install numpy scikit-learn",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# --- Kernel constants (must match subgroup_matrix_tiling_selector.cc) ---------

SG_K = 16
# Multiples of the subgroup-matrix M (8) and N (16), so tile alignment is implicit.
TILE_M_CANDS = (8, 16, 32, 64)
TILE_N_CANDS = (16, 32, 64)
SPLIT_K_CANDS = (1, 2, 4, 8)
MAX_SCRATCH_ELEMS = 16384  # 32 KB of f16 partials

# One .inc per arch; the filename is derived from the arch in the sweep CSV.
DEFAULT_OUTPUT_DIR = Path("onnxruntime/core/providers/webgpu/vendor/intel/math")


@dataclass(frozen=True)
class Config:
    tile_m: int
    tile_n: int
    split_k: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.tile_m, self.tile_n, self.split_k)


@lru_cache(maxsize=None)
def valid_configs(k: int) -> tuple[Config, ...]:
    """Configs satisfying IsTilingValid for a given K.

    Empty for K not a multiple of SG_K: the selector declines those problems, so
    they must never reach the model.
    """
    if k < SG_K or k % SG_K != 0:
        return ()
    k_blocks = k // SG_K
    out: list[Config] = []
    for tm in TILE_M_CANDS:
        for tn in TILE_N_CANDS:
            for sk in SPLIT_K_CANDS:
                if sk < 1 or sk > k_blocks:
                    continue
                if tm * tn * sk > MAX_SCRATCH_ELEMS:
                    continue
                out.append(Config(tm, tn, sk))
    return tuple(out)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# --- Feature engineering ------------------------------------------------------
#
# Cost-model features see (problem, config); student features see the problem only,
# since the tree predicts the config and must run from (M, N, K, batch) at dispatch.
# Keep both builders and the C++ emitter in sync.

PROBLEM_FEATURE_EXPRS = (
    "std::log2(static_cast<float>(M < 1u ? 1u : M))",
    "std::log2(static_cast<float>(N < 1u ? 1u : N))",
    "std::log2(static_cast<float>(K < 1u ? 1u : K))",
    "std::log2(static_cast<float>(batch < 1u ? 1u : batch))",
)


def _log2(v: int) -> float:
    """log2 clamped at 1 so degenerate/zero dimensions cannot raise."""
    return math.log2(max(v, 1))


def problem_features(m: int, n: int, k: int, batch: int) -> list[float]:
    return [
        _log2(m),
        _log2(n),
        _log2(k),
        _log2(batch),
    ]


def cost_features(m: int, n: int, k: int, batch: int, c: Config) -> list[float]:
    tiles = batch * ceil_div(m, c.tile_m) * ceil_div(n, c.tile_n)
    return [
        _log2(m),
        _log2(n),
        _log2(k),
        _log2(batch),
        _log2(c.tile_m),
        _log2(c.tile_n),
        _log2(c.split_k),
        float(tiles),  # independent output tiles (occupancy)
        float(c.tile_m * c.tile_n * c.split_k),  # scratch elems
        float(k // SG_K),  # k_blocks
    ]


# --- Data loading -------------------------------------------------------------


@dataclass
class Row:
    arch: str
    m: int
    n: int
    k: int
    batch: int
    config: Config
    time_ms: float
    runs: int = 0


def _row_quality(r: Row) -> int:
    """Dedup order across concatenated sweeps: more timed runs win."""
    return r.runs


def load_csv(path: Path) -> tuple[list[Row], str]:
    """Load the sweep, deduplicated to the best row per (problem, config).

    Returns (rows, arch), the arch read from the file. A mixed-arch CSV is a hard
    error rather than a silent pick.
    """
    best: dict[tuple[int, int, int, int, tuple[int, int, int]], Row] = {}
    archs: set[str] = set()
    skipped_untimed = 0
    skipped_invalid = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"arch", "M", "N", "K", "tile_m", "tile_n", "split_k", "gpu_us"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing required column(s): {', '.join(sorted(missing))}")

        for rec in reader:
            # Before any skip, so a mixed-arch file is caught even if its rows are untimed.
            archs.add(rec["arch"])

            try:
                gpu_us = float(rec["gpu_us"])
                row = Row(
                    arch=rec["arch"],
                    m=int(rec["M"]),
                    n=int(rec["N"]),
                    k=int(rec["K"]),
                    batch=int(rec.get("batch") or 1),
                    config=Config(int(rec["tile_m"]), int(rec["tile_n"]), int(rec["split_k"])),
                    time_ms=gpu_us / 1000.0,
                    runs=int(rec.get("runs") or 0),
                )
            except ValueError:
                skipped_invalid += 1
                continue

            if gpu_us <= 0.0:  # -1 when the device produced no timestamps
                skipped_untimed += 1
                continue

            key = (row.m, row.n, row.k, row.batch, row.config.as_tuple())
            prev = best.get(key)
            if prev is None or (_row_quality(row), -row.time_ms) > (_row_quality(prev), -prev.time_ms):
                best[key] = row

    if skipped_untimed or skipped_invalid:
        print(
            f"  skipped {skipped_untimed} untimed and {skipped_invalid} unparsable rows.",
            file=sys.stderr,
        )
    if not best:
        raise SystemExit(f"{path}: no rows with a positive gpu_us; the device produced no timestamps.")
    if len(archs) > 1:
        raise SystemExit(f"{path}: mixes archs ({', '.join(sorted(archs))}); split it and run once per arch.")
    return list(best.values()), archs.pop()


# --- Model training + distillation --------------------------------------------


def train_cost_model(rows: list[Row], seed: int = 0) -> HistGradientBoostingRegressor:
    x = np.array([cost_features(r.m, r.n, r.k, r.batch, r.config) for r in rows], dtype=np.float64)
    y = np.log(np.array([r.time_ms for r in rows], dtype=np.float64))  # log-time target
    model = HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.1,
        max_iter=400,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def cost_model_pick(model: HistGradientBoostingRegressor, m: int, n: int, k: int, batch: int) -> Config:
    cands = valid_configs(k)
    if not cands:
        raise ValueError(f"No valid config for K={k}; caller must filter these problems out.")
    x = np.array([cost_features(m, n, k, batch, c) for c in cands], dtype=np.float64)
    preds = model.predict(x)
    return cands[int(np.argmin(preds))]


# Teacher grid: denser than the sweep (a ~1.5x midpoint between each power of two),
# with a per-dimension span mirroring the sweep envelope in
# subgroup_matrix_matmul_tuning_test.cc ([batch, M, N, K] <= [1K, 4K, 64K, 64K]).
# Keeping each axis at its swept cap avoids labelling the student from cost-model
# predictions extrapolated past where the model has seen data.
TEACHER_M_AXIS = (
    8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096,
)
TEACHER_NK_AXIS = (
    16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
)
TEACHER_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def build_teacher(
    model: HistGradientBoostingRegressor, block_size: int = 4096
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Dense problem grid -> best config predicted by the cost model.

    Batched in blocks: ~125k problems x ~40 configs, where per-problem predict()
    calls are dominated by sklearn overhead and one big matrix would be ~5M rows.
    """
    # The selector only dispatches K aligned to the subgroup-matrix K tile.
    k_axis = [v for v in TEACHER_NK_AXIS if valid_configs(v)]
    problems = [
        (m, n, k, batch)
        for m in TEACHER_M_AXIS
        for n in TEACHER_NK_AXIS
        for k in k_axis
        for batch in TEACHER_BATCHES
    ]

    feats: list[list[float]] = []
    labels: list[tuple[int, int, int]] = []
    for start in range(0, len(problems), block_size):
        block = problems[start : start + block_size]
        rows: list[list[float]] = []
        spans: list[tuple[int, int]] = []
        for m, n, k, batch in block:
            cands = valid_configs(k)
            spans.append((len(rows), len(cands)))
            rows.extend(cost_features(m, n, k, batch, c) for c in cands)
        preds = model.predict(np.array(rows, dtype=np.float64))
        for (m, n, k, batch), (offset, count) in zip(block, spans, strict=True):
            best = valid_configs(k)[int(np.argmin(preds[offset : offset + count]))]
            feats.append(problem_features(m, n, k, batch))
            labels.append(best.as_tuple())
    return np.array(feats, dtype=np.float64), labels


def distill(
    x: np.ndarray, labels: list[tuple[int, int, int]], max_depth: int, max_leaves: int | None, seed: int = 0
) -> tuple[DecisionTreeClassifier, list[tuple[int, int, int]]]:
    classes = sorted(set(labels))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in labels])
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaves,
        min_samples_leaf=8,
        random_state=seed,
    )
    clf.fit(x, y)
    return clf, classes


def make_tree_pick(
    clf: DecisionTreeClassifier, classes: list[tuple[int, int, int]]
) -> Callable[[int, int, int, int], Config]:
    def tree_pick(m: int, n: int, k: int, batch: int) -> Config:
        x = np.array([problem_features(m, n, k, batch)], dtype=np.float64)
        tm, tn, sk = classes[int(clf.predict(x)[0])]
        return Config(tm, tn, sk)

    return tree_pick


# --- Regret evaluation --------------------------------------------------------


# Policy signature: (M, N, K, batch) -> Config
PolicyFn = Callable[[int, int, int, int], Config]


def evaluate_regret(
    rows: list[Row],
    policies: dict[str, PolicyFn],
    cost_model: HistGradientBoostingRegressor,
    large_threshold: int = 4096,
) -> dict[str, dict[str, float]]:
    """Regret = chosen_time / oracle_best_time - 1, aggregated per policy.

    Uses measured times where the pick was swept, else the cost-model prediction
    (tracked as 'coverage'). That makes the 'cost_model' row optimistic -- it is
    scored partly against itself -- so read it as a bound, not a competitor.

    A policy may propose a config IsTilingValid rejects (the tree does not see K's
    block count). Those have no runtime to score: excluded from the aggregates and
    reported as 'invalid_pct'.
    """
    measured: dict[tuple[int, int, int, int], dict[tuple[int, int, int], float]] = {}
    for r in rows:
        measured.setdefault((r.m, r.n, r.k, r.batch), {})[r.config.as_tuple()] = r.time_ms

    # Resolve all choices first, then price the unmeasured ones in one batched
    # predict instead of a sklearn call per (policy, problem).
    choices: dict[str, list[Config | None]] = {name: [] for name in policies}
    problems = [p for p, by_cfg in measured.items() if min(by_cfg.values()) > 0]
    pending: list[tuple[int, int, int, int, Config]] = []
    for m, n, k, batch in problems:
        for name, fn in policies.items():
            c = fn(m, n, k, batch)
            if c not in valid_configs(k):
                choices[name].append(None)  # unrunnable; counted, not scored
                continue
            choices[name].append(c)
            if c.as_tuple() not in measured[(m, n, k, batch)]:
                pending.append((m, n, k, batch, c))

    predicted: dict[tuple[int, int, int, int, tuple[int, int, int]], float] = {}
    if pending:
        x = np.array([cost_features(m, n, k, b, c) for m, n, k, b, c in pending], dtype=np.float64)
        for (m, n, k, b, c), t in zip(pending, np.exp(cost_model.predict(x)), strict=True):
            predicted[(m, n, k, b, c.as_tuple())] = float(t)

    results: dict[str, dict[str, float]] = {}
    for name in policies:
        regrets: list[float] = []
        large: list[float] = []
        covered = 0
        invalid = 0
        for (m, n, k, batch), c in zip(problems, choices[name], strict=True):
            if c is None:
                invalid += 1
                continue
            by_cfg = measured[(m, n, k, batch)]
            oracle = min(by_cfg.values())
            key = c.as_tuple()
            if key in by_cfg:
                t = by_cfg[key]
                covered += 1
            else:
                t = predicted[(m, n, k, batch, key)]
            regret = t / oracle - 1.0
            regrets.append(regret)
            if max(m, n, k) > large_threshold:
                large.append(regret)

        if not regrets:
            continue
        arr = np.array(regrets)
        results[name] = {
            "mean_regret_pct": float(np.mean(arr) * 100.0),
            "p99_regret_pct": float(np.percentile(arr, 99) * 100.0),
            "max_regret_pct": float(np.max(arr) * 100.0),
            "large_mean_regret_pct": float(np.mean(large) * 100.0) if large else float("nan"),
            "measured_coverage_pct": 100.0 * covered / len(regrets),
            "invalid_pct": 100.0 * invalid / len(problems),
        }
    return results


# --- C++ emitter --------------------------------------------------------------


def emit_cpp(
    clf: DecisionTreeClassifier,
    classes: list[tuple[int, int, int]],
    arch: str,
    out_path: Path,
) -> None:
    # sklearn's Tree is a Cython extension type; its arrays are untyped in the stubs.
    tree: Any = clf.tree_

    def recurse(node: int, depth: int) -> str:
        pad = "  " * depth
        if tree.children_left[node] == tree.children_right[node]:  # leaf
            cid = int(np.argmax(tree.value[node][0]))
            tm, tn, sk = classes[cid]
            return f"{pad}return {{{tm}, {tn}, {sk}}};\n"
        feat = int(tree.feature[node])
        thr = float(tree.threshold[node])
        s = f"{pad}if (f{feat} <= {thr:.6f}f) {{\n"
        s += recurse(int(tree.children_left[node]), depth + 1)
        s += f"{pad}}} else {{\n"
        s += recurse(int(tree.children_right[node]), depth + 1)
        s += f"{pad}}}\n"
        return s

    # Features the tree never splits on must still be declared to keep indices
    # aligned with PROBLEM_FEATURE_EXPRS, hence [[maybe_unused]].
    feat_decls = "".join(
        f"  [[maybe_unused]] const float f{i} = {expr};\n" for i, expr in enumerate(PROBLEM_FEATURE_EXPRS)
    )
    body = recurse(0, 1)
    depth = int(tree.max_depth)
    n_leaves = int(tree.n_leaves)

    content = f"""// Auto-generated. Do not edit.
// Distilled decision tree (depth {depth}, {n_leaves} leaves) approximating the offline
// GBM cost model for arch: {arch}.
//
// Generated from sweep timing data by training a gradient boosting cost model,
// labeling a dense problem grid, and distilling into a depth-capped decision tree.
//
// Arch-specific: dispatch on arch. One .inc per arch.
//
// Note: Common heuristic rules could potentially be derived from cross-arch analysis
// of raw sweep timing data across all Intel GPU architectures, which may reveal
// universal performance patterns applicable as a simpler baseline or fallback.
//
// Performance oracle only, like HeuristicTiling in
// subgroup_matrix_tiling_selector.cc: the caller MUST still run IsTilingValid.
// The tree does not see K's block count, so it can propose
// split_k > K / {SG_K}.
//
// The including TU must provide <cmath>, <cstdint> and SubgroupMatrixTiling.

namespace onnxruntime::webgpu::intel::{arch} {{

inline SubgroupMatrixTiling PredictSgmmTilingTree(uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {{
{feat_decls}{body}}}

}}  // namespace onnxruntime::webgpu::intel::{arch}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


# --- Main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Sweep CSV emitted by WebGpuSgMatMulTuning.")
    parser.add_argument("--max-depth", type=int, default=8, help="Depth cap of the distilled student tree.")
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=None,
        help="Optional leaf cap; bounds the size of the emitted if/else chain more directly than depth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Where to write the generated PredictTilingTree. Defaults to "
        f"{DEFAULT_OUTPUT_DIR}/subgroup_matrix_tiling_tree_<arch>.inc.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, arch = load_csv(args.input)
    arch = arch.replace("-", "")  # normalize once to a valid C++ identifier (filename + namespace)
    out_path = args.output or DEFAULT_OUTPUT_DIR / f"subgroup_matrix_tiling_tree_{arch}.inc"

    problems = {(r.m, r.n, r.k, r.batch) for r in rows}
    print(f"Loaded {len(rows)} (problem, config) measurements over {len(problems)} problems; arch={arch}.")

    print("Training GBM cost model ...")
    cost_model = train_cost_model(rows, args.seed)

    print("Building teacher labels over dense grid ...")
    x_teacher, labels = build_teacher(cost_model)
    print(f"  {len(labels)} teacher samples, {len(set(labels))} distinct best configs.")

    print(f"Distilling to a depth-<= {args.max_depth} decision tree ...")
    clf, classes = distill(x_teacher, labels, args.max_depth, args.max_leaves, args.seed)
    tree_pick = make_tree_pick(clf, classes)
    student: Any = clf.tree_
    print(f"  student: depth {student.max_depth}, {student.n_leaves} leaves, {len(classes)} tilings.")

    # Assemble the policies to compare.
    policies: dict[str, PolicyFn] = {
        "cost_model": lambda m, n, k, b: cost_model_pick(cost_model, m, n, k, b),
        "distilled_tree": tree_pick,
    }

    print("Evaluating regret ...\n")
    results = evaluate_regret(rows, policies, cost_model)

    header = (
        f"{'policy':<16}{'mean%':>9}{'p99%':>9}{'max%':>9}"
        f"{'>4K mean%':>11}{'measured%':>11}{'invalid%':>10}"
    )
    print(header)
    print("-" * len(header))
    for name in ["distilled_tree", "cost_model"]:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<16}{r['mean_regret_pct']:>9.2f}{r['p99_regret_pct']:>9.2f}"
            f"{r['max_regret_pct']:>9.2f}{r['large_mean_regret_pct']:>11.2f}"
            f"{r['measured_coverage_pct']:>11.1f}{r['invalid_pct']:>10.1f}"
        )
    print("\n(regret = chosen_time / oracle_best_time - 1; lower is better)")
    print("(cost_model is scored partly against its own predictions; treat it as a bound)")

    emit_cpp(clf, classes, arch, out_path)
    print(f"\nWrote distilled selector: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
