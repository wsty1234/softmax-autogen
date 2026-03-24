# Softmax Optimization Log

## Setup

- Date: 2026-03-24
- Branch: `softmax-autogen/mar23b`
- Editable files: `fused_softmax.py`, `optimization.md`
- Environment: `conda` env `unik3d`
- Status: experimentation active, current best checkpointed

## Best Result

- Best accepted speedup over torch: `5.71%`
- Best variant: tuned per-size launch dispatch with single-row kernel defaults and a special 2-row path for width `256`

## Experiments

| Iteration | Variant | Correctness | Triton speedup over torch | Decision | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Current worktree baseline, single-row kernel, fixed `num_warps=4` | pass | `4.02%` | accepted | First validated baseline from the dirty worktree starting point |
| 1 | Persistent-row kernel with cached occupancy-derived launch count | pass | `-1.61%` | rejected | Reduced average throughput, especially on larger widths |
| 2 | Single-row kernel with `num_warps` heuristic `2/4/8` by block size and `num_stages=2` | pass | `5.31%` | accepted | Replaced the baseline as the best validated variant |
| 3 | Multi-row kernel with `ROWS_PER_PROGRAM=4/2/1` and `num_warps=2/4/8` | pass | `4.80%` | rejected | Small-width gains were not enough to offset broader regressions |
| 4 | Tuned per-size launch dispatch with exact benchmark-width configs and fallback heuristics | pass | `5.71%` | accepted | Current best; checkpoint copied to `fused_softmax_bak.py` |
