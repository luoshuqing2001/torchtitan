# Opt-in Engram GPU optimizations

This experiment optimizes the existing GPU-resident Engram variant. It does
not enable DRAM Engram, reuse stale rows, change hash functions, select sparse
Adam, or change optimizer algorithms. The backbone still uses Moonlight Muon;
the Engram table and update parameters keep their existing AdamW routing.
Parameter names and model/optimizer checkpoint layouts are unchanged.

All four switches default to `0`:

| Switch | Behavior |
| --- | --- |
| `ENGRAM_FUSED_LOOKUP=1` | Fuse exact int64 n-gram hashing and GPU table lookup. Keep native dense embedding backward, including strict deterministic behavior. |
| `ENGRAM_FUSED_CONV=1` | Fuse segmented short-convolution forward and input gradients, preserving per-tap activation-dtype rounding and accumulation order. Keep native weight-gradient reduction shapes and kernels. |
| `ENGRAM_UPDATE_CHECKPOINT=1` | Checkpoint only the Engram update with PyTorch non-reentrant checkpointing. Table lookup remains outside. This trades projection recomputation for lower activation memory. |
| `ENGRAM_COMPILE_FP8=1` | Enable the existing `float8_quantize` local compile region. Enable it equally in baseline comparisons. |

The lookup falls back to the original table for CPU, noncontiguous token or
weight tensors, and token dtypes other than int32/int64. The convolution fast
path supports contiguous CUDA BF16/FP32 values with matching weight dtype;
other layouts/dtypes use the original function. Higher-order convolution
gradients use native autograd. Document positions and EOS segmentation, the
sixteen head salts, integer overflow/remainder behavior, the table remainder
parameter, and per-tap BF16 rounding are preserved. CUDA kernels are imported
only when a CUDA optimization is actually used.

## Run

```bash
export NGPU=4 FLAVOR=d200m PLE=engram PLE_DIM=1024 FP8=1
export PLE_ASYNC_TRAINING=0 PLE_UNBIND_INPUTS=0
export LOCAL_BATCH_SIZE=128 TRAIN_STEPS=100 LOG_FREQ=1
export ENGRAM_FUSED_LOOKUP=1 ENGRAM_FUSED_CONV=1 ENGRAM_COMPILE_FP8=1
export ENGRAM_UPDATE_CHECKPOINT=0
export DUMP_FOLDER=./outputs/my_engram_fused
.venv/bin/torchrun --standalone --nproc-per-node=4 --numa-binding=exclusive \
  -m torchtitan.train \
  --module torchtitan.experiments.engram_optim.recipe --config glimmer_dense_dp
```

`glimmer_dense_dp` selects the replicated DP backbone used in the measurements.
`glimmer_dense` preserves the base recipe's parallelism. The original
`PLE_ASYNC_TRAINING=1` remains specific to PLE and is not accepted for Engram.
The usual recipe retains validation/checkpointing settings. Dedicated timing
runs below disable both, save TensorBoard metrics on all ranks, and use fresh
seed-42 initialization with strict deterministic CUDA FlashAttention.

## Validation and measurement

The final targeted suite passed 63 CPU/GPU tests. It covers exact forward and
first-order gradients, CPU native fallback, BF16/FP32, non-power-of-two head
widths, int64 hash overflow, negative token IDs, EOS/document boundaries,
noncontiguous tokens, frozen inputs/weights, checkpoint state reload, aliased
mutation detection and native higher-order gradients. CPU AdamW tests compare
eight complete model/optimizer updates and include state reload.

Four H100 80GB GPUs, d200m, PLE dimension 1024, real Climbmix and sequence 2048
were used for the initial 30-step trials. All implementations use identical
FP8 quantization compilation and the same optimizer/parallelism settings.
Latency averages the slowest rank each step, reporting steps 6--30; memory
peaks include all ranks and all steps, including warmup.

| Configuration | Local batch | Seconds/step | Peak active GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Original Engram | 64 | 1.562275 | 41.1787 | 47.3066 |
| Convolution prototype | 64 | 1.556830 | 40.4293 | 46.5664 |
| Original Engram | 128 | 2.934714 | 61.9919 | 67.8945 |
| Lookup + convolution prototype | 128 | 2.907310 | 60.4931 | 66.3887 |
| Above + update-checkpoint prototype | 128 | 2.988681 | 49.7387 | 55.6270 |

Each optimized 30-step trial matches its same-batch baseline's entire loss
and gradient-norm trajectory bitwise on all four ranks. Prototypes remain
separate from the production profiles. Reproduction scripts, exact launch
settings and source snapshots are in `outputs/engram_round1`.

## Scope and remaining opportunities

Fusing memory-bound operations, owner-routed communication and overlapping
independent work are reusable principles. PLE's root-context checkpoint scope,
its wide-row transport schedule, and a host-specific CPU thread count do not
transfer unchanged to Engram. Engram has sixteen hash heads, narrow table rows
and one wide value bottleneck rather than one PLE slice per decoder layer.
Measured performance remains specific to the hardware, batch and dimensions
above; fallback correctness does not establish performance on other devices.

Further candidates include sharing sort/group metadata across translated
bigram heads and across translated trigram heads for valid model-token ranges,
without changing per-row gradient reductions; fusing convolution weight-gradient temporaries
while preserving the native reduction order, coordinating repeated segmentation
work, and a dedicated DRAM owner-routing path for Engram's narrow hash rows.
The latter needs separate index-processing/communication measurements and
numerical validation. Consecutive index overlap alone does not permit reusing
previous-step values after dense AdamW updates. None of these remaining
candidates is implied to have a demonstrated speedup by the current flags.

## Completed production profiles

Fresh paired 100-step replicated-DP runs use the exact settings above. The
baseline disables lookup/convolution/update-checkpoint flags; common FP8
quantization compilation stays enabled in both. Latency averages steps 6--100,
with steps 51--100 reported separately. Memory includes warmup and all ranks.

| DP configuration | Seconds, 6--100 | Seconds, 51--100 | Active GiB | Reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.923925 | 2.924932 | 61.9919 | 67.8945 |
| Fused lookup + convolution | 2.911953 | 2.907568 | 60.4931 | 66.3887 |

The fused configuration is 0.4095% faster on the full measured interval and
0.5937% faster in the last 50 steps, while saving 1.4988 GiB (2.4177%) of active
GPU memory. This is a small measured gain, not a general speedup guarantee.
Every loss and gradient-norm value over all 100 steps matches bitwise on all
four ranks. Parameters use fresh seed-42 initialization; these are not the
PLE runs' initial checkpoints.

Separate 30-step production profiles validate the regular recipe's FSDP
parallelism and the optional replicated-DP update checkpoint:

| Configuration, 30 steps | Mean seconds, 6--30 | Active GiB | Reserved GiB |
| --- | ---: | ---: | ---: |
| FSDP baseline | 2.885827 | 49.4249 | 54.1172 |
| FSDP fused lookup + convolution | 2.882852 | 47.9262 | 52.7480 |
| Replicated DP fused + update checkpoint | 2.978557 | 49.7387 | 55.6270 |

The FSDP fusion pair saves another 1.4988 GiB; its 0.1031% latency difference
should be treated as approximate parity, not evidence of a stable speedup.
Its full 30-step trajectories match bitwise. The production update-checkpoint
profile also matches the original DP 30-step reference bitwise, saving 19.77%
active memory at 1.49% higher mean latency (reference 2.934714 seconds,
61.9919 GiB). Parallelism changes and different step schedules are kept
separate from the fusion comparisons. No cross-DP/FSDP numerical equivalence
is asserted.

An additional higher-order-plus-hook audit exposed duplicate leaf-hook calls
in the native higher-order fallback. Taking the nested gradient at input views
fixes this while preserving higher-order graphs. The final 63-test suite
includes both input/weight hook cases. This fix affects the higher-order branch;
the measured ordinary first-order training kernels and their reductions are
unchanged. FSDP profiles include the final source.

`outputs/engram_round1/comparison.json` records the complete measurements and
numerical gates. Regenerate it with `.venv/bin/python outputs/engram_round1/compare.py`. Benchmark runs are serialized under the same
GPU lock as the PLE harness, refuse occupied GPUs and existing output folders,
and retain source hashes, launch settings, logs and TensorBoard metrics.
