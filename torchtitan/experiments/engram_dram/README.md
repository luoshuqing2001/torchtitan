# Engram with asynchronous DRAM storage

This experiment keeps the Engram table, projections, gradients and FP32 AdamW
state in CPU DRAM. The table and its optimizer state are partitioned by row
across the data-parallel ranks. Projection parameters remain replicated in
DRAM. The d200m backbone and its Moonlight Muon optimizer are unchanged.

`ENGRAM_ASYNC_TRAINING=0` is the default. It returns the ordinary Engram recipe
with device-resident parameters. Set `PLE=engram` to select the architecture;
`PLE_ASYNC_TRAINING` remains off for this experiment.

```bash
export NGPU=4 FLAVOR=d200m PLE=engram PLE_DIM=1024 FP8=1
export LOCAL_BATCH_SIZE=128 TRAIN_STEPS=100 OMP_NUM_THREADS=16
export ENGRAM_ASYNC_TRAINING=1
.venv/bin/torchrun --standalone --nproc-per-node=4 --numa-binding=exclusive \
  -m torchtitan.train --module torchtitan.experiments.engram_dram.recipe \
  --config glimmer_dense_dp
```

This recipe uses the regular data, scheduler and checkpoint settings and
defaults to pure data parallelism. The study also tests a sharded backbone;
prepared GPU caching requires `fsdp_reshard_after_forward=never` in that mode.
Checkpoint hooks expose the table
and Adam moments as global CPU DTensors; runtime parameters are local tensors.

The optional optimizations below are independent and default to zero. They
are experimental; enable only combinations validated for the intended shape.

| Environment variable | Behavior |
| --- | --- |
| `ENGRAM_DRAM_COMPACT_INDICES` | Exchange two hash families and expand the sixteen translated heads exactly. Falls back for overflowing token IDs or unsupported rank counts. |
| `ENGRAM_DRAM_TENSOR_EXCHANGE` | Exchange family lengths and packed int64 payloads with Gloo tensor collectives, avoiding Python object serialization. Requires compact indices. |
| `ENGRAM_DRAM_BOUNDED_UNIQUE` | Use an exact CPU lookup-based unique operation for the bounded hash domain. Requires compact indices and the AVX512 extension used by fused updates. |
| `ENGRAM_DRAM_GPU_INDICES` | Transfer two inverse arrays and construct the sixteen-head inverse on the copy stream. Requires compact indices. |
| `ENGRAM_DRAM_EARLY_FETCH` | Fetch the consumed current batch before vocabulary statistics. Does not advance the next batch. |
| `ENGRAM_DRAM_CACHE_STAGES` | Stage projection weights once per consumed batch. Always refresh after CPU AdamW, which does not bump native parameter versions. |
| `ENGRAM_DRAM_FUSED_GATHER` | Gather and cast CPU rows into the pinned transfer buffer in one compiled operation. |
| `ENGRAM_DRAM_OWNER_ROUTING` | Route FP32 gradients to their row owner in original rank order. |
| `ENGRAM_DRAM_PREPARE_ROUTING` | Prepare backward routing during forward. Requires compact indices and owner routing. |
| `ENGRAM_DRAM_DEFER_GRADIENT` | Complete CPU gradients in the same backward task's final callback. |
| `ENGRAM_DRAM_EARLY_GRADIENT` | Allocate the dense CPU gradient during forward. Requires prepared routing and deferred gradients. |
| `ENGRAM_DRAM_DIRECT_GRADIENT` | Transfer gradients directly into their dense CPU destination. Requires deferred gradients; narrow Engram rows can make this slower. |
| `ENGRAM_DRAM_GPU_NORM` | Compute CPU-equivalent chunk summaries on the GPU. Requires direct gradients. |
| `ENGRAM_DRAM_CPU_OVERLAP` | Overlap CPU AdamW with the backbone's GPU optimizer and join within the same step. |
| `ENGRAM_DRAM_FUSED_UPDATE` | Fuse CPU clipping with native-arithmetic dense AdamW. Requires optimizer overlap and AVX512. |
| `ENGRAM_DRAM_PREPARE_UPDATE` | Double-buffer the CPU table update. Requires early compact fetch, deferred gradients and fused updates. |
| `ENGRAM_COMPILE_FP8` | Compile the existing FP8 quantization region. Independent of DRAM storage; enable identically for DP comparisons. |

`ENGRAM_DRAM_CPU_THREADS` sets the scoped fused-update thread count; it requires
fused updates. `ENGRAM_DRAM_FETCH_THREADS` independently scopes CPU fetch work.
Both default to zero, using the existing thread configuration.

## Optional single-buffer scheduling

The following switches also default to zero and require asynchronous DRAM
training. They do not change the ordinary GPU recipe.

| Environment variable | Behavior |
| --- | --- |
| `ENGRAM_DRAM_SINGLE_BUFFER_CACHE` | Reuse one full BF16 table buffer after the current lookup's CUDA last-reader event. Requires the existing double-buffer preparation schedule, workspace reuse and early fetch. Supports one training lookup per optimizer step; additional microbatches are rejected. |
| `ENGRAM_DRAM_CACHE_PATCH_CHUNK_ROWS` | Rows per owner in the correction workspace; positive integer, default 262144. Corrections are gathered and scattered in chunks without a second full table. |
| `ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE` | Allocate the patch workspace and upload patch indices only at the late correction phase. Requires single buffering. |
| `ENGRAM_DRAM_DEFER_CPU_JOIN` | Overlap the CPU-only optimizer tail with metric processing. Wait for cache communication submission first, then join CPU work and advance the LR scheduler before the trainer step returns. Requires early fetch and fused CPU updates. |

For single buffering, `DOUBLE_BUFFER_CACHE` enables the existing preparation
protocol; it does not allocate two table buffers when `SINGLE_BUFFER_CACHE=1`.
The BF16 cache is invalid while it holds an uncommitted candidate. A failed
step must refresh it from the authoritative DRAM master. Evaluation suppresses
candidate overwrites, and trainer cleanup cancels a blocked preload before
draining workers. The scheme still performs dense AdamW updates for rows
without current gradients; it does not reuse stale values based on index overlap.

Prepared cache collectives use an ordered window. A preload waits for decoder
forward and completes before the final chunked-loss gradient synchronization.
CPU table arithmetic overlaps the GPU optimizer, but correction communication
waits for the GPU optimizer completion event. Metric communication then waits
for cache readiness. This prevents background cache NCCL from racing Muon's
all-to-all or FSDP collectives on another communicator. The schedule requires
one microbatch per optimizer step and chunked loss with at least two chunks.
These are correctness dependencies; no timeout is extended to hide a race.

Prepared updates compute the zero-gradient next state in scratch DRAM during
forward. At the update boundary, requested rows receive the actual clipped
AdamW update into that scratch state. The parameter and moment tensor objects
retain their identity while their DRAM storage rotates. Applications must not
retain storage views across updates when enabling this option. Scratch state
uses about four additional table-shard sizes of host memory, and is excluded
from checkpoints. Every row still receives every AdamW step, including momentum
and weight decay. No stale weights, sparse Adam or skipped updates are used.
Preparation is discarded if parameter/state versions, learning rate or gradient
coverage change. Native optimizer hooks use the ordinary update path. Failed
steps join preparation without committing scratch state.

The controlled performance harness, source snapshots and numerical comparisons
are under `outputs/engram_dram_round1`. Compare each model to its own ordinary
GPU DP implementation with equal data, batch, precision, scheduler and backbone.
Report the mean per-step maximum over all ranks for steps 6-100 and 51-100;
report memory maxima over all ranks and steps, including warmup. CPU versus GPU
AdamW can differ in rounding; internal DRAM optimizations must preserve the
complete DRAM loss and gradient-norm trajectories bit for bit.

## Follow-up profiles

Starting from the validated profile below, the low-latency follow-up uses:

```bash
export ENGRAM_DRAM_SINGLE_BUFFER_CACHE=1
export ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE=1
export ENGRAM_DRAM_DEFER_CPU_JOIN=1
```

The lower-active-memory alternative fetches current rows and drops the full
GPU table cache. It has a separate latency measurement:

```bash
export ENGRAM_DRAM_GPU_TABLE_CACHE=0 ENGRAM_DRAM_DOUBLE_BUFFER_CACHE=0
export ENGRAM_DRAM_REUSE_CACHE_WORKSPACE=0 ENGRAM_DRAM_SINGLE_BUFFER_CACHE=0
export ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE=0 ENGRAM_DRAM_DEFER_CPU_JOIN=1
```

These are incremental flags, not complete launch environments. Full frozen
launch configurations, source snapshots, and exact numerical checks are in
`outputs/mlsys2027_study`. The follow-up study additionally applies backbone
FSDP through its study converter; that is distinct from the standalone pure-DP
recipe. Figure 2 identifies the backbone placement for each configuration.
Report active and reserved allocator counters separately; neither measures
total device usage. The two follow-up profiles are a memory/latency tradeoff.

## First-round validated d200m profile

The selected profile keeps FP32 master parameters and AdamW state in DRAM.
It retains two BF16 table lookup buffers on the GPU. The next zero-gradient
state is prepared during forward; only the exact active-row corrections need
GPU patching after the CPU update. Both buffers count toward reported GPU memory.
Streaming CPU stores reduce host write traffic. GPU scatter accepts runtime
row counts, avoiding one Triton recompilation per changing batch.

This is a hardware-specific measured configuration, not a universal speedup.
The prepared path needs AVX512; CPU memory bandwidth, affinity, batch size and
GPU interconnect affect whether the update is hidden.

Additional switches, all disabled by default:

| Environment variable | Behavior |
| --- | --- |
| `ENGRAM_DRAM_GPU_TABLE_CACHE` | Retain a BF16 GPU lookup copy with authoritative FP32 master weights in DRAM. |
| `ENGRAM_DRAM_DOUBLE_BUFFER_CACHE` | Prepare the next lookup copy during forward and patch exact active rows after update. |
| `ENGRAM_DRAM_ASYNC_CLIP` | Transfer the clipping coefficient asynchronously and clip CPU gradients in the update worker. |
| `ENGRAM_DRAM_STREAMING_UPDATE` | Use non-temporal CPU stores with a per-thread fence before consumers read results. |
| `ENGRAM_DRAM_GPU_PROJECTIONS` | Optionally keep projection parameters on GPU; disabled in the selected profile. |

Use the following environment with the launch command above:

```bash
export ENGRAM_ASYNC_TRAINING=1
export ENGRAM_COMPILE_FP8=1
export ENGRAM_DRAM_ASYNC_CLIP=1
export ENGRAM_DRAM_BOUNDED_UNIQUE=1
export ENGRAM_DRAM_CACHE_STAGES=1
export ENGRAM_DRAM_COMPACT_INDICES=1
export ENGRAM_DRAM_CPU_OVERLAP=1
export ENGRAM_DRAM_CPU_THREADS=26
export ENGRAM_DRAM_DEFER_GRADIENT=1
export ENGRAM_DRAM_DOUBLE_BUFFER_CACHE=1
export ENGRAM_DRAM_EARLY_FETCH=1
export ENGRAM_DRAM_EARLY_GRADIENT=1
export ENGRAM_DRAM_FETCH_THREADS=8
export ENGRAM_DRAM_FUSED_GATHER=1
export ENGRAM_DRAM_FUSED_UPDATE=1
export ENGRAM_DRAM_GPU_INDICES=1
export ENGRAM_DRAM_GPU_TABLE_CACHE=1
export ENGRAM_DRAM_OWNER_ROUTING=1
export ENGRAM_DRAM_PREPARE_ROUTING=1
export ENGRAM_DRAM_PREPARE_UPDATE=1
export ENGRAM_DRAM_STREAMING_UPDATE=1
export ENGRAM_DRAM_TENSOR_EXCHANGE=1
```

## Controlled results

Four H100 GPUs, d200m, local batch 128, sequence length 2048, seed 42,
strict determinism, 100 steps on Climbmix. The backbone uses Moonlight Muon;
PLE/Engram use AdamW. Each baseline is its corresponding ordinary GPU DP
implementation, with identical settings and common FP8 compilation.
Baseline implementations were not optimized for this comparison.

| Variant | DP seconds/step | DRAM seconds/step | Difference | DP active GiB | DRAM active GiB | Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PLE | 2.9350 | 2.9420 | +0.24% | 55.21 | 44.69 | 19.1% |
| Engram | 2.9292 | 2.9539 | +0.84% | 61.99 | 50.76 | 18.1% |

Latency averages cover steps 6-100, taking the slowest rank at each step.
The differences for steps 51-100 are +0.43% and +0.86%, respectively.
Memory is the maximum active allocation across all ranks and all steps.
Maximum reserved memory changes from 61.45 to 47.25 GiB for PLE and from
67.89 to 53.05 GiB for Engram. With a 0.5% latency tolerance and 15% active
memory reduction requirement, PLE meets both targets; Engram narrowly misses
the latency target. These are single controlled runs, not confidence intervals.

Both selected DRAM configurations reproduce the complete 100-step loss and
grad_norm trajectories of their own pre-optimization DRAM references bitwise.
CPU and GPU optimizer arithmetic can round differently: this is not a claim
of bitwise equality between DRAM and GPU DP training.

Machine-readable results and numerical checks:
`outputs/engram_dram_round1/final_d200m_comparison.json`.
Exact per-case environments and source checksums are in
`outputs/engram_dram_round1/case_sources/` and
`outputs/ple_async_round12/case_sources/`.
The selected Engram case is `dram_runtime_scatter_b128_100`; the selected PLE
case is `dram_shard_b128_engram_ple_clean100`.

## Further activation and workspace reductions

`ENGRAM_DRAM_REUSE_CACHE_WORKSPACE=1` reuses the consumed lookup table for the
post-update patch collective, and uses the receiving table's own rank slice
for full-cache collectives. It requires double buffering. The old lookup has
no backward readers when the optimizer reaches this point. The next lookup
remains separate until its patches are complete.

`ENGRAM_DRAM_PREFETCH_VALUE=1` replaces the saved wide value activation with
its small projection inputs and stages recomputation on a separate CUDA
stream during lower decoder backward. PyTorch's original linear backward
still computes all gradients. Saved input versions are checked, original
activation storage is released after forward, and the trainer joins outstanding
work on a failed step. Inference takes the ordinary projection path.

`ENGRAM_DRAM_ALIGN_VALUE=1` additionally pads a four-aligned, non-eight-aligned
value width by four rows **only during recomputation**, then restores the
original shape. The original forward and its gradient graph are unchanged.
Other widths use ordinary recomputation. This is opt-in and requires BF16
prefetched values. GEMM padding can change rounding for other shapes; the
reported d200m configuration needs its own full-trajectory equivalence check.
No generic bitwise-equivalence claim is made for arbitrary configurations.

`ENGRAM_DRAM_CHECKPOINT_VALUE=1` is a simpler alternative using PyTorch
non-reentrant checkpointing for the value projections. It saves the wide
activation but adds exposed recomputation. It cannot be combined with
prefetched recomputation or the earlier value-offload prototype. Activation
DMA and ordinary recomputation were screened separately; neither should be
selected solely from its memory saving without checking latency.

All these switches default to zero. They do not alter the ordinary GPU DP
implementation, model dimensions, optimizer hyperparameters or data order.

`ENGRAM_DRAM_ALIGN_VALUE_FORWARD=1` also aligns the value-up forward GEMM in
the DRAM variant. It requires aligned prefetched values. Padding the ordinary
linear's weight and relying on automatic padding backward changes the backward
reduction dimensions and can change BF16 results. `AlignedValueLinear` therefore
uses temporary padding only for forward storage, while backward uses the
original input and weight dimensions. Model and optimizer tensors keep their
original shapes; the GPU DP implementation is untouched. Both input and weight
gradients, and second derivatives, have dedicated tests.

To replay the lower-memory profile with the strict controlled benchmark
configuration (including seed 42 and deterministic mode), use a new case name:

```bash
ENGRAM_DRAM_PREFETCH_VALUE=1 ENGRAM_DRAM_ALIGN_VALUE=1 \
ENGRAM_DRAM_ALIGN_VALUE_FORWARD=1 \
.venv/bin/python outputs/engram_dram_round1/run_workspace.py \
  my_engram_lower_memory_b128_100 100
```

This local harness fixes batch 128 on four GPUs and loads the validated DRAM
transport flags, including workspace reuse. It rejects existing case names and
occupied GPUs. The direct recipe command at the top uses normal recipe debug,
validation and checkpoint defaults instead of the controlled harness overrides.
For other model shapes, batches or hardware, repeat numerical and performance
validation before selecting alignment flags.

## Selected lower-memory results

The completed 100-step profiles use the same unchanged GPU DP controls,
four H100 GPUs, d200m, local batch 128, sequence 2048, real Climbmix data,
seed 42 and strict determinism as above. Both use Moonlight Muon for the
backbone and AdamW for the auxiliary parameters.

| Variant | DP seconds/step | Selected DRAM seconds/step | Mean difference | Late difference | DP active GiB | DRAM active GiB | Saving vs DP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PLE | 2.935001 | 2.946211 | +0.3819% | +0.3392% | 55.2114 | 38.6508 | 29.99% |
| Engram | 2.929214 | 2.930016 | +0.0274% | +0.2701% | 61.9919 | 41.9209 | 32.38% |

Mean latency covers steps 6-100; late latency covers steps 51-100. Each step
uses the maximum across ranks. Memory peaks include every rank and all steps,
including warmup. Selected reserved peaks are 41.2363 GiB (PLE) and
43.1309 GiB (Engram), versus DP's 61.4473 and 67.8945 GiB.

Compared with the previous selected DRAM profiles, active peaks drop another
6.0352 GiB for PLE (44.6860 -> 38.6508, -13.51%) and 8.8364 GiB for Engram
(50.7573 -> 41.9209, -17.41%). Both selected profiles meet the 0.5% tolerance
for mean and late latency. These measurements establish parity on this setup;
they do not establish a stable speedup or a bound for other configurations.

The selected PLE case is `dram_shard_b128_memory_context_async100`; its replay
command is in the PLE DRAM experiment README. The selected Engram case is
`dram_aligned_forward_b128_100`; its replay command is directly above.
`dram_workspace_aligned_release_b128_100` is an earlier ablation that saves
similar memory but misses the latency target and is not selected.

Both selected 100-step runs reproduce their own pre-optimization DRAM
reference's complete loss, global maximum loss and grad_norm bitwise on all
four ranks. Dedicated real-shape GPU checks also compare the value projection's
output and every input/parameter gradient exactly. Ordinary GPU-versus-CPU
optimizer rounding remains outside this bitwise-equivalence claim.

`outputs/engram_dram_round1/compare_memory.py` regenerates
`outputs/engram_dram_round1/memory_comparison.json`, including the discarded
ablations and both performance/numerical gates. Each final case has a frozen
source manifest and launch environment in `case_sources/`.
