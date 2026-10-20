# CPU row-sharded PLE

This opt-in experiment partitions the DRAM PLE table and its FP32 AdamW state
by vocabulary row across the entire data-parallel world. Other PLE parameters
remain replicated in CPU memory when `PLE_ASYNC_TRAINING=1` is set.
The switch defaults to `0` in both this recipe and the regular Glimmer recipe:
PLE then uses ordinary device-resident parameters, native autograd and FSDP,
without DRAM fetch workers or copy streams. It does not disable PLE itself.

Set `PLE_ASYNC_TRAINING=1` with this recipe to select CPU row ownership and
asynchronous fetch. With the regular Glimmer recipe, the same flag selects
the original replicated DRAM implementation. Python configs expose the switch
as `config.model.ple_async_training`, defaulting to `False`.

The request worker exchanges unique token IDs through a separate Gloo group,
gathers only the rows owned by this rank, packs them in the compute dtype, and
stages rows and index metadata on a CUDA copy stream. The caller broadcasts
owned rows over NCCL and selects directly into its own lookup rows, without
allocating a per-rank GPU matrix for the global union. Checkpoint recomputation
reuses these current-step rows, with stream events protecting their lifetime.
The next training step starts a fresh fetch after optimizer updates complete.

Backward accumulates gradients in the original rank order in FP32, but retains
and transfers only the global union of rows owned by this rank. The resulting
CPU gradient is dense within the owned shard. Native fused AdamW updates every
owned row, including zero-gradient rows with momentum or weight decay. There
is no sparse-Adam substitution or delayed parameter update.

Gradient clipping gathers the existing CPU L2 chunk summaries in global row
order. A replicated summary participates in the ordinary mixed CPU/FSDP norm
calculation; the resulting global coefficient also clips the owned table.
Shard boundaries must align with norm chunks so sharding preserves the CPU
reduction order. Model and Adam checkpoint hooks expose globally shaped CPU
DTensors with row placements while training tensors remain ordinary local
CPU tensors. Native DCP can read the global layout rather than treating local
shards as replicated parameters.

## Run

Use the same environment and backbone settings as the regular local recipe:

```bash
export NGPU=4 FLAVOR=d200m PLE=ple PLE_DIM=1024 FP8=1
export PLE_ASYNC_TRAINING=1
export LOCAL_BATCH_SIZE=64 TRAIN_STEPS=100 LOG_FREQ=10
export PLE_CHECKPOINT=1 PLE_CPU_FUSED_ADAM=1
export PLE_MIXED_PRECISION_BACKWARD=1 PLE_RECOMPUTE_LINEAR=1
export CPU_NORM_CHUNK_SIZE=65536 OMP_NUM_THREADS=16
export DUMP_FOLDER=./outputs/ple_dram_row_shard
.venv/bin/torchrun --standalone --nproc-per-node=4 --numa-binding=exclusive \
  -m torchtitan.train \
  --module torchtitan.experiments.ple_dram_shard.recipe --config glimmer_dense
```

This uses the regular recipe's dataset, checkpoint and scheduler configuration.

## Additional opt-in scheduling experiments

All of these switches default to zero and require `PLE_ASYNC_TRAINING=1`.

| Environment variable | Behavior |
| --- | --- |
| `PLE_PREPARE_UPDATE` | Prepare the next dense zero-gradient AdamW state in scratch DRAM, then replace accessed rows using actual clipped gradients. Requires early fetch, deferred table gradients and fused CPU updates. |
| `PLE_PREPARE_ROUTING` | Build the ordered backward routing plan during fetch. Requires owner routing. |
| `PLE_EARLY_GRADIENT` | Allocate the dense owned gradient before backward. Requires prepared routing and deferred gradients. |
| `PLE_STREAMING_UPDATE` | Use fenced non-temporal stores for prepared updates. Requires prepared updates. |
| `PLE_DEFER_CPU_JOIN` | Join CPU updates after metric processing but before the trainer step returns; advance the LR scheduler only after that join. Requires early fetch and fused CPU updates. |
| `PLE_REUSE_CONTEXT_NORM_INPUT` | Reuse the sole-use scaled context projection as RMSNorm's input-gradient output. Uses the identical native kernel and reduction grid. Requires context checkpointing; the validated path is the installed Hopper native backend at width 1024. Other paths and retained graphs use the original backward. |
| `PLE_OFFLOAD_SAVED_INPUTS` | Store checkpointed per-layer PLE inputs in pinned DRAM and prefetch one layer ahead in backward. Requires saved matmuls and early fetch; supports one microbatch per step. |
| `PLE_COMPACT_SAVED_INPUTS` | Store identical-token rows once per layer and restore the complete saved input before backward. Requires saved-input staging. Does not deduplicate GEMMs or gradient accumulation and never reuses stale values across steps. |
| `PLE_AUDIT_COMPACT_INPUTS` | Compare every restored byte with the original saved input. Requires compact inputs; intended for correctness diagnostics, not latency measurement. |

Preparation consumes additional host scratch and is not assumed to improve
end-to-end time. The study evaluates it separately from deferred joining.
The selected saved-input path is implemented in `saved_inputs.py`. Its staging
manager belongs to the model; it does not patch global checkpoint functions.
The study retains the earlier prototype for reproducibility. Native context
normalization reuse did not lower the whole-step peak in this profile and is
not required by the selected compact-input configuration.
`PLE_COMPILE_FP8=1` additionally enables the existing local compiler for FP8
dynamic quantization. It fuses GPU scaling/casting operations and reduces their
temporary buffers; table ownership and CPU AdamW updates are unchanged. The
flag defaults to zero. Initial compilation adds startup time.
It is independent of `PLE_ASYNC_TRAINING`; disabling asynchronous training
does not implicitly enable or disable FP8 quantization compilation.

`PLE_UNBIND_INPUTS=1` optionally routes the per-layer inputs through one
`unbind` operation. Its backward stacks layer gradients once, avoiding the
full token-by-all-layers zero-fill and gradient additions of independent
`[:, layer]` slices. This works with both ordinary GPU PLE and DRAM PLE;
weights, AdamW updates and checkpoint layouts are unchanged. It defaults to
zero (`config.model.ple_unbind_inputs=False`). Exact routing/optimizer tests
pass, but strict full-training numerical equivalence remains unresolved,
including nondeterminism in unchanged baseline repetitions. This option is
experimental and does not enable asynchronous training by itself.

The controlled large-batch comparison and reproduction scripts are in
`outputs/ple_large_batch`; both implementations use the same compilation
settings and load the same initial model checkpoint.

`PLE_OVERLAP_CPU_OPTIMIZER=1` selects an experimental optimizer container
that overlaps CPU AdamW with the GPU Moonlight Muon/AdamW updates. It requires
`PLE_ASYNC_TRAINING=1` and separate CPU/device optimizers (the run example's
`PLE_CPU_FUSED_ADAM=1` provides that separation). Each optimizer must contain
parameters on one device. CPU optimizers must be native AdamW; each rank
supports one CUDA device. The flag defaults to zero, including in this recipe.

Clipping still completes first. A CUDA event gates the CPU update behind the
preceding stream work and finite checks. The container reuses the idle fetch
worker, updates disjoint CPU/device parameters in parallel, and joins the CPU
future even if a device update raises. It returns only after CPU updates
complete, before learning-rate advancement, the next fetch, zero_grad or a
checkpoint. Optimizer algorithms, gradient accumulation and checkpoint tensor
layouts are unchanged. CUDA graph capture and closures are not supported.
Container hooks still run on the caller; CPU AdamW step hooks run on the worker.
Exact fixed-gradient update/state and bidirectional checkpoint tests pass;
strict full-training equivalence remains unresolved as described above, so
this option remains explicit and experimental. Timings and rejected alternatives
are recorded in `outputs/ple_async_round10`.
The controlled performance harness in `outputs/ple_async_round6` additionally
sets seed 42 and deterministic algorithms, logs every step, and disables
checkpointing/validation for both compared variants.
Historical asynchronous benchmark launchers also require the explicit
`PLE_ASYNC_TRAINING=1` setting with the current code.

## Current scope

- FP32 CPU master weights and native FP32 AdamW state.
- Pure data parallelism, with one model part per rank; no TP, CP, EP or PP.
- Vocabulary rows divide evenly across ranks; each local table contains whole
  CPU norm chunks, and the global norm summary fits within one such chunk.
- No EMA checkpoint support.
- Parameters and Adam states stay in DRAM; temporary row and gradient buffers
  use GPU memory. Forward routing needs additional temporary buffers, so check
  the peak for the intended batch size and PLE width.
- Initialization still materializes the replicated table before retaining its
  owned shard. The steady-state memory reduction does not apply to that initial
  allocation.

## Validation

`tests/unit_tests/cpu/test_ple_dram_shard.py` covers exact forward rows, ordered
embedding gradients, global clipping, native AdamW parameters and moments,
checkpoint resume and interoperability with replicated table checkpoints.
The CUDA test adds a sharded GPU DTensor to the norm and checks cached row
consumption on a second stream and mixed CPU/GPU DCP state.

```bash
.venv/bin/python -m pytest tests/unit_tests/cpu/test_ple_dram_shard.py -q
```

Four-GPU primitive comparisons and end-to-end measurements are recorded in
`outputs/ple_async_round6`. Full training reproducibility and long convergence
must be distinguished from primitive equality; this remains an experimental
recipe.

The real four-GPU d200m save/restart check also restored a full step-2 checkpoint
and continued through step 3. Cross-world-size resume, HF export and asynchronous
checkpoint modes have not been exercised in this experiment.

## Optional compiled FP8 quantization

Round 7 added the `PLE_COMPILE_FP8=1` option. On four H100s, d200m, batch
64/rank and sequence 2048, 50-step runs (steps 6-50 measured) changed latency
from 2.575 to 2.069 seconds/step and active peak from 24.745 to 21.862 GiB.
The equally compiled no-PLE baseline was 1.4516 seconds/step and 16.051 GiB;
near-zero PLE overhead is still not achieved. Compilation startup is excluded
from these steady-state measurements.

The adopted change uses the existing `float8_quantize` local compile region;
the all-to-all routing and deferred-gradient prototypes remain disabled.
Exact quantization bytes/scales and Float8Linear training tests pass, as does
a compiled four-GPU checkpoint save/restart. Full-system trajectories retain
the previously observed reproducibility limitation. Settings, profiles,
numerical checks and measurements are in `outputs/ple_async_round7/README.md`.

The subsequent full-model check in `outputs/ple_async_numerics8/README.md`
restored one complete checkpoint for three 50-update runs with identical
batches: eager reference, eager repetition, and compiled quantization.
Strict numerical equivalence failed. Maximum loss differences were 0.003026
for eager repetition and 0.002359 for compilation, but final all-parameter
relative L2 differences were 8.98% and 8.99%, respectively. The baseline's
own divergence prevents attributing these differences to quantization fusion;
similar error scales also do not prove equivalence. The option remains off
by default pending full-system reproducibility and convergence validation.

## Optional packed CPU row gather

`PLE_FUSED_ROW_GATHER=1` fuses CPU row indexing and dtype conversion directly
into the pinned transfer buffer using a local PyTorch Inductor region. It
requires `PLE_ASYNC_TRAINING=1` in this row-sharded recipe and defaults to zero.
The previous path materializes gathered FP32 rows, a converted CPU tensor and
then a pinned copy. The fused path avoids these intermediate allocations and
DRAM copies; it still transfers all current requested row values to the GPU.
Compilation startup is separate from steady-state training time. Empty and
single-row requests use native operators to avoid zero/one specialization
consuming compilation-cache entries.

No weights are reused across optimizer updates. Even a row absent from the
current batch can change through dense AdamW momentum and weight decay, so
repeated token IDs do not establish that a cached value remains current.
Optimizer state, row ownership, gradient reductions, clipping and checkpoints
are unchanged. The switch composes with `PLE_OVERLAP_CPU_OPTIMIZER=1`.
CPU/CUDA row-shard training and checkpoint comparisons cover the packed path
with exact outputs, gradients, parameters and Adam moments. Full-model
reproducibility retains the unresolved limitation documented above.

The controlled round-11 tests (`outputs/ple_async_round11/README.md`) measured
30 updates per timing run, reporting steps 6-30, on four H100s with FP8
compilation and unbind enabled for every implementation. With optimizer overlap
already enabled, packed gather changed batch 64/rank from 1.951 to 1.821 s/step
and batch 128/rank from 3.370 to 3.224 s/step. A second batch-128 pair reproduced
3.370 to 3.222 s/step. GPU active peaks stayed at 19.68 and 37.26 GiB/rank.
Equally compiled fully replicated GPU DP with PLE took 1.585 and 2.925 s/step,
so the packed DRAM path remains 14.9% and 10.2% slower, respectively.

A separate 15-update live-data check found exact packed values in all 60
four-rank fetches (593728 rows checked). Consecutive global requests had 78.7%
index overlap, but none of the 436115 repeated owned rows retained identical
BF16 values after the intervening updates. This checked run used its own
15-update LR schedule; it does not establish cache reuse rates for other
schedules or datasets. It supports avoiding stale whole-row reuse here.

`PLE_FUSED_EMBEDDING=1` enables experimental CUDA kernels for BF16 lookup,
scaling, and context blending, with the original intermediate BF16 rounding.
The table backward sorts token positions once, then accumulates each row in
stable token order in FP32 without materializing FP32 token-width tiles.
Widths at most 1024, other dtypes/devices, and higher-order gradient paths use
the existing Torch implementation. Large element offsets use int64. This flag
requires asynchronous training and defaults to zero; CPU-only recipe imports
still do not import Triton.

`PLE_DEFER_TABLE_GRADIENT=1` optionally transfers and scatters the owned table
gradient while the remaining context projection backward runs. An autograd
completion callback accumulates the result before the same `backward()`
returns. Parameter gradient hooks, explicit-input backward/autograd.grad and
higher-order gradients use the synchronous path. A failed graph's data-only
futures are discarded on the next task. CPU chunk norm summaries can be
computed in the transfer worker and reused for clipping while their tensor
version and chunk size match. Gradient accumulation or ordinary in-place
mutation invalidates that cache. Optimizer updates still join before `step()`
returns; no update or row value is reused from an earlier training step.
This option also requires asynchronous training and defaults to zero.

With both options plus the earlier fused row gather and CPU optimizer overlap,
the round-12 four-GPU batch-128 run measured 3.066 s per step and 37.26 GiB
active memory per GPU, versus a refreshed fully replicated DP run at 2.929 s
and 55.21 GiB. Batch 64 measured 1.746 s and 19.68 GiB for DRAM PLE. These
measurements use 30 steps, reporting steps 6--30, real Climbmix, sequence 2048,
seed 42, deterministic algorithms, and matching FP8 compilation/input routing
settings. Exact isolated/update/checkpoint tests pass; the full-training
numerical limitation described above still applies. Further candidate tests,
rejected approaches, and reproducible measurements are recorded in
`outputs/ple_async_round12`; prototype-only changes there are not enabled by
these flags.

## Additional opt-in latency experiments

All switches below default to `0` and require `PLE_ASYNC_TRAINING=1`.
They do not select a different backbone optimizer; the benchmark still uses
Moonlight Muon for the backbone and AdamW for PLE.

| Switch | Behavior and requirements |
| --- | --- |
| `PLE_FUSED_EMBEDDING=1` | Fuse BF16 row lookup and PLE combination; use ordered FP32 row-gradient accumulation. |
| `PLE_DEFER_TABLE_GRADIENT=1` | Finish owned CPU gradients on the same backward task's final callback. Hooks, explicit gradient requests and higher-order backward use the synchronous path. |
| `PLE_DIRECT_GRADIENT=1` | Copy sorted row runs directly into the private pinned CPU gradient, avoiding a second CPU scatter. Requires CUDA 13 and deferred table gradients. |
| `PLE_GPU_NORM=1` | Compute CPU-order chunk summaries on a separate GPU stream beside direct gradient DMA. Requires direct gradient copies. A private-RNG probe selects a matching native CPU reduction; unsupported layouts warn and retain CPU summaries. |
| `PLE_FUSED_TABLE_UPDATE=1` | Fuse clipping and native-arithmetic CPU AdamW in one AVX512 pass. Requires CPU optimizer overlap. Native optimizer hooks and unsupported AdamW options fall back to native clipping/update. |
| `PLE_CPU_UPDATE_THREADS=N` | Scope an OpenMP thread count to the CPU table optimizer, including other parameters sharing its AdamW instance. Defaults to `0`, which inherits the worker setting. A positive count requires fused table updates and is restored after success or failure. |
| `PLE_SAVE_MATMULS=1` | Retain staged PLE linear outputs in selective activation checkpointing. Trades GPU activation memory for less recomputation. `PLE_SAVE_LAYERS` limits the retained layer prefix; `0` recomputes all layers while preserving the overlapped gradient path. |
| `PLE_CONTEXT_CHECKPOINT=1` | Checkpoint only the root context projection and normalization, leaving row lookup and blending outside. Requires replicated-gradient overlap. Defaults to `0`; use `PLE_CHECKPOINT=0` to avoid nesting the full-root checkpoint. |
| `PLE_OVERLAP_REPLICATED_GRADIENT=1` | Reduce replicated CPU PLE gradients on a separate CUDA stream and finish them before backward returns. Requires saved PLE matmuls. |
| `PLE_OWNER_ROUTING=1` | Send FP32 table gradients only to their row owner, merge in rank order in one kernel, and broadcast only initialized forward row payloads. Higher-order backward uses the original path. |
| `PLE_EARLY_FETCH=1` | Start the consumed current batch's fetch before CPU vocabulary accounting, using experiment-specific Trainer/model subclasses. Does not read ahead in the dataloader. |
| `PLE_RELEASE_FETCH_CACHE=1` | Release the consumed batch's fetch reference after passing it to the model. Requires early fetch; lowers the bounded current-step cache lifetime. |

The fused table update keeps native dense AdamW semantics. Rows with existing
momentum still update even when absent from the current batch. The zero-state
shortcut only omits moment arithmetic when both moment tensors are bitwise zero
and the row has not been requested. All rows still receive weight decay.
Restored or externally modified moment tensors invalidate that shortcut's proof.
CPU master parameters and optimizer state are complete before optimizer step
returns. CUDA batch copies explicitly advance gradient version counters so
cached norms cannot survive a gradient mutation unnoticed.

These remain experimental. Exact primitive/optimizer tests do not establish
bitwise equivalence of complete model training; full-model deterministic loss
comparison remains an outstanding validation requirement. See the round-12
artifacts under `outputs/ple_async_round12` for measured candidates and source
snapshots. Compare repeated runs of the same batch, checkpoint, precision,
parallelism and data; do not infer a latency match from one favorable step.

### Replicated backbone and latency measurements

`--config glimmer_dense_dp` selects a fully replicated GPU backbone using the
launched world size. With `PLE_ASYNC_TRAINING=0` (the default), PLE also stays on
GPU and uses fused AdamW. With `PLE_ASYNC_TRAINING=1`, only the CPU table is
partitioned by vocabulary row; the backbone is still replicated. Moonlight
Muon remains the backbone optimizer in both cases.

The round-12 batch-128 latency candidate enables the switches in the preceding
table, existing CPU optimizer overlap, fused CPU row gathering, FP8 quantization
compilation and unbound layer inputs. It retains nine PLE layer matmul pairs
(`PLE_SAVE_LAYERS=9`) and disables root PLE checkpointing (`PLE_CHECKPOINT=0`).
It targets four H100 GPUs, d200m, PLE width 1024 and sequence length 2048.
This profile trades activation memory for less recomputation and is not a
universal setting for other batch sizes or model shapes.

`EarlyFetchTrainer.Config.release_fetch_cache` optionally releases the request
from the consumed CPU batch after transferring it to the model. It defaults to
`False`, retaining the bounded current-step request until the step returns.
This does not reuse old weights in the following step. Controlled tests distinguish
this lifetime choice because it changes GPU allocation reuse and peak memory.
It does not change dataset consumption, parameter values or checkpoint layout.
Benchmark results with different lifetimes must not be pooled as identical runs.
The recipe exposes the lifetime choice through `PLE_RELEASE_FETCH_CACHE=1`,
which defaults to `0` and requires `PLE_EARLY_FETCH=1`.

### Lower-memory replicated-backbone profile

Retaining only one PLE layer's matmul pair and releasing the current fetch
reference substantially lowers memory compared with the nine-layer latency
profile. The remaining asynchronous transport/update switches stay enabled.
Root PLE checkpointing stays disabled. Both sides retain a replicated GPU
backbone, Moonlight Muon, the same model-only initial checkpoint, Climbmix,
FP8 settings, local batch 128, global batch 512 and sequence length 2048.

In the paired four-H100 100-step comparison, steps 6--100 measured:

| Profile | Seconds/step | Peak active GiB/GPU | Peak reserved GiB/GPU |
| --- | ---: | ---: | ---: |
| GPU DP including GPU PLE | 2.926935 | 55.2114 | 66.5254 |
| DRAM PLE, replicated backbone, one retained layer | 2.944625 | 45.6860 | 53.1289 |

The DRAM profile reduces active GPU memory by 17.25% with 0.60% greater
latency. Steps 51--100 measured 2.947094 versus 2.924784 seconds (+0.76%).
This is near DP latency, but does not meet a strict 0.5% latency threshold.
Memory peaks include warmup steps and all four ranks. CPU table parameters,
both Adam moments and the dense gradient require about 4.59 GiB per rank in
DRAM; GPU memory savings are not total-system memory savings.

Reproduce using unique tags and the existing local benchmark assets:

```bash
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dram --backbone dp --saved-layers 1 --release-fetch-cache \
  --batch 128 --steps 100 --tag my_low_memory_dram
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dp --batch 128 --steps 100 --tag my_low_memory_dp
```

Asynchronous training remains disabled by default. The 133-test regression
passed before exposing the fetch-lifetime environment switch; the subsequent
switch/checkpoint-policy regression passed 44 tests. Full-model bitwise
equivalence remains unresolved: even unchanged DP repeats diverge. Over the
paired 100-step runs, maximum loss difference was 0.019915 (0.3495% relative),
while unchanged DP long repeats differed by up to 0.016391. Gradient norms
also differ; these measurements do not prove numerical equivalence or
convergence. See `comparison_low_memory.json` and `numerics_low_memory.json`
in the round-12 output directory.

The alternative FSDP-backbone profile retains four PLE layer pairs and also
releases the fetch reference. Its 100-step run measured 2.940186 seconds,
46.7715 GiB active and 53.4707 GiB reserved: +0.4527% latency and -15.2865%
active memory versus the same GPU DP control. This meets the overall
0.5%/15% target, with a different backbone parallelism. Its late interval,
steps 51--100, measured 2.943915 seconds (+0.6541%), so the strict latency
threshold does not hold in every measurement interval. Use
`--backbone fsdp --saved-layers 4 --release-fetch-cache` with the DRAM
reproduction command to select this profile. A three-layer FSDP run used
46.3163 GiB but measured 2.943370 seconds (+0.5615%); it was not the faster
long-run choice.

A bounded persistent GPU Adam mirror was also tested while committing CPU
master parameters and moments before every step returned. Mirrors covering
25% and 12.5% of the owned table measured 2.991839 and 2.940987 seconds per
step over steps 6--30, respectively. Neither improved on the one-layer
non-mirror short run (2.936933 seconds). Their peak active memory was
46.8344 and 46.2602 GiB. These rejected candidates remain output-directory
prototypes, with per-case source snapshots; no production switch enables
them. Isolated mirror tests matched native CPU AdamW bitwise for three beta
pairs over 32 updates, including zero-gradient rows, parameter/moment
mutation and optimizer-state restore. Correct arithmetic alone does not
establish a performance benefit.

Moving row broadcasts into the current batch's CPU vocabulary-accounting
window was also tested. A separate NCCL group and a completion event ordered
worker-issued broadcasts before main-group collectives. Four-GPU BF16/FP32
fetch checks matched bitwise, including empty requests and changed weights.
Nevertheless, the 100-step replicated-backbone run measured 2.947636 seconds
and 45.6860 GiB active, versus 2.944625 seconds without background broadcasts.
This candidate was rejected and its provisional production switch/code was
removed. An earlier unjoined prototype was stopped and marked invalid; its
timing is excluded. The synchronized prototype and per-case snapshots remain
in the output directory for reproducibility.

### Device chunk norms

The output-directory `gpu_cpu_norm` prototype computes chunk norms from the
already merged GPU row gradients on a separate stream while the direct DMA
commits the unchanged dense gradient to CPU. Only the small summaries are
copied back. This strategy is integrated as `PLE_GPU_NORM=1`, defaulting to
zero and requiring `PLE_DIRECT_GRADIENT=1`. CPU cache version checks,
accumulation fallback and dense AdamW
updates are unchanged. On this build, reproducing the actual CPU norm requires
eight sequential FP32 accumulators, separate multiply/add rounding, a left-fold
lane reduction and correctly rounded square root. The CPU capability string
alone does not identify this norm implementation. The integrated implementation
tests native norms on the actual CPU transfer worker using a private generator,
selects matching lane/FMA parameters, and warns/falls back to CPU summaries
if it cannot match. It does not consume the training RNG state.

All 78 dense/sparse numerical cases matched native CPU norms bitwise, including
empty rows, two table widths, chunks 256/65536 and subnormal products. The
two-rank fused-embedding test also passed eight Adam updates, accumulation,
hooks, failure recovery, external gradient mutation and checkpoint restore.

The first paired 30-step replicated-backbone test measured 2.927712 seconds
and 45.6860 GiB active for DRAM versus 2.922836 seconds and 55.2114 GiB for
GPU DP: +0.1668% latency and -17.2526% active GPU memory. Steps 6--30 are
reported. The prototype's paired 100-step run measured 2.933537 versus
2.924944 seconds (+0.2938%), with the same 17.2526% active-memory reduction.
Steps 51--100 measured 2.938199 versus 2.921968 seconds (+0.5555%). These
are near-DP results, not proof of the strict threshold in every interval.
The integrated implementation passed 78 regression tests, including native
norm matching, RNG preservation, unsupported-plan fallback, switch validation
and the full distributed gradient/optimizer/checkpoint checks. Its separate
100-step performance repetition is tracked in `gpu_norm_comparison.json`.
Prototype and integrated measurements are not pooled as identical runs.

The integrated implementation's paired 100-step repetition measured 2.938062
versus 2.928254 seconds (+0.3349%), with 45.6860 versus 55.2114 GiB active
(-17.2526%). Steps 51--100 measured 2.947887 versus 2.925971 seconds
(+0.7490%). The overall average meets the 0.5%/15% target, but the late
interval still exceeds the strict latency threshold. This does not establish
stable parity in every interval or full-model numerical equivalence.

Use the lower-memory replicated-backbone reproduction command with
`--gpu-norm` to exercise the integrated path. Both asynchronous training and
GPU norm summaries remain disabled by default. CPU master parameters,
moments and gradients are complete at the existing backward/step boundaries.

Full-model reproducibility diagnostics are separate from these timing runs.
Across two unchanged DP runs, all ranks had identical first batches and
initial parameters, but 138 of 149 gradient fingerprints already differed
before the first optimizer update. Module labels identified the final global
attention layer's cuDNN varlen backward as the first divergence. Repeating
that backward with identical captured inputs changed 115 valid query-gradient
elements on rank zero (maximum absolute difference 3.64e-12); key/value
gradients matched. This was observed with torch 2.16.0.dev20261013+cu130 and
cuDNN 92600, despite strict deterministic settings. It was not tail padding.

Strict deterministic CUDA varlen attention now selects FlashAttention; normal
non-deterministic backend selection is unchanged. Two independent ten-step DP
runs then matched all 149 initial parameter/gradient fingerprints, updated
parameter fingerprints, and full loss/grad-norm trajectories on all four ranks.
The integrated backend selection also matched the diagnostic selection exactly.
With that policy, enabling GPU chunk norms matched the disabled path's entire
30-step loss and grad-norm trajectories bitwise on all four ranks, at local
batch 128 and sequence length 2048. The complete transport/update optimization
stack also matched the plain row-sharded path's full 30-step loss and grad-norm
trajectories bitwise on all four ranks. These are separate full-model checks;
the ordinary DRAM reference has all optional transport/update flags disabled
and uses the same replicated backbone, norm chunk size and learning-rate schedule.
The timings above
predate this backend change and must not be pooled with Flash-only results.
Diagnostic fingerprints and captured tensors remain local in the round-12
output directory and are excluded from timings.

### Deterministic backend remeasurement

A fresh paired 100-step run uses the same FlashAttention policy for both
fully replicated GPU DP and the optimized DRAM path. All other measurement
settings remain unchanged: four H100s, local batch 128, sequence 2048,
real Climbmix, seed 42 and the same initial model checkpoint/schedule.
The DRAM path saves one layer's PLE matmuls, releases consumed fetch-cache
references, and enables GPU chunk norms. Source snapshots include shared
attention code and all launch flags.

| Metric | GPU DP | DRAM PLE | Difference |
| --- | ---: | ---: | ---: |
| Mean seconds, steps 6--100 | 2.933289 | 2.938922 | +0.1920% |
| Mean seconds, steps 51--100 | 2.930724 | 2.939539 | +0.3008% |
| Peak active GiB, all ranks/steps | 55.2114 | 45.6860 | -17.2526% |
| Peak reserved GiB, all ranks/steps | 61.4473 | 48.2480 | -21.48% |

Both latency intervals satisfy the 0.5% threshold, alongside more than 15%
active GPU memory savings, on this configuration. This is measured parity
within the stated tolerance, not literal equal latency or a guarantee for
other hardware/batches. DRAM uses host RAM instead: sampled process RSS was
about 18 GiB per rank, which includes allocations outside the PLE table.
`outputs/ple_async_round12/deterministic_comparison.json` records the complete
results and separate numerical validation gates.

Both numerical gates and the paired performance target are complete. The
validation runs are `dram_shard_b128_det_norm_off30`,
`dram_shard_b128_det_norm_on30` and `dram_shard_b128_det_plain30`; performance
runs are `gpu_ddp_b128_det_control100` and `dram_shard_b128_det_norm100`.
Use `compare_deterministic.py` in the output directory to regenerate the report.
The new norm/transport regression suite passed 78 tests, and the attention
targeted suite passed 15 tests. All asynchronous options remain disabled by
default. No diagnostic fingerprint monkeypatch is enabled in these timing runs.

### Lower-memory CPU update tuning

`PLE_SAVE_LAYERS=0` retains the staged-linear and replicated-gradient transport
paths but recomputes every layer's PLE matmul outputs. At local batch 128 this
removes another 1 GiB of active GPU memory compared with the one-layer cache.
`PLE_CPU_UPDATE_THREADS=26` scopes the OpenMP team around the CPU table optimizer,
including the native AdamW update of other PLE parameters sharing that optimizer,
and restores the caller's previous setting, including on errors. Fetch workers,
vocabulary statistics and gradient norm reductions keep their existing setting.
The value 26 is specific to this host's per-rank CPU allocation; the default
remains zero (inherit), and asynchronous training remains disabled by default.

The fused CPU kernel also compares clipped gradient bits before storing them.
It skips writes whose value is already identical, without skipping arithmetic,
row updates, tensor version bumps, or changes to signed zero/NaN payloads.
Parameter and moment arithmetic remains the native FP32 AdamW sequence.

The first scoped-thread, configuration, gradient-bit and recomputation suite passed
78 tests (`tests_r13_cpu_final.log`). It covers 32 native AdamW updates for
three beta settings at thread counts 0/1/4/26, mutation/restore, and restoration
of the thread setting after both success and a kernel exception. Separate
30-step prototypes matched the original full-model loss/grad-norm trajectories
bitwise on all four ranks. Fresh production 100-step profiles and the complete
100-step trajectory comparison are recorded in `round13_comparison.json`.
Extending the scope to the other parameters sharing the CPU optimizer passed
81 tests (`tests_r13_worker.log`), including native fallback, auxiliary updates,
and restoration after optimizer-hook failures. Its separate performance runs
are recorded in `round13_worker_comparison.json`.

```bash
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dram --backbone dp --batch 128 --steps 100 --saved-layers 0 \
  --cpu-update-threads 26 --release-fetch-cache --gpu-norm --tag my_dram_tuned
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dp --batch 128 --steps 100 --tag my_dp_control
```

### Final whole-optimizer measurements and rejected prototypes

The final implementation uses zero saved PLE matmul layers and scopes 26 CPU
threads around the complete PLE optimizer. A fresh paired 100-step measurement
uses four H100s, real Climbmix, local batch 128, sequence 2048, strict seed-42
determinism and the same FlashAttention backend and Moonlight Muon backbone
optimizer. Latency averages the slowest rank at each step, excluding steps 1--5;
memory peaks include all ranks and all steps, including warmup.

| Configuration | Seconds, 6--100 | Seconds, 51--100 | Active GiB | Reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| GPU PLE, replicated DP backbone | 2.925450 | 2.925283 | 55.2114 | 61.4473 |
| DRAM PLE, replicated DP backbone | 2.928888 | 2.930230 | 44.6860 | 47.2520 |
| DRAM PLE, FSDP backbone, inherited CPU threads | 2.926631 | 2.928280 | 42.7715 | 44.8398 |

The identical-backbone DRAM configuration reduces active GPU memory by 19.06%
versus GPU DP, and by another 1 GiB versus the previous DRAM configuration.
Its mean latency is 0.118% higher than DP (0.169% for steps 51--100): measured
parity within 0.5%, without evidence of a stable speedup. Its complete 100-step
loss and gradient-norm trajectories match the prior DRAM reference bitwise on
all four ranks. The CPU regression suite passed 81 tests.

The optional FSDP backbone reduces active memory by 22.53% versus replicated
GPU DP, with mean latency 0.040% higher (0.102% for steps 51--100). This changes
backbone parallelism; the extra savings are not attributable solely to PLE.
The 26-thread setting regresses performance for this FSDP configuration, so
leave it at zero. Its 30-step inherited-versus-tuned thread comparison matches
bitwise; this does not establish bitwise equivalence between FSDP and DP.
See `outputs/ple_async_round12/round13_worker_comparison.json` for the separate
performance and numerical gates.

Additional 30-step trials did not justify production changes:

- Increasing CPU optimizer threads to 52 takes 2.943276 seconds per step,
  versus a fresh GPU DP control at 2.922163 seconds. Keep 26 for this DP host
  allocation, rather than assuming more SMT threads improve throughput.
- Offloading the 6 GiB root PLE normalization input to pinned DRAM lowers active
  memory to about 41.50 GiB with buffer reuse, but takes 3.470557 seconds per
  step. A 64 MiB paced-copy prototype improves this to 3.120975 seconds, still
  slower than the 2.922400-second DRAM control. Increasing CUDA connection
  counts or launch-queue capacity does not recover parity; the latter trial
  takes 3.483166 seconds.
- Profiling shows prefetch completed before consumption, with roughly 0.013 ms
  unpack wait. Large DMA transfers instead coincide with long host CUDA kernel
  submission calls; one call spends about 386 ms submitting a roughly 1 us
  device kernel. Smaller paced copies mitigate backward submission delay but
  add forward completion time. This observation does not establish a general
  CUDA driver defect.

These activation-offload and runtime-queue trials remain output-directory
prototypes, with no production hooks or new defaults. All four tested final
activation/thread variants match their replicated-backbone DRAM control's
complete 30-step loss/gradient-norm trajectories bitwise on all ranks; see
`outputs/ple_async_round12/round13_tail_comparison.json`. Prototype small-tensor
checks also cover aliased in-place mutation, GPU-storage release, and first/
second derivatives, including mutation between derivative evaluations.
Asynchronous training and all optional transport/update switches remain off
by default. These measurements do not imply speedups on other configurations.

### Context-only checkpointing

The follow-up profile additionally stages saved per-layer PLE inputs in pinned
DRAM, retains one current-step row per unique token, and reconstructs each
full input one layer ahead of backward. This changes storage and transfers,
not the GEMM dimensions or gradient aggregation order. It requires one
microbatch per optimizer step and the existing saved-matmul/early-fetch path.
Starting from the optimized context-checkpoint profile, enable:

```bash
export PLE_CHECKPOINT=0 PLE_CONTEXT_CHECKPOINT=1
export PLE_OFFLOAD_SAVED_INPUTS=1 PLE_COMPACT_SAVED_INPUTS=1
export PLE_DEFER_CPU_JOIN=1
```

All new switches default to zero. `PLE_AUDIT_COMPACT_INPUTS=1` compares restored
bytes with the original saved inputs and is a diagnostic, not a timing setting.
The model owns its staging manager; ordinary GPU training does not install it.
The selected compact profile does not need `PLE_REUSE_CONTEXT_NORM_INPUT`.
Copy-stream dependencies and allocator stream records protect asynchronous
buffer lifetimes; shutdown explicitly joins the staging stream. Active and
reserved GPU allocation peaks are reported separately in the study artifact.
Full immutable launch environments and comparison data are under
`outputs/mlsys2027_study`; these exports are incremental flags.

`PLE_CONTEXT_CHECKPOINT=1` uses PyTorch non-reentrant checkpointing for
`SavedContext`, keeping its projection and normalization out of the saved
activation set. Row lookup and blending remain outside this checkpoint, so
current-step row caches can be released after the forward pass. This avoids
retaining the large normalization input without adding activation DMA or
changing projection, normalization, clipping or optimizer arithmetic.

The switch requires `PLE_OVERLAP_REPLICATED_GRADIENT=1` (and its existing
prerequisites), and defaults to zero. It works independently of the full-root
`PLE_CHECKPOINT` setting. For the measured configuration, set `PLE_CHECKPOINT=0`
in direct training launches; enabling both nests the checkpoints and retains
the full-root row-cache lifetime. Inference bypasses recomputation. Parameter
and optimizer checkpoint layouts are unchanged.

The targeted CPU suite passes 92 tests, including both input-gradient modes,
both backward-precision settings, eight exact AdamW updates with model/state
reload, saved-activation shapes, and aliased input mutation errors. Separate
four-rank, real-data 30-step runs preserve the complete loss/gradient-norm
trajectories bitwise. The hand-written root-norm-input recomputation variant
remains a prototype: its small short-run speed difference does not justify
extra autograd code and its higher reserved-memory peak.

Reproduce the production configuration with local benchmark assets:

```bash
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dram --backbone dp --batch 128 --steps 100 --saved-layers 0 \
  --cpu-update-threads 26 --release-fetch-cache --gpu-norm \
  --context-checkpoint --tag my_context_checkpoint
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dp --batch 128 --steps 100 --tag my_context_dp_control
```

The harness explicitly disables the full-root checkpoint for DRAM profiles,
sets this new flag to zero unless requested, and captures source/launch flags.
`outputs/ple_async_round12/compare_round14.py` regenerates the comparison and
marks it complete only after both 100-step profiles and the full same-backbone
trajectory equivalence gate finish.

The completed paired 100-step production measurement uses the same four-H100,
Climbmix, local-batch-128, sequence-2048, strict deterministic FlashAttention
configuration and replicated Moonlight Muon backbone as the preceding runs.

| Metric | GPU DP PLE | DRAM context checkpoint | Difference |
| --- | ---: | ---: | ---: |
| Mean seconds, steps 6--100 | 2.926850 | 2.946015 | +0.6548% |
| Mean seconds, steps 51--100 | 2.927501 | 2.946824 | +0.6601% |
| Peak active GiB, all ranks/steps | 55.2114 | 38.6508 | -29.9948% |
| Peak reserved GiB, all ranks/steps | 61.4473 | 41.2363 | -32.8915% |

This saves another 6.0352 GiB of active GPU memory relative to the previous
44.6860 GiB DRAM configuration, but does not meet the 0.5% latency-parity
threshold. Keep the flag disabled for the previous latency-oriented profile;
enable it when the additional memory savings justify about 19 ms per step
versus the paired DP control on this configuration. The full 100-step loss and
gradient-norm trajectories match the prior DRAM reference bitwise on all four
ranks. This equivalence is between DRAM configurations, not a claim of bitwise
DP-versus-DRAM equivalence. See `round14_comparison.json` in the round-12 output
directory for complete measurements, separate short-run prototypes and gates.

`PLE_TENSOR_EXCHANGE=1` replaces Python-object request exchange with two Gloo
tensor collectives: per-rank lengths followed by padded int64 IDs. IDs retain
their original rank and element order; model arithmetic is unchanged. It is
shared with Engram DRAM, defaults to zero, and requires asynchronous training.

### Lower-memory profile with asynchronous clipping

A subsequent full 100-step run combines context-only checkpointing with
`PLE_ASYNC_CLIP=1` and `PLE_TENSOR_EXCHANGE=0`. On the same four-H100 d200m
configuration (batch 128 per rank, sequence 2048), it records 2.946211 seconds
per step over steps 6--100 versus 2.935001 for the unchanged GPU DP control
(+0.3819%). Steps 51--100 differ by +0.3392%. Peak active allocation is
38.6508 GiB versus DP's 55.2114 GiB (-29.99%) and the prior DRAM profile's
44.6860 GiB (another 6.0352 GiB saved). Peak reserved allocation is
41.2363 GiB versus DP's 61.4473 GiB. Peaks include warmup and all ranks.

This particular run meets the 0.5% mean and late-window latency tolerance;
it does not establish a latency speedup or a universal bound. Its complete
100-step loss and gradient-norm trajectory matches
`dram_shard_b128_engram_ple_clean100` bitwise. GPU DP versus CPU DRAM optimizer
rounding remains a separate numerical comparison.

```bash
PLE_ASYNC_CLIP=1 PLE_TENSOR_EXCHANGE=0 \
.venv/bin/python outputs/ple_async_round12/run_latency_profile.py \
  --mode dram --backbone dp --batch 128 --steps 100 --saved-layers 0 \
  --cpu-update-threads 26 --release-fetch-cache --gpu-norm \
  --context-checkpoint --tag my_lower_memory_profile
```

The measured case is `dram_shard_b128_memory_context_async100`.
The combined PLE/Engram memory report is generated by
`outputs/engram_dram_round1/compare_memory.py`.
