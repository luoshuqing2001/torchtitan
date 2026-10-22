# Glimmer dense / PLE / Engram ladder

Scaling ladder of `torchtitan_recipes.models.glimmer_dense` on ClimbMix with the
OLMo3 tokenizer: flavors `d200m`, `d400m`, `d700m`, `d1b`, each as dense,
PLE (`PLE=ple`) and Engram (`PLE=engram`). PLE and Engram run synchronously
(`PLE_ASYNC_TRAINING=0`). Each run uses one 4xH100 node.

## New node

```bash
git clone -b <branch> https://github.com/luoshuqing2001/torchtitan.git ~/torchtitan
cd ~/torchtitan
./scripts/ladder/setup_node.sh          # venv, FA3, OLMo3 tokenizer, ClimbMix 100B
```

To mirror logs and checkpoints to R2, put the credentials file at
`~/r2-credentials.env` (or set `R2_ENV`) and have `rclone` on `PATH` (or set
`RCLONE`). Without it, everything stays under `outputs/ladder/`.

## Run

```bash
nohup ./scripts/ladder/run_ladder.sh scripts/ladder/plans/dense.txt \
    >> ~/ladder.out 2>&1 < /dev/null &
./scripts/ladder/ladder_status.sh
```

A plan line is `<flavor> <variant> <micro_batch> [KEY=VALUE ...]`; the
optional pairs are extra environment for that run, e.g.
`d200m ple 64 TRAIN_STEPS=9922`. Runs go one after another. Relaunching the
runner skips runs with a `DONE` marker and resumes unfinished ones from their
latest checkpoint.

When several nodes share the ladder, give each node different plan lines. Run
names are unique, so all nodes can write to the same `R2_DEST`. Never run the
same run on two nodes.

## Comparability

Everything below is fixed in the recipe, so dense, PLE and Engram at one flavor
see the same data in the same order and are measured at the same token counts.

| | d200m | d400m | d700m | d1b |
|---|---|---|---|---|
| tokens per step (`GLOBAL_BATCH_TOKENS`) | 512K | 512K | 1M | 1M |
| micro x accumulation | 64 x 1 | 64 x 1 | 64 x 2 | 64 x 2 |
| steps | 9922 * | 17408 | 14336 | 28160 |
| training tokens | 5.20B * | 9.13B | 15.03B | 29.53B |
| dense tokens/s per GPU (H100) | 67.9k | 45.1k | 32.3k | 20.5k |

- Global batch is near the compute-optimal batch for each flavor
  (B = 0.292 * C^0.3271, DeepSeek LLM). `LOCAL_BATCH_SIZE` only sets the
  microbatch; gradient accumulation makes up the rest.
- The token budget is 20 tokens per dense parameter, rounded up to whole
  2^28-token intervals so the last step validates and checkpoints. PLE and
  Engram use the dense budget.
- Validation runs every 2^28 tokens (268M) on about 8.4M held-out ClimbMix
  tokens, so val points of all runs land on the same token counts.
- Resumable checkpoints are saved every 2^28 tokens, and the newest 2 are
  kept. Snapshots at doubling token counts (2^28, 2^29, ..., 2^34) and the
  final step keep the full training state for lm-eval-harness and WSD decay
  branches.
- Optimizer: Moonlight Muon for matrices, AdamW for the rest, both at the
  flavor's LR with weight decay 0.1. WSD schedule: 2% warmup, 88% stable,
  10% cosine decay to 0.1x the peak LR. Snapshots before the decay are in the
  stable phase.

\* d200m dense started before the budget rounding, so it ran 9922 steps and
its last validation is at step 9728. d200m PLE and Engram must set
`TRAIN_STEPS=9922` to match it. The final d200m val loss comes from the final
checkpoint.
