"""Check that fewer GPUs with DATA_SHARDS read exactly the 4-rank data, token by token.

    python scripts/ladder/check_data_shards.py [steps]

Modes: 2 GPUs x 1 microbatch of 2 shards (accumulation 1), and 1 GPU x 2
microbatches of 2 shards (accumulation 2). Each step must equal the 4-rank step.
"""
import os
import sys

import torch

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan_recipes.models.climbmix import climbmix_dataset, climbmix_train_and_val_files

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEQ, SHARD = 2048, 64 * 2048  # 4-rank per-rank microbatch: 64 rows of 2048
FIELDS = ("input", "labels", "positions", "padding_mask")
tok = HuggingFaceTokenizer.Config().build(tokenizer_path="./assets/hf/OLMo3")
train, val = climbmix_train_and_val_files("./data/climbmix-100b")


def loader(files, world, rank, tokens, shards=None):
    cfg = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=climbmix_dataset(files)),
        num_data_shards=shards,
        data_shard_tokens=SHARD if shards else None,
    )
    return iter(cfg.build(dp_world_size=world, dp_rank=rank, tokenizer=tok,
                          max_context_length=SEQ, num_tokens_per_microbatch=tokens))


def rank_order(mbs):
    # Split each microbatch into shard-sized pieces in order.
    pieces = []
    for mb in mbs:
        for i in range(mb.input.shape[0] // SHARD):
            pieces.append({f: getattr(mb, f)[i * SHARD:(i + 1) * SHARD] for f in FIELDS})
    return pieces


ok = True
for name, files in (("train", train), ("val", val)):
    ref = [loader(files, 4, r, SHARD) for r in range(4)]
    two = [loader(files, 2, r, 2 * SHARD, shards=4) for r in range(2)]
    one = loader(files, 1, 0, 2 * SHARD, shards=4)
    for step in range(STEPS):
        want = [next(it) for it in ref]                         # ranks 0..3
        want = [{f: getattr(mb, f) for f in FIELDS} for mb in want]
        got2 = rank_order([next(two[0]), next(two[1])])         # [r0, r2, r1, r3]
        got1 = rank_order([next(one), next(one)])               # [r0, r1, r2, r3]
        same2 = all(torch.equal(got2[i][f], want[r][f]) for i, r in enumerate((0, 2, 1, 3)) for f in FIELDS)
        same1 = all(torch.equal(got1[i][f], want[i][f]) for i in range(4) for f in FIELDS)
        starts = all(int(p["positions"][0]) == 0 for p in got1 + got2)
        ok &= same2 and same1 and starts
        print(f"{name} step {step}: 2gpu/accum1 {'MATCH' if same2 else 'DIFF'}  "
              f"1gpu/accum2 {'MATCH' if same1 else 'DIFF'}  shard starts at position 0: {starts}")
print("ALL MATCH" if ok else "MISMATCH")
sys.stdout.flush()
os._exit(0 if ok else 1)  # skip Grain prefetch-thread teardown
