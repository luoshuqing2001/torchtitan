"""Check that 2 ranks x 2 virtual shards read exactly the 4-rank data, token by token."""
import hashlib
import sys

import torch

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan_recipes.models.climbmix import climbmix_dataset, climbmix_train_and_val_files

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEQ, LOCAL4 = 2048, 64
tok = HuggingFaceTokenizer.Config().build(tokenizer_path="./assets/hf/OLMo3")
train, val = climbmix_train_and_val_files("./data/climbmix-100b")


def loader(files, world, rank, tokens, shards=None):
    cfg = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=climbmix_dataset(files)),
        num_data_shards=shards,
    )
    return cfg.build(dp_world_size=world, dp_rank=rank, tokenizer=tok,
                     max_context_length=SEQ, num_tokens_per_microbatch=tokens)


def digest(mb):
    h = hashlib.sha1()
    for t in (mb.input, mb.labels, mb.positions, mb.padding_mask):
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:12], mb.num_valid_tokens


ok = True
for name, files in (("train", train), ("val", val)):
    four = [iter(loader(files, 4, r, LOCAL4 * SEQ)) for r in range(4)]
    two = [iter(loader(files, 2, r, 2 * LOCAL4 * SEQ, shards=4)) for r in range(2)]
    for step in range(STEPS):
        ref = [next(it) for it in four]
        got = [next(it) for it in two]
        for r in range(2):
            a, b = ref[r], ref[r + 2]
            g = got[r]
            same = all(
                torch.equal(getattr(g, f), torch.cat([getattr(a, f), getattr(b, f)]))
                for f in ("input", "labels", "positions", "padding_mask")
            ) and g.num_valid_tokens == a.num_valid_tokens + b.num_valid_tokens
            # The second virtual shard must start a document (position 0).
            boundary = int(g.positions[LOCAL4 * SEQ]) == 0
            ok &= same and boundary
            print(f"{name} step {step} gpu {r}: virtual = 4-rank[{r}] + 4-rank[{r + 2}] "
                  f"{'MATCH' if same else 'DIFF'}  boundary@{LOCAL4 * SEQ}={'yes' if boundary else 'NO'}  "
                  f"{digest(g)}")
print("ALL MATCH" if ok else "MISMATCH")
