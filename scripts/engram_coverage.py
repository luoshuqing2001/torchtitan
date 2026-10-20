#!/usr/bin/env python3
"""Measure how many Engram hash rows a pretraining scan actually touches.

The hash matches ``EngramTable``: 8 bigram heads and 8 trigram heads, the
same salts and mix constants, and ``eos`` filled across a document boundary.
Each document is tokenized like ``TextProcessor`` (BOS and EOS when the
tokenizer has them). Engram sees ``input_ids[:-1]`` relative to that full
sequence. Rows per head are the d200m counts that match a Gemma PLE module
at the padded tokenizer vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

_HOLDOUT_SHARDS = 16
_BATCH = 8192
_NGRAM_HEADS = 16
_BIGRAM_SALTS = np.array((1, 3, 5, 7, 9, 11, 13, 15), dtype=np.int64)
_TRIGRAM_SALTS = np.array((17, 19, 21, 23, 25, 27, 29, 31), dtype=np.int64)
_MIX_PREV = np.int64(1_000_003)
_MIX_PREV2 = np.int64(1_000_003 * 1_000_003)
_PLE_DIMS = (256, 512, 1024)
_DIM = 1024
_N_LAYERS = 12
_PAD_MULTIPLE = 256


def _token_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _load_tokenizer(tokenizer_dir: str) -> tuple[Tokenizer, int | None, int | None]:
    path = Path(tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    config_path = path / "tokenizer_config.json"
    bos_id = None
    eos_id = None
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        bos = _token_str(config.get("bos_token"))
        eos = _token_str(config.get("eos_token"))
        if bos is not None:
            bos_id = tokenizer.token_to_id(bos)
        if eos is not None:
            eos_id = tokenizer.token_to_id(eos)
    return tokenizer, bos_id, eos_id


def _padded_vocab(vocab: int) -> int:
    return ((vocab + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


def _rows_per_head(vocab_size: int, ple_dim: int) -> int:
    """Hash rows for d200m. The rectangular table is the PLE lookup.

    ``outputs/engram_coverage_olmo2_d200m.md`` was measured with the older
    projection budget, which put the leftover active parameters into the
    table instead of the value path.
    """
    target_table = vocab_size * _N_LAYERS * ple_dim
    return max(1, target_table // _DIM)


def _train_shards(data_dir: str) -> list[str]:
    paths = sorted(Path(data_dir).glob("shard_*.parquet"))
    if len(paths) <= _HOLDOUT_SHARDS:
        raise SystemExit(
            f"{data_dir} has {len(paths)} shards; need more than {_HOLDOUT_SHARDS}."
        )
    return [str(path) for path in paths[:-_HOLDOUT_SHARDS]]


_TOKENIZER: Tokenizer | None = None
_BOS_ID: int | None = None
_EOS_ID: int | None = None
_ROWS: dict[int, int] = {}


def _init_worker(tokenizer_dir: str, rows: dict[int, int]) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _ROWS
    _TOKENIZER, _BOS_ID, _EOS_ID = _load_tokenizer(tokenizer_dir)
    if _EOS_ID is None:
        raise RuntimeError(f"{tokenizer_dir} has no eos token")
    _ROWS = rows


def _model_inputs(ids: list[int]) -> np.ndarray | None:
    """Token ids Engram sees: TextProcessor's full encode, without the last label."""
    full: list[int] = []
    if _BOS_ID is not None:
        full.append(_BOS_ID)
    full.extend(ids)
    if _EOS_ID is not None:
        full.append(_EOS_ID)
    if len(full) < 2:
        return None
    return np.asarray(full[:-1], dtype=np.int64)


def _context(tokens: np.ndarray, doc_starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Previous tokens, with eos filled across a segment boundary."""
    assert _EOS_ID is not None
    after_eos = np.zeros(tokens.shape[0], dtype=np.bool_)
    if tokens.shape[0] > 1:
        after_eos[1:] = tokens[:-1] == _EOS_ID
    starts = doc_starts | after_eos
    segments = np.cumsum(starts, dtype=np.int64)
    prev = np.full(tokens.shape, _EOS_ID, dtype=np.int64)
    prev2 = np.full(tokens.shape, _EOS_ID, dtype=np.int64)
    if tokens.shape[0] > 1:
        same = segments[1:] == segments[:-1]
        prev[1:] = np.where(same, tokens[:-1], prev[1:])
    if tokens.shape[0] > 2:
        same2 = segments[2:] == segments[:-2]
        prev2[2:] = np.where(same2, tokens[:-2], prev2[2:])
    return prev, prev2


def _hashes(tokens: np.ndarray, prev: np.ndarray, prev2: np.ndarray) -> np.ndarray:
    current = tokens[:, None]
    previous = prev[:, None]
    previous2 = prev2[:, None]
    bigram = current + previous * _MIX_PREV + _BIGRAM_SALTS * _MIX_PREV2
    trigram = current + previous * _MIX_PREV + previous2 * _MIX_PREV2 + _TRIGRAM_SALTS
    return np.concatenate((bigram, trigram), axis=1)


def _flat_indices(tokens: np.ndarray, prev: np.ndarray, prev2: np.ndarray, rows: int) -> np.ndarray:
    return _rows_from_hashes(_hashes(tokens, prev, prev2), rows)


def _rows_from_hashes(hashed: np.ndarray, rows: int) -> np.ndarray:
    local = np.remainder(hashed, np.int64(rows))
    offsets = np.arange(_NGRAM_HEADS, dtype=np.int64) * np.int64(rows)
    return (local + offsets).reshape(-1)


def _hit_from_hashes(hashed: np.ndarray, rows: int, mask: np.ndarray) -> np.ndarray:
    return _newly_hit(_rows_from_hashes(hashed, rows), mask)


def _newly_hit(flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rows in ``flat`` that ``mask`` has not recorded yet. ``np.unique`` sorts and dominates."""
    counts = np.bincount(flat, minlength=mask.shape[0])
    new_rows = np.flatnonzero((counts != 0) & ~mask)
    if new_rows.size:
        mask[new_rows] = True
    return new_rows.astype(np.uint32, copy=False)


def _scan_shard(path: str) -> tuple[int, list[tuple[int, dict[int, np.ndarray]]]]:
    """Return training tokens and rows first touched at each batch boundary."""
    assert _TOKENIZER is not None and _EOS_ID is not None
    table = pq.read_table(path, columns=["text"])
    texts = table.column("text").to_pylist()
    seen = {ple_dim: np.zeros(_NGRAM_HEADS * rows, dtype=np.bool_) for ple_dim, rows in _ROWS.items()}
    n_tokens = 0
    first_seen: list[tuple[int, dict[int, np.ndarray]]] = []
    # The three moduli are independent, and bincount releases the GIL.
    pool = ThreadPoolExecutor(max_workers=len(_ROWS))
    for start in range(0, len(texts), _BATCH):
        chunk = [
            text if isinstance(text, str) else ""
            for text in texts[start : start + _BATCH]
        ]
        encoded = _TOKENIZER.encode_batch(chunk, add_special_tokens=False)
        sequences: list[np.ndarray] = []
        for encoding in encoded:
            inputs = _model_inputs(encoding.ids)
            if inputs is not None:
                sequences.append(inputs)
        if not sequences:
            continue
        tokens = np.concatenate(sequences)
        doc_starts = np.zeros(tokens.shape[0], dtype=np.bool_)
        offset = 0
        for sequence in sequences:
            doc_starts[offset] = True
            offset += int(sequence.shape[0])
        prev, prev2 = _context(tokens, doc_starts)
        n_tokens += int(tokens.shape[0])
        hashed = _hashes(tokens, prev, prev2)
        futures = {
            ple_dim: pool.submit(_hit_from_hashes, hashed, rows, seen[ple_dim])
            for ple_dim, rows in _ROWS.items()
        }
        fresh: dict[int, np.ndarray] = {}
        for ple_dim, future in futures.items():
            new_rows = future.result()
            if new_rows.size:
                fresh[ple_dim] = new_rows
        if fresh:
            first_seen.append((n_tokens, fresh))
    pool.shutdown(wait=True)
    return n_tokens, first_seen


def _check_hash_matches_torch(rows: int) -> None:
    """Fail if the numpy hash drifts from ``EngramTable``."""
    import torch

    fill_id = int(_EOS_ID)
    tokens_np = np.array([fill_id, 15, 200, 3, fill_id, 9, 8], dtype=np.int64)
    doc_starts = np.zeros(tokens_np.shape[0], dtype=np.bool_)
    doc_starts[0] = True
    doc_starts[4] = True
    prev, prev2 = _context(tokens_np, doc_starts)
    got = _flat_indices(tokens_np, prev, prev2, rows)

    tokens = torch.from_numpy(tokens_np.copy())
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)
    fill = int(_EOS_ID)
    after_eos = torch.zeros_like(tokens, dtype=torch.bool)
    after_eos[1:] = tokens[:-1] == fill
    starts = (positions == 0) | after_eos
    segments = starts.to(dtype=torch.int64).cumsum(dim=0)
    prev_t = torch.full_like(tokens, fill)
    prev2_t = torch.full_like(tokens, fill)
    same = segments[1:] == segments[:-1]
    prev_t[1:] = torch.where(same, tokens[:-1], prev_t[1:])
    same2 = segments[2:] == segments[:-2]
    prev2_t[2:] = torch.where(same2, tokens[:-2], prev2_t[2:])
    current = tokens.unsqueeze(-1)
    bigram_salt = torch.tensor(_BIGRAM_SALTS.tolist(), dtype=torch.int64)
    trigram_salt = torch.tensor(_TRIGRAM_SALTS.tolist(), dtype=torch.int64)
    bigram = current + prev_t.unsqueeze(-1) * int(_MIX_PREV) + bigram_salt * int(_MIX_PREV2)
    trigram = (
        current
        + prev_t.unsqueeze(-1) * int(_MIX_PREV)
        + prev2_t.unsqueeze(-1) * int(_MIX_PREV2)
        + trigram_salt
    )
    hashed = torch.cat((bigram, trigram), dim=-1)
    local = torch.remainder(hashed, rows)
    offsets = torch.arange(_NGRAM_HEADS) * rows
    expected = (local + offsets).reshape(-1).numpy()
    if not np.array_equal(got, expected):
        raise SystemExit("numpy Engram hash does not match the torch formula")


def _write_report(
    path: str,
    *,
    tokenizer_dir: str,
    data_dir: str,
    vocab: int,
    padded_vocab: int,
    eos_id: int,
    rows: dict[int, int],
    n_shards: int,
    shards_done: int,
    tokens: int,
    covered: dict[int, int],
    milestones: dict[int, list[dict]],
    finished: bool,
    table_full: bool,
) -> None:
    lines = [
        "# ClimbMix Engram 行覆盖",
        "",
        f"- Tokenizer：`{tokenizer_dir}`，{vocab:,} 个 id，eos {eos_id}",
        f"- 词表按 {_PAD_MULTIPLE} 对齐后是 {padded_vocab:,}。d200m 的 Engram 行数按这个词表去配平 PLE。",
        f"- 数据：`{data_dir}`，训练 shard {n_shards:,} 个，最后 {_HOLDOUT_SHARDS} 个留作验证",
        "- 每个样本按 TextProcessor 分词。Engram 看到的是去掉最后一个 label 的 input。",
        "  文档之间和 eos 之后不借用上文。8 个 bigram head、8 个 trigram head。",
        "  覆盖率的分母是 16 个 head 的哈希行，一行被任意 token 取到即算覆盖。",
        "",
    ]
    for ple_dim in _PLE_DIMS:
        total = _NGRAM_HEADS * rows[ple_dim]
        lines.extend(
            [
                f"## ple_dim {ple_dim}",
                "",
                f"- 每个 head {rows[ple_dim]:,} 行，合计 {total:,} 行",
                "",
                "| 覆盖率 | token | 已见行 |",
                "|---:|---:|---:|",
            ]
        )
        tier_rows = milestones[ple_dim]
        if not tier_rows:
            lines.append("| — | — | — |")
        for row in tier_rows:
            lines.append(
                f"| {row['percent']}% | {row['tokens'] / 1e9:.4f}B | {row['covered']:,} |"
            )
        hit = covered[ple_dim]
        lines.extend(
            [
                "",
                f"- 已覆盖：{hit / total * 100:.2f}%（{hit:,}/{total:,}）",
                f"- 未见行：{total - hit:,}",
                "",
            ]
        )
    lines.extend(
        [
            "## 当前进度",
            "",
            f"- 已扫 shard：{shards_done:,}/{n_shards:,}",
            f"- 已扫 token：{tokens / 1e9:.4f}B",
            "",
        ]
    )
    if table_full:
        lines.append("三档哈希表都已经全部被取到，后面的 shard 不再读。")
    elif finished:
        lines.append("训练 shard 已扫完。上面的未见行在这份训练集里没有被哈希命中。")
    else:
        lines.append("扫描还在进行，这份文档会在每个 shard 结束后更新。")
    lines.append("")
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("CLIMBMIX_DIR", "/home/sky/torchtitan/data/climbmix-100b"),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("TOKENIZER_DIR", "./assets/hf/OLMo2"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-shards", type=int, default=0, help="Stop after N train shards. 0 scans all.")
    parser.add_argument("--output", default="", help="Optional JSONL path.")
    parser.add_argument("--report", default="outputs/engram_coverage_olmo2_d200m.md")
    args = parser.parse_args()

    tokenizer, bos_id, eos_id = _load_tokenizer(args.tokenizer_dir)
    if eos_id is None:
        raise SystemExit(f"{args.tokenizer_dir} has no eos token")
    global _EOS_ID, _BOS_ID
    _EOS_ID = eos_id
    _BOS_ID = bos_id
    vocab = tokenizer.get_vocab_size()
    padded = _padded_vocab(vocab)
    rows = {ple_dim: _rows_per_head(padded, ple_dim) for ple_dim in _PLE_DIMS}
    _check_hash_matches_torch(rows[256])

    all_shards = _train_shards(args.data_dir)
    shards = all_shards[: args.max_shards] if args.max_shards > 0 else all_shards
    print(
        f"tokenizer {args.tokenizer_dir}  vocab {vocab:,}  padded {padded:,}  "
        f"bos {bos_id}  eos {eos_id}",
        flush=True,
    )
    print(
        "d200m rows/head "
        + "  ".join(f"ple{ple_dim}={rows[ple_dim]:,}" for ple_dim in _PLE_DIMS),
        flush=True,
    )
    print(
        f"train shards {len(shards):,}  holdout {_HOLDOUT_SHARDS}  workers {args.workers}",
        flush=True,
    )

    seen = {ple_dim: np.zeros(_NGRAM_HEADS * rows[ple_dim], dtype=np.bool_) for ple_dim in _PLE_DIMS}
    covered = {ple_dim: 0 for ple_dim in _PLE_DIMS}
    next_percent = {ple_dim: 1 for ple_dim in _PLE_DIMS}
    milestones: dict[int, list[dict]] = {ple_dim: [] for ple_dim in _PLE_DIMS}
    tokens = 0
    output = open(args.output, "w") if args.output else None
    started = time.perf_counter()
    table_full = False

    def emit(record: dict) -> None:
        if output is not None:
            output.write(json.dumps(record) + "\n")
            output.flush()

    def note_hits(ple_dim: int, tokens_at: int) -> None:
        total = _NGRAM_HEADS * rows[ple_dim]
        hit = covered[ple_dim]
        crossed = []
        while next_percent[ple_dim] <= 100 and hit / total * 100 >= next_percent[ple_dim]:
            crossed.append(next_percent[ple_dim])
            emit(
                {
                    "kind": "milestone",
                    "ple_dim": ple_dim,
                    "percent": next_percent[ple_dim],
                    "tokens": tokens_at,
                    "covered": hit,
                    "rows": total,
                }
            )
            next_percent[ple_dim] += 1
        if not crossed:
            return
        for percent in crossed:
            milestones[ple_dim].append(
                {"percent": percent, "tokens": tokens_at, "covered": hit}
            )
        label = f"{crossed[0]}%" if len(crossed) == 1 else f"{crossed[0]}%–{crossed[-1]}%"
        print(f"ple{ple_dim} 覆盖 {label} 需要 {tokens_at / 1e9:.4f}B token", flush=True)

    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer_dir, rows),
    )
    try:
        for index, (path, (shard_tokens, first_seen)) in enumerate(
            zip(shards, pool.map(_scan_shard, shards, chunksize=1), strict=True),
            start=1,
        ):
            del path
            tokens_before = tokens
            for shard_tokens_at, fresh in first_seen:
                tokens_at = tokens_before + shard_tokens_at
                for ple_dim, new_rows in fresh.items():
                    mask = seen[ple_dim]
                    unseen = new_rows[~mask[new_rows]]
                    if unseen.size == 0:
                        continue
                    mask[unseen] = True
                    covered[ple_dim] += int(unseen.size)
                    note_hits(ple_dim, tokens_at)
            tokens += shard_tokens
            elapsed = time.perf_counter() - started
            parts = []
            for ple_dim in _PLE_DIMS:
                total = _NGRAM_HEADS * rows[ple_dim]
                parts.append(f"ple{ple_dim} {covered[ple_dim] / total * 100:.2f}%")
            print(
                f"shard {index}/{len(shards)}  tokens {tokens / 1e9:.3f}B  "
                + "  ".join(parts)
                + f"  {elapsed / 60:.1f}min",
                flush=True,
            )
            emit(
                {
                    "kind": "shard",
                    "shard": index,
                    "tokens": tokens,
                    "covered": covered,
                }
            )
            table_full = all(
                covered[ple_dim] == _NGRAM_HEADS * rows[ple_dim] for ple_dim in _PLE_DIMS
            )
            _write_report(
                args.report,
                tokenizer_dir=args.tokenizer_dir,
                data_dir=args.data_dir,
                vocab=vocab,
                padded_vocab=padded,
                eos_id=eos_id,
                rows=rows,
                n_shards=len(all_shards),
                shards_done=index,
                tokens=tokens,
                covered=covered,
                milestones=milestones,
                finished=table_full or index == len(shards),
                table_full=table_full,
            )
            if table_full:
                break
    finally:
        # map() submits every shard up front. Cancel the rest once the tables are full.
        pool.shutdown(wait=False, cancel_futures=True)

    if table_full:
        print(f"三档表在 {tokens / 1e9:.4f}B token 时全部覆盖。", flush=True)
    else:
        print(f"扫完 {tokens / 1e9:.3f}B token，表还没有盖满。", flush=True)
    print(f"报告写在 {args.report}", flush=True)
    if output is not None:
        output.close()


if __name__ == "__main__":
    main()
