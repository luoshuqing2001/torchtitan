#!/usr/bin/env python3
"""Measure how many training tokens are needed to cover a tokenizer.

Shards are read in the same order as dense training: sorted ``shard_*.parquet``,
with the last 16 held out for validation. Each document is tokenized the way
``TextProcessor`` does (EOS appended when the tokenizer has one, BOS prepended
when it has one). Sequences shorter than two tokens are skipped. The token
count is the number of input positions training would see.

Coverage is the fraction of tokenizer ids that have occurred. The padded model
vocabulary is not included.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

_HOLDOUT_SHARDS = 16
_BATCH = 2048


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


_TOKENIZER: Tokenizer | None = None
_BOS_ID: int | None = None
_EOS_ID: int | None = None
_VOCAB: int = 0


def _init_worker(tokenizer_dir: str) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _VOCAB
    _TOKENIZER, _BOS_ID, _EOS_ID = _load_tokenizer(tokenizer_dir)
    _VOCAB = _TOKENIZER.get_vocab_size()


def _scan_shard(path: str) -> tuple[int, list[tuple[int, np.ndarray]]]:
    """Return training tokens and the ids first seen at each batch boundary.

    Each entry is ``(tokens_in_this_shard, new_ids)``. The token count is the
    total at the end of that batch, so coverage milestones land on the batch
    that crossed them.
    """
    assert _TOKENIZER is not None
    texts = _load_texts(path)
    seen = np.zeros(_VOCAB, dtype=np.bool_)
    n_tokens = 0
    extra = int(_BOS_ID is not None) + int(_EOS_ID is not None)
    first_seen: list[tuple[int, np.ndarray]] = []
    for start in range(0, len(texts), _BATCH):
        chunk = [
            text if isinstance(text, str) else ""
            for text in texts[start : start + _BATCH]
        ]
        encoded = _TOKENIZER.encode_batch(chunk, add_special_tokens=False)
        pieces: list[np.ndarray] = []
        batch_tokens = 0
        for encoding in encoded:
            ids = encoding.ids
            length = len(ids) + extra
            if length < 2:
                continue
            batch_tokens += length - 1
            if ids:
                pieces.append(np.asarray(ids, dtype=np.int64))
        n_tokens += batch_tokens
        if not pieces and batch_tokens == 0:
            continue
        fresh = np.zeros(0, dtype=np.int64)
        if pieces:
            flat = np.concatenate(pieces)
            flat = np.unique(flat[flat < _VOCAB])
            fresh = flat[~seen[flat]]
        specials = [
            token_id
            for token_id in (_BOS_ID, _EOS_ID)
            if token_id is not None and batch_tokens and not seen[token_id]
        ]
        if specials:
            fresh = np.unique(np.concatenate([fresh, np.asarray(specials, dtype=np.int64)]))
        if fresh.size:
            seen[fresh] = True
            first_seen.append((n_tokens, fresh))
    return n_tokens, first_seen


def _load_texts(path: str) -> list[str]:
    if path.endswith(".parquet"):
        table = pq.read_table(path, columns=["text"])
        return table.column("text").to_pylist()
    if path.endswith(".jsonl.zst"):
        import zstandard as zstd

        texts: list[str] = []
        with open(path, "rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip():
                    continue
                value = json.loads(line).get("text")
                texts.append(value if isinstance(value, str) else "")
        return texts
    raise ValueError(f"unsupported shard format: {path}")


def _train_shards(data_dir: str, holdout: int) -> list[str]:
    root = Path(data_dir)
    paths = sorted(root.glob("shard_*.parquet"))
    if not paths:
        paths = sorted(
            path
            for path in root.rglob("*.jsonl.zst")
            if path.is_file() and ".cache" not in path.parts
        )
    if holdout < 0:
        raise SystemExit(f"holdout must be >= 0, got {holdout}")
    if len(paths) <= holdout:
        raise SystemExit(
            f"{data_dir} has {len(paths)} shards; need more than {holdout}."
        )
    if holdout:
        paths = paths[:-holdout]
    return [str(path) for path in paths]


def _write_report(
    path: str,
    *,
    tokenizer_dir: str,
    data_dir: str,
    vocab: int,
    bos_id: int | None,
    eos_id: int | None,
    n_shards: int,
    holdout: int,
    shards_done: int,
    tokens: int,
    covered: int,
    milestones: list[dict],
    finished: bool,
) -> None:
    missing = vocab - covered
    lines = [
        "# ClimbMix 词表覆盖",
        "",
        f"- Tokenizer：`{tokenizer_dir}`，{vocab:,} 个 id，bos {bos_id}，eos {eos_id}",
        (
            f"- 数据：`{data_dir}`，shard {n_shards:,} 个，全部计入覆盖率"
            if holdout == 0
            else f"- 数据：`{data_dir}`，训练 shard {n_shards:,} 个，最后 {holdout} 个留作验证"
        ),
        "- 顺序与训练相同。每个样本按 TextProcessor 分词，短于 2 个 token 的样本不计。",
        "  token 数是训练时的 input 位置数。覆盖率的分母是 tokenizer 词表，不含 embedding 补齐行。",
        "",
        "## 覆盖到每个百分点需要的 token",
        "",
        "| 覆盖率 | token | 已见 id |",
        "|---:|---:|---:|",
    ]
    for row in milestones:
        lines.append(
            f"| {row['percent']}% | {row['tokens'] / 1e9:.4f}B | {row['covered']:,} |"
        )
    if not milestones:
        lines.append("| — | — | — |")
    lines.extend(
        [
            "",
            "## 当前进度",
            "",
            f"- 已扫 shard：{shards_done:,}/{n_shards:,}",
            f"- 已扫 token：{tokens / 1e9:.4f}B",
            f"- 已覆盖：{covered / vocab * 100:.2f}%（{covered:,}/{vocab:,}）",
            f"- 未见 id：{missing:,}",
            "",
        ]
    )
    if finished and missing == 0:
        lines.append(f"词表在 {tokens / 1e9:.4f}B token 时覆盖完。")
    elif finished:
        lines.append(
            "训练 shard 全部扫完，词表没有被覆盖完。"
            f"最后覆盖率 {covered / vocab * 100:.2f}%。"
        )
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
        default=os.environ.get("TOKENIZER_DIR", "./assets/hf/Qwen3.8"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-shards", type=int, default=0, help="Stop after N train shards. 0 scans all.")
    parser.add_argument(
        "--holdout",
        type=int,
        default=_HOLDOUT_SHARDS,
        help="Drop this many trailing shards, matching the ClimbMix validation split. 0 scans every file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSONL path. One record per shard, plus milestone records.",
    )
    parser.add_argument(
        "--report",
        default="outputs/vocab_coverage.md",
        help="Markdown report, rewritten after every shard.",
    )
    args = parser.parse_args()

    tokenizer, bos_id, eos_id = _load_tokenizer(args.tokenizer_dir)
    vocab = tokenizer.get_vocab_size()
    all_shards = _train_shards(args.data_dir, args.holdout)
    shards = all_shards[: args.max_shards] if args.max_shards > 0 else all_shards
    milestones: list[dict] = []
    print(
        f"tokenizer {args.tokenizer_dir}  vocab {vocab:,}  "
        f"bos {bos_id}  eos {eos_id}",
        flush=True,
    )
    print(
        f"train shards {len(shards):,}  holdout {args.holdout}  workers {args.workers}",
        flush=True,
    )

    seen = np.zeros(vocab, dtype=np.bool_)
    tokens = 0
    covered = 0
    next_percent = 1
    output = open(args.output, "w") if args.output else None
    started = time.perf_counter()

    def emit(record: dict) -> None:
        if output is not None:
            output.write(json.dumps(record) + "\n")
            output.flush()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer_dir,),
    ) as pool:
        for index, (path, (shard_tokens, first_seen)) in enumerate(
            zip(shards, pool.map(_scan_shard, shards, chunksize=1), strict=True),
            start=1,
        ):
            del path
            tokens_before = tokens
            for shard_tokens_at, new_ids in first_seen:
                seen[new_ids] = True
                tokens_at = tokens_before + shard_tokens_at
                covered = int(seen.sum())
                crossed = []
                while next_percent <= 100 and covered / vocab * 100 >= next_percent:
                    crossed.append(next_percent)
                    emit(
                        {
                            "kind": "milestone",
                            "percent": next_percent,
                            "tokens": tokens_at,
                            "covered": covered,
                            "vocab": vocab,
                        }
                    )
                    next_percent += 1
                if crossed:
                    for percent in crossed:
                        milestones.append(
                            {
                                "percent": percent,
                                "tokens": tokens_at,
                                "covered": covered,
                            }
                        )
                    label = (
                        f"{crossed[0]}%"
                        if len(crossed) == 1
                        else f"{crossed[0]}%–{crossed[-1]}%"
                    )
                    print(
                        f"覆盖 {label} 需要 {tokens_at / 1e9:.4f}B token",
                        flush=True,
                    )
            tokens += shard_tokens
            covered = int(seen.sum())
            elapsed = time.perf_counter() - started
            print(
                f"shard {index}/{len(shards)}  tokens {tokens / 1e9:.3f}B  "
                f"vocab {covered / vocab * 100:.2f}% ({covered:,}/{vocab:,})  "
                f"{elapsed / 60:.1f}min",
                flush=True,
            )
            emit(
                {
                    "kind": "shard",
                    "shard": index,
                    "tokens": tokens,
                    "covered": covered,
                    "vocab": vocab,
                }
            )
            _write_report(
                args.report,
                tokenizer_dir=args.tokenizer_dir,
                data_dir=args.data_dir,
                vocab=vocab,
                bos_id=bos_id,
                eos_id=eos_id,
                n_shards=len(shards),
                holdout=args.holdout,
                shards_done=index,
                tokens=tokens,
                covered=covered,
                milestones=milestones,
                finished=False,
            )

    missing = vocab - covered
    stopped_early = len(shards) < len(all_shards)
    if missing == 0:
        print(f"词表在 {tokens / 1e9:.3f}B token 时覆盖完。", flush=True)
    elif stopped_early:
        print(
            f"停在前 {len(shards)} 个 shard，{tokens / 1e9:.3f}B token，"
            f"覆盖 {covered / vocab * 100:.2f}%（{covered:,}/{vocab:,}），"
            f"还缺 {missing:,} 个 id。",
            flush=True,
        )
    else:
        print(
            f"训练 shard 扫完仍缺 {missing:,} 个 id。"
            f"共 {tokens / 1e9:.3f}B token，覆盖 {covered / vocab * 100:.2f}%"
            f"（{covered:,}/{vocab:,}）。",
            flush=True,
        )
    _write_report(
        args.report,
        tokenizer_dir=args.tokenizer_dir,
        data_dir=args.data_dir,
        vocab=vocab,
        bos_id=bos_id,
        eos_id=eos_id,
        n_shards=len(shards),
        holdout=args.holdout,
        shards_done=len(shards),
        tokens=tokens,
        covered=covered,
        milestones=milestones,
        finished=not stopped_early,
    )
    print(f"报告写在 {args.report}", flush=True)
    if output is not None:
        output.close()


if __name__ == "__main__":
    main()
