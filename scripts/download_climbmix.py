#!/usr/bin/env python3
"""Download ~100B tokens from karpathy/climbmix-400b-shuffle.

The dataset is a global shuffle of nvidia/Nemotron-ClimbMix: 400B GPT-2
tokens, decoded to text and packed into equal parquet shards of about 250M
characters each (see karpathy/nanochat dev/repackage_data_reference.py).
A prefix of the shards is therefore an unbiased subset, and

    n = ceil(num_shards * target_tokens / 400e9)

shards is the 100B-token slice (1636 of 6543 on the current repo).

Shards are written under the repository ``data/`` directory, which is
gitignored. Re-running skips files whose size already matches the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "karpathy/climbmix-400b-shuffle"
SOURCE_TOKENS = 400_000_000_000
API_TREE = (
    "https://huggingface.co/api/datasets/"
    f"{REPO_ID}/tree/main?recursive=true&limit=1000"
)
RESOLVE_URL = (
    "https://huggingface.co/datasets/"
    f"{REPO_ID}/resolve/main/{{name}}"
)


def _request(url: str) -> tuple[object, str | None]:
    headers = {"User-Agent": "download-climbmix/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
        return payload, response.headers.get("Link")


def list_shards() -> list[dict]:
    """Return every parquet shard sorted by filename, with size and sha256."""
    shards: list[dict] = []
    url: str | None = API_TREE
    while url:
        page, link = _request(url)
        if not isinstance(page, list):
            raise RuntimeError(f"unexpected HF tree response: {type(page)}")
        for item in page:
            path = item.get("path", "")
            if not path.endswith(".parquet"):
                continue
            lfs = item.get("lfs") or {}
            shards.append(
                {
                    "name": path,
                    "size": int(item["size"]),
                    "sha256": lfs.get("oid"),
                }
            )
        url = None
        if link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";", 1)[0].strip().strip("<>")
    shards.sort(key=lambda shard: shard["name"])
    expected = [f"shard_{index:05d}.parquet" for index in range(len(shards))]
    names = [shard["name"] for shard in shards]
    if names != expected:
        raise RuntimeError(
            "shard names are not a contiguous shard_00000.. sequence; "
            f"got {names[:3]} .. {names[-3:]}"
        )
    return shards


def select_shards(shards: list[dict], target_tokens: int) -> list[dict]:
    if target_tokens <= 0 or target_tokens > SOURCE_TOKENS:
        raise ValueError(
            f"target tokens must be in (0, {SOURCE_TOKENS}], got {target_tokens}"
        )
    count = -(-len(shards) * target_tokens // SOURCE_TOKENS)  # ceil div
    return shards[:count]


def estimated_tokens(num_selected: int, num_total: int) -> int:
    return num_selected * SOURCE_TOKENS // num_total


def _curl_download(url: str, partial: Path) -> None:
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        command[1:1] = ["--header", f"Authorization: Bearer {token}"]
    subprocess.run(command, check=True)


def download_one(shard: dict, out_dir: Path) -> tuple[str, int, bool]:
    dest = out_dir / shard["name"]
    expected = shard["size"]
    if dest.is_file() and dest.stat().st_size == expected:
        return shard["name"], expected, False
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.is_file() and partial.stat().st_size > expected:
        partial.unlink()
    url = RESOLVE_URL.format(name=shard["name"])
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            _curl_download(url, partial)
            if partial.stat().st_size == expected:
                partial.replace(dest)
                return shard["name"], expected, True
            last_error = RuntimeError(
                f"{shard['name']} size {partial.stat().st_size} != {expected}"
            )
        except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
            last_error = error
        time.sleep(2)
    raise RuntimeError(f"failed to download {shard['name']}") from last_error


def _format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes:.0f}B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "climbmix-100b",
        help="directory to write the parquet shards into (default: <repo>/data/climbmix-100b, gitignored)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100_000_000_000,
        help="target token count, scaled from the 400B source (default: 100B)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel curl downloads",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"listing {REPO_ID} ...", flush=True)
    shards = list_shards()
    selected = select_shards(shards, args.tokens)
    tokens = estimated_tokens(len(selected), len(shards))
    total_bytes = sum(shard["size"] for shard in selected)
    manifest = {
        "repo": REPO_ID,
        "source_tokens": SOURCE_TOKENS,
        "target_tokens": args.tokens,
        "estimated_tokens": tokens,
        "num_shards": len(selected),
        "num_shards_total": len(shards),
        "bytes": total_bytes,
        "files": selected,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"selected {len(selected)}/{len(shards)} shards, "
        f"~{tokens / 1e9:.3f}B tokens, {_format_bytes(total_bytes)}",
        flush=True,
    )

    started = time.time()
    done_bytes = 0
    done_count = 0
    lock = threading.Lock()
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, shard, out_dir) for shard in selected]
        for future in as_completed(futures):
            try:
                name, size, fetched = future.result()
            except Exception as error:  # noqa: BLE001 - report and keep going
                with lock:
                    failures.append(str(error))
                    print(f"ERROR {error}", flush=True)
                continue
            with lock:
                done_count += 1
                done_bytes += size
                elapsed = max(time.time() - started, 1e-6)
                rate = done_bytes / elapsed
                remaining = max(total_bytes - done_bytes, 0)
                eta = remaining / rate if rate else 0
                action = "got" if fetched else "skip"
                print(
                    f"[{done_count}/{len(selected)}] {action} {name}  "
                    f"{_format_bytes(done_bytes)}/{_format_bytes(total_bytes)}  "
                    f"{_format_bytes(rate)}/s  ETA {eta / 60:.0f}m",
                    flush=True,
                )

    if failures:
        print(f"{len(failures)} downloads failed. Re-run to resume.", file=sys.stderr)
        return 1
    print(
        f"done: {out_dir}  ~{tokens / 1e9:.3f}B tokens  {_format_bytes(total_bytes)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
