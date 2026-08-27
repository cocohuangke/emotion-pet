"""预提取所有音频的 MFCC 并缓存到磁盘(pickle)。

一次性计算,后续训练直接加载进内存,避免每个 epoch 重复提取。
MFCC 是确定性特征(同文件同参数 = 逐位相同),只要音频文件不变,
缓存永远有效,与标签 / 采样 / split 无关。

并行方式选多线程(ThreadPoolExecutor):librosa 底层是 numpy/scipy 的
C 实现,会释放 GIL,线程能真正并行;内存共享、零 IPC 开销,优于多进程。

用法:
    python scripts/precompute_mfcc.py [--csv data/raw/labels.csv] [--workers 8]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from emotion_recognition.dataset import (  # noqa: E402
    AUDIO_PATH_COLUMN,
    DEFAULT_MAX_AUDIO_FRAMES,
    DEFAULT_N_MFCC,
    extract_mfcc,
)

DATA_ROOT: Path = PROJECT_ROOT / "data"
DEFAULT_CSV: Path = DATA_ROOT / "raw" / "labels.csv"
DEFAULT_CACHE: Path = DATA_ROOT / "raw" / "mfcc_cache.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="预提取 MFCC 缓存")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="labels.csv 路径")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="缓存输出路径(.pkl)")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"labels.csv 不存在: {args.csv}")

    df: pd.DataFrame = pd.read_csv(args.csv)
    df[AUDIO_PATH_COLUMN] = df[AUDIO_PATH_COLUMN].fillna("")
    audio_rels = sorted({str(r) for r in df[AUDIO_PATH_COLUMN] if str(r)})

    print(f"[mfcc] 唯一音频数 = {len(audio_rels)}")
    print(f"[mfcc] 线程数 = {args.workers}, 输出 = {args.cache}")

    cache: Dict[str, np.ndarray] = {}
    if args.cache.exists():
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)
        print(f"[mfcc] 续传:已加载 {len(cache)} 条缓存")
    audio_rels = [r for r in audio_rels if r not in cache]
    if not audio_rels:
        print("[mfcc] 全部已完成,无需提取")
        return
    print(f"[mfcc] 待提取 {len(audio_rels)} 条")
    missing: list = []
    t0: float = time.time()

    # 预热:单线程提取第一个文件,触发 librosa/numba 的 JIT 编译完成。
    # 否则多线程并发触发 JIT 编译会导致 race condition,进程原生崩溃(退出码 -1)。
    if audio_rels:
        _warm = extract_mfcc(DATA_ROOT / audio_rels[0])
        print(f"[mfcc] 预热完成 shape={_warm.shape}")

    def work(rel: str) -> tuple:
        mfcc = extract_mfcc(
            DATA_ROOT / rel,
            n_mfcc=DEFAULT_N_MFCC,
            max_frames=DEFAULT_MAX_AUDIO_FRAMES,
        )
        return rel, mfcc

    done: int = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, rel): rel for rel in audio_rels}
        for fut in as_completed(futures):
            rel = futures[fut]
            rel_key, mfcc = fut.result()
            cache[rel_key] = mfcc
            if float(mfcc.sum()) == 0.0:
                missing.append(rel_key)
            done += 1
            if done % 1000 == 0 or done == len(audio_rels):
                # 定期 checkpoint 写入,中断后可续传。
                with open(args.cache, "wb") as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(
                    f"[mfcc] {done}/{len(audio_rels)} 完成 "
                    f"({elapsed:.0f}s, {speed:.0f} 条/s, checkpoint 已写)"
                )

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with open(args.cache, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = args.cache.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"[mfcc] 缓存写入 {args.cache} ({size_mb:.1f} MB)")
    print(f"[mfcc] 总耗时 {elapsed:.0f}s, 命中 {len(cache)} 条, 全零(缺失) {len(missing)} 个")
    if missing:
        print(f"[mfcc] 缺失样本前 5 个: {missing[:5]}")


if __name__ == "__main__":
    main()
