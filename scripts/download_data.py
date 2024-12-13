"""数据下载与预处理脚本

下载公开情感识别数据集并整理为本项目统一格式。
详见 docs/dataset.md。
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument("--target", type=str, default="./data/raw",
                        help="Target directory for raw data")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "text", "audio"],
                        help="Which dataset to download")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Data target: {target}")
    print("[INFO] This script downloads public emotion datasets.")
    print("[INFO] See docs/dataset.md for manual instructions.")
    print(f"[INFO] Requested: {args.dataset}")

    # 占位：实际下载逻辑由用户根据网络环境执行
    # 推荐数据集见 docs/dataset.md
    print("[DONE] Scaffold created. Populate data/raw/ manually per docs/dataset.md.")


if __name__ == "__main__":
    main()
