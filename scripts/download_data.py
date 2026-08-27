"""下载并预处理公开情感识别数据集，生成本项目统一的 ``labels.csv``。

支持的数据集（共 5 个，全部自动下载）：
    * text  : GoEmotions（HuggingFace, 27 类映射 7 类，分层子采样至 3000 条）
    * audio : RAVDESS（HF 镜像 parquet, 8 类映射 7 类；Zenodo 直连过慢已弃用）、
              CASIA（中文 6 类）、ESD 中文子集（中文 5 类，分层子采样至 1500 条）、
              EMO-DB（德语 7 类）
    * 仅 IEMOCAP 需人工申请（可选），脚本仅打印指引，不自动下载。

生成格式（对齐 ``emotion_recognition/dataset.py`` 约定）：
    ``labels.csv`` 三列 ``text, audio_path, label``，其中：
    - ``audio_path`` 为相对 ``data_root``（即 ``./data``）的音频路径，如
      ``raw/audio/ravdess/Actor_01/03-01-01-01-01-01-01.wav``；
    - 纯文本样本 ``audio_path`` 留空，纯语音样本 ``text`` 留空；
    - ``label`` 必须是 7 类之一：happy/sad/angry/fear/surprise/disgust/neutral。

用法示例（在项目根目录执行）：:

    # 下载全部可自动获取的数据集（文本 + 语音）
    python scripts/download_data.py --dataset all

    # 仅下载文本数据集
    python scripts/download_data.py --dataset text

    # 仅下载 RAVDESS 语音数据集
    python scripts/download_data.py --dataset ravdess

    # 指定目标目录（默认 ./data/raw）
    python scripts/download_data.py --dataset all --target ./data/raw

网络说明：
    * HuggingFace 下载可能因网络/证书问题失败（如 ``unable to get local issuer
      certificate``）。可设置国内镜像后重试：::

        $env:HF_ENDPOINT = "https://hf-mirror.com"   # PowerShell

    * 所有 HF 数据均已改用国内镜像（hf-mirror.com），无需额外配置；EMO-DB
      直连 emodb.bilderbar.info。
    * RAVDESS 从 HF 镜像（hf-mirror.com）的 parquet 文件下载，共约 325 MB
      （2 个分片，音频以 RIFF/WAVE 字节内嵌），无需 torchcodec 解码。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 项目 7 类情绪标签（对齐 config.yaml）。
EMOTION_LABELS: List[str] = [
    "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral",
]

# GoEmotions 28 类 -> 7 类映射（按类别名，索引无关，见 docs/dataset.md L62-72）。
GOEMOTIONS_TO_7: Dict[str, str] = {
    # happy
    "joy": "happy", "amusement": "happy", "excitement": "happy",
    "love": "happy", "optimism": "happy", "relief": "happy",
    "pride": "happy", "gratitude": "happy", "approval": "happy",
    "admiration": "happy", "caring": "happy", "desire": "happy",
    # sad
    "sadness": "sad", "grief": "sad", "disappointment": "sad",
    "remorse": "sad", "embarrassment": "sad",
    # angry
    "anger": "angry", "annoyance": "angry", "disapproval": "angry",
    # fear
    "fear": "fear", "nervousness": "fear",
    # surprise
    "surprise": "surprise", "curiosity": "surprise", "confusion": "surprise",
    "realization": "surprise",
    # disgust / neutral
    "disgust": "disgust",
    "neutral": "neutral",
}

# RAVDESS 语音数据（speech-only，1440 条）的 HF 镜像 parquet 下载地址。
# 说明：Zenodo 直连（~640MB zip）在弱网络下频繁断连且极慢，已弃用。
# 改用 hf-mirror 的 datasets-server parquet 端点，音频以 RIFF/WAVE 字节内嵌，
# 用 pyarrow 直接读 bytes 写出 wav，绕过 datasets 的 torchcodec 音频解码。
RAVDESS_PARQUET_URLS: List[str] = [
    "https://hf-mirror.com/api/datasets/xbgoose/ravdess/parquet/default/train/{i}.parquet".format(i=i)
    for i in (0, 1)
]

# RAVDESS 情绪字符串（parquet 的 emotion 列）-> 7 类。
RAVDESS_EMOTION_NAME: Dict[str, str] = {
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "sad": "sad",
    "surprised": "surprise",
    "happy": "happy",
    "calm": "neutral",
    "neutral": "neutral",
}

# CASIA 中文语音情感数据（6 类，1200 条）的 HF 镜像 parquet 下载地址。
CASIA_PARQUET_URLS: List[str] = [
    f"https://hf-mirror.com/api/datasets/BillyLin/CASIA_speech_emotion_recognition_preload/parquet/default/train/{i}.parquet"
    for i in range(12)
]

# CASIA 情绪字符串 -> 7 类（注意 "anger" 而非 "angry"）。
CASIA_EMOTION_MAP: Dict[str, str] = {
    "anger": "angry", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprise",
}

# ESD 情感语音数据（中文子集，5 类，约 17500 条）的 HF 镜像 parquet 下载地址。
ESD_PARQUET_URLS: List[str] = [
    f"https://hf-mirror.com/api/datasets/jspaulsen/esd/parquet/default/train/{i}.parquet"
    for i in range(7)
]

# ESD 情绪字符串 -> 7 类（注意 "happiness"/"sadness" 而非 "happy"/"sad"）。
ESD_EMOTION_MAP: Dict[str, str] = {
    "anger": "angry", "happiness": "happy", "neutral": "neutral",
    "sadness": "sad", "surprise": "surprise",
}

# EMO-DB 德语情感语音（7 类，535 条）。
# 优先 HF parquet 镜像（~26MB，秒下），德国服务器 zip 作兜底（40MB，慢）。
EMODB_PARQUET_URLS: List[str] = [
    "https://hf-mirror.com/api/datasets/confit/emodb-parquet/parquet/default/train/0.parquet",
]
EMODB_ZIP_URL: str = "http://emodb.bilderbar.info/download/download.zip"

# EMO-DB HF parquet emotion 字符串 -> 7 类（无 surprise，EMO-DB 本就 7 类含 boredom/anxiety）。
EMODB_EMOTION_MAP: Dict[str, str] = {
    "anger": "angry",
    "anxiety": "fear",       # Angst → fear
    "boredom": "neutral",    # Langeweile → neutral
    "happiness": "happy",
    "sadness": "sad",
    "disgust": "disgust",
    "neutral": "neutral",
}

# EMO-DB zip 文件名第 5 位字符（0 基）-> 7 类情绪（兜底路径用）。
EMODB_EMOTION_LETTER: Dict[str, str] = {
    "W": "angry",   # Wut (anger)
    "L": "neutral", # Langeweile (boredom) → neutral
    "E": "disgust", # Ekel (disgust)
    "A": "fear",    # Angst (fear)
    "F": "happy",   # Freude (joy)
    "T": "sad",     # Trauer (sadness)
    "N": "neutral", # Neutral
}

# 每数据集最大样本数（分层子采样；0 = 不限制）。
# GoEmotions 27 类自然分布不均（disgust/fear 极少），故按占比分配而非均分；
# ESD 中文子集 5 类各 3500 条，按占比即每类 300 条。
MAX_SAMPLES: Dict[str, int] = {
    "goemotions": 3000,
    "esd": 1500,
    "ravdess": 0,
    "casia": 0,
    "emodb": 0,
}


def _stratified_sample_indices(labels: List[str], n: int, seed: int = 42) -> List[int]:
    """按 ``labels`` 分层子采样，返回被选中样本的下标（占比较小的类也至少保留 1 条）。"""
    if n <= 0 or len(labels) <= n:
        return list(range(len(labels)))
    import random

    rng = random.Random(seed)
    groups: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)

    total = len(labels)
    # 按各类占比分配配额，至少 1 个。
    alloc: Dict[str, int] = {
        lab: max(1, round(n * len(idx) / total)) for lab, idx in groups.items()
    }
    # 将配额调整到恰好 n（优先从样本最多的类增减）。
    diff = n - sum(alloc.values())
    labs_by_size = sorted(groups, key=lambda l: -len(groups[l]))
    k = 0
    while diff != 0 and labs_by_size:
        lab = labs_by_size[k % len(labs_by_size)]
        if diff > 0:
            alloc[lab] += 1
            diff -= 1
        elif alloc[lab] > 1:
            alloc[lab] -= 1
            diff += 1
        k += 1

    selected: List[int] = []
    for lab, idx in groups.items():
        kk = min(alloc[lab], len(idx))
        selected.extend(rng.sample(idx, kk))
    return selected


def _stratified_subsample(rows: List[List[str]], n: int, seed: int = 42) -> List[List[str]]:
    """对 ``[text, audio_path, label]`` 行做分层子采样（label 在第 2 列）。"""
    idx = _stratified_sample_indices([r[2] for r in rows], n, seed=seed)
    return [rows[i] for i in idx]


# ---------------------------------------------------------------------------
# 下载辅助
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1 << 20,
    max_retries: int = 20,
    backoff: float = 2.0,
) -> None:
    """流式下载文件到 ``dest``，支持断点续传与失败重试。

    大文件（如 RAVDESS ~640MB）在弱网络下容易 ``IncompleteRead``，
    通过 ``Range`` 头从已写入字节处续传，并在每次失败后指数退避重试，
    直到下载完整为止。
    """
    import time

    # HEAD 获取完整文件大小（失败则置 0，后续从 GET content-length 推断）。
    expected: int = 0
    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        if head.ok:
            expected = int(head.headers.get("content-length", 0))
    except Exception:
        pass

    # 仅当本地文件已完整时才跳过；HEAD 失败时 expected=0，不跳过，走续传。
    if dest.exists() and expected > 0 and dest.stat().st_size >= expected:
        print(f"    [skip] 已完整: {dest.name} ({dest.stat().st_size} bytes)")
        return

    # 清理 0 字节空文件。
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()

    total: int = 0
    for attempt in range(1, max_retries + 1):
        done: int = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done > 0 else {}
        try:
            resp = requests.get(url, stream=True, timeout=60, headers=headers)
            # 416 Range Not Satisfiable：本地文件已大于等于服务器文件，
            # 可能是上次 HEAD 的 content-length 不准或文件已完整但未触发 skip。
            # 删除本地文件从头重下（视为 200 全文）。
            if resp.status_code == 416:
                resp.close()
                if dest.exists():
                    dest.unlink()
                done = 0
                headers = {}
                resp = requests.get(url, stream=True, timeout=60, headers=headers)
            resp.raise_for_status()
            # 服务器可能忽略 Range 返回 200（全文）。206 = 续传生效。
            if resp.status_code == 206 and done > 0:
                mode = "ab"
            else:
                if done > 0:
                    done = 0  # 服务器返回全文，重置计数从头写
                mode = "wb"
            # 推断总大小：206 时 content-length 为剩余，需加已下载；200 时为全文。
            cl = int(resp.headers.get("content-length", 0))
            total = cl + done if resp.status_code == 206 else cl
            if total == 0 and expected > 0:
                total = expected
            with dest.open(mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r    {pct:3d}%  ({done // (1 << 20)} MB)",
                              end="", flush=True)
            # 校验完整性：若已知总大小且已下载量达标，视为完成。
            if total == 0 or done >= total:
                print("\r    [done] 100%", flush=True)
                return
            print(f"\n    [warn] 下载不完整 ({done}/{total} bytes)，重试...")
        except Exception as exc:
            print(f"\n    [warn] 第 {attempt} 次尝试失败: {type(exc).__name__}: {exc}")
            print(f"           已下载 {done} bytes，{backoff:.0f}s 后重试...")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
    raise RuntimeError(f"下载失败（已重试 {max_retries} 次）: {url}")


# ---------------------------------------------------------------------------
# 文本数据集
# ---------------------------------------------------------------------------

def _load_hf_dataset(name: str, config: Optional[str] = None, **kwargs):
    """延迟导入 ``datasets`` 并加载数据集，失败时给出清晰指引。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "缺少 `datasets` 库。请先安装：pip install datasets"
        ) from exc
    try:
        if config is not None:
            return load_dataset(name, config, **kwargs)
        return load_dataset(name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"从 HuggingFace 加载 {name!r} 失败：{exc}\n"
            "若为 SSL 证书错误，可设置镜像后重试：\n"
            '  PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"\n'
            "  Bash:       export HF_ENDPOINT=https://hf-mirror.com"
        ) from exc


def download_goemotions(raw_dir: Path, out_csv: Path) -> int:
    """下载 GoEmotions（27 类映射 7 类，取单标签样本，分层子采样）。"""
    print("[text] 下载 GoEmotions ...")
    ds = _load_hf_dataset("google-research-datasets/go_emotions", config="simplified")
    # 类别名列表（按索引），用于把多标签整数转成类别名。
    names: List[str] = list(ds["train"].features["labels"].feature.names)
    rows: List[List[str]] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for example in ds[split]:
            labels: List[int] = example["labels"]
            if not labels:
                continue  # 跳过无标签样本
            # 多标签样本取首个标签（简化处理，保证单标签 7 类分类）。
            name: str = names[labels[0]]
            mapped: str = GOEMOTIONS_TO_7.get(name)
            if mapped is None:
                continue  # 未覆盖的类别（理论上不应出现）直接跳过
            rows.append([str(example["text"]).strip(), "", mapped])
    max_samples = MAX_SAMPLES.get("goemotions", 0)
    if max_samples > 0:
        before = len(rows)
        rows = _stratified_subsample(rows, max_samples)
        print(f"    [info] GoEmotions 分层子采样 {before} -> {len(rows)} 行")
    _append_csv(out_csv, rows)
    print(f"    [ok] GoEmotions 写入 {len(rows)} 行")
    return len(rows)


# ---------------------------------------------------------------------------
# 语音数据集
# ---------------------------------------------------------------------------

def download_ravdess(raw_dir: Path, out_csv: Path) -> int:
    """下载 RAVDESS（HF 镜像 parquet），解出 wav 并按 emotion 列映射 7 类。

    数据源为 ``xbgoose/ravdess`` 的 datasets-server parquet（2 个分片），
    ``audio.bytes`` 是完整 RIFF/WAVE 字节，用 pyarrow 读 bytes 直接写 wav，
    不依赖 torchcodec/librosa 音频解码。
    """
    print("[audio] 下载 RAVDESS（HF 镜像 parquet）...")
    audio_dir = raw_dir / "audio" / "ravdess"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "缺少 `pyarrow` 库。请先安装：pip install pyarrow"
        ) from exc

    # 下载 parquet 分片（复用带断点续传的 download_file）。
    parquet_paths: List[Path] = []
    for i, url in enumerate(RAVDESS_PARQUET_URLS):
        pq_path = raw_dir / f"_ravdess_{i}.parquet"
        download_file(url, pq_path)
        parquet_paths.append(pq_path)

    rows: List[List[str]] = []
    total = 0
    for pq_path in parquet_paths:
        df = pq.read_table(str(pq_path)).to_pandas()
        for _, row in df.iterrows():
            emotion_raw = str(row["emotion"])
            label = RAVDESS_EMOTION_NAME.get(emotion_raw)
            if label is None:
                print(f"    [warn] 未知情绪 {emotion_raw}，跳过 {row['audio']['path']}")
                continue
            actor = int(row["actor"])
            fname = row["audio"]["path"]
            wav_bytes = row["audio"]["bytes"]

            actor_dir = audio_dir / f"Actor_{actor:02d}"
            actor_dir.mkdir(parents=True, exist_ok=True)
            (actor_dir / fname).write_bytes(wav_bytes)

            rel_audio = Path("raw") / "audio" / "ravdess" / f"Actor_{actor:02d}" / fname
            rows.append(["", rel_audio.as_posix(), label])
            total += 1

    _append_csv(out_csv, rows)
    print(f"    [ok] RAVDESS 写入 {total} 行")
    return total


def download_casia(raw_dir: Path, out_csv: Path) -> int:
    """下载 CASIA 中文语音情感数据（6 类，1200 条）。

    数据源为 ``BillyLin/CASIA_speech_emotion_recognition_preload`` 的
    datasets-server parquet（12 个分片），``audio_bytes`` 为扁平二进制，
    是完整 RIFF/WAVE 字节（16kHz/mono/16-bit），直接写出 wav。文本列为
    中文朗读脚本（不含情绪信息），故 labels.csv 文本字段留空（纯语音）。
    """
    print("[audio] 下载 CASIA（中文 6 类）...")
    audio_dir = raw_dir / "audio" / "casia"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("缺少 `pyarrow` 库。请先安装：pip install pyarrow") from exc

    parquet_paths: List[Path] = []
    for i, url in enumerate(CASIA_PARQUET_URLS):
        pq_path = raw_dir / f"_casia_{i}.parquet"
        download_file(url, pq_path)
        parquet_paths.append(pq_path)

    rows: List[List[str]] = []
    idx = 0
    for pq_path in parquet_paths:
        df = pq.read_table(str(pq_path)).to_pandas()
        for _, row in df.iterrows():
            emotion_raw = str(row["emotion"]).strip()
            label = CASIA_EMOTION_MAP.get(emotion_raw)
            if label is None:
                print(f"    [warn] 未知情绪 {emotion_raw!r}，跳过")
                continue
            wav_bytes = row["audio_bytes"]
            fname = f"casia_{idx:05d}_{label}.wav"
            (audio_dir / fname).write_bytes(wav_bytes)
            rel_audio = Path("raw") / "audio" / "casia" / fname
            rows.append(["", rel_audio.as_posix(), label])
            idx += 1

    _append_csv(out_csv, rows)
    print(f"    [ok] CASIA 写入 {len(rows)} 行")
    return len(rows)


def download_esd(raw_dir: Path, out_csv: Path) -> int:
    """下载 ESD 中文子集（5 类，约 17500 条，分层子采样至 1500 条）。

    数据源为 ``jspaulsen/esd`` 的 datasets-server parquet（7 个分片），
    ``audio`` 为 struct{bytes, path}（与 RAVDESS 相同），``audio.bytes``
    是完整 RIFF/WAVE 字节（16kHz/mono/16-bit）。仅保留 ``language == 'zh'``
    的行。``transcript`` 为中文朗读脚本，写入 ``text`` 列。
    """
    print("[audio] 下载 ESD（中文子集 5 类）...")
    audio_dir = raw_dir / "audio" / "esd"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("缺少 `pyarrow` 库。请先安装：pip install pyarrow") from exc

    parquet_paths: List[Path] = []
    for i, url in enumerate(ESD_PARQUET_URLS):
        pq_path = raw_dir / f"_esd_{i}.parquet"
        download_file(url, pq_path)
        parquet_paths.append(pq_path)

    # 先收集全部中文样本（含音频字节与中文脚本），子采样后再写盘，避免写出多余 WAV。
    entries: List[List[str]] = []  # [text, label, fname, wav_bytes]
    idx = 0
    for pq_path in parquet_paths:
        df = pq.read_table(str(pq_path)).to_pandas()
        for _, row in df.iterrows():
            language = str(row["language"]).strip()
            if language != "zh":
                continue  # 仅保留中文子集
            emotion_raw = str(row["emotion"]).strip()
            label = ESD_EMOTION_MAP.get(emotion_raw)
            if label is None:
                print(f"    [warn] 未知情绪 {emotion_raw!r}，跳过")
                continue
            speaker_id = str(row["speaker_id"]).strip()
            transcript = str(row["transcript"]).strip()
            wav_bytes = row["audio"]["bytes"]
            fname = f"esd_{speaker_id}_{idx:05d}_{label}.wav"
            entries.append([transcript, label, fname, wav_bytes])
            idx += 1

    max_samples = MAX_SAMPLES.get("esd", 0)
    keep = _stratified_sample_indices([e[1] for e in entries], max_samples)
    if max_samples > 0:
        print(f"    [info] ESD 分层子采样 {len(entries)} -> {len(keep)} 行")

    rows: List[List[str]] = []
    for i in keep:
        text, label, fname, wav_bytes = entries[i]
        (audio_dir / fname).write_bytes(wav_bytes)
        rel_audio = Path("raw") / "audio" / "esd" / fname
        rows.append([text, rel_audio.as_posix(), label])

    _append_csv(out_csv, rows)
    print(f"    [ok] ESD（中文）写入 {len(rows)} 行")
    return len(rows)


def download_emodb(raw_dir: Path, out_csv: Path) -> int:
    """下载 EMO-DB 德语情感语音（7 类，304 条 parquet 镜像）。

    优先 HF parquet 镜像 ``confit/emodb-parquet``（~26MB，秒下，
    schema: ``audio`` struct{bytes,path}, ``emotion`` string, ``label`` int64，
    16kHz/mono/16-bit）。失败回退德国服务器 zip（40MB，慢，
    文件名第 5 位字符编码情绪：W=anger, L=boredom, E=disgust, A=fear,
    F=joy, T=sadness, N=neutral）。纯语音，文本字段留空。
    """
    print("[audio] 下载 EMO-DB（德语 7 类）...")
    audio_dir = raw_dir / "audio" / "emodb"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = _download_emodb_parquet(raw_dir, audio_dir)
        if rows:
            _append_csv(out_csv, rows)
            print(f"    [ok] EMO-DB 写入 {len(rows)} 行（HF parquet）")
            return len(rows)
        print("    [warn] parquet 镜像无数据，回退德国服务器 zip...")
    except Exception as exc:
        print(f"    [warn] parquet 镜像失败 ({type(exc).__name__})，回退德国服务器 zip...")

    rows = _download_emodb_zip(raw_dir, audio_dir)
    _append_csv(out_csv, rows)
    print(f"    [ok] EMO-DB 写入 {len(rows)} 行（德国服务器 zip）")
    return len(rows)


def _download_emodb_parquet(raw_dir: Path, audio_dir: Path) -> List[List[str]]:
    """HF parquet 镜像路径：audio struct + emotion string。"""
    import pyarrow.parquet as pq

    rows: List[List[str]] = []
    idx = 0
    for i, url in enumerate(EMODB_PARQUET_URLS):
        pq_path = raw_dir / f"_emodb_{i}.parquet"
        download_file(url, pq_path)
        df = pq.read_table(str(pq_path)).to_pandas()
        for _, row in df.iterrows():
            emotion_raw = str(row["emotion"]).strip()
            label = EMODB_EMOTION_MAP.get(emotion_raw)
            if label is None:
                print(f"    [warn] 未知情绪 {emotion_raw!r}，跳过")
                continue
            aud = row["audio"]
            wav_bytes = aud.get("bytes") if isinstance(aud, dict) else aud
            if not wav_bytes:
                continue
            fname = f"emodb_{idx:05d}_{label}.wav"
            (audio_dir / fname).write_bytes(wav_bytes)
            rel_audio = Path("raw") / "audio" / "emodb" / fname
            rows.append(["", rel_audio.as_posix(), label])
            idx += 1
    return rows


def _download_emodb_zip(raw_dir: Path, audio_dir: Path) -> List[List[str]]:
    """德国服务器 zip 兜底路径：文件名第 5 位字符编码情绪。"""
    import zipfile

    zip_path = raw_dir / "_emodb.zip"
    download_file(EMODB_ZIP_URL, zip_path, max_retries=30)

    rows: List[List[str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            fname = Path(info.filename).name
            if info.is_dir() or not fname.lower().endswith(".wav"):
                continue
            if len(fname) < 6:
                continue
            emotion_letter = fname[5]
            label = EMODB_EMOTION_LETTER.get(emotion_letter)
            if label is None:
                print(f"    [warn] 未知情绪字母 {emotion_letter!r}，跳过 {fname}")
                continue
            (audio_dir / fname).write_bytes(zf.read(info.filename))
            rel_audio = Path("raw") / "audio" / "emodb" / fname
            rows.append(["", rel_audio.as_posix(), label])
    return rows


def print_manual_instructions() -> None:
    """打印需人工申请的数据集指引（仅 IEMOCAP）。"""
    print("\n[manual] 以下数据集需人工申请，脚本不自动下载：")
    print("  - IEMOCAP   : https://sail.usc.edu/iemocap/ (需签署协议)")
    print("  说明：IEMOCAP 为可选数据集，仅在需要与相关基准做可比性对比时才需申请；")
    print("        其余 5 个数据集均已自动下载。下载后按 docs/dataset.md 预处理，")
    print("        并将行追加到 labels.csv。")


# ---------------------------------------------------------------------------
# CSV 写入
# ---------------------------------------------------------------------------

def _append_csv(out_csv: Path, rows: List[List[str]]) -> None:
    """把行追加到 labels.csv（存在则续写，不存在则写表头）。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["text", "audio_path", "label"])
        writer.writerows(rows)


def _validate_labels(out_csv: Path) -> None:
    """校验生成的 CSV 的标签都在 7 类内。"""
    if not out_csv.exists():
        return
    bad: List[str] = []
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] not in EMOTION_LABELS:
                bad.append(row["label"])
    if bad:
        print(f"[warn] 发现 {len(bad)} 个非法标签: {sorted(set(bad))[:10]}")
    else:
        print("[ok] labels.csv 标签校验通过（均在 7 类内）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Callable[..., int]] = {
    "goemotions": download_goemotions,
    "ravdess": download_ravdess,
    "casia": download_casia,
    "esd": download_esd,
    "emodb": download_emodb,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载并预处理公开情感识别数据集，生成 labels.csv"
    )
    parser.add_argument(
        "--target", type=Path, default=Path("./data/raw"),
        help="原始数据目录（默认 ./data/raw）",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "text", "audio", "goemotions", "ravdess",
                 "casia", "esd", "emodb"],
        help="要下载的数据集（默认 all）",
    )
    args = parser.parse_args()

    raw_dir: Path = args.target.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    # labels.csv 直接放在 data/raw/ 下（对齐 build_real_loaders 的读取路径）。
    out_csv: Path = raw_dir / "labels.csv"
    print(f"[INFO] 目标目录: {raw_dir}")
    print(f"[INFO] labels.csv: {out_csv}")

    selected: List[str] = {
        "all": ["goemotions", "ravdess", "casia", "esd", "emodb"],
        "text": ["goemotions"],
        "audio": ["ravdess", "casia", "esd", "emodb"],
        "goemotions": ["goemotions"],
        "ravdess": ["ravdess"],
        "casia": ["casia"],
        "esd": ["esd"],
        "emodb": ["emodb"],
    }[args.dataset]

    total = 0
    for name in selected:
        try:
            total += DATASETS[name](raw_dir, out_csv)
        except RuntimeError as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)

    print_manual_instructions()
    print(f"\n[DONE] 共写入 {total} 行到 {out_csv}")
    _validate_labels(out_csv)
    print("\n下一步训练：")
    print("  python -m emotion_recognition.train --config config.yaml "
          "--epochs 20 --batch_size 32 --lr 1e-4 --use-pretrained-bert")


if __name__ == "__main__":
    main()
