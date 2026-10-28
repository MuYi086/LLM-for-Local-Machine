"""IndexTTS-2.5 主模型和辅助资源检查及只读运行视图。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "config.yaml",
    "gpt.pth",
    "codec.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
)
W2V_BERT_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "conformer_shaw.pt",
)
BIGVGAN_REQUIRED_FILES = ("config.json", "bigvgan_generator.pt")


def missing_model_resources(
    model_dir: str | os.PathLike[str],
    w2v_bert_dir: str | os.PathLike[str],
    campplus_path: str | os.PathLike[str],
    bigvgan_dir: str | os.PathLike[str],
) -> list[str]:
    """列出 IndexTTS-2.5 推理所需但本机缺少的模型文件。"""
    model_path = Path(model_dir).expanduser()
    w2v_path = Path(w2v_bert_dir).expanduser()
    campplus = Path(campplus_path).expanduser()
    bigvgan = Path(bigvgan_dir).expanduser()
    missing = [
        f"model/{name}" for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()
    ]
    missing.extend(
        f"w2v-bert-2.0/{name}"
        for name in W2V_BERT_REQUIRED_FILES
        if not (w2v_path / name).is_file()
    )
    if not campplus.is_file():
        missing.append("campplus/campplus_cn_common.bin")
    missing.extend(
        f"bigvgan/{name}" for name in BIGVGAN_REQUIRED_FILES if not (bigvgan / name).is_file()
    )
    return missing


def build_model_view(
    model_dir: Path,
    view_dir: Path,
    w2v_bert_dir: Path,
    campplus_path: Path,
    bigvgan_dir: Path,
) -> Path:
    """创建指向本地只读权重的临时目录视图。"""
    model_path = model_dir.expanduser().resolve()
    view_path = view_dir.expanduser().resolve()
    w2v_path = w2v_bert_dir.expanduser().resolve()
    campplus = campplus_path.expanduser().resolve()
    bigvgan = bigvgan_dir.expanduser().resolve()
    missing = missing_model_resources(model_path, w2v_path, campplus, bigvgan)
    if missing:
        raise FileNotFoundError("IndexTTS-2.5 本地模型资源不完整: " + ", ".join(missing))
    if view_path == model_path or model_path in view_path.parents:
        raise ValueError("IndexTTS-2.5 运行视图不能创建在模型源目录内。")
    if view_path.exists():
        raise FileExistsError(f"IndexTTS-2.5 运行视图已存在: {view_path}")

    view_path.mkdir(parents=True)
    try:
        for filename in REQUIRED_MODEL_FILES:
            os.symlink(model_path / filename, view_path / filename)

        qwen_emotion_dir = model_path / "qwen0.6bemo4-merge"
        if qwen_emotion_dir.is_dir():
            os.symlink(qwen_emotion_dir, view_path / qwen_emotion_dir.name)

        cache_path = view_path / "hf_cache"
        cache_path.mkdir()
        os.symlink(w2v_path, cache_path / "w2v-bert-2.0")
        os.symlink(campplus, cache_path / "campplus_cn_common.bin")
        os.symlink(bigvgan, cache_path / "bigvgan")
    except Exception:
        shutil.rmtree(view_path, ignore_errors=True)
        raise
    return view_path
