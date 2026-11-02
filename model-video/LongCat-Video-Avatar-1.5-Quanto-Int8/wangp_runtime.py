"""WanGP 权重目录映射与本地配置工具。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

MODEL_WEIGHT_NAMES = (
    "longcat_avatar_v1_5_quanto_bf16_int8.safetensors",
    "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
)
TOKENIZER_FILE_NAMES = (
    "tokenizer.json",
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
DEFAULT_MEMORY_PROFILE = 0
SUPPORTED_MEMORY_PROFILES = frozenset({0, 4})


def validate_memory_profile(value: int | str) -> int:
    """校验本服务支持的 WanGP 显存档位。"""
    try:
        profile = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("LongCat Avatar 8390 仅支持 MMGP Profile 0 或 Profile 4。") from exc
    if profile not in SUPPORTED_MEMORY_PROFILES:
        raise ValueError("LongCat Avatar 8390 仅支持 MMGP Profile 0 或 Profile 4。")
    return profile


def configured_paths(
    wangp_dir: str | Path,
    model_dir: str | Path,
    base_model_dir: str | Path,
) -> dict[Path, Path]:
    """返回 WanGP 名称到已下载 LongCat 资源的只读链接映射。"""
    root = Path(wangp_dir).expanduser().resolve()
    avatar = Path(model_dir).expanduser().resolve()
    base = Path(base_model_dir).expanduser().resolve()
    checkpoints = root / "ckpts"
    tokenizer_dir = checkpoints / "umt5-xxl"
    return {
        checkpoints / MODEL_WEIGHT_NAMES[0]: avatar / MODEL_WEIGHT_NAMES[0],
        checkpoints / "umt5-xxl" / MODEL_WEIGHT_NAMES[1]: avatar / MODEL_WEIGHT_NAMES[1],
        **{tokenizer_dir / name: avatar / "umt5-xxl" / name for name in TOKENIZER_FILE_NAMES},
        checkpoints / "whisper-large-v3": avatar / "whisper-large-v3",
        checkpoints / "Wan2.1_VAE_bf16.safetensors": (
            base / "vae" / "diffusion_pytorch_model.safetensors"
        ),
        root / "loras" / "longcat_avatar_v1_5" / "dmd_lora.safetensors": (
            avatar / "lora" / "dmd_lora.safetensors"
        ),
    }


def link_status(
    wangp_dir: str | Path,
    model_dir: str | Path,
    base_model_dir: str | Path,
) -> dict[str, bool]:
    """检查 WanGP 所需别名是否指向预期的本地资源。"""
    status: dict[str, bool] = {}
    for destination, source in configured_paths(wangp_dir, model_dir, base_model_dir).items():
        try:
            status[str(destination.relative_to(Path(wangp_dir).expanduser().resolve()))] = (
                (
                    source.is_file() and source.stat().st_size > 0
                    if source.is_file()
                    else source.is_dir()
                )
                and destination.exists()
                and destination.resolve() == source.resolve()
            )
        except (OSError, ValueError):
            status[str(destination)] = False
    return status


def ensure_links(
    wangp_dir: str | Path,
    model_dir: str | Path,
    base_model_dir: str | Path,
) -> dict[str, bool]:
    """建立指向现有权重的软链接，遇到同名冲突时停止而不覆盖。"""
    mapping = configured_paths(wangp_dir, model_dir, base_model_dir)
    for destination, source in mapping.items():
        if not source.exists():
            raise FileNotFoundError(f"WanGP 所需 LongCat 文件不存在：{source}")
        if source.is_file() and source.stat().st_size <= 0:
            raise ValueError(f"WanGP 所需 LongCat 文件为空：{source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            raise FileExistsError(f"WanGP 权重别名已指向其他位置：{destination}")
        if destination.exists():
            try:
                if destination.samefile(source):
                    continue
            except OSError:
                pass
            raise FileExistsError(f"WanGP 权重别名已存在且不是目标文件：{destination}")
        destination.symlink_to(source, target_is_directory=source.is_dir())
    return link_status(wangp_dir, model_dir, base_model_dir)


def update_wangp_config(
    wangp_dir: str | Path,
    *,
    memory_profile: int = DEFAULT_MEMORY_PROFILE,
    attention_mode: str = "sdpa",
) -> Path:
    """更新 WanGP 配置中的检查点目录和显存运行档。"""
    memory_profile = validate_memory_profile(memory_profile)
    root = Path(wangp_dir).expanduser().resolve()
    config_path = root / "wgp_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"WanGP 配置尚未生成：{config_path}")
    try:
        config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 WanGP 配置：{config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("WanGP wgp_config.json 顶层必须是 JSON 对象。")

    config.update(
        {
            "checkpoints_paths": ["ckpts"],
            "loras_root": "loras",
            "profile": memory_profile,
            "video_profile": memory_profile,
            "image_profile": memory_profile,
            "attention_mode": attention_mode,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=".wgp_config_", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(config, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)
    return config_path
