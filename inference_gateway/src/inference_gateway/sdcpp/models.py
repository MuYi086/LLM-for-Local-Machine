"""模型清单和固定 CLI 参数映射，不推断模型 family 的能力。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import local_path, positive_number

TASKS = {"text-to-image", "image-to-image", "text-to-video", "image-to-video"}
COMPONENTS = {
    "model",
    "diffusion_model",
    "high_noise_diffusion_model",
    "vae",
    "clip_l",
    "clip_g",
    "t5xxl",
    "llm",
    "llm_vision",
    "clip_vision",
    "tokenizer",
}
DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 1.0,
    "sampler": "euler",
    "frames": 1,
    "fps": 16,
}
LIMITS = {
    "max_width": 1024,
    "max_height": 1024,
    "max_pixels": 1048576,
    "max_steps": 50,
    "max_frames": 1,
    "width_multiple": 8,
    "height_multiple": 8,
    "frame_multiple": 1,
    "frame_offset": 0,
}
RUNTIME = {
    "backend": "cuda0",
    "params_backend": "cpu",
    "flash_attn": "auto",
    "vae_tiling": True,
    "temporal_tiling": False,
    "offload_to_cpu": False,
    "image_to_image_flag": "init-img",
    "max_vram_gib": 14,
}


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须为不小于 {minimum} 的整数。")
    return value


def _fields(value: object, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"{name} 包含未知字段或不是对象。")
    return value


@dataclass(frozen=True)
class GenerationRequest:
    """一次生成的结构化参数；路径仅供受信任的本机调用方使用。"""

    task: str
    prompt: str
    negative_prompt: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    frames: int | None = None
    fps: int | None = None
    init_image: Path | None = None


@dataclass(frozen=True)
class ModelSpec:
    """本地操作者声明的模型组件、任务和资源上限。"""

    id: str
    tasks: tuple[str, ...]
    components: dict[str, Path]
    defaults: dict
    limits: dict[str, int]
    runtime: dict

    def validate_files(self) -> None:
        """只检查组件可读且非空，不读取权重或声称模型兼容性。"""
        for name, path in self.components.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"模型组件缺失或为空：{name}：{path}")
            with path.open("rb"):
                pass

    def arguments(self, request: GenerationRequest, output: Path) -> list[str]:
        """验证请求并生成固定 argv；不接受任意额外 CLI flags。"""
        if request.task not in self.tasks:
            raise ValueError(f"模型 {self.id} 未声明任务 {request.task}。")
        if (
            not isinstance(request.prompt, str)
            or not request.prompt.strip()
            or len(request.prompt) > 65536
            or "\0" in request.prompt
        ):
            raise ValueError("prompt 必须为非空文本，长度不超过 65536。")
        if (
            not isinstance(request.negative_prompt, str)
            or len(request.negative_prompt) > 65536
            or "\0" in request.negative_prompt
        ):
            raise ValueError("negative_prompt 无效。")
        video = request.task.endswith("video")
        values = {
            key: getattr(request, key) if getattr(request, key) is not None else self.defaults[key]
            for key in ("width", "height", "steps", "cfg_scale", "frames", "fps")
        }
        for key in ("width", "height", "steps"):
            value = _integer(values[key], key)
            if value > self.limits[f"max_{key}"]:
                raise ValueError(f"{key} 超过模型上限。")
        for key in ("width", "height"):
            if values[key] % self.limits[f"{key}_multiple"]:
                raise ValueError(f"{key} 不符合模型对齐要求。")
        if values["width"] * values["height"] > self.limits["max_pixels"]:
            raise ValueError("像素数超过模型上限。")
        positive_number(values["cfg_scale"], "cfg_scale", allow_zero=True)
        _integer(request.seed, "seed", -1)
        if request.seed > 2147483647:
            raise ValueError("seed 超出 32 位有符号整数范围。")
        if not video and (request.frames is not None or request.fps is not None):
            raise ValueError("图片任务不能设置 frames 或 fps。")
        if video:
            # 固定版本的 CLI 对单帧 vid_gen 写 PNG，无法履行本 provider 的 AVI 合同。
            frames = _integer(values["frames"], "frames", 2)
            if (
                frames > self.limits["max_frames"]
                or (frames - self.limits["frame_offset"]) % self.limits["frame_multiple"]
            ):
                raise ValueError("frames 超出模型限制或不符合时序对齐要求。")
            if _integer(values["fps"], "fps") > 120:
                raise ValueError("fps 不能超过 120。")
        image_required = request.task.startswith("image-to-")
        if image_required != (request.init_image is not None):
            raise ValueError("参考图必须与 image-to-* 任务匹配。")
        arguments = ["--mode", "vid_gen" if video else "img_gen"]
        for name, path in self.components.items():
            arguments += [
                "--"
                + (
                    name.replace("_", "-")
                    if name in {"diffusion_model", "high_noise_diffusion_model"}
                    else name
                ),
                str(path),
            ]
        for name in ("backend", "params_backend", "flash_attn", "max_vram_gib"):
            flag = "max-vram" if name == "max_vram_gib" else name.replace("_", "-")
            arguments += ["--" + flag, str(self.runtime[name])]
        for name in ("vae_tiling", "temporal_tiling", "offload_to_cpu"):
            if self.runtime[name]:
                arguments.append("--" + name.replace("_", "-"))
        arguments += [
            "--prompt",
            request.prompt,
            "--negative-prompt",
            request.negative_prompt,
            "--width",
            str(values["width"]),
            "--height",
            str(values["height"]),
            "--steps",
            str(values["steps"]),
            "--cfg-scale",
            str(values["cfg_scale"]),
            "--seed",
            str(request.seed),
            "--sampling-method",
            self.defaults["sampler"],
            "--output",
            str(output),
        ]
        if video:
            arguments += ["--video-frames", str(values["frames"]), "--fps", str(values["fps"])]
        if request.init_image is not None:
            image = Path(request.init_image).resolve()
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError("参考图不存在或为空。")
            image_flag = (
                self.runtime["image_to_image_flag"]
                if request.task == "image-to-image"
                else "init-img"
            )
            arguments += ["--" + image_flag, str(image)]
        return arguments


def load_models(path: Path) -> dict[str, ModelSpec]:
    """读取严格版本清单；空清单表示没有已配置模型。"""
    document = json.loads(path.read_text(encoding="utf-8"))
    _fields(document, {"schema_version", "models"}, "manifest")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(document.get("models"), list)
    ):
        raise ValueError("模型清单必须使用 schema_version=1 和 models 数组。")
    models: dict[str, ModelSpec] = {}
    for entry in document["models"]:
        _fields(entry, {"id", "tasks", "components", "defaults", "limits", "runtime"}, "model")
        model_id = entry.get("id")
        if (
            not isinstance(model_id, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", model_id)
            or model_id in models
        ):
            raise ValueError("模型 ID 无效或重复。")
        tasks = entry.get("tasks")
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(not isinstance(t, str) or t not in TASKS for t in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            raise ValueError("tasks 必须是非空且不重复的已支持任务数组。")
        components = _fields(entry.get("components"), COMPONENTS, "components")
        if ("model" in components) == ("diffusion_model" in components):
            raise ValueError("必须且只能提供 model 或 diffusion_model。")
        defaults = DEFAULTS | _fields(entry.get("defaults", {}), set(DEFAULTS), "defaults")
        limits = LIMITS | _fields(entry.get("limits", {}), set(LIMITS), "limits")
        for name, value in limits.items():
            _integer(value, name, 0 if name == "frame_offset" else 1)
        if not isinstance(defaults["sampler"], str) or defaults["sampler"] not in {
            "euler",
            "euler_a",
            "heun",
            "dpm2",
            "dpm++2m",
            "lcm",
        }:
            raise ValueError("sampler 不在允许列表内。")
        runtime = RUNTIME | _fields(entry.get("runtime", {}), set(RUNTIME), "runtime")
        for name, allowed in (
            ("backend", {"cuda0", "cpu"}),
            ("params_backend", {"cuda0", "cpu"}),
            ("flash_attn", {"auto", "on", "off"}),
            ("image_to_image_flag", {"init-img", "ref-image"}),
        ):
            if not isinstance(runtime[name], str) or runtime[name] not in allowed:
                raise ValueError("runtime 后端或 Flash Attention 设置无效。")
        for name in ("vae_tiling", "temporal_tiling", "offload_to_cpu"):
            if type(runtime[name]) is not bool:
                raise ValueError(f"{name} 必须为布尔值。")
        positive_number(runtime["max_vram_gib"], "max_vram_gib")
        model = ModelSpec(
            model_id,
            tuple(tasks),
            {key: local_path(value, path.resolve().parent) for key, value in components.items()},
            defaults,
            limits,
            runtime,
        )
        # 所有默认数值都在加载时检查，避免将坏配置拖到持锁阶段。
        for name in ("frames", "fps"):
            _integer(defaults[name], name)
        if (
            defaults["frames"] > limits["max_frames"]
            or (defaults["frames"] - limits["frame_offset"]) % limits["frame_multiple"]
            or defaults["fps"] > 120
        ):
            raise ValueError("默认帧数或帧率不符合模型限制。")
        if any(task.endswith("video") for task in tasks) and defaults["frames"] < 2:
            raise ValueError("视频任务的默认 frames 必须至少为 2。")
        # 参考图文件在请求时检查；此处使用等价的纯文本任务验证共同默认值。
        probe = ModelSpec(model.id, ("text-to-image",), model.components, defaults, limits, runtime)
        probe.arguments(
            GenerationRequest(task="text-to-image", prompt="validate"), Path("validate.png")
        )
        models[model_id] = model
    return models
