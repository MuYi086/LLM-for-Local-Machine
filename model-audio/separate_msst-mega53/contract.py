"""MSST 重点声部分离的轻量请求合同，不导入模型依赖。"""

from __future__ import annotations

SAMPLE_RATE = 44_100
WEIGHT_FILENAME = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
CONFIG_FILENAME = "mvsep_mega_model_bs_roformer_53_stems.yaml"


def parse_stems(value: str, supported: list[str]) -> list[str]:
    """按本地模型配置校验调用方声部选择，不编码业务配器策略。"""
    stems = [item.strip() for item in value.split(",")]
    if not 1 <= len(stems) <= len(supported):
        raise ValueError(f"stems 必须选择 1–{len(supported)} 个声部。")
    if len(set(stems)) != len(stems) or any(stem not in supported for stem in stems):
        raise ValueError(f"stems 不得重复或为空，可选值为 {', '.join(supported)}。")
    return stems


def supported_stems(config_path) -> list[str]:
    """安全读取模型配置中的声部，不执行 YAML Python 对象构造。"""
    import yaml

    class ModelLoader(yaml.SafeLoader):
        pass

    # 官方 YAML 包含 tuple，仅允许构造普通序列，其余 Python 标签继续拒绝。
    ModelLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=ModelLoader)
    stems = config["training"]["instruments"]
    if (
        not isinstance(stems, list)
        or not stems
        or any(
            not isinstance(stem, str)
            or not stem
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in stem)
            for stem in stems
        )
        or len(set(stems)) != len(stems)
        or len(stems) != config["model"]["num_stems"]
    ):
        raise ValueError("模型配置中的声部清单无效。")
    return stems
