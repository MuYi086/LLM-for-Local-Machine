#!/usr/bin/env python3
"""使用 Transformers 和可选 LoRA 对隔离测试集执行确定性推理。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.training.build_audio_polisher_poc import SYSTEM_PROMPT, USER_INSTRUCTION


def build_messages(source: str) -> list[dict[str, str]]:
    """构建与训练模板一致的 system/user 消息。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            # LLaMA-Factory 的 Alpaca converter 会用单个换行拼接 prompt 与 query。
            "content": f"{USER_INSTRUCTION}\n{source}",
        },
    ]


def normalize_prediction_text(text: str) -> str:
    """移除行末 Markdown 硬换行空格，并限制连续空行为一个。"""
    without_trailing_spaces = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", without_trailing_spaces).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def run_inference(
    *,
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    adapter_path: Path | None,
    max_new_tokens: int,
) -> None:
    """加载基础模型和可选 LoRA，逐条生成测试集预测。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(dataset, start=1):
        messages = build_messages(str(record["input"]))
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = generated[0, model_inputs.input_ids.shape[1] :]
        prediction = normalize_prediction_text(
            tokenizer.decode(generated_tokens, skip_special_tokens=True)
        )
        predictions.append(
            {
                "prompt": prompt,
                "predict": prediction,
                "label": str(record["output"]),
                "metadata": record.get("metadata", {}),
            }
        )
        print(f"[{index}/{len(dataset)}] 已生成 {len(prediction)} 字", flush=True)

    _write_jsonl(output_path, predictions)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    """执行命令行推理。"""
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_path=args.output,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
