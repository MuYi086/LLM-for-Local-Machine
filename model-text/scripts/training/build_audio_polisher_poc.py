#!/usr/bin/env python3
"""从本地 Markdown 来源构建有声书文本润色 POC 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """你是有声书 TTS 文本润色编辑。你的任务是把原始小说片段改成适合直接朗读的纯文本。
保持事实、情节、人物、视角和信息量，不续写，不总结，不添加原文没有的情绪、音效或配乐。
修复明显 OCR 错字、断行和标点，改善长句停顿与对白可读性。
只输出润色后的正文，不输出标题、解释、标签、Markdown、JSON 或思考过程。"""

USER_INSTRUCTION = "请润色下面的原文，使它适合有声书 TTS 直接朗读。"

_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\([^)]*\)")
_HTML_PATTERN = re.compile(r"<[^>]+>")
_URL_PATTERN = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])")
_CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")
_OCR_SENTINELS = (
    "the image contains",
    "the image depicts",
    "the image shows",
    "i'm sorry, but i can't",
    "i cannot assist with",
)
_UNSUITABLE_HEADINGS = re.compile(
    r"目录|广告|评语|编辑部|大本营|beauty|笋子|星座|布局|好运|黑小?猫|"
    r"发言|留言|读者|大礼|怪社|茶楼|互动",
    re.IGNORECASE,
)
_FORBIDDEN_OUTPUT_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"</?think>", re.IGNORECASE),
    re.compile(r"\[(?:sfx|bgm|emotion|音效|配乐|情绪)[^]]*]", re.IGNORECASE),
    re.compile(r"(?:schemaVersion|LLM Local Backend)\s*[:：]?", re.IGNORECASE),
    re.compile(r"^\s*(?:润色结果|输出|正文|答案)\s*[:：]"),
)


@dataclass(frozen=True)
class TextChunk:
    """记录一个不跨标题边界的候选文本片段。"""

    heading: str
    text: str


@dataclass(frozen=True)
class SourceManifest:
    """记录数据来源的可审计元数据。"""

    file_name: str
    sha256: str
    source_chars: int
    split: str
    sample_count: int
    rights_status: str = "research-only"
    review_status: str = "automated-poc"


Teacher = Callable[[str], str]


def select_source_files(source_dir: Path, story_count: int, seed: int = 42) -> list[Path]:
    """使用固定随机种子选择有效的 Markdown 来源。

    内容只有错误占位符的文件会被排除；先按文件名排序再抽样，确保跨机器可复现。

    Args:
        source_dir: Markdown 来源目录。
        story_count: 需要选择的来源数量。
        seed: 随机抽样种子。

    Returns:
        可复现的随机路径列表。

    Raises:
        ValueError: 有效文件数量不足或数量参数无效。
    """
    if story_count < 1:
        raise ValueError("story_count 必须大于 0")

    candidates: list[Path] = []
    for path in source_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 100 or text.casefold() in {"error", "failed", "null"}:
            continue
        candidates.append(path)

    candidates.sort(key=lambda path: path.name)
    if len(candidates) < story_count:
        raise ValueError(f"有效 Markdown 只有 {len(candidates)} 个，少于要求的 {story_count} 个")
    return random.Random(seed).sample(candidates, story_count)


def clean_markdown(raw_text: str) -> str:
    """移除图片、链接和常见 OCR 占位内容，同时保留标题结构。"""
    # NFC 保留中文全角标点；NFKC 会把逗号等转换成不适合中文朗读的半角符号。
    normalized = unicodedata.normalize("NFC", raw_text).replace("\r\n", "\n")
    normalized = _IMAGE_PATTERN.sub("", normalized)
    cleaned_lines: list[str] = []
    previous_blank = False

    for line in normalized.splitlines():
        stripped = _HTML_PATTERN.sub("", line).strip()
        lowered = stripped.casefold()
        if _URL_PATTERN.match(stripped) or any(marker in lowered for marker in _OCR_SENTINELS):
            continue
        stripped = re.sub(r"[ \t\u3000]+", " ", stripped)
        if not stripped:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(stripped)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def _split_long_paragraph(paragraph: str, maximum_chars: int) -> list[str]:
    if len(paragraph) <= maximum_chars:
        return [paragraph]

    sentences = [part.strip() for part in _SENTENCE_PATTERN.split(paragraph) if part.strip()]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > maximum_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                sentence[start : start + maximum_chars]
                for start in range(0, len(sentence), maximum_chars)
            )
        elif current and len(current) + len(sentence) > maximum_chars:
            parts.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        parts.append(current)
    return parts


def _section_paragraphs(document: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading = "未命名片段"
    paragraphs: list[str] = []
    paragraph_lines: list[str] = []

    def commit_paragraph() -> None:
        if paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
            paragraph_lines.clear()

    def commit_section() -> None:
        commit_paragraph()
        if paragraphs:
            sections.append((heading, paragraphs.copy()))
            paragraphs.clear()

    for line in document.splitlines():
        stripped = line.strip()
        if not stripped:
            commit_paragraph()
            continue
        heading_match = _HEADING_PATTERN.match(stripped)
        if heading_match:
            commit_section()
            heading = heading_match.group(1).strip()
            continue
        paragraph_lines.append(stripped)
    commit_section()
    return sections


def extract_candidate_chunks(
    document: str,
    *,
    minimum_chars: int = 320,
    target_chars: int = 720,
    maximum_chars: int = 960,
) -> list[TextChunk]:
    """在标题边界内构建适合教师模型处理的候选片段。"""
    if not 0 < minimum_chars <= target_chars <= maximum_chars:
        raise ValueError("片段长度必须满足 0 < minimum <= target <= maximum")

    chunks: list[TextChunk] = []
    for heading, paragraphs in _section_paragraphs(clean_markdown(document)):
        if _UNSUITABLE_HEADINGS.search(heading):
            continue
        expanded = [
            part
            for paragraph in paragraphs
            for part in _split_long_paragraph(paragraph, maximum_chars)
        ]
        current: list[str] = []
        current_chars = 0
        for paragraph in expanded:
            separator_chars = 1 if current else 0
            would_exceed = current_chars + separator_chars + len(paragraph) > maximum_chars
            if current and would_exceed:
                text = "\n".join(current)
                if len(text) >= minimum_chars:
                    chunks.append(TextChunk(heading=heading, text=text))
                current = []
                current_chars = 0
            current.append(paragraph)
            current_chars += (1 if current_chars else 0) + len(paragraph)
            if current_chars >= target_chars:
                text = "\n".join(current)
                if len(text) >= minimum_chars:
                    chunks.append(TextChunk(heading=heading, text=text))
                current = []
                current_chars = 0
        if current:
            text = "\n".join(current)
            if len(text) >= minimum_chars:
                chunks.append(TextChunk(heading=heading, text=text))

    return [chunk for chunk in chunks if _is_narrative_candidate(chunk.text)]


def _is_narrative_candidate(text: str) -> bool:
    chinese_chars = len(_CHINESE_PATTERN.findall(text))
    visible_chars = len(re.sub(r"\s", "", text))
    if visible_chars == 0 or chinese_chars / visible_chars < 0.55:
        return False
    return sum(text.count(mark) for mark in '。！？!?“”"') >= 3


def _choose_evenly(chunks: Sequence[TextChunk], count: int) -> list[TextChunk]:
    if len(chunks) < count:
        raise ValueError(f"候选片段只有 {len(chunks)} 个，少于要求的 {count} 个")
    # 杂志 OCR 的卷首和卷尾通常是目录、广告或读者互动区，取内部等分点更接近正文。
    indexes = [round((index + 1) * (len(chunks) + 1) / (count + 1)) - 1 for index in range(count)]
    return [chunks[index] for index in indexes]


def split_source_names(source_names: Sequence[str], seed: int) -> dict[str, str]:
    """以来源为单位稳定划分训练、验证和测试集合。"""
    unique_names = sorted(set(source_names))
    if len(unique_names) < 3:
        raise ValueError("来源至少需要 3 个，才能隔离训练、验证和测试集合")
    random.Random(seed).shuffle(unique_names)

    test_count = max(1, round(len(unique_names) * 0.1))
    validation_count = max(1, round(len(unique_names) * 0.1))
    if test_count + validation_count >= len(unique_names):
        validation_count = 1
        test_count = 1

    split_map = {name: "train" for name in unique_names}
    for name in unique_names[:test_count]:
        split_map[name] = "test"
    for name in unique_names[test_count : test_count + validation_count]:
        split_map[name] = "validation"
    return split_map


def validate_teacher_output(source: str, output: str) -> str:
    """校验教师输出确实是可直接用于监督微调的纯文本。"""
    normalized = unicodedata.normalize("NFC", output).strip()
    if not normalized:
        raise ValueError("教师输出为空")
    for pattern in _FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.search(normalized):
            raise ValueError(f"教师输出包含禁止格式: {pattern.pattern}")
    if normalized.startswith("{") or normalized.startswith("["):
        raise ValueError("教师输出疑似 JSON 或标签列表")

    source_length = max(1, len(re.sub(r"\s", "", source)))
    output_length = len(re.sub(r"\s", "", normalized))
    ratio = output_length / source_length
    if output_length < 20 or not 0.35 <= ratio <= 2.2:
        raise ValueError(f"教师输出长度异常: {output_length} 字，比例 {ratio:.2f}")
    return normalized


def make_ollama_teacher(
    *,
    endpoint: str,
    model: str,
    seed: int,
    timeout_seconds: float,
    retries: int,
) -> Teacher:
    """创建调用本机 Ollama 的教师函数。"""

    def generate(source: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{USER_INSTRUCTION}\n\n原文：\n{source}",
                },
            ],
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "seed": seed,
                "num_ctx": 4096,
                "num_predict": 1400,
            },
        }
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    result = json.load(response)
                return str(result["message"]["content"])
            except (KeyError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Ollama 教师连续 {retries} 次调用失败") from last_error

    return generate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _dataset_record(source: str, target: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": USER_INSTRUCTION,
        "input": source,
        "output": target,
        "system": SYSTEM_PROMPT,
        "metadata": metadata,
    }


def _easy_dataset_record(record: dict[str, Any], teacher_model: str) -> dict[str, Any]:
    metadata = record["metadata"]
    question = f"{record['instruction']}\n\n原文：\n{record['input']}"
    return {
        "question": question,
        "answer": record["output"],
        "chunkName": f"{metadata['source_file']} / {metadata['heading']}",
        "chunkContent": record["input"],
        "model": teacher_model,
        "questionLabel": "audio-polisher-a-to-b",
        "cot": "",
        "confirmed": False,
        "score": 0,
        "tags": ["audio-polisher", "automated-poc", metadata["split"]],
        "other": json.dumps(metadata, ensure_ascii=False),
        "note": "自动生成的流程 POC 候选，未经人工 Gold 审核，不得直接用于生产训练。",
    }


def build_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    story_count: int,
    samples_per_story: int,
    seed: int,
    teacher_model: str,
    teacher: Teacher,
) -> dict[str, Any]:
    """构建数据集并以原子替换方式写入所有产物。"""
    selected_files = select_source_files(source_dir, story_count, seed)
    split_map = split_source_names([path.name for path in selected_files], seed)
    resume_path = output_dir / "data" / "generation_resume.json"
    completed: dict[str, str] = {}
    if resume_path.exists():
        raw_completed = json.loads(resume_path.read_text(encoding="utf-8"))
        if isinstance(raw_completed, dict):
            completed = {str(key): str(value) for key, value in raw_completed.items()}

    datasets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    manifests: list[SourceManifest] = []
    sample_number = 0

    for file_number, source_path in enumerate(selected_files, start=1):
        raw_text = source_path.read_text(encoding="utf-8", errors="replace")
        chunks = _choose_evenly(extract_candidate_chunks(raw_text), samples_per_story)
        split = split_map[source_path.name]
        manifests.append(
            SourceManifest(
                file_name=source_path.name,
                sha256=_sha256(source_path),
                source_chars=len(raw_text),
                split=split,
                sample_count=len(chunks),
            )
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            sample_number += 1
            sample_id = hashlib.sha256(
                f"{source_path.name}\0{chunk.heading}\0{chunk.text}".encode()
            ).hexdigest()[:16]
            target = completed.get(sample_id)
            if target is None:
                print(
                    f"[{sample_number}/{story_count * samples_per_story}] "
                    f"生成 {source_path.name} / {chunk.heading}",
                    flush=True,
                )
                target = validate_teacher_output(chunk.text, teacher(chunk.text))
                completed[sample_id] = target
                _write_json(resume_path, completed)
            else:
                target = validate_teacher_output(chunk.text, target)
                print(
                    f"[{sample_number}/{story_count * samples_per_story}] "
                    f"复用 {source_path.name} / {chunk.heading}",
                    flush=True,
                )

            metadata = {
                "sample_id": sample_id,
                "source_file": source_path.name,
                "source_sha256": manifests[-1].sha256,
                "heading": chunk.heading,
                "chunk_number": chunk_number,
                "split": split,
                "rights_status": "research-only",
                "review_status": "automated-poc",
                "teacher_model": teacher_model,
            }
            datasets[split].append(_dataset_record(chunk.text, target, metadata))

        print(f"[{file_number}/{story_count}] 完成来源 {source_path.name}", flush=True)

    data_dir = output_dir / "data"
    names = {
        "train": "story_audio_polish_train",
        "validation": "story_audio_polish_validation",
        "test": "story_audio_polish_test",
    }
    for split, records in datasets.items():
        _write_json(data_dir / f"{names[split]}.json", records)

    dataset_info = {
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
        for dataset_name in names.values()
    }
    _write_json(data_dir / "dataset_info.json", dataset_info)

    all_records = [
        record for split in ("train", "validation", "test") for record in datasets[split]
    ]
    _write_json(
        output_dir / "easy_dataset_import.json",
        {
            "datasets": [_easy_dataset_record(record, teacher_model) for record in all_records],
            "sourceInfo": {
                "kind": "audio-polisher-poc",
                "rights_status": "research-only",
                "review_status": "automated-poc",
            },
        },
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "teacher_model": teacher_model,
        "task": "A 原始小说片段 -> B 纯净可朗读文本",
        "rights_status": "research-only",
        "review_status": "automated-poc",
        "sources": [asdict(item) for item in manifests],
        "counts": {split: len(records) for split, records in datasets.items()},
    }
    _write_json(data_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--story-count", type=int, default=10)
    parser.add_argument("--samples-per-story", type=int, default=3)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:11434")
    parser.add_argument("--teacher-model", default="qwen3.5:9b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """构建默认 10 来源、每来源 3 条样本的 POC 数据集。"""
    args = parse_args()
    teacher = make_ollama_teacher(
        endpoint=args.teacher_url,
        model=args.teacher_model,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    manifest = build_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        story_count=args.story_count,
        samples_per_story=args.samples_per_story,
        seed=args.seed,
        teacher_model=args.teacher_model,
        teacher=teacher,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
