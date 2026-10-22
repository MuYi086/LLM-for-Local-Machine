"""有声书文本润色 POC 数据构建器的无模型测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

# model-text 按模态保留目录名，动态导入其 namespace package，避免修改全局 sys.path。
_training = "model-text.scripts.training"
_builder = import_module(f"{_training}.build_audio_polisher_poc")
clean_markdown = _builder.clean_markdown
extract_candidate_chunks = _builder.extract_candidate_chunks
select_source_files = _builder.select_source_files
split_source_names = _builder.split_source_names
validate_teacher_output = _builder.validate_teacher_output
compare_predictions = import_module(
    f"{_training}.compare_audio_polisher_predictions"
).compare_predictions
decide_dataset_import = import_module(
    f"{_training}.import_audio_polisher_to_easy_dataset"
).decide_dataset_import
_inference = import_module(f"{_training}.run_audio_polisher_inference")
build_messages = _inference.build_messages
normalize_prediction_text = _inference.normalize_prediction_text


class AudioPolisherPocTests(unittest.TestCase):
    def test_select_source_files_is_seeded_and_ignores_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            for index in range(6):
                (source_dir / f"source-{index}.md").write_text("中" * 200, encoding="utf-8")
            (source_dir / "placeholder.md").write_text("error", encoding="utf-8")
            (source_dir / "ignored.error").write_text("中" * 100, encoding="utf-8")

            first_selection = select_source_files(source_dir, story_count=3, seed=42)
            second_selection = select_source_files(source_dir, story_count=3, seed=42)

        self.assertEqual(first_selection, second_selection)
        self.assertEqual(len(first_selection), 3)
        self.assertNotIn("placeholder.md", {path.name for path in first_selection})

    def test_clean_markdown_removes_images_urls_and_ocr_sentinel(self) -> None:
        raw_text = """# 第一幕
![广告](https://example.com/ad.jpg)
https://example.com
The image contains a magazine advertisement.
　门外忽然响起第三次敲门声。  她没有回答。
"""

        cleaned = clean_markdown(raw_text)

        self.assertIn("# 第一幕", cleaned)
        self.assertIn("门外忽然响起第三次敲门声。 她没有回答。", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("The image contains", cleaned)

    def test_extract_candidate_chunks_never_crosses_heading_boundaries(self) -> None:
        first_scene = "。".join(f"第一幕的第{index}句话" for index in range(80)) + "。"
        second_scene = "。".join(f"第二幕的第{index}句话" for index in range(80)) + "。"
        document = f"# 第一幕\n{first_scene}\n# 第二幕\n{second_scene}"

        chunks = extract_candidate_chunks(
            document,
            minimum_chars=120,
            target_chars=260,
            maximum_chars=360,
        )

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk.heading in {"第一幕", "第二幕"} for chunk in chunks))
        self.assertTrue(
            all(not ("第一幕" in chunk.text and "第二幕" in chunk.text) for chunk in chunks)
        )

    def test_split_source_names_keeps_source_groups_isolated(self) -> None:
        source_names = [f"story-{index}.md" for index in range(10)]

        split_map = split_source_names(source_names, seed=42)

        split_counts = {
            split_name: sum(value == split_name for value in split_map.values())
            for split_name in {"train", "validation", "test"}
        }
        self.assertEqual(split_counts, {"train": 8, "validation": 1, "test": 1})
        self.assertEqual(set(split_map), set(source_names))

    def test_validate_teacher_output_rejects_metadata_and_accepts_plain_text(self) -> None:
        source = "夜里，楼道尽头传来脚步声。她握住门把手，却没有立刻开门。"
        valid_output = "夜里，楼道尽头传来脚步声。她握紧门把手，却迟迟没有开门。"

        self.assertEqual(validate_teacher_output(source, valid_output), valid_output)

        invalid_outputs = (
            '```json\n{"text": "不要返回 JSON"}\n```',
            "[SFX: 脚步声] 夜里有人来了。",
            "<think>先分析</think>夜里有人来了。",
            "润色结果：夜里有人来了。",
        )
        for invalid_output in invalid_outputs:
            with self.subTest(invalid_output=invalid_output):
                with self.assertRaises(ValueError):
                    validate_teacher_output(source, invalid_output)

    def test_compare_predictions_reports_adapter_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            baseline_path = output_dir / "baseline.jsonl"
            adapter_path = output_dir / "adapter.jsonl"
            baseline_path.write_text(
                json.dumps({"predict": "完全不同", "label": "门外响起脚步声"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            adapter_path.write_text(
                json.dumps(
                    {"predict": "门外响起了脚步声", "label": "门外响起脚步声"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = compare_predictions(baseline_path, adapter_path)

        self.assertGreater(report["character_similarity_delta"], 0)
        self.assertEqual(report["adapter"]["plain_text_pass"], 1)

    def test_inference_messages_keep_source_out_of_system_prompt(self) -> None:
        source = "门外响起了三次敲门声。"

        messages = build_messages(source)

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertNotIn(source, messages[0]["content"])
        self.assertEqual(
            messages[1]["content"], f"请润色下面的原文，使它适合有声书 TTS 直接朗读。\n{source}"
        )

    def test_normalize_prediction_text_removes_markdown_line_break_spaces(self) -> None:
        raw_text = "第一段。  \n第二段。\t\n\n\n第三段。"

        normalized = normalize_prediction_text(raw_text)

        self.assertEqual(normalized, "第一段。\n第二段。\n\n第三段。")

    def test_easy_dataset_import_decision_prevents_partial_duplicate_import(self) -> None:
        self.assertEqual(decide_dataset_import(0, 30), "import")
        self.assertEqual(decide_dataset_import(30, 30), "skip")
        with self.assertRaises(ValueError):
            decide_dataset_import(5, 30)


if __name__ == "__main__":
    unittest.main()
