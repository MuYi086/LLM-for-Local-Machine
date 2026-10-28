import tempfile
import unittest
from pathlib import Path
import i2vbench as b

class TestCore(unittest.TestCase):
    def test_ui_graph_compiler_bypass_and_widgets(self):
        ui={"nodes":[
           {"id":1,"type":"Source","mode":0,"inputs":[],"widgets_values":["hello"],"outputs":[{"type":"IMAGE"}]},
           {"id":2,"type":"Identity","mode":4,"inputs":[{"name":"image","type":"IMAGE","link":10}],"outputs":[{"type":"IMAGE"}]},
           {"id":3,"type":"SaveVideo","mode":0,"inputs":[{"name":"video","type":"IMAGE","link":11}],"outputs":[],"widgets_values":["prefix","auto","h264"]},
        ],"links":[[10,1,0,2,0,"IMAGE"],[11,2,0,3,0,"IMAGE"]]}
        info={"Source":{"input":{"required":{"text":["STRING",{}]}}},"SaveVideo":{"input":{"required":{"video":["IMAGE"]},"optional":{"filename_prefix":["STRING"],"format":["DYNAMIC_COMBO"],"codec":[["h264","auto"]]}}}}
        g=b.convert_ui_to_api(ui,info)
        self.assertEqual(g["3"]["inputs"]["video"],["1",0])
        self.assertEqual(g["3"]["inputs"]["codec"],"h264")
        self.assertEqual(g["1"]["inputs"]["text"],"hello")
        self.assertNotIn("2",g)

    def test_ui_graph_compiler_nested_dynamic_combo_widgets(self):
        ui={"nodes":[
           {"id":1,"type":"Source","mode":0,"inputs":[],"widgets_values":[],"outputs":[{"type":"IMAGE"}]},
           {"id":2,"type":"SaveVideo","mode":0,"inputs":[{"name":"video","type":"IMAGE","link":10}],"outputs":[],"widgets_values":["prefix","auto","h264"]},
        ],"links":[[10,1,0,2,0,"IMAGE"]]}
        codec_schema=["COMFY_DYNAMICCOMBO_V3",{"options":[{"key":"auto","inputs":{"required":{}}},{"key":"h264","inputs":{"required":{}}}]}]
        format_schema=["COMFY_DYNAMICCOMBO_V3",{"options":[{"key":"auto","inputs":{"required":{"codec":codec_schema}}}]}]
        info={
           "Source":{"input":{"required":{}}},
           "SaveVideo":{"input":{"required":{"video":["IMAGE"],"filename_prefix":["STRING",{}],"format":format_schema},"optional":{"codec":codec_schema}}},
        }
        graph=b.convert_ui_to_api(ui,info)
        self.assertEqual(graph["2"]["inputs"]["format"],"auto")
        self.assertEqual(graph["2"]["inputs"]["format.codec"],"h264")

    def test_expected_trial_matrix(self):
        c=b.cfg()
        self.assertEqual(len(c["models"]),1)
        self.assertEqual(len(c["scenes"]),3)
        self.assertEqual(len(c["seeds"]),3)
        self.assertEqual(len(c["models"])*len(c["scenes"])*len(c["seeds"]),9)
        self.assertIn("step_distilled",c["models"]["hunyuan15_step_fp8"]["unet_filename"])

    def test_live_choice_check_allows_api_upload_schema(self):
        b.validate_live_choice("hunyuan15_step_fp8", "56", "LoadImage", "image", "example.png", [["old.png"], {"image_upload": True}])
        with self.assertRaisesRegex(RuntimeError, "not present in live ComfyUI choices"):
            b.validate_live_choice("hunyuan15_step_fp8", "12", "UNETLoader", "unet_name", "wrong.safetensors", [["model.safetensors"]])

    def test_video_metadata(self):
        self.assertEqual(b.median([10,30,20]),20)
        self.assertEqual(b.median([10,30]),20)

    def test_benchmark_runs_are_sorted_by_start_time(self):
        rows=[
            {"run_id":"a-later","timestamp":"2026-10-27T10:02:00+00:00"},
            {"run_id":"z-earlier","timestamp":"2026-10-27T10:01:00+00:00"},
        ]
        self.assertEqual([row["run_id"] for row in b.chronological_runs(rows)], ["z-earlier","a-later"])

    def test_blind_video_alias_hides_original_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "hunyuan15_step_fp8__rainy_city__s42.mp4"
            source.write_bytes(b"video bytes")
            alias = b.blind_video_alias(source, Path(temp_dir) / "blind_videos", "VABC123")
            self.assertEqual(alias.name, "VABC123.mp4")
            self.assertEqual(alias.read_bytes(), source.read_bytes())

if __name__ == "__main__": unittest.main()
