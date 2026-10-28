"""HunyuanVideo I2V benchmark orchestrator. Requires running ComfyUI for real inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import psutil
import requests
import yaml
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
TEMPLATES = {
    "hunyuan15_step_fp8": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_hunyuan_video_1.5_720p_i2v.json",
}
RUN_FIELDS = ["run_id", "model", "scene", "seed", "status", "wall_seconds", "gpu_peak_mib", "gpu_baseline_mib", "gpu_delta_peak_mib", "gpu_util_peak_pct", "wsl_mem_peak_mib", "width", "height", "fps", "video_seconds", "render_seconds_per_video_second", "output", "prompt_id", "timestamp", "error"]
SCORE_FIELDS = ["blind_id", "visual_quality", "motion_stability", "prompt_adherence", "notes"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cfg():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel):
    return ROOT / rel


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write_json(file, obj):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    Path(file).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(file):
    file = Path(file)
    if not file.exists():
        return []
    with file.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(file, fields, rows):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_name(file.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(file)


def api(server, endpoint, method="GET", **kwargs):
    url = server.rstrip("/") + "/" + endpoint.lstrip("/")
    r = requests.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
    if not r.ok:
        raise RuntimeError(f"{method} {endpoint}: HTTP {r.status_code}: {r.text[:2000]}")
    return r


def test_server(server):
    system = api(server, "system_stats").json()
    object_info = api(server, "object_info", timeout=90).json()
    return system, object_info


def fetch_templates(args):
    c = cfg()
    for name, url in TEMPLATES.items():
        dst = path(c["models"][name]["source_template"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.force:
            print("exists:", dst)
            continue
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        obj = r.json()
        if not isinstance(obj.get("nodes"), list) or not isinstance(obj.get("links"), list):
            raise RuntimeError(f"Invalid UI graph at {url}")
        write_json(dst, obj)
        print("saved:", dst)


def _patch_ui(workflow, config):
    """Patch *specific officially linked* UI templates; check expected node types first."""
    nodes = {str(n["id"]): n for n in workflow["nodes"]}
    def set_node(node_id, expected, *, values=None, mode=None, node_type=None):
        n = nodes[str(node_id)]
        if n["type"] != expected:
            raise RuntimeError(f"Official template changed: node {node_id} is {n['type']} not {expected}. Inspect template before proceeding.")
        if values is not None:
            n["widgets_values"] = values
        if mode is not None:
            n["mode"] = mode
        if node_type is not None:
            n["type"] = node_type
    set_node(12, "UNETLoader", values=[config["unet_filename"], "default"])
    set_node(126, "BasicScheduler", values=["simple", config["steps"], 1])
    set_node(129, "CFGGuider", values=[config["cfg"]])
    set_node(130, "ModelSamplingSD3", values=[config["flow_shift"]])
    return workflow


def convert_ui_to_api(ui, object_info):
    """Compile stock LiteGraph-style UI workflow using running ComfyUI /object_info.

    Only reachable active SaveVideo branches are compiled. Muted 4/bypass nodes pass
    through a compatible same-typed input. Unsupported nodes fail loudly.
    """
    nodes = {str(n["id"]): n for n in ui["nodes"]}
    links = {str(l[0]): l for l in ui["links"]}
    cache = {}
    visiting = set()

    def resolve(linkid):
        l = links.get(str(linkid))
        if not l:
            raise RuntimeError(f"Missing link {linkid}")
        src, slot = str(l[1]), int(l[2])
        n = nodes[src]
        if n.get("mode") == 4:
            outtype = n["outputs"][slot]["type"]
            candidates = [i for i in n.get("inputs", []) if i.get("link") is not None and i.get("type") == outtype]
            if len(candidates) != 1:
                raise RuntimeError(f"Cannot bypass inactive node {src} ({n['type']}), output type {outtype}, matches={len(candidates)}")
            return resolve(candidates[0]["link"])
        if n.get("mode", 0) != 0:
            raise RuntimeError(f"Unsupported node mode {n.get('mode')}: {src} {n['type']}")
        compile_node(src)
        return [src, slot]

    def compile_node(nid):
        if nid in cache:
            return
        if nid in visiting:
            raise RuntimeError("Graph cycle detected")
        n = nodes[nid]
        cls = n["type"]
        info = object_info.get(cls)
        if not info:
            raise RuntimeError(f"ComfyUI lacks node {cls!r}. Update ComfyUI / install required custom node.")
        visiting.add(nid)
        inputs = {}
        declared = {}
        for section in ("required", "optional"):
            declared.update(info.get("input", {}).get(section, {}))
        ui_inputs = {i["name"]: i for i in n.get("inputs", [])}
        widget_values = list(n.get("widgets_values") or [])
        widget_index = 0

        def is_widget_schema(schema):
            desc = schema[0] if isinstance(schema, (list, tuple)) and schema else schema
            return desc in {
                "INT", "FLOAT", "BOOLEAN", "STRING", "DYNAMIC_COMBO", "COMBO",
                "COMFY_DYNAMICCOMBO_V3",
            } if isinstance(desc, str) else isinstance(desc, (list, tuple))

        def consume_widget(schema, name, parent_path=None):
            nonlocal widget_index
            if widget_index >= len(widget_values):
                raise RuntimeError(f"Missing UI widget value for node {nid} {cls}.{name}")
            value = widget_values[widget_index]
            widget_index += 1
            desc = schema[0] if isinstance(schema, (list, tuple)) and schema else schema
            input_path = f"{parent_path}.{name}" if parent_path else name
            if desc != "COMFY_DYNAMICCOMBO_V3":
                return {input_path: value}

            metadata = schema[1] if len(schema) > 1 and isinstance(schema[1], dict) else {}
            option = next((item for item in metadata.get("options", []) if item.get("key") == value), None)
            if option is None:
                raise RuntimeError(f"Unsupported dynamic choice {value!r} for node {nid} {cls}.{name}")
            result = {input_path: value}
            nested = option.get("inputs", {})
            for nested_section in ("required", "optional"):
                for nested_name, nested_schema in nested.get(nested_section, {}).items():
                    if is_widget_schema(nested_schema) and widget_index < len(widget_values):
                        result.update(consume_widget(nested_schema, nested_name, input_path))
                    elif nested_section == "required":
                        raise RuntimeError(
                            f"Cannot compile required dynamic input {nid} {cls}.{name}.{nested_name}"
                        )
            return result

        for name, schema in declared.items():
            link = ui_inputs.get(name, {}).get("link")
            is_widget = is_widget_schema(schema)
            if link is not None:
                inputs[name] = resolve(link)
                # A linked widget can retain its original UI value (rare); skip one when needed.
                if is_widget and ui_inputs.get(name, {}).get("widget") and widget_index < len(widget_values):
                    consume_widget(schema, name)
            elif is_widget and widget_index < len(widget_values):
                values = consume_widget(schema, name)
                if name in ("seed", "noise_seed") and widget_index < len(widget_values) and widget_values[widget_index] in ("fixed", "randomize", "increment", "decrement"):
                    widget_index += 1
                inputs.update(values)
        missing = [k for k in info.get("input", {}).get("required", {}) if k not in inputs]
        if missing:
            raise RuntimeError(f"Missing required args for node {nid} {cls}: {missing}. The upstream UI template/schema likely changed.")
        cache[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": n.get("title", cls)}}
        visiting.remove(nid)

    sinks = [n for n in ui["nodes"] if n.get("mode", 0) == 0 and n["type"] == "SaveVideo"]
    if len(sinks) != 1:
        raise RuntimeError(f"Expected exactly one active SaveVideo sink; found {len(sinks)}")
    compile_node(str(sinks[0]["id"]))
    return cache


def prepare(args):
    c = cfg()
    _, object_info = test_server(c["server"])
    for name, config in c["models"].items():
        source = path(config["source_template"])
        if not source.exists():
            raise RuntimeError(f"Missing {source}. Run fetch-templates or copy official UI JSON.")
        ui = _patch_ui(read_json(source), config)
        workflow = convert_ui_to_api(ui, object_info)
        expected = config["unet_filename"]
        if not any(expected in str(n["inputs"].values()) for n in workflow.values()):
            raise RuntimeError(f"Missing expected model {expected} in compiled graph")
        dst = path(config["workflow"])
        write_json(dst, workflow)
        print("Compiled:", dst, "nodes", len(workflow))


def _select_node(graph, cls, contains_title=None):
    matches = [(nid, node) for nid, node in graph.items() if node["class_type"] == cls]
    if contains_title is not None:
        matches = [(nid, node) for nid, node in matches if contains_title in node.get("_meta", {}).get("title", "")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {cls} {contains_title or ''}, found {len(matches)}")
    return matches[0][1]["inputs"]


def patch_graph(graph, image_name, prompt, seed, conf, global_conf, prefix):
    graph = deepcopy(graph)
    image_input = _select_node(graph, "LoadImage")
    image_input["image"] = image_name
    prompt_input = _select_node(graph, "CLIPTextEncode", "Positive")
    prompt_input["text"] = prompt
    latent = _select_node(graph, "HunyuanVideo15ImageToVideo")
    latent.update(width=global_conf["width"], height=global_conf["height"], length=global_conf["frames"], batch_size=1)
    _select_node(graph, "RandomNoise")["noise_seed"] = seed
    _select_node(graph, "BasicScheduler")["steps"] = conf["steps"]
    _select_node(graph, "CFGGuider")["cfg"] = conf["cfg"]
    _select_node(graph, "ModelSamplingSD3")["shift"] = conf["flow_shift"]
    _select_node(graph, "SaveVideo")["filename_prefix"] = prefix
    return graph


def _assert_near(a, b, label):
    if a != b:
        raise RuntimeError(f"{label}: expected {b}, actual {a}")


def check_workflow(name, graph, config):
    loader = _select_node(graph, "UNETLoader")
    if config["unet_filename"] not in str(loader.values()):
        raise RuntimeError(f"{name}: wrong checkpoint; expected {config['unet_filename']}")
    _select_node(graph, "LoadImage")
    _select_node(graph, "SaveVideo")
    _select_node(graph, "CLIPTextEncode", "Positive")
    if ("step_distilled" not in config["unet_filename"] or config["steps"] not in (8, 12)
            or config["cfg"] != 1 or config["flow_shift"] != 7):
        raise RuntimeError("Hunyuan must use 480p I2V STEP-distilled 8 or 12 steps, CFG=1, shift=7")
    _assert_near(_select_node(graph, "BasicScheduler")["steps"], config["steps"], "Hunyuan steps")
    _assert_near(_select_node(graph, "CFGGuider")["cfg"], config["cfg"], "Hunyuan CFG")
    _assert_near(_select_node(graph, "ModelSamplingSD3")["shift"], config["flow_shift"], "Hunyuan shift")
    _select_node(graph, "HunyuanVideo15ImageToVideo")


def validate_live_choice(model, node_id, node_type, input_name, value, schema):
    """只校验固定选项；上传输入由运行时替换，不按当前文件列表拒绝。"""
    metadata = schema[1] if isinstance(schema, (tuple, list)) and len(schema) > 1 and isinstance(schema[1], dict) else {}
    if metadata.get("image_upload"):
        return
    choices = schema[0] if isinstance(schema, (tuple, list)) and schema else None
    if isinstance(choices, list) and choices and isinstance(value, str) and value not in choices:
        raise RuntimeError(
            f"{model}: node {node_id} {node_type}.{input_name}: {value!r} "
            "not present in live ComfyUI choices. Check installed weights / loader / UI version."
        )


def preflight(args):
    c = cfg()
    system, object_info = test_server(c["server"])
    print("ComfyUI available; runtime:", str(system.get("system", {}))[:250])
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe missing; install ffmpeg")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi missing; GPU telemetry unavailable")
    print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip())
    for model, spec in c["models"].items():
        p = path(spec["workflow"])
        if not p.exists():
            raise RuntimeError(f"Workflow not compiled: {p}; run prepare")
        graph = read_json(p)
        check_workflow(model, graph, spec)
        for nid, node in graph.items():
            cls = node["class_type"]
            if cls not in object_info:
                raise RuntimeError(f"{model}: missing {cls} node at id {nid}")
            required = object_info[cls].get("input", {}).get("required", {})
            missing = set(required) - set(node["inputs"])
            if missing:
                raise RuntimeError(f"{model}: node {nid} {cls} missing {sorted(missing)}")
            for input_name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    continue  # Graph edge, not a widget value.
                schema = required.get(input_name) or object_info[cls].get("input", {}).get("optional", {}).get(input_name)
                if schema:
                    validate_live_choice(model, nid, cls, input_name, value, schema)
        print(model, "OK: compiled nodes", len(graph), "checkpoint", spec["unet_filename"])
    for name, s in c["scenes"].items():
        p = path(s["image"])
        if not p.exists():
            raise RuntimeError(f"Missing real image for {name}: {p}; see README")
        with Image.open(p) as im:
            print(name, im.size, im.mode)
    if c["frames"] <= 1 or c["frames"] % 4 != 1:
        raise RuntimeError("Expected frames = 4n+1, e.g. 121")
    print("Preflight passed; API format node schemas and local files checked. Runtime sample not yet verified.")


def query_gpu():
    cmd = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "--id=0"]
    try:
        line = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=True).stdout.strip().splitlines()[0]
        nums = [x.strip() for x in line.split(",")]
        return int(nums[0]), float(nums[1])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None, None


class Telemetry:
    def __init__(self, interval, outfile):
        self.interval = interval
        self.outfile = outfile
        self.event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._work, daemon=True)

    def _work(self):
        while not self.event.is_set():
            vram, util = query_gpu()
            mem = psutil.virtual_memory().used / 1024**2
            self.samples.append({"elapsed_seconds": round(time.monotonic()-self.start, 2), "gpu_memory_used_mib": vram, "gpu_utilization_pct": util, "wsl_mem_used_mib": round(mem, 1)})
            self.event.wait(self.interval)

    def __enter__(self):
        self.start = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.event.set()
        self.thread.join(timeout=10)
        write_csv(self.outfile, ["elapsed_seconds", "gpu_memory_used_mib", "gpu_utilization_pct", "wsl_mem_used_mib"], self.samples)

    def metric(self, key):
        vals = [r[key] for r in self.samples if r[key] is not None]
        return max(vals) if vals else None


def upload_image(server, file):
    p = Path(file)
    name = f"i2vbench_{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}{p.suffix.lower()}"
    with p.open("rb") as f:
        r = api(server, "upload/image", "POST", files={"image": (name, f, "image/png")}, data={"overwrite": "true"}, timeout=120).json()
    return r.get("name", name)


def walk_files(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("filename"), str) and obj["filename"].lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
            yield obj
        for v in obj.values():
            yield from walk_files(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_files(item)


def wait_history(server, prompt_id, timeout, poll):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = api(server, "history/" + prompt_id, timeout=35).json()
        job = history.get(prompt_id)
        if job:
            status = job.get("status") or {}
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError("ComfyUI execution failed: " + json.dumps(status.get("messages", []), ensure_ascii=False)[-2000:])
            return job
        time.sleep(poll)
    try:
        api(server, "interrupt", "POST", json={}, timeout=15)
    except Exception:
        pass
    raise TimeoutError(f"Timeout {timeout}s waiting for prompt {prompt_id}; sent interrupt")


def save_result_video(server, job, dest):
    candidates = list(walk_files(job.get("outputs") or {}))
    if not candidates:
        raise RuntimeError("No video file returned by ComfyUI history outputs. Check SaveVideo and current ComfyUI output format.")
    item = candidates[-1]
    params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
    r = api(server, "view", params=params, timeout=120, stream=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(1048576):
            f.write(chunk)
    if dest.stat().st_size < 4096:
        raise RuntimeError("Returned video too small / empty")


def probe(file):
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(file)], capture_output=True, text=True, timeout=30, check=True)
    data = json.loads(p.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("No video stream in downloaded file")
    rate = video.get("avg_frame_rate") or "0/1"
    a, _, b = rate.partition("/")
    fps = float(a) / float(b or 1) if float(b or 1) else 0
    dur = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    return int(video["width"]), int(video["height"]), round(fps, 2), round(dur, 3)


def run(args):
    c = cfg()
    preflight(args)
    res = path(c["output_dir"])
    rows_path = res / "runs.csv"
    rows = read_csv(rows_path)
    existing = {r["run_id"]: r for r in rows}
    targets = [(model, scene, seed) for model in c["models"] for scene in c["scenes"] for seed in c["seeds"]]
    submitted = 0
    input_map = {}
    for model, scene, seed in targets:
        rid = f"{model}__{scene}__s{seed}"
        # No invisible retries: a failed attempt remains a failed trial in the 9-run protocol.
        if rid in existing:
            continue
        if args.limit is not None and submitted >= args.limit:
            break
        spec = c["models"][model]
        scenario = c["scenes"][scene]
        record = dict.fromkeys(RUN_FIELDS, "")
        record.update(run_id=rid, model=model, scene=scene, seed=seed, status="failed", timestamp=now())
        baseline, _ = query_gpu()
        record["gpu_baseline_mib"] = baseline if baseline is not None else ""
        started = time.monotonic()
        try:
            if scene not in input_map:
                input_map[scene] = upload_image(c["server"], path(scenario["image"]))
            workflow = read_json(path(spec["workflow"]))
            prepared = patch_graph(workflow, input_map[scene], scenario["prompt"], seed, spec, c, "i2vbench/" + rid)
            write_json(res / "raw" / (rid + ".prompt.json"), prepared)
            with Telemetry(c["sample_interval_s"], res / "gpu" / (rid + ".csv")) as monitor:
                payload = {"prompt": prepared, "client_id": str(uuid.uuid4())}
                response = api(c["server"], "prompt", "POST", json=payload, timeout=90).json()
                if "prompt_id" not in response:
                    raise RuntimeError("Prompt rejected: " + json.dumps(response, ensure_ascii=False)[:2000])
                pid = response["prompt_id"]
                record["prompt_id"] = pid
                job = wait_history(c["server"], pid, c["job_timeout_s"], c["poll_interval_s"])
                write_json(res / "raw" / (rid + ".history.json"), job)
                out = res / "videos" / (rid + ".mp4")
                save_result_video(c["server"], job, out)
            wall = time.monotonic() - started
            w, h, fps, duration = probe(out)
            if duration < 2 or duration > 9:
                raise RuntimeError(f"Output duration {duration}s outside acceptable 2..9s range")
            if abs(w - c["width"]) > 16 or abs(h - c["height"]) > 16:
                raise RuntimeError(f"Output size {w}x{h} differs from target {c['width']}x{c['height']} by >16 pixels")
            if abs(fps - c["fps_target"]) > 0.5:
                raise RuntimeError(f"Output FPS {fps} differs from target {c['fps_target']} by >0.5")
            record.update(status="success", wall_seconds=round(wall, 2), width=w, height=h, fps=fps, video_seconds=duration, render_seconds_per_video_second=round(wall/duration, 2), output=str(out.relative_to(ROOT)))
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"[:1800]
            print("FAILED", rid, record["error"], file=sys.stderr)
            if isinstance(e, TimeoutError):
                print("Timeout interrupted; stopping batch to avoid overlapping unfinished work", file=sys.stderr)
                args.stop_after = True
        finally:
            record["wall_seconds"] = record["wall_seconds"] or round(time.monotonic()-started, 2)
            if "monitor" in locals() and isinstance(monitor, Telemetry):
                peak = monitor.metric("gpu_memory_used_mib")
                record["gpu_peak_mib"] = peak if peak is not None else ""
                record["gpu_delta_peak_mib"] = max(0, peak-baseline) if peak is not None and baseline is not None else ""
                record["gpu_util_peak_pct"] = monitor.metric("gpu_utilization_pct") or ""
                record["wsl_mem_peak_mib"] = monitor.metric("wsl_mem_used_mib") or ""
                del monitor
            existing[rid] = record
            write_csv(rows_path, RUN_FIELDS, [existing[key] for key in sorted(existing)])
            print(f"{rid} => {record['status']} time={record['wall_seconds']}s vram={record['gpu_peak_mib']} MiB")
            submitted += 1
            if getattr(args, "stop_after", False):
                break
    print("Results:", rows_path)


def demo_images(args):
    """Deliberately schematic first-frames for pipeline smoke testing, not image-quality research."""
    W,H=480,832
    for name,mode in [("rainy_city",0),("windy_portrait",1),("calm_mountains",2)]:
        img=Image.new("RGB",(W,H))
        d=ImageDraw.Draw(img)
        for y in range(H):
            t=y/H
            if mode==0: col=(int(7+30*t),int(15+18*t),int(40+36*t))
            elif mode==1: col=(int(170+48*t),int(105+47*t),int(84+39*t))
            else: col=(int(35+79*t),int(86+82*t),int(132+50*t))
            d.line([(0,y),(W,y)],fill=col)
        rng=random.Random(mode)
        if mode==0:
            for i in range(27):
                x=rng.randrange(-15,W); high=rng.randrange(200,580); width=rng.randrange(28,77)
                d.rectangle((x,high,x+width,720),fill=(19,27,57))
                for wx in range(x+7,x+width-5,12):
                    for wy in range(high+15,690,24):
                        if rng.random()<.55:d.rectangle((wx,wy,wx+4,wy+9),fill=rng.choice([(222,112,91),(60,132,190),(237,188,76)]))
            d.ellipse((190,330,290,590),fill=(22,25,42));d.line([(0,750),(W,660)],fill=(70,122,147),width=6)
        elif mode==1:
            d.ellipse((88,120,393,510),fill=(84,50,48));d.ellipse((148,180,348,495),fill=(223,159,123));d.ellipse((180,310,197,326),fill=(70,45,33));d.ellipse((287,310,305,326),fill=(70,45,33));d.arc((211,340,271,397),0,180,fill=(120,74,72),width=5)
            d.polygon([(65,810),(105,530),(375,530),(440,810)],fill=(67,67,85))
        else:
            d.ellipse((320,94,420,194),fill=(251,210,133))
            d.polygon([(0,580),(115,310),(240,585)],fill=(66,107,135));d.polygon([(110,600),(330,255),(510,630)],fill=(52,84,126));d.polygon([(0,730),(280,480),(500,700),(500,832),(0,832)],fill=(29,74,89))
        dst=path("assets"+"/"+name+".png")
        dst.parent.mkdir(exist_ok=True,parents=True)
        img.save(dst)
        print("Smoke illustration only:",dst)


def blind_video_alias(source: Path, target_dir: Path, blind_id: str) -> Path:
    """创建只使用盲评 ID 命名的视频链接。

    Args:
        source: 正式结果目录中的原视频。
        target_dir: 盲评页使用的视频目录。
        blind_id: 不含模型信息的匿名 ID。

    Returns:
        以盲评 ID 命名的视频文件路径。
    """
    source = Path(source)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    alias = target_dir / f"{blind_id}{source.suffix.lower()}"
    alias.unlink(missing_ok=True)
    try:
        alias.hardlink_to(source)
    except OSError:
        shutil.copy2(source, alias)
    return alias


def review_sheet(args):
    c=cfg(); res=path(c["output_dir"])
    rows=[r for r in read_csv(res/"runs.csv") if r["status"]=="success" and path(r["output"]).exists()]
    if not rows:
        raise RuntimeError("No successful real runs, cannot create review sheet")
    secret_path=res/"blind_salt.txt"
    if not secret_path.exists():
        secret_path.write_text(uuid.uuid4().hex,encoding="utf-8")
    salt=secret_path.read_text().strip()
    blinds=[]
    for r in rows:
        bid="V"+hashlib.sha256((salt+r["run_id"]).encode()).hexdigest()[:10].upper()
        blinds.append((bid,r))
    random.Random(salt).shuffle(blinds)
    write_csv(res/"blind_key.csv",["blind_id","run_id","model","scene","seed"],[dict(blind_id=b,run_id=r["run_id"],model=r["model"],scene=r["scene"],seed=r["seed"]) for b,r in blinds])
    prev={x["blind_id"]:x for x in read_csv(res/"manual_scores.csv")}
    write_csv(res/"manual_scores.csv",SCORE_FIELDS,[dict(prev.get(b,{"blind_id":b,"visual_quality":"","motion_stability":"","prompt_adherence":"","notes":""})) for b,_ in blinds])
    cards=[]
    for b,r in blinds:
        alias=blind_video_alias(path(r["output"]), res/"blind_videos", b)
        src=html.escape(alias.relative_to(res).as_posix())
        cards.append(f'<article><h3>{b}</h3><video controls preload="metadata" src="{src}"></video><p>场景：{html.escape(r["scene"])}；请在 manual_scores.csv 填 1~5 分。</p></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Blind I2V Review</title><style>body{background:#111827;color:#f3f4f6;font:16px system-ui;padding:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}article{border:1px solid #374151;border-radius:12px;padding:16px}video{width:100%;max-height:560px}p{color:#9ca3af}</style><h1>盲评：不显示模型名</h1><p>按画质、运动稳定性、提示词遵循分别打 1~5 分；完成后编辑 manual_scores.csv 并运行 report。</p><main>'''+''.join(cards)+'''</main></html>'''
    (res/"blind_review.html").write_text(page,encoding="utf-8")
    print("Blind review HTML:",res/"blind_review.html")
    print("Scores CSV:",res/"manual_scores.csv")


def chronological_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """按开始时间排列试验记录。

    Args:
        rows: 从 benchmark CSV 读取的试验记录。

    Returns:
        按 timestamp 升序排列的新列表。
    """
    return sorted(rows, key=lambda row: row.get("timestamp", ""))


def median(vals):
    vals=sorted(vals)
    if not vals:return None
    n=len(vals)
    return vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2


def report(args):
    c=cfg(); out=path(c["output_dir"])
    rows=read_csv(out/"runs.csv")
    score_rows=read_csv(out/"manual_scores.csv")
    key={x["run_id"]:x["blind_id"] for x in read_csv(out/"blind_key.csv")}
    scores={r["blind_id"]:r for r in score_rows}
    results={}
    expected=len(c["scenes"])*len(c["seeds"])
    for model in c["models"]:
        group=chronological_runs([r for r in rows if r["model"]==model])
        good=[r for r in group if r["status"]=="success"]
        valid=[]
        for r in good:
            s=scores.get(key.get(r["run_id"],""),{})
            try:
                v,m,p=[float(s[k]) for k in ("visual_quality","motion_stability","prompt_adherence")]
                if not all(1<=x<=5 for x in (v,m,p)):continue
                valid.append(.45*v+.35*m+.20*p)
            except (TypeError,ValueError,KeyError):pass
        # Exclude first successful run from warm timings; cold load is tracked separately.
        warm=good[1:] if len(good)>1 else []
        results[model]={"attempted":len(group),"expected":expected,"success":len(good),"failed":len([r for r in group if r["status"]!="success"]),"quality_n":len(valid),"quality_100":sum(valid)/len(valid)*20 if valid else None,
                        "warm_seconds":median([float(r["wall_seconds"]) for r in warm]),
                        "render_s_per_s":median([float(r["render_seconds_per_video_second"]) for r in warm]),
                        "peak_vram_mib":max([int(r["gpu_peak_mib"]) for r in good if r["gpu_peak_mib"] not in ("",None)],default=None),
                        "peak_vram_delta_mib":max([int(r["gpu_delta_peak_mib"]) for r in good if r["gpu_delta_peak_mib"] not in ("",None)],default=None),
                        "peak_wsl_ram_mib":max([float(r["wsl_mem_peak_mib"]) for r in good if r["wsl_mem_peak_mib"] not in ("",None)],default=None),
                        "cold_seconds":float(good[0]["wall_seconds"]) if good else None,
                        "failed_cases":[r["run_id"] for r in group if r["status"]!="success"],
                        "formats":sorted(set(f"{r['width']}x{r['height']}@{r['fps']}fps" for r in good))}
    lines=["# I2V Benchmark 真实结果", "", f"生成时间：{now()}", "", "口径：完整计划是 1 个模型 × 3 场景 × 3 seed = 9 次；失败率基于已提交/记录的试验数，未执行不得算成功或失败。", "", "| 模型 | 成功/计划 | 失败率/已执行 | 冷启动 s | 热启动中位 s | GPU峰值 MiB | 增量峰值 MiB | WSL RAM峰值 MiB | 已评分/成功 | 画质100分 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):return f"{x:.2f}" if isinstance(x,(float,int)) else "待测"
    for m,r in results.items():
        fail_rate=f"{r['failed']/r['attempted']*100:.1f}% ({r['failed']}/{r['attempted']})" if r['attempted'] else "待测"
        lines.append(f"| {m} | {r['success']}/{expected} | {fail_rate} | {fmt(r['cold_seconds'])} | {fmt(r['warm_seconds'])} | {fmt(r['peak_vram_mib'])} | {fmt(r['peak_vram_delta_mib'])} | {fmt(r['peak_wsl_ram_mib'])} | {r['quality_n']}/{r['success']} | {fmt(r['quality_100'])} |")
    lines.extend(["", "当前报告记录单模型运行基线，不生成模型排名。画质分数在人工评分完成前保持待测。"])
    lines.extend(["", "## 分场景结果", "", "| 模型 | 场景 | 成功/3 | 失败/已执行 | 成功样本中位用时(s) |", "|---|---|---:|---:|---:|"])
    for m in c["models"]:
        for scene in c["scenes"]:
            subset=[r for r in rows if r["model"]==m and r["scene"]==scene]
            good_scene=[r for r in subset if r["status"]=="success"]
            fail_scene=len(subset)-len(good_scene)
            lines.append(f"| {m} | {scene} | {len(good_scene)}/3 | {fail_scene}/{len(subset)} | {fmt(median([float(x['wall_seconds']) for x in good_scene]))} |")
    lines.extend(["", "## 输出规格与失败项", ""])
    for m,r in results.items():
        lines.append(f"- {m}: 输出规格 {r['formats'] or '无'}；失败 {', '.join(r['failed_cases']) or '无'}。")
    lines.extend(["", "## 解释与限制", "", "- GPU 峰值是 nvidia-smi 全卡占用，不是精确独占模型显存；应确保其他 GPU 服务已停止。", "- wall time 包含 Comfy 排队/文件上传之后的推理、解码、输出与文件拉取；首条冷启动尤其包含加载时间。", "- 热启动耗时排除首条成功样本的模型加载；原生解码分辨率与实际帧率应核对。", "- 当前 Hunyuan 使用 Step-Distilled 12 步、CFG=1、shift=7。", "- 雨夜/人物/风景三场景仅供业务评估，不是标准公共视频评测基准。"])
    (out/"report.md").parent.mkdir(parents=True,exist_ok=True)
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Saved",out/"report.md")
    print("\n".join(lines[:10]))


def main():
    parser=argparse.ArgumentParser(description="HunyuanVideo I2V benchmark")
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fetch-templates",help="Download official UI workflow templates")
    f.add_argument("--force", action="store_true")
    for cmd in ("prepare","preflight","make-demo-images","review-sheet","report"):
        sub.add_parser(cmd)
    r=sub.add_parser("run")
    r.add_argument("--limit", type=int, default=None)
    args=parser.parse_args()
    try:
        {"fetch-templates":fetch_templates,"prepare":prepare,"preflight":preflight,"make-demo-images":demo_images,"run":run,"review-sheet":review_sheet,"report":report}[args.command](args)
    except (requests.RequestException, OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2,f"ERROR: {type(exc).__name__}: {exc}\n")

if __name__=="__main__":
    main()
