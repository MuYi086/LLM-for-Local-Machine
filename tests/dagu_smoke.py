"""用固定版本 Dagu scheduler 和假模型服务验证真实入队，不加载模型。"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from test_dagu_jobs import Handler


def smoke_environment(root: Path, port: int) -> dict[str, str]:
    """清除部署路径和配置覆盖，仅保留二进制选择以隔离测试状态。"""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DAGU_") or key == "DAGU_BIN"
    } | {
        "DAGU_HOME": str(root / "server"),
        "DAGU_JOB_ROOT": str(root),
        "DAGU_SIGNAL_PROPAGATION": "true",
        "DAGU_MOSS_BASE_URL": f"http://127.0.0.1:{port}",
        "DAGU_VOXCPM_BASE_URL": f"http://127.0.0.1:{port}",
    }


def main() -> None:
    """在独立临时目录中完成两个 DAG，退出时关闭调度器。"""
    repository = Path(__file__).resolve().parents[1]
    wrapper = repository / "scripts/dagu/start.sh"
    with tempfile.TemporaryDirectory(prefix="dagu-smoke-") as directory:
        root = Path(directory)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            (root / "inputs").mkdir()
            (root / "inputs/image.json").write_text('{"prompt":"test"}')
            (root / "inputs/speech.json").write_text('{"text":"test"}')
            env = smoke_environment(root, server.server_port)
            with (root / "scheduler.log").open("w") as log:
                process = subprocess.Popen(
                    ["bash", str(wrapper), "scheduler"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    for kind, job in (("moss", "image"), ("voxcpm", "speech")):
                        subprocess.run(
                            ["bash", str(wrapper), "enqueue", kind, "--", f"job_id={job}"],
                            env=env,
                            check=True,
                            timeout=15,
                        )
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError("Dagu scheduler 提前退出。")
                        if all(
                            (root / "jobs" / job / "completed.json").exists()
                            for job in ("image", "speech")
                        ):
                            print("Dagu 两个真实入队任务均完成（假模型 HTTP 服务）。")
                            break
                        time.sleep(0.1)
                    else:
                        raise TimeoutError("Dagu 无模型联调超时。")
                except Exception:
                    log.flush()
                    print((root / "scheduler.log").read_text()[-8000:])
                    raise
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
