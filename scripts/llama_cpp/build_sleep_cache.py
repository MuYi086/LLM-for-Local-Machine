#!/usr/bin/env python3
"""复用仓库外 CUDA 构建，独立链接 sleep-cache 服务，保留正在运行的库。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def link_command(
    command: str, output: Path, bin_dir: Path, impl: Path | None = None, archive: Path | None = None
) -> list[str]:
    """替换链接输出和搜索路径，避免原地覆盖活动的共享库。"""
    args = shlex.split(command)
    args[args.index("-o") + 1] = str(output)
    for index, arg in enumerate(args):
        if arg.startswith("-Wl,-rpath,"):
            args[index] = "-Wl,-rpath," + str(bin_dir) + ":" + arg[len("-Wl,-rpath,") :]
        if impl is not None and arg.endswith("/libllama-server-impl.so"):
            args[index] = str(impl)
        if archive is not None and Path(arg).name == "libserver-context.a":
            args[index] = str(archive)
    return args


def build_private(build: Path, output: Path) -> None:
    """直接编译一个翻译单元并复制 archive，完全不执行依赖构建。"""
    commands = json.loads((build / "compile_commands.json").read_text())
    matches = [
        entry for entry in commands if entry["file"].endswith("/tools/server/server-context.cpp")
    ]
    if len(matches) != 1:
        raise RuntimeError("compile_commands.json 中没有唯一的 server-context.cpp 编译命令。")
    entry = matches[0]
    work = build / "tools/server"
    with tempfile.TemporaryDirectory(prefix="sleep-cache-build-", dir=output) as temporary:
        private = Path(temporary)
        obj = private / "server-context.cpp.o"
        compile_args = shlex.split(entry["command"])
        compile_args[compile_args.index("-o") + 1] = str(obj)
        # 依赖文件也写入私有目录，不能触碰原构建树。
        for index, arg in enumerate(compile_args[:-1]):
            if arg == "-MF":
                compile_args[index + 1] = str(private / "context.d")
        subprocess.run(compile_args, cwd=entry["directory"], check=True)
        archive = private / "libserver-context.a"
        shutil.copy2(work / "libserver-context.a", archive)
        subprocess.run(["ar", "r", str(archive), str(obj)], check=True)
        impl = output / "libllama-server-impl.so"
        for target, destination in (
            ("llama-server-impl", impl),
            ("llama-server", output / "llama-server"),
        ):
            command = (work / f"CMakeFiles/{target}.dir/link.txt").read_text()
            new_path = destination.with_name(destination.name + ".new")
            subprocess.run(
                link_command(
                    command, new_path, output, impl if target == "llama-server" else None, archive
                ),
                cwd=work,
                check=True,
            )
            os.replace(new_path, destination)


def main() -> None:
    """在独立目录串行构建，输出用原子替换保护已有运行实例。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    if output == build / "bin":
        parser.error("output-dir 必须与原 build/bin 分离。")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_private(build, output)
    print(output / "llama-server")


if __name__ == "__main__":
    main()
