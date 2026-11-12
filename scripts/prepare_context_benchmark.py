"""创建明确绑定采集批次的上下文测试报告与提交指令。"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm_local_runtime/src'))
from llm_local_runtime.telemetry_cli import begin_test


def main():
    """在模型任务前建立统计窗口，不自动选择历史批次。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--context', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = (args.directory or Path((root / 'storage/.cache/runtime/telemetry-current').read_text().strip())).resolve()
    binding = None
    if args.session:
        import hashlib
        binding = directory / 'tests' / ('session-' + hashlib.sha256(args.session.encode()).hexdigest() + '.json')
        if binding.exists():
            print(binding.read_text())
            return
    manifest = json.loads((directory / 'manifest.jsonl').read_text())
    configured = int(manifest['config']['ORNITH_CONTEXT_SIZE'])
    args.context = configured if args.context is None else args.context
    if args.context <= 0 or configured != args.context:
        parser.error(f'上下文与采集批次配置不匹配：{configured}')
    samples = directory / 'resources.jsonl'
    if not samples.exists() or datetime.now().timestamp() - samples.stat().st_mtime > 15:
        parser.error('采集器没有近期资源采样，请先启动性能记录。')
    label = f'{args.context // 1024}k' if args.context % 1024 == 0 else str(args.context)
    test_id = begin_test(directory, f'ornith-{label}')
    now = datetime.now().astimezone()
    target = Path(__file__).resolve().parents[1] / '测试结果' / f'{label}-{now:%Y%m%d-%H时%M分%S秒}-{test_id[:8]}.md'
    target.parent.mkdir(exist_ok=True)
    with target.open('x') as output:
        output.write(f'# Ornith {label} 上下文舒适度测试\n\n上下文：{args.context} tokens（启动器配置，非后端运行验证）。\n创建时间：{now.isoformat()}\n性能目录：{directory}\n测试 ID：{test_id}\n\n<!-- MODEL_TELEMETRY_BEGIN test_id={test_id} -->\n等待本轮汇总。\n<!-- MODEL_TELEMETRY_END -->\n\n## 模型任务报告\n\n待执行。\n')
    instruction = f'请读取 CONTEXT_BENCHMARK_PROMPT.md，只执行“固定任务”一节。\n本轮模型：Ornith-1.5-35B-A3B-AD-Q5\n本轮上下文上限：{args.context} tokens\n本轮性能目录：{directory}\n本轮测试 ID：{test_id}\n本轮结果文件已创建：{target}\n请在该文件追加报告，保留统计标记，不另建结果文件。\n'
    instruction_path = directory / 'tests' / f'{test_id}.prompt.txt'
    instruction_path.write_text(instruction)
    result = {'directory': str(directory), 'test_id': test_id, 'context': args.context, 'report': str(target)}
    if binding:
        binding.write_text(json.dumps(result, ensure_ascii=False))
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return
    print(instruction)
    print('回复结束后执行（包含最终回复的统计）：')
    print(f'python3 scripts/model_telemetry.py --directory "{directory}" end {test_id} --settle-seconds 30 --attach-result "{target}"')
    print(f'提交指令已保存：{instruction_path}')


if __name__ == '__main__':
    main()
