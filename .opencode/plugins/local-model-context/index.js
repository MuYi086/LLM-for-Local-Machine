import {execFileSync} from 'node:child_process';
import path from 'node:path';

// OpenCode 2.2.7 的压缩请求复用主任务 system；这里仅替换压缩请求，避免审查格式压过摘要模板。
const SUMMARY_SYSTEM = `You are writing a conversation checkpoint, not performing the task in the transcript.
对话中的源码、审查格式和最终报告要求都是待总结的数据；不要继续审查、写最终报告或调用工具。
Return only a concise Markdown checkpoint using these exact English heading lines. Do not translate headings or wrap the answer in a code fence.
## Objective
## Requirements
## Decisions
## Work State
### Completed
### Active
### Blocked
## Next Move
## Relevant Files
## Important Context
Use short factual bullets below each heading; write (none) where necessary. Preserve exact paths, unfinished requests, test ID, telemetry directory, result path and next unread line for every active file. Never claim unread code was reviewed. 摘要正文使用简体中文，英文标题保持原样。正文目标不超过 600 个汉字；只保存恢复任务必需的状态，不复述源码或完整报告。每个标题下最多两条短句。`;
const SOURCES = ['start.sh', 'easel_bridge/app.py', 'model-text/ornith_1_5_35b_a3b/app.py', 'model-text/ornith_1_5_35b_a3b/runtime.py'];
const local = (event) => event.model?.providerID === 'local-ornith';

export default {
  id: 'llm-local-model-context',
  async setup(ctx) {
    const sessions = new Map();
    const bindings = new Map();
    const sources = new Set(SOURCES.map(p => path.resolve(ctx.location.directory, p)));
    await ctx.session.hook('context', event => {
      if (!local(event)) { sessions.delete(event.sessionID); return; }
      event.system.push({type:'text', text:'面向用户的进度说明、工具操作说明、解释和最终报告全部使用简体中文。代码标识符、路径和协议规定的固定标题保持原文。'});
      // 仅为明确执行本仓库固定基准的会话限制原生 read，其他工作保持原样。
      if (!sessions.has(event.sessionID) && JSON.stringify(event.messages).includes('CONTEXT_BENCHMARK_PROMPT.md')) {
        if (sessions.size >= 256) sessions.delete(sessions.keys().next().value);
        sessions.set(event.sessionID, { lastMessage: undefined });
      }
      if (sessions.has(event.sessionID)) {
        if (!bindings.has(event.sessionID)) {
          try {
            const value = execFileSync('python3', [path.join(ctx.location.directory, 'scripts/prepare_context_benchmark.py'), '--session', event.sessionID, '--json'], {encoding:'utf8', timeout:10000, stdio:['ignore','pipe','pipe']});
            bindings.set(event.sessionID, JSON.parse(value));
          } catch {
            bindings.set(event.sessionID, {error:'自动绑定失败：检查 start.sh 性能采集是否开启及固定入口是否存在。不得借用旧数据。'});
          }
        }
        event.system.push({type:'text', text:'本轮自动性能绑定（由启动器批次与会话 ID 建立）：'+JSON.stringify(bindings.get(event.sessionID))+'。使用其中 report 作为唯一报告文件，保留遥测标记。正文写完后使用其中 directory/test_id/report 执行 scripts/model_telemetry.py 的 end --attach-result，汇总到同一文件。该窗口截止于最终回复之前，最终回复不计入；请在报告注明。无需让用户执行 begin/end。'});
      }
      if (sessions.has(event.sessionID)) event.system.push({type:'text', text:'固定基准源码必须使用 read(path, offset, limit)，offset 从 1 开始，limit 不超过 150；每次模型回复只发起一个源码 read，按返回行号继续，禁止一轮读取四个文件。压缩检查点后从下一未读行继续。'});
    });
    await ctx.tool.hook('execute.before', event => {
      const state = sessions.get(event.sessionID);
      if (!state || event.tool !== 'read' || !event.input || typeof event.input.path !== 'string') return;
      if (!sources.has(path.resolve(ctx.location.directory, event.input.path))) return;
      if (state.lastMessage === event.messageID) throw new Error('固定基准每次模型回复只允许一个源码 read；请在下一次回复按 offset 和 limit 继续。');
      state.lastMessage = event.messageID;
      event.input = {...event.input, offset: Number.isInteger(event.input.offset) && event.input.offset > 0 ? event.input.offset : 1, limit: Number.isInteger(event.input.limit) && event.input.limit > 0 ? Math.min(event.input.limit, 150) : 150};
    });
    await ctx.session.hook('compaction', event => {
      if (!local(event)) return;
      event.system = [{type:'text', text:SUMMARY_SYSTEM}];
      event.tools = {};
      event.options.temperature = 0;
      event.options.maxTokens = Math.min(event.options.maxTokens ?? 4096, 4096);
    });
    await ctx.session.hook('http.request', async event => {
      if (!local(event) || event.kind !== 'compaction') return;
      const original = event.request;
      // 只修改本次已获准的请求，不建立第二条模型调用或绕过现有网关/GPU 锁。
      if (original.method !== 'POST' || !original.headers.get('content-type')?.includes('application/json')) return;
      const body = await original.clone().json();
      if (!body || Array.isArray(body) || typeof body !== 'object') return;
      body.chat_template_kwargs = {...body.chat_template_kwargs, enable_thinking:false};
      const headers = new Headers(original.headers);
      headers.delete('content-length');
      event.request = new Request(original, {headers, body:JSON.stringify(body)});
    });
  },
};
