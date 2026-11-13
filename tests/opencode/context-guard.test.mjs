import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import plugin from '../../.opencode/plugins/local-model-context/index.js';

async function setup(directory = '/workspace/repo') {
  const hooks = new Map();
  await plugin.setup({ location: { directory }, session: { hook: async (name, fn) => hooks.set(`session.${name}`, fn) }, tool: { hook: async (name, fn) => hooks.set(`tool.${name}`, fn) } });
  return hooks;
}
const event = () => ({sessionID: 's1', model: {providerID: 'local-ornith', id: 'Ornith-1.5-9B-GGUF'}, system: [{type: 'text', text: 'Write a six-part review in Chinese.'}], messages: [{role:'user',content:[{type:'text',text:'Read CONTEXT_BENCHMARK_PROMPT.md'}]}], tools: {read:{}}, options: {maxTokens:4096}});

test('compaction gets dedicated headings without original review instructions', async () => {
  const hooks = await setup(); const input = event();
  const history = JSON.stringify(input.messages);
  await hooks.get('session.compaction')(input);
  assert.equal(input.tools.read, undefined);
  assert.equal(JSON.stringify(input.messages), history);
  const text = input.system.map(p=>p.text).join('\n');
  assert.ok(!text.includes('six-part review'));
  for(const heading of ['Objective','Requirements','Decisions','Work State','Next Move','Relevant Files','Important Context']) assert.ok(text.includes(`## ${heading}`));
  assert.equal(input.options.temperature, 0);
  assert.equal(input.options.maxTokens, 4096);
  assert.ok(text.includes("摘要正文使用简体中文"));
});
test('non-local model compaction and primary requests retain normal behavior', async () => {
  const hooks = await setup();const input = event();input.model.providerID='other';const before=JSON.stringify(input);
  await hooks.get('session.compaction')(input);assert.equal(JSON.stringify(input),before);
});
test('benchmark reads get capped and another source read in same message is rejected', async () => {
  const hooks = await setup(); await hooks.get('session.context')(event());
  const first={sessionID:'s1',messageID:'m1',tool:'read',input:{path:'/workspace/repo/start.sh'}};
  await hooks.get('tool.execute.before')(first);assert.equal(first.input.limit,150);assert.equal(first.input.offset,1);
  await assert.rejects(async()=>hooks.get('tool.execute.before')({sessionID:'s1',messageID:'m1',tool:'read',input:{path:'/workspace/repo/easel_bridge/app.py'}}), /one|一个/);
  const next={sessionID:'s1',messageID:'m2',tool:'read',input:{path:'/workspace/repo/start.sh',offset:151,limit:9999}};
  await hooks.get('tool.execute.before')(next);assert.equal(next.input.limit,150);assert.equal(next.input.offset,151);
});
test('other sessions and files are not modified', async () => {
 const hooks=await setup();await hooks.get('session.context')(event());
 for(const input of [{sessionID:'other',messageID:'m',tool:'read',input:{path:'/workspace/repo/start.sh'}},{sessionID:'s1',messageID:'m',tool:'read',input:{path:'/workspace/repo/README.md'}}]) {const before=JSON.stringify(input);await hooks.get('tool.execute.before')(input);assert.equal(JSON.stringify(input),before);}
});
test('only summary HTTP body disables thinking and headers remain intact', async () => {
 const hooks=await setup();
 const input={...event(),kind:'compaction',request:new Request('http://127.0.0.1:8500/v1/chat/completions',{method:'POST',headers:{Authorization:'Bearer test', 'Content-Type':'application/json'},body:JSON.stringify({model:'Ornith-1.5-9B-GGUF',chat_template_kwargs:{custom:true,enable_thinking:true},messages:[{role:'user',content:'summary'}]})})};
 await hooks.get('session.http.request')(input);const body=await input.request.json();assert.equal(body.chat_template_kwargs.enable_thinking,false);assert.equal(body.chat_template_kwargs.custom,true);assert.equal(input.request.headers.get('authorization'),'Bearer test');
 assert.equal(body.llm_context_guard, true);
 const primary={...event(),kind:'primary',request:new Request('http://127.0.0.1:8500/v1/chat/completions',{method:'POST',body:'{}'})};const before=primary.request;await hooks.get('session.http.request')(primary);assert.equal(primary.request,before);
});

test('benchmark telemetry JSON is blocked without affecting other sessions', async () => {
 const hooks=await setup();await hooks.get('session.context')(event());
 const read={sessionID:'s1',messageID:'m1',tool:'read',input:{path:'/workspace/repo/storage/.cache/runtime/telemetry/run-x/report-x.json'}};
 await assert.rejects(async()=>hooks.get('tool.execute.before')(read), /brief/);
 await hooks.get('tool.execute.before')({...read,sessionID:'other'});
});
test('benchmark instructions require bounded CLI finish and explain Code Mode', async () => {
 const hooks=await setup();const input=event();await hooks.get('session.context')(input);
 const text=input.system.map(p=>p.text).join('\n');
 assert.ok(text.includes('--brief'));assert.ok(text.includes('JavaScript'));
 assert.ok(text.includes('attached'));assert.ok(text.includes('最终回复'));
});
test('large generated reports and telemetry symlink aliases are guarded', async () => {
 const root=mkdtempSync(path.join(os.tmpdir(),'opencode-brief-'));
 try {
  mkdirSync(path.join(root,'测试结果')); const report=path.join(root,'测试结果','result.md');
  writeFileSync(report,'stub'); const hooks=await setup(root);await hooks.get('session.context')(event());
  const small={sessionID:'s1',messageID:'m1',tool:'read',input:{path:report}};
  await hooks.get('tool.execute.before')(small);assert.equal(small.input.limit,40);
  writeFileSync(report,'x'.repeat(20000));
  await assert.rejects(async()=>hooks.get('tool.execute.before')({...small,messageID:'m2'}),/brief/);
  const run=path.join(root,'storage/.cache/runtime/telemetry/run-test');mkdirSync(run,{recursive:true});
  writeFileSync(path.join(run,'report.json'),'{}');symlinkSync(run,path.join(root,'alias'),'dir');
  await assert.rejects(async()=>hooks.get('tool.execute.before')({sessionID:'s1',messageID:'m3',tool:'read',input:{path:path.join(root,'alias/report.json')}}),/brief/);
  symlinkSync(path.join(run,'report.json'),path.join(root,'alias/disguised.txt'));
  await assert.rejects(async()=>hooks.get('tool.execute.before')({sessionID:'s1',messageID:'m4',tool:'read',input:{path:path.join(root,'alias/disguised.txt')}}),/brief/);
 } finally {rmSync(root,{recursive:true,force:true});}
});

test('local primary context requires Chinese progress and reports', async () => {
 const hooks=await setup(); const input=event();
 await hooks.get('session.context')(input);
 assert.ok(input.system.some(part=>part.text.includes('全部使用简体中文')));
});
