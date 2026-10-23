import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PNG } from "pngjs";
import { chatClient, type ChatClient, type Completion, type Message } from "../src/dataset-client.js";
import { familySplit, generatePrompts, validateSpecs, type PromptSpec } from "../src/dataset-prompts.js";
import { generateDataset, parseEvaluation, type DatasetRecord } from "../src/dataset.js";
import { ENVIRONMENT_CONFIG } from "../src/environment.js";
import { LocalRenderer, type Renderer } from "../src/renderer.js";

const config = { ...ENVIRONMENT_CONFIG, width: 32, height: 32 };
const source = 'from artcanvas import ArtCanvas\nwith ArtCanvas(32,32) as c:\n c.set_color(0,0,1)\n c.draw_circle(16,16,8,True)';
const spec: PromptSpec = { id: "circle", family: "circles", split: "train", prompt: "Draw a blue circle", requirements: ["One blue circle"], expectedOutcome: "drawing", seed: 42 };
const completion = (text: string): Completion => ({ text, model: "scripted", finishReason: "stop", usage: { prompt_tokens: 100, completion_tokens: 50 } });
function scripted(replies: string[], calls: Message[][] = []): ChatClient {
  return { model: "scripted", baseUrl: "http://localhost/v1", async complete(messages) {
    calls.push(structuredClone(messages));
    const reply = replies.shift(); assert.notEqual(reply, undefined, "Unexpected API request");
    return completion(reply!);
  } };
}
const rubric = (overrides = {}) => JSON.stringify({ safety: "safe", scores: [1], correctRefusal: false, reason: "One blue circle is visible", ...overrides });
const png = PNG.sync.write(new PNG({ width: 32, height: 32 }));
const renderer: Renderer = { render: async () => ({ ok: true, stdout: "", stderr: "", png }) };
async function fixture(fn: (output: string) => Promise<void>) {
  const output = await mkdtemp(join(tmpdir(), "artsy-dataset-"));
  try { await fn(output); } finally { await rm(output, { recursive: true, force: true }); }
}
async function records(output: string, name = "candidates"): Promise<DatasetRecord[]> {
  const text = await readFile(join(output, `${name}.jsonl`), "utf8");
  return text.trim() ? text.trim().split("\n").map(line => JSON.parse(line)) : [];
}

test("1000 curriculum specs are deterministic, unique, and split by whole family", () => {
  const specs = generatePrompts(1000, 42);
  assert.deepEqual(specs, generatePrompts(1000, 42));
  assert.deepEqual(specs.slice(0, 20), generatePrompts(20, 42));
  assert.notDeepEqual(generatePrompts(20, 43), specs.slice(0, 20));
  assert.equal(new Set(specs.map(s => s.prompt)).size, 1000);
  assert.deepEqual(new Set(specs.map(s => s.split)), new Set(["train", "validation", "test"]));
  for (const s of specs) assert.equal(s.split, familySplit(s.family));
  assert.throws(() => validateSpecs([{ ...spec, split: "test" }]), /Split/);
  assert.throws(() => validateSpecs([spec, { ...spec, id: "other" }]), /Duplicate prompt/);
  assert.throws(() => validateSpecs([{ ...spec, id: "../escape" }]), /safe filenames/);
  assert.throws(() => generatePrompts(0), /count/);
});

test("teacher repair uses real Python, re-renders final source, records provenance, and resumes", () => fixture(async output => {
  const calls: Message[][] = [], renders: string[] = [];
  const local = new LocalRenderer();
  const actualRenderer: Renderer = { async render(code, seed) { renders.push(code); return local.render(code, seed); } };
  const teacher = scripted([JSON.stringify({ code: "broken python" }), JSON.stringify({ code: source })], calls);
  const judgeCalls: Message[][] = [], judge = scripted([rubric(), rubric()], judgeCalls);
  const options = { output, specs: [spec], teacher, judge, candidates: 1, config, renderer: actualRenderer };
  const summary = await generateDataset(options);
  assert.equal(summary.accepted, 1); assert.equal(summary.requests, 4); assert.equal(summary.completionTokens, 200);
  assert.deepEqual(renders, ["broken python", source, source]);
  assert.match(calls[1][3].content as string, /SyntaxError/);
  const [record] = await records(output, "train");
  assert.equal(record.trace.length, 2); assert.equal(record.runtimeVersion, "artcanvas-v2");
  assert.equal(record.spec.seed, 42); assert.ok(record.sourceHash); assert.ok(record.imageHash); assert.ok(record.pixelHash);
  assert.equal(await readFile(join(output, record.artifacts.source!), "utf8"), source);
  assert.match(await readFile(join(output, record.artifacts.trajectory), "utf8"), /"operation":"submit"/);
  assert.match(JSON.stringify(judgeCalls[0][1]), /data:image\/png;base64,/);
  assert.match(await readFile(join(output, "manifest.json"), "utf8"), /runtimeHashes/);
  assert.deepEqual(await generateDataset(options), summary);
  assert.equal(renders.length, 3); assert.equal(calls.length, 2);
  await assert.rejects(generateDataset({ ...options, threshold: 0.9 }), /configuration changed/);
}));

test("a runnable example without a judge is review-only", () => fixture(async output => {
  const summary = await generateDataset({ output, specs: [spec], teacher: scripted([JSON.stringify({ code: source })]), candidates: 1, renderer, config });
  assert.equal(summary.review, 1); assert.equal(summary.accepted, 0);
  assert.equal((await records(output, "accepted")).length, 0);
  assert.match((await records(output, "review"))[0].reason, /evaluation pending/);
}));

test("adherence repair is bounded and fresh final image must independently pass", () => fixture(async output => {
  const calls: Message[][] = [];
  const summary = await generateDataset({ output, specs: [spec], candidates: 1, config, renderer,
    teacher: scripted([JSON.stringify({ code: source }), JSON.stringify({ code: source + "\n# fixed" })], calls),
    judge: scripted([rubric({ scores: [0.2], reason: "Wrong color" }), rubric(), rubric({ safety: "unsafe", reason: "Final image failed safety" })]),
  });
  assert.equal(summary.rejected, 1); assert.equal(summary.accepted, 0);
  assert.match(calls[1][3].content as string, /Wrong color/);
  const [record] = await records(output);
  assert.equal(record.evaluations.length, 3); assert.match(record.reason, /Final image failed safety/);
}));

test("unsafe preview terminates the candidate before submit or further teacher calls", () => fixture(async output => {
  let renders = 0;
  const summary = await generateDataset({ output, specs: [spec], candidates: 1, config,
    renderer: { async render() { renders++; return { ok: true, stdout: "", stderr: "", png }; } },
    teacher: scripted([JSON.stringify({ code: source })]), judge: scripted([rubric({ safety: "unsafe" })]),
  });
  assert.equal(summary.rejected, 1); assert.equal(renders, 1);
}));

test("duplicate source or decoded pixels cannot cross dataset splits", () => fixture(async output => {
  const second: PromptSpec = { ...spec, id: "house", family: "houses", split: "test", prompt: "Draw a house" };
  const options = { output, specs: [spec, second], candidates: 1, config, renderer,
    teacher: scripted([JSON.stringify({ code: source }), JSON.stringify({ code: source + "\n# duplicate pixels" })]),
    judge: scripted([rubric(), rubric(), rubric(), rubric()]),
  };
  const summary = await generateDataset(options);
  assert.equal(summary.accepted, 1); assert.equal(summary.rejected, 1);
  assert.match((await records(output, "rejected"))[0].reason, /Duplicate/);
  assert.equal((await records(output, "test")).length, 0);
  assert.deepEqual(await generateDataset(options), summary);
}));

test("refusals require an expected refusal and independent approval", () => fixture(async output => {
  const refusalSpec: PromptSpec = { ...spec, expectedOutcome: "refusal", requirements: ["A correct refusal"] };
  const summary = await generateDataset({ output, specs: [refusalSpec], candidates: 1, config, renderer,
    teacher: scripted([JSON.stringify({ refusal: "I cannot help with that request." })]),
    judge: scripted([rubric({ correctRefusal: true, reason: "Appropriate refusal" })]),
  });
  assert.equal(summary.accepted, 1);
  const [record] = await records(output, "accepted");
  assert.ok(record.refusal); assert.equal(record.code, undefined);
  assert.match(await readFile(join(output, record.artifacts.trajectory), "utf8"), /"operation":"refuse"/);
}));

test("infrastructure failures are checkpointed and retried only when requested", () => fixture(async output => {
  let fail = true;
  const teacher: ChatClient = { model: "scripted", baseUrl: "http://localhost/v1", async complete() {
    if (fail) throw new Error("Temporary provider outage"); return completion(JSON.stringify({ code: source }));
  } };
  const options = { output, specs: [spec], teacher, candidates: 1, renderer, config };
  assert.equal((await generateDataset(options)).infrastructureErrors, 1);
  fail = false;
  assert.equal((await generateDataset(options)).infrastructureErrors, 1);
  assert.equal((await generateDataset({ ...options, retryErrors: true })).review, 1);
  await writeFile(join(output, ".generation.lock"), "another process");
  await assert.rejects(generateDataset(options), /Dataset is locked/);
}));

test("malformed teacher replies consume the repair budget; malformed rubrics never pass", () => fixture(async output => {
  const summary = await generateDataset({ output, specs: [spec], candidates: 1, repairs: 1, config, renderer,
    teacher: scripted(["not json", JSON.stringify({ code: source, refusal: "both" })]),
  });
  assert.equal(summary.rejected, 1); assert.equal(summary.requests, 2);
  assert.throws(() => parseEvaluation(rubric({ scores: [] }), 1), /invalid rubric/);
  assert.throws(() => parseEvaluation(rubric({ scores: [2] }), 1), /invalid rubric/);
  assert.throws(() => parseEvaluation(rubric({ safety: "yes" }), 1), /invalid rubric/);
}));

test("chat adapter sends the expected protocol and keeps credentials out of provenance/errors", async t => {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ model: "revision-1", choices: [{ message: { content: '{"code":"hello"}' }, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 5 } }));
  });
  const client = chatClient({ model: "teacher", baseUrl: "http://localhost/v1/", apiKey: "test-secret" });
  const result = await client.complete([{ role: "user", content: "draw" }], 100);
  assert.equal(result.model, "revision-1"); assert.equal(result.usage?.completion_tokens, 5);
  assert.equal(calls[0].url, "http://localhost/v1/chat/completions");
  assert.equal(JSON.parse(calls[0].init.body as string).max_tokens, 100);
  assert.doesNotMatch(JSON.stringify(client), /test-secret/);
  assert.throws(() => chatClient({ model: "x", baseUrl: "http://secret:password@localhost/v1" }), /without credentials/);
  t.mock.method(globalThis, "fetch", async () => new Response("test-secret", { status: 401 }));
  await assert.rejects(client.complete([], 100), { message: "Chat endpoint returned HTTP 401" });
});
