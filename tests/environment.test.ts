import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PNG } from "pngjs";
import { DrawingEnvironment, ENVIRONMENT_CONFIG } from "../src/environment.js";
import { LocalRenderer, type Renderer } from "../src/renderer.js";

const png = PNG.sync.write(new PNG({ width: 32, height: 32 }));
const config = { ...ENVIRONMENT_CONFIG, width: 32, height: 32 };
const good = 'from artcanvas import ArtCanvas\nwith ArtCanvas(32,32) as c:\n c.draw_circle(16,16,8,True)';
async function fixture(fn: (root: string) => Promise<void>) {
  const root = await mkdtemp(join(tmpdir(), "artsy-environment-"));
  try { await fn(root); } finally { await rm(root, { recursive: true, force: true }); }
}

test("core repairs, executes final source afresh, preserves records and resets all state", () => fixture(async root => {
  const calls: { source: string; seed: number }[] = [];
  const renderer: Renderer = { async render(source, seed) {
    calls.push({ source, seed });
    return source === "broken" ? { ok: false, stdout: "", stderr: "SyntaxError" } : { ok: true, stdout: "", stderr: "", png };
  } };
  const env = new DrawingEnvironment(root, renderer, config);
  const first = await env.reset("circle", 42);
  await env.write_code("broken");
  assert.equal((await env.run_code()).ok, false);
  await env.write_code(good);
  assert.equal((await env.run_code()).ok, true);
  const final = await env.submit();
  assert.equal(final.outcome, "submitted");
  assert.deepEqual(calls, [{ source: "broken", seed: 42 }, { source: good, seed: 42 }, { source: good, seed: 42 }]);
  assert.equal(await readFile(join(first.directory, "program.py"), "utf8"), good);
  const trajectory = (await readFile(join(first.directory, "trajectory.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line));
  assert.deepEqual(trajectory.map(t => t.operation), ["reset", "write_code", "run_code", "write_code", "run_code", "submit"]);
  assert.equal((await env.write_code("late edit")).ok, false);
  const second = await env.reset("new prompt", 7);
  assert.notEqual(first.directory, second.directory);
  assert.match(await readFile(join(first.directory, "trajectory.jsonl"), "utf8"), /SyntaxError/);
  assert.match((await env.submit()).error!, /Write code first/);
  assert.equal((await env.refuse("Unsupported request")).outcome, "refused");
  assert.equal((await env.run_code()).outcome, "refused");
}));

test("core enforces dimensions, preview/call/token budgets and episode deadline", () => fixture(async root => {
  const renderer: Renderer = { render: async () => ({ ok: true, stdout: "", stderr: "", png }) };
  const wrongSize = new DrawingEnvironment(root, renderer);
  await wrongSize.reset("circle"); await wrongSize.write_code(good);
  assert.match((await wrongSize.submit()).error!, /Expected 1920/);
  const env = new DrawingEnvironment(root, renderer, { ...config, maxAttempts: 1, maxCalls: 5 });
  await env.reset("circle"); await env.write_code(good); await env.run_code();
  assert.match((await env.run_code()).error!, /Preview budget/);
  assert.equal((await env.submit()).outcome, "submitted");
  await env.reset("circle");
  for (let i = 0; i < 5; i++) await env.run_code();
  assert.equal((await env.submit()).outcome, "budget_exhausted");
  await env.reset("circle"); await env.chargeTokens(config.maxPolicyTokens);
  assert.equal((await env.submit()).outcome, "budget_exhausted");
  const expired = new DrawingEnvironment(root, renderer, { ...config, episodeMs: 0 });
  await expired.reset("circle"); assert.equal((await expired.submit()).outcome, "timed_out");
}));

test("infrastructure errors are recorded and can recover on the next attempt", () => fixture(async root => {
  let calls = 0;
  const renderer: Renderer = { async render() { if (!calls++) throw new Error("worker crashed"); return { ok: true, stdout: "", stderr: "", png }; } };
  const env = new DrawingEnvironment(root, renderer, config);
  await env.reset("circle"); await env.write_code(good);
  assert.equal((await env.run_code()).failureKind, "infrastructure");
  assert.equal((await env.submit()).outcome, "submitted");
}));

test("local runtime reproduces seeded transforms and recovers after timeout/invalid output", async () => {
  const renderer = new LocalRenderer(undefined, 1);
  const source = await readFile(new URL("../examples/transforms.py", import.meta.url), "utf8");
  const first = await renderer.render(source, 42), second = await renderer.render(source, 42);
  assert.equal(first.ok, true, JSON.stringify(first)); assert.deepEqual(first.png, second.png);
  assert.notDeepEqual(first.png, (await renderer.render(source, 43)).png);
  assert.equal((await renderer.render("while True: pass", 0)).ok, false);
  assert.equal((await renderer.render("open('output.png','wb').write(b'x' * (8*1024*1024 + 1))", 0)).ok, false);
  assert.equal((await renderer.render("from artcanvas import ArtCanvas\nArtCanvas(5000,5000)", 0)).ok, false);
  assert.equal((await renderer.render(good, 0)).ok, true);
  const missing = await new LocalRenderer("/nonexistent-artsy-python").render(good, 0);
  assert.equal(missing.failureKind, "infrastructure");
});

test("local render cleans temporary state, bounds logs, and handles repeated saves", async () => {
  const renderer = new LocalRenderer();
  const result = await renderer.render("import os\nprint(os.getcwd())\nprint('x'*20000)\nfrom artcanvas import ArtCanvas\nwith ArtCanvas(32,32) as c:\n c.save()\n c.save()", 0);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.ok(Buffer.byteLength(result.stdout) <= 8192);
  const { access } = await import("node:fs/promises");
  await assert.rejects(access(result.stdout.split("\n")[0]));
});
