import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PNG } from "pngjs";
import { createModels } from "@earendil-works/pi-ai";
import { fauxProvider, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { context, draw, hash, openDrawing } from "../src/harness.js";
import { LocalRenderer, type Renderer } from "../src/renderer.js";
import { liquidModels } from "../src/model.js";
import { MODEL_ID, PROVIDER, LIMITS } from "../src/config.js";

const call = (name: string, args = {}) => fauxAssistantMessage([fauxToolCall(name, args)], { stopReason: "toolUse" });
const png = PNG.sync.write(new PNG({ width: 16, height: 16 }));
const good = 'from artcanvas import ArtCanvas\nwith ArtCanvas(32, 32) as c:\n c.set_color(0,0,1)\n c.draw_circle(16,16,8,True)\n';
const model = { provider: "faux", modelId: "faux-1" };

test("Liquid provider selects the exact requested model", () => {
  const configured = liquidModels().getModel(PROVIDER, MODEL_ID);
  assert.equal(configured?.id, MODEL_ID); assert.equal(configured?.contextWindow, 32768);
});

test("write/run/fix/submit uses Pi tasks, fresh renders, persisted state and idempotent resume", async () => {
  const directory = await mkdtemp(join(tmpdir(), "artsy-test-"));
  const faux = fauxProvider(); const models = createModels(); models.setProvider(faux.provider);
  faux.setResponses([call("write_code", { source: "broken" }), call("run_code"), call("write_code", { source: good }), call("run_code"), call("submit")]);
  const sources: string[] = [];
  const renderer: Renderer = { async render(source) { sources.push(source); return source === "broken" ? { ok: false, stdout: "", stderr: "SyntaxError" } : { ok: true, stdout: "", stderr: "", png, width: 16, height: 16 }; } };
  try {
    const state = await draw("blue circle", directory, { models, model, renderer, seed: 42 });
    assert.equal(state.submitted, true); assert.equal(state.attempts, 2); assert.equal(state.seed, 42); assert.equal(state.calls, 5);
    assert.deepEqual(sources, ["broken", good, good]);
    assert.equal(await readFile(join(directory, "program.py"), "utf8"), good);
    assert.equal(hash(await readFile(join(directory, "image.png"))), state.imageHash);
    const reopened = await openDrawing({ directory, models, model, renderer });
    try {
      const request = await reopened.root.submit({ type: "input", content: "blue circle", requestId: hash("blue circle") }, context);
      assert.equal((await request.wait(context)).status, "done");
      assert.equal((await reopened.state())?.submitted, true);
      assert.equal(sources.length, 3);
      const transcript = await reopened.root.entries({}, 100, undefined, context);
      assert.ok(transcript.items.some(e => e.kind === "pi.tool-result"));
    } finally { await reopened.close(); }
    await assert.rejects(draw("another prompt", directory, { models, model, renderer }), /another prompt/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("text-only answers exhaust a bounded turn budget instead of claiming success", async () => {
  const directory = await mkdtemp(join(tmpdir(), "artsy-budget-"));
  const faux = fauxProvider(); const models = createModels(); models.setProvider(faux.provider);
  faux.setResponses(Array.from({ length: LIMITS.maxTurns + 2 }, () => fauxAssistantMessage("Done!")));
  try {
    await assert.rejects(draw("circle", directory, { models, model, renderer: { render: async () => { throw new Error("must not execute"); } } }), /No valid submission/);
    assert.ok(faux.state.callCount <= LIMITS.maxTurns);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("VM runner: original API renders and reports code failures", async () => {
  const renderer = new LocalRenderer();
  const result = await renderer.render(good, 0);
  assert.equal(result.ok, true, JSON.stringify(result)); assert.equal(result.width, 32);
  const bad = await renderer.render('raise RuntimeError("repair me")', 0);
  assert.equal(bad.ok, false); assert.match(bad.stderr, /repair me/);

});

test("VM runner: timeout terminates code and a symlink cannot be submitted", async () => {
  const renderer = new LocalRenderer(undefined, 1);
  const result = await renderer.render("while True: pass", 0);
  assert.equal(result.ok, false);
  const symlink = await renderer.render("import os; os.symlink('/etc/passwd', 'output.png')", 0);
  assert.equal(symlink.ok, false);
});

test("VM runner: complete repair episode through the real Pi harness", async () => {
  const directory = await mkdtemp(join(tmpdir(), "artsy-real-"));
  const faux = fauxProvider(); const models = createModels(); models.setProvider(faux.provider);
  faux.setResponses([call("write_code", { source: "invalid python!" }), call("run_code"), call("write_code", { source: good }), call("run_code"), call("submit")]);
  try { assert.equal((await draw("blue circle", directory, { models, model })).submitted, true); }
  finally { await rm(directory, { recursive: true, force: true }); }
});
