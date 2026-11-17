import assert from "node:assert/strict";
import { test } from "node:test";
import { readDrawings, completedDrawing } from "./sketchbook.ts";
import type { Drawing, Run } from "./types.ts";
const parent = "a".repeat(32),
  child = "b".repeat(32);
const drawing: Drawing = {
  id: parent,
  title: "A fox",
  selectedId: parent,
  versions: [{ id: parent, prompt: "A fox" }],
};
const run: Run = {
  id: child,
  prompt: "Make it blue",
  parent_id: parent,
  state: "completed",
  artifacts: ["image.png", "program.py"],
  result: { outcome: "submitted" },
};
test("restores the old sketchbook format and rejects malformed storage", () => {
  assert.deepEqual(
    readDrawings(
      JSON.stringify([
        drawing,
        { ...drawing, id: "../secret" },
        { ...drawing, versions: [{}] },
      ]),
    ),
    [drawing],
  );
  assert.deepEqual(readDrawings("{broken"), []);
  assert.deepEqual(readDrawings("{}"), []);
});
test("a resumed edit retains its original drawing and versions", () => {
  const result = completedDrawing(run, null, [drawing])!;
  assert.equal(result.id, parent);
  assert.equal(result.selectedId, child);
  assert.deepEqual(
    result.versions.map((v) => v.id),
    [parent, child],
  );
  assert.equal(drawing.versions.length, 1);
  assert.equal(completedDrawing(run, result, [drawing])!.versions.length, 2);
});
test("refused and failed drawings do not enter the sketchbook", () => {
  assert.equal(
    completedDrawing({ ...run, result: { outcome: "refused" } }, drawing, []),
    null,
  );
  assert.equal(
    completedDrawing({ ...run, state: "failed" }, drawing, []),
    null,
  );
  assert.equal(completedDrawing({ ...run, artifacts: [] }, drawing, []), null);
});
test("turn-limited responses retain their image and edit availability", () => {
  const paused = {
    ...run,
    result: { outcome: "incomplete", turnLimitReached: true },
    artifacts: [],
  };
  assert.equal(
    completedDrawing(paused, drawing, [])!.versions.at(-1)!.editable,
    false,
  );
  assert.equal(
    completedDrawing(paused, drawing, [])!.versions.at(-1)!.hasImage,
    false,
  );
  assert.equal(
    completedDrawing(
      { ...paused, artifacts: ["program.py"] },
      drawing,
      [],
    )!.versions.at(-1)!.editable,
    true,
  );
});

test('server history restores version chains and excludes active or failed jobs', async () => {
  const { historyDrawings } = await import('./sketchbook.ts');
  const original = { ...run, id: parent, prompt: 'A fox', parent_id: undefined };
  const drawings = historyDrawings([{ ...run, id: 'c'.repeat(32), state: 'failed' }, run, original]);
  assert.equal(drawings.length, 1);
  assert.equal(drawings[0].id, parent);
  assert.equal(drawings[0].selectedId, child);
  assert.deepEqual(drawings[0].versions.map(v => v.id), [parent, child]);
});
