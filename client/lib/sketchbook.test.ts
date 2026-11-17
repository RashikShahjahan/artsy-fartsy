import assert from "node:assert/strict";
import { test } from "node:test";
import { completedDrawing, historyDrawings, promptParent, readDrawings, savedDrawing } from "./sketchbook.ts";
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
test("refused or cancelled runs do not create sketchbook entries", () => {
  assert.equal(
    savedDrawing({ ...run, result: { outcome: "refused" } }, drawing, []),
    null,
  );
  assert.equal(
    savedDrawing({ ...run, state: "cancelled" }, drawing, []),
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

test("server history restores version chains and keeps failures as separate entries", () => {
  const original = { ...run, id: parent, prompt: "A fox", parent_id: undefined };
  const failure = { ...run, id: "c".repeat(32), state: "failed" };
  const drawings = historyDrawings([failure, run, original]);
  assert.equal(drawings.length, 2);
  assert.equal(drawings[0].id, failure.id);
  assert.equal(drawings[0].versions[0].failed, true);
  assert.equal(drawings[0].versions[0].prompt, failure.prompt);
  assert.equal(promptParent(drawings[0].versions[0]), parent);
  assert.equal(drawings[1].id, parent);
  assert.equal(drawings[1].selectedId, child);
  assert.deepEqual(drawings[1].versions.map((v) => v.id), [parent, child]);
});

test("failed retries retain the original prompt and parent, and return to the parent drawing", () => {
  const failure = savedDrawing(
    { ...run, state: "failed", result: { reason: "Service unavailable" } },
    drawing,
    [drawing],
  )!;
  assert.equal(failure.id, child);
  assert.equal(failure.versions[0].prompt, run.prompt);
  assert.equal(failure.versions[0].error, "Service unavailable");
  const [reopened] = readDrawings(JSON.stringify([failure]));
  const parentId = promptParent(reopened.versions[0]);
  assert.equal(parentId, parent);
  const retried = savedDrawing(
    { ...run, id: "d".repeat(32), parent_id: parentId },
    reopened,
    [reopened, drawing],
  )!;
  assert.equal(retried.id, parent);
  assert.deepEqual(retried.versions.map((v) => v.id), [parent, "d".repeat(32)]);
  assert.equal(retried.selectedId, "d".repeat(32));
  assert.equal(drawing.selectedId, parent);
});

test("failed root retries create fresh drawings without a failed parent", () => {
  const failure = savedDrawing({ ...run, parent_id: undefined, state: "failed" }, null, [])!;
  assert.equal(promptParent(failure.versions[0]), undefined);
  const retried = completedDrawing({ ...run, id: "d".repeat(32), parent_id: undefined }, failure, [failure])!;
  assert.equal(retried.id, "d".repeat(32));
  assert.equal(retried.versions.length, 1);
});
