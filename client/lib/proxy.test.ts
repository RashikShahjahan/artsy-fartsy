import assert from "node:assert/strict";
import { test } from "node:test";
import { allowedRoute, trustedOrigin } from "./proxy.ts";
const id = "a".repeat(32);
test("permits consumer generation, speech, polling, images and cancellation", () => {
  for (const path of ["artsy/runs", "artsy/speech", `runs/${id}/cancel`])
    assert.equal(allowedRoute("POST", path), true);
  for (const path of [
    `runs/${id}`,
    `runs/${id}/image.png`,
    `runs/${id}/preview-1-2.png`,
  ])
    assert.equal(allowedRoute("GET", path), true);
});
test("does not proxy internal launchers, review APIs or arbitrary paths", () => {
  for (const path of [
    "launches",
    "runs",
    "models",
    "preferences",
    "artsy/../launches",
    `runs/${id}/program.py`,
    `runs/${id}/trace.json`,
    "runs/../../secret",
  ]) {
    assert.equal(allowedRoute("GET", path), false);
    assert.equal(allowedRoute("POST", path), false);
  }
  assert.equal(allowedRoute("DELETE", `runs/${id}`), false);
  assert.equal(allowedRoute("GET", "artsy/runs"), false);
});

test("validates browser origins against the public Host header", () => {
  assert.equal(trustedOrigin("http://127.0.0.1:3000", "127.0.0.1:3000"), true);
  assert.equal(trustedOrigin("http://localhost:3000", "localhost:3000"), true);
  assert.equal(
    trustedOrigin("https://unrelated.example", "127.0.0.1:3000"),
    false,
  );
  assert.equal(trustedOrigin("http://127.0.0.1:4000", "127.0.0.1:3000"), false);
  assert.equal(trustedOrigin("null", "127.0.0.1:3000"), false);
});
