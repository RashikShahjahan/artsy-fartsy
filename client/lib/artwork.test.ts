import assert from "node:assert/strict";
import { test } from "node:test";
import { artworkFailure } from "./artwork.ts";

test("missing artwork requires a fresh start rather than connection retries", () => {
  assert.equal(artworkFailure(404), "missing");
  assert.equal(artworkFailure(410), "missing");
  assert.equal(artworkFailure(502), "connection");
  assert.equal(artworkFailure(503), "connection");
  assert.equal(artworkFailure(403), "invalid");
});
