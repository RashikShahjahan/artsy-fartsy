import assert from "node:assert/strict";
import { test } from "node:test";
import { trustedOrigin } from "./proxy.ts";
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
