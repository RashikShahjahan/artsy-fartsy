import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { draw } from "./harness.js";

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  episode: { type: "string" }, seed: { type: "string", default: "0" }, help: { type: "boolean", short: "h" },
} });
if (values.help || !positionals.length) {
  console.log('npm run draw -- "Draw a blue circle" [--episode episodes/my-run] [--seed 0]');
} else {
  const seed = Number(values.seed);
  if (!Number.isSafeInteger(seed) || seed < 0) throw new Error("seed must be a nonnegative integer");
  const directory = resolve(values.episode ?? `episodes/${randomUUID()}`);
  try { const result = await draw(positionals.join(" "), directory, { seed }); console.log(JSON.stringify(result, null, 2)); }
  catch (error) { console.error(String(error)); process.exitCode = 1; }
}
