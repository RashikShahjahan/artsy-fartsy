import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { DrawingEnvironment } from "./environment.js";

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  prompt: { type: "string" }, episode: { type: "string" }, seed: { type: "string", default: "0" }, help: { type: "boolean", short: "h" },
} });
if (values.help || positionals.length !== 1 || !values.prompt) {
  console.log('npm run render -- examples/circle.py --prompt "Draw a blue circle" [--episode episodes/local] [--seed 0]');
  if (!values.help) process.exitCode = 1;
} else {
  try {
    const env = new DrawingEnvironment(resolve(values.episode ?? `episodes/${randomUUID()}`));
    const episode = await env.reset(values.prompt, Number(values.seed));
    const written = await env.write_code(await readFile(resolve(positionals[0]), "utf8"));
    const result = written.ok ? await env.submit() : written;
    console.log(JSON.stringify({ directory: episode.directory, ...result }, null, 2));
    if (!result.ok) process.exitCode = 1;
  } catch (error) { console.error(String(error)); process.exitCode = 1; }
}
