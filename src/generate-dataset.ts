import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { chatClient } from "./dataset-client.js";
import { generatePrompts, validateSpecs } from "./dataset-prompts.js";
import { generateDataset } from "./dataset.js";

const help = `npm run dataset -- [options]
  --output DIR          Dataset directory (default: datasets/pilot)
  --count N             Structured prompts to generate (default: 20)
  --seed N              Curriculum seed (default: 42)
  --input FILE          Custom JSONL specs instead of the built-in curriculum
  --candidates N        Programs per prompt (default: 2)
  --repairs N           Repairs per candidate, 0–2 (default: 2)
  --max-tokens N        Teacher tokens per request (default: 4096)
  --threshold N         Minimum score for EVERY requirement (default: 0.85)
  --teacher-model ID    Required, or set ARTSY_TEACHER_MODEL
  --teacher-url URL     Chat base URL (default: http://127.0.0.1:8080/v1)
  --judge-model ID      Optional vision evaluator, or ARTSY_JUDGE_MODEL
  --judge-url URL       Evaluator chat base URL (defaults to teacher URL)
  --timeout-ms N        Per-request deadline (default: 60000)
  --dry-run             Print prompt JSONL without network calls or rendering
  --retry-errors        Retry checkpointed infrastructure failures on resume
  --help, -h            Show this help

Credentials: ARTSY_TEACHER_API_KEY and ARTSY_JUDGE_API_KEY environment variables.
No judge: executable examples go to review.jsonl; accepted/train/validation/test
remain empty. Repeating the same command resumes completed candidate records.
Generated Python runs locally with your current permissions, as with npm run draw.`;

try {
  const { values } = parseArgs({ options: {
    output: { type: "string", default: "datasets/pilot" }, count: { type: "string", default: "20" }, seed: { type: "string", default: "42" }, input: { type: "string" },
    candidates: { type: "string", default: "2" }, repairs: { type: "string", default: "2" }, "max-tokens": { type: "string", default: "4096" }, threshold: { type: "string", default: "0.85" },
    "teacher-model": { type: "string" }, "teacher-url": { type: "string" }, "judge-model": { type: "string" }, "judge-url": { type: "string" },
    "timeout-ms": { type: "string", default: "60000" }, "dry-run": { type: "boolean" }, "retry-errors": { type: "boolean" }, help: { type: "boolean", short: "h" },
  } });
  if (values.help) console.log(help);
  else {
    const specs = values.input ? validateSpecs((await readFile(values.input, "utf8")).split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
      try { return JSON.parse(line); } catch { throw new Error(`Invalid JSON on input line ${i + 1}`); }
    })) : generatePrompts(Number(values.count), Number(values.seed));
    if (values["dry-run"]) console.log(specs.map(spec => JSON.stringify(spec)).join("\n"));
    else {
      const model = values["teacher-model"] ?? process.env.ARTSY_TEACHER_MODEL;
      if (!model) throw new Error("Set --teacher-model or ARTSY_TEACHER_MODEL to your stronger teacher model");
      const baseUrl = values["teacher-url"] ?? process.env.ARTSY_TEACHER_URL ?? "http://127.0.0.1:8080/v1";
      const judgeModel = values["judge-model"] ?? process.env.ARTSY_JUDGE_MODEL;
      const timeoutMs = Number(values["timeout-ms"]);
      if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 300_000) throw new Error("timeout-ms must be an integer from 1 to 300000");
      const summary = await generateDataset({ output: values.output!, specs,
        candidates: Number(values.candidates), repairs: Number(values.repairs), maxTokens: Number(values["max-tokens"]), threshold: Number(values.threshold),
        retryErrors: values["retry-errors"],
        teacher: chatClient({ model, baseUrl, apiKey: process.env.ARTSY_TEACHER_API_KEY, timeoutMs }),
        judge: judgeModel ? chatClient({ model: judgeModel, baseUrl: values["judge-url"] ?? process.env.ARTSY_JUDGE_URL ?? baseUrl, apiKey: process.env.ARTSY_JUDGE_API_KEY, timeoutMs }) : undefined,
        onRecord: record => console.error(`${record.id} ${record.status}: ${record.reason}`),
      });
      console.log(JSON.stringify(summary, null, 2));
      if (summary.infrastructureErrors) process.exitCode = 1;
    }
  }
} catch (error) { console.error(String(error)); process.exitCode = 1; }
