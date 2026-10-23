import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, rename } from "node:fs/promises";
import { join, resolve } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { Type, type Models } from "@earendil-works/pi-ai";
import { createRegistry, defineDoc, defineExtension, defineTool, GenerationTask, Harness, hook, section, type ToolRegistration } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { LIMITS, MODEL_ID, PROVIDER } from "./config.js";
import { liquidModels } from "./model.js";
import { LocalRenderer, RUNTIME_VERSION, type Renderer } from "./renderer.js";

export const context = BACKGROUND_CONTEXT;
export const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
type EpisodeState = { chargedTasks: string[]; requestHash: string; source: string; seed: number; attempts: number; calls: number; submitted: boolean; refusal?: string; artifact: string; sourceHash: string; imageHash: string };
export const Episode = defineDoc<EpisodeState>({
  kind: "artsy.episode", version: 1, scope: "conversation", history: "latest", fork: "initial",
  initial: () => ({ chargedTasks: [], requestHash: "", source: "", seed: 0, attempts: 0, calls: 0, submitted: false, artifact: "", sourceHash: "", imageHash: "" }),
  checkpointWhen: () => true,
});

const reply = (value: unknown, isError = false) => ({ output: [{ type: "text" as const, text: JSON.stringify(value) }], isError });
async function atomicFile(path: string, data: string | Buffer) {
  await writeFile(`${path}.tmp`, data); await rename(`${path}.tmp`, path);
}

export async function openDrawing(options: { directory: string; renderer?: Renderer; models?: Models; model?: {provider: string; modelId: string}; seed?: number }) {
  if (options.seed !== undefined && (!Number.isSafeInteger(options.seed) || options.seed < 0)) throw new Error("seed must be a nonnegative integer");
  const directory = resolve(options.directory);
  await mkdir(directory, { recursive: true });
  const renderer = options.renderer ?? new LocalRenderer();
  const apiDoc = await readFile(new URL("../docs/artcanvas.md", import.meta.url), "utf8");
  const drawing = defineExtension({
    name: "artsy.drawing",
    wraps: ["write_code", "read_code", "run_code", "submit", "refuse"].map(name => ({ tool: name, wrap: (tool: ToolRegistration) => ({
      ...tool,
      execute: async (...[args, api, ctx]: Parameters<ToolRegistration["execute"]>) => {
        await api.commit(async tx => {
          const doc = await tx.doc(Episode, api.conversationId);
          // A replay-safe tool's budget is charged once, even if its phase restarts.
          if (!doc.chargedTasks.includes(String(api.taskId))) {
            if (doc.calls >= LIMITS.maxCalls) throw new Error("Tool budget exhausted");
            doc.calls += 1;
            doc.chargedTasks.push(String(api.taskId));
          }
        }, ctx);
        return tool.execute(args, api, ctx);
      },
    }) })),
    sections: [section("drawing", () => `Draw the user's request by writing a Python program. Use write_code, run_code, fix errors with write_code, then submit. A text answer alone is not a submission. Call refuse(reason) to finish a request you cannot fulfill. You have ${LIMITS.maxAttempts} preview attempts. submit always executes the latest source again in a fresh working directory. The policy is text-only: inspect execution errors, not images. Use math/random as needed.\n${apiDoc}`)],
    tools: [
      defineTool({
        name: "write_code", description: "Replace program.py with complete Python source. Fix code by calling this again.",
        parameters: Type.Object({ source: Type.String({ minLength: 1, maxLength: LIMITS.sourceBytes }) }),
        replay: "safe", executionMode: "sequential",
        execute: async ({ source }, api, ctx) => {
          if (Buffer.byteLength(source) > LIMITS.sourceBytes) throw new Error("Source exceeds 64 KiB");
          await api.commit(async tx => { const doc = await tx.doc(Episode, api.conversationId); if (doc.submitted || doc.refusal) throw new Error("Episode already finished"); doc.source = source; }, ctx);
          return reply({ written: true, bytes: Buffer.byteLength(source) });
        },
      }),
      defineTool({
        name: "read_code", description: "Read the current program.py source.", parameters: Type.Object({}),
        replay: "safe", executionMode: "sequential",
        execute: async (_, api, ctx) => reply({ source: (await api.snapshot(Episode, api.conversationId, ctx))?.source ?? "" }),
      }),
      defineTool({
        name: "run_code", description: "Run the current code in a fresh Python/Cairo process in the VM. Return errors and PNG dimensions.", parameters: Type.Object({}),
        replay: "unsafe", executionMode: "sequential",
        execute: async (_, api, ctx) => {
          const state = await api.commit(async tx => {
            const doc = await tx.doc(Episode, api.conversationId);
            if (doc.submitted || doc.refusal || !doc.source) throw new Error("Write code before running");
            if (doc.attempts >= LIMITS.maxAttempts) throw new Error("Preview attempt budget exhausted; fix and submit the final code");
            doc.attempts += 1; return { source: doc.source, seed: doc.seed, attempt: doc.attempts };
          }, ctx);
          const result = await renderer.render(state.source, state.seed);
          if (result.png) await atomicFile(join(directory, `preview-${state.attempt}.png`), result.png);
          const { png, ...feedback } = result;
          return reply(feedback, !result.ok);
        },
      }),
      defineTool({
        name: "submit", description: "Execute the latest code in a fresh working directory and submit a validated PNG and source. Fix errors and retry if rejected.", parameters: Type.Object({}),
        replay: "unsafe", executionMode: "sequential",
        execute: async (_, api, ctx) => {
          const state = await api.snapshot(Episode, api.conversationId, ctx);
          if (state?.refusal) throw new Error("Episode already refused");
          if (!state?.source) throw new Error("Write code first");
          if (state.submitted) return { ...reply({ submitted: true, artifact: state.artifact }), control: { terminate: true } };
          const result = await renderer.render(state.source, state.seed);
          if (!result.ok || !result.png) { const { png, ...feedback } = result; return reply(feedback, true); }
          await atomicFile(join(directory, "program.py"), state.source);
          await atomicFile(join(directory, "image.png"), result.png);
          const sourceHash = hash(state.source), imageHash = hash(result.png);
          await api.commit(async tx => { const doc = await tx.doc(Episode, api.conversationId); doc.submitted = true; doc.artifact = join(directory, "image.png"); doc.sourceHash = sourceHash; doc.imageHash = imageHash; }, ctx);
          return { ...reply({ submitted: true, width: result.width, height: result.height, sourceHash, imageHash }), control: { terminate: true } };
        },
      }),
      defineTool({
        name: "refuse", description: "Finish with a structured refusal and a concise reason.",
        parameters: Type.Object({ reason: Type.String({ minLength: 1, maxLength: 2048 }) }),
        replay: "safe", executionMode: "sequential",
        execute: async ({ reason }, api, ctx) => {
          if (!reason.trim() || Buffer.byteLength(reason) > 2048) throw new Error("Expected a reason of 1–2048 bytes");
          await api.commit(async tx => {
            const doc = await tx.doc(Episode, api.conversationId);
            if (doc.submitted) throw new Error("Episode already submitted");
            doc.refusal = reason;
          }, ctx);
          return { ...reply({ refused: true, reason }), control: { terminate: true } };
        },
      }),
    ],
    hooks: [hook(GenerationTask, {
      onYield: async (_, api, ctx) => {
        const state = await api.snapshot(Episode, api.conversationId, ctx);
        if (!state?.submitted && !state?.refusal) return { continue: "Use the drawing tools. Write code, run it, fix errors, and call submit to finish." };
      },
    })],
  });
  const registry = createRegistry(); registry.install(drawing);
  const originalModels = options.models ?? liquidModels();
  const boundedStream: Models["streamSimple"] = (model, transcript, settings) => {
    const assistants = transcript.messages.filter(m => m.role === "assistant");
    const calls = assistants.flatMap(m => m.role === "assistant" ? m.content : []).filter(c => c.type === "toolCall").length;
    if (assistants.length >= LIMITS.maxTurns || calls >= LIMITS.maxCalls) throw new Error("Episode model/tool budget exhausted");
    return originalModels.streamSimple(model, transcript, settings);
  };
  const boundedModels: Models = new Proxy(originalModels, { get(target, property) {
    if (property === "streamSimple") return boundedStream;
    const value = Reflect.get(target, property);
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const harness = await Harness.open(await openNodeSqliteStorage(join(directory, "session.sqlite")), {
    models: boundedModels, registry,
    settings: { toolExecution: "sequential", retry: { enabled: false }, compaction: { enabled: false }, stream: { timeoutMs: 120_000, maxRetries: 0 } },
  }, context);
  const root = await harness.root(context, {
    agent: { model: options.model ?? { provider: PROVIDER, modelId: MODEL_ID }, extensions: [drawing] },
    init: async (tx, id) => { (await tx.doc(Episode, id)).seed = options.seed ?? 0; },
  });
  return { harness, root, async close() { await harness.close(context); }, async state() { return harness.snapshot(Episode, root.id, context); } };
}

export async function draw(prompt: string, directory: string, options: Omit<Parameters<typeof openDrawing>[0], "directory"> = {}) {
  const session = await openDrawing({ directory, ...options });
  const timer = setTimeout(() => { void session.root.abort(context); }, LIMITS.episodeMs);
  try {
    await session.root.commit(async tx => {
      const doc = await tx.doc(Episode, session.root.id);
      if (doc.requestHash && doc.requestHash !== hash(prompt)) throw new Error("This episode belongs to another prompt; choose a new --episode directory");
      doc.requestHash = hash(prompt);
    }, context);
    // A stable request ID makes reopening an interrupted episode resume its existing work.
    const request = await session.root.submit({ type: "input", content: prompt, requestId: hash(prompt) }, context);
    await request.wait(context);
    const state = await session.state();
    await atomicFile(join(directory, "result.json"), JSON.stringify({ runtimeVersion: RUNTIME_VERSION, outcome: state?.submitted ? "submitted" : state?.refusal ? "refused" : "incomplete", model: options.model ?? { provider: PROVIDER, modelId: MODEL_ID }, prompt, ...state }, null, 2));
    if (!state?.submitted && !state?.refusal) throw new Error(`No valid submission; inspect ${directory}/session.sqlite`);
    return state;
  } finally { clearTimeout(timer); await session.close(); }
}
