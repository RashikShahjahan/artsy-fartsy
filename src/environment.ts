import { createHash } from "node:crypto";
import { appendFile, mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { LIMITS } from "./config.js";
import { LocalRenderer, RUNTIME_VERSION, validatePng, type Render, type Renderer } from "./renderer.js";

export const ENVIRONMENT_CONFIG = Object.freeze({ version: 1, runtimeVersion: RUNTIME_VERSION, width: 1920, height: 1200,
  maxCalls: LIMITS.maxCalls, maxAttempts: LIMITS.maxAttempts, renderSeconds: LIMITS.renderSeconds,
  episodeMs: LIMITS.episodeMs, sourceBytes: LIMITS.sourceBytes, maxPolicyTokens: 32768, pngBytes: 8 * 1024 * 1024, feedbackBytes: 8192 });
export type EnvironmentConfig = { readonly [K in keyof typeof ENVIRONMENT_CONFIG]: typeof ENVIRONMENT_CONFIG[K] extends number ? number : typeof ENVIRONMENT_CONFIG[K] };
export type Outcome = "active" | "submitted" | "refused" | "budget_exhausted" | "timed_out";
export type Feedback = { ok: boolean; outcome: Outcome; error?: string; artifact?: string; width?: number; height?: number; stdout?: string; stderr?: string; exitCode?: number; elapsedMs?: number; failureKind?: Render["failureKind"]; reason?: string };
const digest = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");

/** Model-independent, sequential episode operations for generation, rollouts and evaluation. */
export class DrawingEnvironment {
  private source = "";
  private seed = 0;
  private prompt = "";
  private started = 0;
  private calls = 0;
  private attempts = 0;
  private tokens = 0;
  private busy = false;
  private outcome: Outcome = "active";
  private directory = "";
  private sequence = 0;
  readonly renderer: Renderer;
  constructor(readonly artifactRoot: string, renderer?: Renderer, readonly config: EnvironmentConfig = ENVIRONMENT_CONFIG) {
    if (![config.width, config.height].every(n => Number.isInteger(n) && n >= 1 && n <= 4096)) throw new Error("Invalid configured canvas dimensions");
    this.renderer = renderer ?? new LocalRenderer(undefined, config.renderSeconds);
  }

  private async exclusive<T>(fn: () => Promise<T>): Promise<T> {
    if (this.busy) throw new Error("Episode operations must be sequential");
    this.busy = true;
    try { return await fn(); } finally { this.busy = false; }
  }
  async reset(prompt: string, seed = 0) {
    return this.exclusive(async () => {
      if (!prompt.trim() || !Number.isSafeInteger(seed) || seed < 0) throw new Error("Expected a prompt and nonnegative integer seed");
      await mkdir(resolve(this.artifactRoot), { recursive: true });
      this.directory = await mkdtemp(join(resolve(this.artifactRoot), "episode-"));
      this.source = ""; this.seed = seed; this.prompt = prompt; this.calls = 0; this.attempts = 0; this.tokens = 0; this.sequence = 0;
      this.started = Date.now(); this.outcome = "active";
      await writeFile(join(this.directory, "episode.json"), JSON.stringify({ prompt, seed, config: this.config }, null, 2));
      await this.record("reset", { prompt, seed }, { ok: true, outcome: this.outcome });
      return { prompt, seed, config: this.config, directory: this.directory };
    });
  }
  /** The training adapter supplies actual generated token counts, including tool arguments. */
  async chargeTokens(count: number) {
    return this.exclusive(async () => {
      if (!Number.isSafeInteger(count) || count < 0) throw new Error("Invalid token count");
      if (!this.directory) throw new Error("Call reset first");
      if (this.outcome !== "active") return { ok: false, outcome: this.outcome, error: "Episode is terminal" };
      this.tokens += count;
      if (this.tokens >= this.config.maxPolicyTokens) this.outcome = "budget_exhausted";
      const feedback = { ok: this.outcome === "active", outcome: this.outcome };
      await this.record("tokens", { count }, feedback); return feedback;
    });
  }
  private async record(operation: string, input: unknown, feedback: Feedback) {
    if (!this.directory) throw new Error("Call reset first");
    await appendFile(join(this.directory, "trajectory.jsonl"), JSON.stringify({ sequence: this.sequence++, operation, input, feedback,
      seed: this.seed, runtimeVersion: this.config.runtimeVersion, calls: this.calls, attempts: this.attempts, tokens: this.tokens }) + "\n");
    await writeFile(join(this.directory, "state.json"), JSON.stringify({ outcome: this.outcome, prompt: this.prompt, seed: this.seed,
      source: this.source, calls: this.calls, attempts: this.attempts, tokens: this.tokens, runtimeVersion: this.config.runtimeVersion }));
  }
  private async operation(name: string, input: unknown, action: () => Promise<Feedback>) {
    return this.exclusive(async () => {
      if (!this.directory) throw new Error("Call reset first");
      let result: Feedback;
      if (this.outcome !== "active") result = { ok: false, outcome: this.outcome, error: "Episode is terminal" };
      else if (Date.now() - this.started >= this.config.episodeMs) { this.outcome = "timed_out"; result = { ok: false, outcome: this.outcome, error: "Episode deadline exceeded" }; }
      else if (this.calls >= this.config.maxCalls || this.tokens >= this.config.maxPolicyTokens) { this.outcome = "budget_exhausted"; result = { ok: false, outcome: this.outcome, error: "Episode budget exhausted" }; }
      else {
        this.calls++;
        result = await action();
        if (Date.now() - this.started >= this.config.episodeMs && result.outcome === "active") this.outcome = "timed_out";
        else if (this.calls >= this.config.maxCalls && result.outcome === "active") this.outcome = "budget_exhausted";
        result.outcome = this.outcome;
      }
      await this.record(name, input, result); return result;
    });
  }
  async write_code(source: string) {
    return this.operation("write_code", { source }, async () => {
      if (!source.trim() || Buffer.byteLength(source) > this.config.sourceBytes) return { ok: false, outcome: this.outcome, error: "Expected 1–65536 bytes of source" };
      this.source = source; return { ok: true, outcome: this.outcome };
    });
  }
  private async render(final: boolean): Promise<Feedback> {
    if (!this.source) return { ok: false, outcome: this.outcome, error: "Write code first" };
    if (!final && this.attempts >= this.config.maxAttempts) return { ok: false, outcome: this.outcome, error: "Preview budget exhausted; submit final code" };
    if (!final) this.attempts++;
    let result: Render;
    try { result = await this.renderer.render(this.source, this.seed); }
    catch (error) { result = { ok: false, stdout: "", stderr: "", error: String(error), failureKind: "infrastructure" }; }
    const feedback: Feedback = { ok: result.ok, outcome: this.outcome, stdout: result.stdout.slice(0, this.config.feedbackBytes), stderr: result.stderr.slice(0, this.config.feedbackBytes),
      error: result.error, exitCode: result.exitCode, elapsedMs: result.elapsedMs, failureKind: result.failureKind };
    if (!result.ok || !result.png) { feedback.ok = false; feedback.error ??= "No valid PNG"; return feedback; }
    try {
      if (result.png.length > this.config.pngBytes) throw new Error("PNG exceeds configured byte limit");
      Object.assign(feedback, validatePng(result.png, this.config));
    }
    catch (error) { return { ...feedback, ok: false, error: String(error), failureKind: "invalid_code" }; }
    if (Date.now() - this.started >= this.config.episodeMs) { this.outcome = "timed_out"; return { ...feedback, ok: false, outcome: this.outcome, error: "Episode deadline exceeded" }; }
    const artifact = final ? "image.png" : `preview-${this.attempts}.png`;
    await writeFile(join(this.directory, artifact), result.png);
    feedback.artifact = join(this.directory, artifact);
    if (final) {
      await writeFile(join(this.directory, "program.py"), this.source);
      await writeFile(join(this.directory, "submission.json"), JSON.stringify({ sourceHash: digest(this.source), imageHash: digest(result.png), seed: this.seed, runtimeVersion: this.config.runtimeVersion }));
      this.outcome = "submitted"; feedback.outcome = this.outcome;
    }
    return feedback;
  }
  async run_code() { return this.operation("run_code", {}, () => this.render(false)); }
  async submit() { return this.operation("submit", {}, () => this.render(true)); }
  async refuse(reason: string) {
    return this.operation("refuse", { reason }, async () => {
      if (!reason.trim() || Buffer.byteLength(reason) > 2048) return { ok: false, outcome: this.outcome, error: "Expected a refusal reason of 1–2048 bytes" };
      this.outcome = "refused"; return { ok: true, outcome: this.outcome, reason };
    });
  }
}
