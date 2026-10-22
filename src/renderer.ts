import { spawn } from "node:child_process";
import { constants, existsSync } from "node:fs";
import { mkdtemp, open, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import { LIMITS } from "./config.js";

export type Render = { ok: boolean; stdout: string; stderr: string; error?: string; exitCode?: number; png?: Buffer; width?: number; height?: number };
export interface Renderer { render(source: string, seed: number): Promise<Render> }

/** Runs directly inside the externally isolated VM; provides no sandbox. */
export class LocalRenderer implements Renderer {
  constructor(
    readonly python = process.env.ARTSY_PYTHON ?? (existsSync(".venv/bin/python") ? resolve(".venv/bin/python") : "python3"),
    readonly seconds = LIMITS.renderSeconds,
  ) {}

  async render(source: string, seed: number): Promise<Render> {
    const directory = await mkdtemp(join(tmpdir(), "artsy-render-"));
    const runtime = fileURLToPath(new URL("../runtime/", import.meta.url));
    try {
      await writeFile(join(directory, "program.py"), source);
      const result = await new Promise<Render>((resolveResult, reject) => {
        const child = spawn(this.python, ["-c", `import random, runpy; random.seed(${JSON.stringify(seed)}); runpy.run_path('program.py', run_name='__main__')`], {
          cwd: directory, detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"],
          env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1",
            PYTHONPATH: [runtime, process.env.PYTHONPATH].filter(Boolean).join(delimiter),
            ...(process.platform === "darwin" ? { DYLD_FALLBACK_LIBRARY_PATH: process.env.DYLD_FALLBACK_LIBRARY_PATH ?? "/opt/homebrew/lib:/usr/local/lib:/usr/lib" } : {}),
          },
        });
        let stdout = "", stderr = "", timedOut = false;
        const stop = () => {
          if (!child.pid) return;
          try { process.platform === "win32" ? child.kill("SIGKILL") : process.kill(-child.pid, "SIGKILL"); } catch {}
        };
        const timer = setTimeout(() => { timedOut = true; stop(); }, this.seconds * 1000);
        child.stdout.on("data", (data: Buffer) => { stdout = (stdout + data.toString()).slice(0, 8192); });
        child.stderr.on("data", (data: Buffer) => { stderr = (stderr + data.toString()).slice(0, 8192); });
        child.once("error", error => { clearTimeout(timer); reject(error); });
        child.once("exit", () => stop());
        child.once("close", code => {
          clearTimeout(timer);
          resolveResult({ ok: code === 0 && !timedOut, stdout, stderr,
            ...(code !== null ? { exitCode: code } : {}), ...(timedOut ? { error: "Rendering timed out" } : {}) });
        });
      });
      if (!result.ok) return result;
      try {
        const file = await open(join(directory, "output.png"), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
        let png: Buffer;
        try {
          const stat = await file.stat();
          if (!stat.isFile() || stat.size < 24 || stat.size > 8 * 1024 * 1024) throw new Error("Expected a regular PNG of at most 8 MiB");
          png = await file.readFile();
        } finally { await file.close(); }
        if (!png.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"))) throw new Error("Invalid PNG header");
        const width = png.readUInt32BE(16), height = png.readUInt32BE(20);
        if (width < 1 || height < 1 || width > 4096 || height > 4096) throw new Error("PNG dimensions exceed 4096×4096");
        PNG.sync.read(png, { checkCRC: true });
        return { ...result, png, width, height };
      } catch (error) { return { ...result, ok: false, error: String(error) }; }
    } finally { await rm(directory, { recursive: true, force: true }); }
  }
}
