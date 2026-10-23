import { createHash } from "node:crypto";

export const PROMPT_VERSION = "artsy-curriculum-v1";
export type Split = "train" | "validation" | "test";
export type PromptSpec = {
  id: string; family: string; split: Split; prompt: string; requirements: string[];
  expectedOutcome: "drawing" | "refusal"; seed: number;
};
export const sha256 = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");

// Assign whole task families before sampling parameters or wording. Keep this
// mapping fixed within a curriculum version so increasing count cannot move data.
const families = [
  ["circles", "train"], ["rectangles", "train"], ["triangles", "train"], ["lines", "train"],
  ["rows", "train"], ["columns", "train"], ["landscapes", "train"], ["flowers", "train"],
  ["grids", "validation"], ["houses", "test"], ["rings", "validation"], ["orbits", "test"],
] as const;
const colors = ["red", "blue", "green", "orange", "purple", "teal"];
const backgrounds = ["white", "light gray", "cream"];

export function familySplit(family: string): Split {
  const builtin = families.find(([name]) => name === family);
  if (builtin) return builtin[1];
  const bucket = parseInt(sha256(family).slice(0, 8), 16) % 10;
  return bucket < 8 ? "train" : bucket === 8 ? "validation" : "test";
}

export function validateSpecs(value: unknown): PromptSpec[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error("Expected at least one prompt specification");
  const ids = new Set<string>(), prompts = new Set<string>();
  return value.map((row: any) => {
    if (!row || typeof row.family !== "string" || !row.family.trim() || typeof row.prompt !== "string" || !row.prompt.trim() ||
        !Array.isArray(row.requirements) || !row.requirements.length || !row.requirements.every((r: unknown) => typeof r === "string" && r.trim()) ||
        !["drawing", "refusal"].includes(row.expectedOutcome) || !Number.isSafeInteger(row.seed) || row.seed < 0) {
      throw new Error("Each prompt needs family, prompt, nonempty requirements, expectedOutcome (drawing/refusal), and a nonnegative integer seed");
    }
    const split = familySplit(row.family);
    if (row.split !== undefined && row.split !== split) throw new Error(`Split for family ${row.family} must be ${split}`);
    const id = row.id ?? sha256(JSON.stringify([row.family, row.prompt, row.seed])).slice(0, 24);
    if (typeof id !== "string" || !/^[a-zA-Z0-9_-]{1,80}$/.test(id) || ids.has(id)) throw new Error("Prompt IDs must be unique safe filenames (1–80 characters)");
    const normalized = row.prompt.trim().replace(/\s+/g, " ").toLowerCase();
    if (prompts.has(normalized)) throw new Error("Duplicate prompt in generation plan");
    ids.add(id); prompts.add(normalized);
    return { id, family: row.family, split, prompt: row.prompt, requirements: [...row.requirements], expectedOutcome: row.expectedOutcome, seed: row.seed };
  });
}

/** Stateless parameter sampling makes prefixes stable across count changes. */
export function generatePrompts(count = 20, seed = 42): PromptSpec[] {
  if (!Number.isSafeInteger(count) || count < 1 || count > 10_000 || !Number.isSafeInteger(seed) || seed < 0) throw new Error("count must be an integer from 1 to 10000 and seed a nonnegative safe integer");
  const prompts = Array.from({ length: count }, (_, i) => {
    const [family, split] = families[i % families.length];
    const entropy = sha256(`${PROMPT_VERSION}:${seed}:${i}`);
    const sample = (offset: number, size: number) => parseInt(entropy.slice(offset, offset + 8), 16) % size;
    const color = colors[sample(0, colors.length)], accent = colors[(colors.indexOf(color) + 1 + sample(8, 5)) % colors.length];
    const n = 2 + sample(16, 5), background = backgrounds[sample(24, backgrounds.length)];
    // An explicit, visible variation also prevents repeated templates at larger counts.
    const margin = 60 + Math.floor(i / families.length) / 10;
    const tasks: Record<string, string[]> = {
      circles: [`Draw one filled ${color} circle centered on the canvas`, `The circle has a radius of ${100 + sample(32, 200)} pixels`],
      rectangles: [`Draw one filled ${color} rectangle centered on the canvas`, `Its width is ${300 + sample(32, 400)} pixels and its height is 200 pixels`],
      triangles: [`Draw one filled ${color} equilateral triangle centered on the canvas`, "The triangle points upward and has a side length of 400 pixels"],
      lines: [`Draw exactly ${n} parallel ${color} horizontal lines`, `Use 12-pixel-wide strokes, equal spacing, and endpoints ${200 + sample(32, 200)} pixels from both sides`],
      rows: [`Draw exactly ${n} filled ${color} circles in one centered horizontal row`, "Use equal radii and spacing, with no overlaps"],
      columns: [`Draw exactly ${n} filled ${color} squares in one centered vertical column`, "Use equal sizes and spacing, with no overlaps"],
      landscapes: [`Draw a flat landscape with exactly ${n} ${color} triangular mountains`, `Place a single ${accent} sun above the mountains and a green ground strip below them`],
      flowers: [`Draw one flower with exactly ${n + 3} ${color} petals around a ${accent} circular center`, "Add a green stem below the flower and exactly two green leaves"],
      grids: [`Draw a grid of ${n} columns and 3 rows of filled ${color} squares`, "Use equal square sizes and equal gaps, with the grid centered"],
      houses: [`Draw one house with a ${color} rectangular body and a ${accent} triangular roof`, `Add exactly ${n} blue windows in a row and one brown door below the windows`],
      rings: [`Draw exactly ${n} concentric ${color} circle outlines centered on the canvas`, "Use 10-pixel strokes, equal radial gaps, and no filled circles"],
      orbits: [`Draw one ${accent} filled circle at the center with exactly ${n} small ${color} filled circles around it`, "Space the small circles equally around a circular orbit and draw the orbit as a thin black outline"],
    };
    const requirements = [...tasks[family], `Use a ${background} background`, `Leave at least ${margin} pixels clear along every canvas edge`, "Do not include text, labels, borders, or extra objects"];
    return { id: sha256(`${PROMPT_VERSION}:${seed}:${i}`).slice(0, 24), family, split,
      prompt: `${["Please create this drawing:", "Make an illustration with these requirements:", "Draw the following composition:"][sample(40, 3)]} ${requirements.join(". ")}.`,
      requirements, expectedOutcome: "drawing" as const, seed: parseInt(entropy.slice(0, 8), 16) };
  });
  return validateSpecs(prompts);
}
