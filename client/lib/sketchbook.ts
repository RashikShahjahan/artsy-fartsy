import { RUN_ID, type Drawing, type Run } from "./types.ts";
export const STORAGE = "artsy.sketchbook.v1";
export const ACTIVE = "artsy.active.v1";
export function readDrawings(raw: string | null): Drawing[] {
  try {
    const data = JSON.parse(raw || "[]");
    return Array.isArray(data)
      ? data
          .filter(
            (d): d is Drawing =>
              d &&
              RUN_ID.test(d.id) &&
              typeof d.title === "string" &&
              Array.isArray(d.versions) &&
              d.versions.length > 0 &&
              d.versions.every(
                (v: Drawing["versions"][number]) =>
                  v && RUN_ID.test(v.id) && typeof v.prompt === "string",
              ),
          )
          .slice(0, 50)
      : [];
  } catch {
    return [];
  }
}
export function completedDrawing(
  run: Run,
  current: Drawing | null,
  drawings: Drawing[],
): Drawing | null {
  const paused =
    run.result?.outcome === "incomplete" &&
    run.result.turnLimitReached === true;
  if (
    run.state !== "completed" ||
    !(
      paused ||
      (run.result?.outcome === "submitted" &&
        run.artifacts?.includes("image.png"))
    )
  )
    return null;
  const base = current ||
    drawings.find((d) => d.versions.some((v) => v.id === run.parent_id)) || {
      id: run.id,
      title: run.prompt,
      versions: [],
    };
  return {
    ...base,
    selectedId: run.id,
    versions: [
      ...base.versions.filter((v) => v.id !== run.id),
      {
        id: run.id,
        prompt: run.prompt,
        hasImage: !!run.artifacts?.includes("image.png"),
        editable: !paused || !!run.artifacts?.includes("program.py"),
      },
    ],
  };
}

export function historyDrawings(runs: Run[]): Drawing[] {
  let drawings: Drawing[] = [];
  // History arrives newest first. Build parent/version relationships oldest first.
  for (const run of [...runs].reverse()) {
    const drawing = completedDrawing(run, null, drawings);
    if (drawing) drawings = [drawing, ...drawings.filter(d => d.id !== drawing.id)];
  }
  return drawings.slice(0, 50);
}
