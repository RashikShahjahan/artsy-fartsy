"use client";
import { useState } from "react";
import type { Run, Step } from "@/lib/types";
import { Icon } from "./icon";
import { textButton } from "./ui";
const phases: Record<string, string> = {
  thinking: "Imagining your drawing…",
  drawing: "Putting it on paper…",
  previewing: "Taking a closer look…",
  rendering: "Finishing up…",
};
function label(step: Step) {
  if (step.kind === "model" || step.kind === "message")
    return "Imagining your drawing";
  if (step.kind === "render" || step.tool === "preview_drawing")
    return "Checking the preview";
  if (step.kind === "submit" || step.tool === "submit_drawing")
    return "Finishing your drawing";
  return "Refining your drawing";
}
export function Activity({ run, busy, status }: { run: Run | null; busy: boolean; status?: string }) {
  const [index, setIndex] = useState(0);
  const steps = (run?.progress?.steps || []).filter(
    (step) => step.kind !== "model" || step.output || step.previews?.length,
  );
  const selected = busy
    ? steps.at(-1)
    : steps[Math.min(index, steps.length - 1)];
  const previewSteps = busy ? steps : selected ? [selected] : [];
  const previewName = previewSteps.flatMap((step) => step.previews || []).findLast(
    (src) =>
      /^data:image\/png;base64,/.test(src) ||
      /^preview-\d+-\d+\.png$/.test(src),
  );
  const preview = previewName?.startsWith("preview-")
    ? `/api/runs/${run?.id}/${previewName}`
    : previewName;
  if (!busy && !steps.length) return null;
  return (
    <section className="mx-auto my-5 w-full max-w-2xl text-center" aria-label="Drawing steps">
      {preview && (
        <img className="mx-auto mb-6 block max-h-[48svh] max-w-full rounded-md object-contain" src={preview} alt="Drawing in progress" />
      )}
      <div className="flex items-center justify-center gap-3 text-sm [&>svg]:text-coral" role="status">
        <Icon name="spark" className={busy ? "motion-safe:animate-pulse" : ""} />
        <span>
          {busy
            ? status || phases[run?.progress?.phase || "thinking"] ||
              (selected ? label(selected) : "Getting ready…")
            : selected ? label(selected) : "Drawing steps"}
        </span>
      </div>
      {selected &&
        (selected.kind === "model" || selected.kind === "message") &&
        selected.output && <p className="mt-4 max-h-[35svh] overflow-y-auto text-left text-sm leading-relaxed whitespace-pre-wrap wrap-anywhere">{selected.output}</p>}
      {!busy && steps.length > 0 && (
        <div className="mt-5 flex flex-wrap items-center justify-between gap-2 text-xs sm:gap-3">
          <button
            className={textButton}
            onClick={() => setIndex((i) => Math.max(0, i - 1))}
            disabled={index === 0}
          >
            Previous
          </button>
          <span>
            Step {Math.min(index + 1, steps.length)} of {steps.length} ·{" "}
            {selected?.status}
          </span>
          <button
            className={textButton}
            onClick={() => setIndex((i) => Math.min(steps.length - 1, i + 1))}
            disabled={index >= steps.length - 1}
          >
            Next
          </button>
        </div>
      )}
    </section>
  );
}
