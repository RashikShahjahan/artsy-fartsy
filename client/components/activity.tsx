"use client";
import { useState } from "react";
import type { Run, Step } from "@/lib/types";
import { Icon } from "./icon";
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
export function Activity({ run, busy }: { run: Run | null; busy: boolean }) {
  const [index, setIndex] = useState(0);
  const steps = (run?.progress?.steps || []).filter(
    (step) => step.kind !== "model" || step.output || step.previews?.length,
  );
  const selected = busy
    ? steps.at(-1)
    : steps[Math.min(index, steps.length - 1)];
  const previewName = selected?.previews?.findLast(
    (src) =>
      /^data:image\/png;base64,/.test(src) ||
      /^preview-\d+-\d+\.png$/.test(src),
  );
  const preview = previewName?.startsWith("preview-")
    ? `/api/runs/${run?.id}/${previewName}`
    : previewName;
  if (!busy && !steps.length) return null;
  return (
    <section className="activity" aria-label="Drawing steps">
      {preview && (
        <img className="preview" src={preview} alt="Drawing in progress" />
      )}
      <div className="activity-label" role="status">
        <Icon name="spark" className={busy ? "breathing" : ""} />
        <span>
          {selected
            ? label(selected)
            : phases[run?.progress?.phase || "thinking"] || "Getting ready…"}
        </span>
      </div>
      {selected &&
        (selected.kind === "model" || selected.kind === "message") &&
        selected.output && <p className="agent-message">{selected.output}</p>}
      {!busy && steps.length > 0 && (
        <div className="step-navigation">
          <button
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
