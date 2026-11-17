"use client";
import { useEffect, useState } from "react";
import { artworkFailure, type ArtworkFailure } from "@/lib/artwork";
import type { Version } from "@/lib/types";
import { Icon } from "./icon";
import { textButton } from "./ui";

export function Artwork({ version, hasAlternatives, onUnavailable, onMissing, onStartAgain }: {
  version: Version;
  hasAlternatives: boolean;
  onUnavailable: (id: string) => void;
  onMissing: (id: string) => void;
  onStartAgain: () => void;
}) {
  const [attempt, setAttempt] = useState(0);
  const [source, setSource] = useState("");
  const [failure, setFailure] = useState<ArtworkFailure | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    setSource("");
    setFailure(null);
    onUnavailable(version.id);
    onMissing("");
    if (version.hasImage === false) return () => controller.abort();
    void (async () => {
      try {
        const response = await fetch(`/api/runs/${version.id}/image.png`, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok) {
          const reason = artworkFailure(response.status);
          if (!controller.signal.aborted) {
            setFailure(reason);
            if (reason === "missing") onMissing(version.id);
          }
          return;
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      } catch {
        if (!controller.signal.aborted) setFailure("connection");
      }
    })();
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [version.id, version.hasImage, attempt, onUnavailable, onMissing]);

  if (source && !failure) return (
    <img className="max-h-[48svh] max-w-full rounded-md object-contain shadow-lg sm:max-h-[52svh]" src={source} alt={version.prompt}
      onLoad={() => onUnavailable("")}
      onError={() => { setFailure("invalid"); onUnavailable(version.id); }} />
  );
  const noImage = version.hasImage === false;
  const missing = failure === "missing";
  const loading = !noImage && !failure;
  return (
    <div className="flex min-h-[min(320px,40svh)] w-full flex-col items-center justify-center rounded-2xl border border-dashed border-line bg-surface/60 p-6 text-center sm:p-8" role="status" aria-live="polite" aria-busy={loading}>
      <Icon name="image" />
      <h2 className="mt-4 text-lg font-medium">{loading ? "Loading your drawing…" : noImage ? "No image for this version" : missing
        ? "This drawing is no longer available" : failure === "connection"
          ? "Can’t reach your drawing right now" : "This image couldn’t be opened"}</h2>
      {!loading && <>
        <p className="mt-2 mb-5 max-w-sm text-sm leading-relaxed text-muted">{noImage ? `${hasAlternatives ? "Choose another version below, or continue" : "Continue"} with a new drawing instruction.`
          : missing ? `The saved image is missing. ${hasAlternatives ? "Choose another version below, or start" : "Start"} a new drawing from the original idea.`
          : failure === "connection" ? "There’s a connection problem with the drawing service. Your saved versions are still here. Try loading again in a moment."
          : `Try loading it again${hasAlternatives ? ", or choose another version below" : ""}.`}</p>
        {missing ? <button className={textButton} onClick={onStartAgain}>Start from the original idea</button>
          : !noImage && <button className={textButton} onClick={() => setAttempt((value) => value + 1)}>Try loading again</button>}
      </>}
    </div>
  );
}
