"use client";
import { useEffect, useRef, useState } from "react";
import { isActive, type Run } from "@/lib/types";
export function useRun(
  id: string | null,
  onComplete: (run: Run) => void,
  onMissing: () => void,
) {
  const [run, setRun] = useState<Run | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const callbacks = useRef({ onComplete, onMissing });
  useEffect(() => {
    callbacks.current = { onComplete, onMissing };
  });
  useEffect(() => {
    setRun(null);
    setConnectionError("");
    if (!id) return;
    let disposed = false,
      failures = 0;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      try {
        const response = await fetch(`/api/runs/${id}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (disposed) return;
        if (response.status === 404) {
          callbacks.current.onMissing();
          return;
        }
        if (!response.ok) throw new Error("Connection unavailable");
        const next: Run = { ...(await response.json()), id };
        if (disposed) return;
        failures = 0;
        setConnectionError("");
        setRun(next);
        if (!isActive(next)) {
          callbacks.current.onComplete(next);
          return;
        }
        timer = setTimeout(poll, 1500);
      } catch {
        if (disposed) return;
        setConnectionError("Connection lost. Reconnecting…");
        timer = setTimeout(
          poll,
          Math.min(10000, 1500 * 2 ** Math.min(failures++, 4)),
        );
      }
    }
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller.abort();
    };
  }, [id]);
  return { run, connectionError };
}
