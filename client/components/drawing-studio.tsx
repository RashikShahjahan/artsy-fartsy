"use client";
import { useEffect, useRef, useState } from "react";
import {
  ACTIVE,
  STORAGE,
  savedDrawing,
  promptParent,
  readDrawings,
  historyDrawings,
} from "@/lib/sketchbook";
import { RUN_ID, type Drawing, type Run } from "@/lib/types";
import { useRun } from "@/hooks/use-run";
import { useVoice } from "@/hooks/use-voice";
import { Activity } from "./activity";
import { Artwork } from "./artwork";
import { Icon } from "./icon";
import { iconButton, textButton } from "./ui";
const examples = [
  {
    image: "landscape",
    title: "A sunny landscape",
    prompt: "Draw a sunny landscape with green hills, trees, birds, and a warm cream sky.",
  },
  {
    image: "fractal",
    title: "A colorful fractal",
    prompt: "A highly detailed, colorful fractal design with infinite complexity, featuring swirling patterns that resemble spirals, paisley shapes, and branching tendrils. The fractal is rendered in vibrant gradients of blue, green, and gold, transitioning into purples and pinks at the edges. The pattern feels infinite, with intricate, repeating details visible at every zoom level. The background is dark to highlight the luminous fractal shapes, creating a mesmerizing and otherworldly effect.",
  },
  {
    image: "metropolis",
    title: "A future metropolis",
    prompt: "A glowing metropolis with flying cars, towering skyscrapers, and neon lights reflecting on wet streets.",
  },
];

export function DrawingStudio() {
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [current, setCurrent] = useState<Drawing | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [ready, setReady] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [failedImageId, setFailedImageId] = useState("");
  const [missingVersionId, setMissingVersionId] = useState("");
  const [historyAttempt, setHistoryAttempt] = useState(0);
  const [canRetry, setCanRetry] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [savedRun, setSavedRun] = useState<Run | null>(null);
  const [canceling, setCanceling] = useState(false);
  const submitting = useRef(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const historyButton = useRef<HTMLButtonElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const busy = !ready || starting || !!activeId;
  const version =
    current?.versions.find((v) => v.id === current.selectedId) ||
    current?.versions.at(-1);
  const versionIndex = current?.versions.findIndex((item) => item.id === version?.id);
  const canRevise = !!version && !version.failed;
  const voice = useVoice(
    (text) => {
      setPrompt((draft) =>
        [draft.trim(), text].filter(Boolean).join(" ").slice(0, 4000),
      );
      input.current?.focus();
    },
    (message) => {
      setError(message);
    },
  );
  function rememberRun(id: string | null) {
    setActiveId(id);
    try {
      if (id) localStorage.setItem(ACTIVE, id);
      else localStorage.removeItem(ACTIVE);
    } catch {
      /* Optional storage. */
    }
  }
  function finish(run: Run) {
    const drawing = savedDrawing(run, current, drawings);
    setSavedRun(run);
    rememberRun(null);
    setCanceling(false);
    if (drawing) {
      setCurrent(drawing);
      setDrawings((items) =>
        [drawing, ...items.filter((d) => d.id !== drawing.id)].slice(0, 50),
      );
      if (run.state === "failed") {
        setError(drawing.versions[0].error || "");
        setPrompt(run.prompt);
        return;
      }
      const message =
        run.result?.outcome === "incomplete"
          ? "Here’s the latest step. What would you like to do next?"
          : "Your drawing is ready. Anything you’d like to change?";
      setNotice(message);
      void voice.speak(message);
    } else {
      setError(
        run.result?.refusalReason ||
          run.result?.reason ||
          (["stopped", "cancelled", "canceled"].includes(run.state)
            ? "Drawing stopped."
            : "I couldn’t finish that. Try again?"),
      );
      setPrompt(run.prompt);
      setCanRetry(true);
    }
  }
  const { run, connectionError } = useRun(activeId, finish, () => {
    rememberRun(null);
    setCanceling(false);
    setError("This drawing is unavailable. Please try again.");
  });
  useEffect(() => {
    try {
      setDrawings(readDrawings(localStorage.getItem(STORAGE)));
      const resume = localStorage.getItem(ACTIVE);
      if (RUN_ID.test(resume || "")) setActiveId(resume);
    } catch {
      /* Start with an empty sketchbook. */
    }
    const controller = new AbortController();
    fetch('/api/runs', { cache: 'no-store', signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error('History unavailable');
        const { runs } = await response.json() as { runs: Run[] };
        if (controller.signal.aborted) return;
        setDrawings(historyDrawings(runs));
        const active = runs.find(r => ['queued', 'running', 'stopping'].includes(r.state));
        rememberRun(active?.id || null);
      })
      .catch(() => { /* Keep the browser sketchbook while reconnecting. */ })
      .finally(() => { if (!controller.signal.aborted) setReady(true); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (ready) {
      try {
        localStorage.setItem(STORAGE, JSON.stringify(drawings));
      } catch {
        /* Artifacts stay on the server. */
      }
    }
  }, [drawings, ready]);
  useEffect(() => {
    if (!version?.id || activeId) return;
    const controller = new AbortController();
    setSavedRun(null);
    fetch(`/api/runs/${version.id}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) throw new Error();
        const result = await response.json();
        if (!controller.signal.aborted)
          setSavedRun({ ...result, id: version.id });
      })
      .catch(() => {
        /* The saved image remains accessible without step history. */
      });
    return () => controller.abort();
  }, [version?.id, activeId]);
  function closeHistory() {
    dialog.current?.close();
    historyButton.current?.focus();
  }
  function openDrawing(drawing: Drawing) {
    if (busy) return;
    voice.stop();
    voice.cancelSpeech();
    setCurrent(drawing);
    setFailedImageId("");
    setMissingVersionId("");
    setSavedRun(null);
    setNotice("");
    const selected = drawing.versions.find((v) => v.id === drawing.selectedId) || drawing.versions.at(-1);
    setPrompt(selected?.failed ? selected.prompt : "");
    setError(selected?.error || "");
    setCanRetry(false);
    closeHistory();
  }
  function newDrawing() {
    if (busy) return;
    voice.stop();
    voice.cancelSpeech();
    setCurrent(null);
    setSavedRun(null);
    setPrompt("");
    setError("");
    setCanRetry(false);
    setNotice("");
  }
  async function submit(text: string) {
    text = text.trim();
    if (busy || submitting.current || voice.listening || !text || (version && missingVersionId === version.id)) return;
    if (text.length > 4000) {
      setError("Please keep your idea under 4,000 characters.");
      return;
    }
    submitting.current = true;
    setStarting(true);
    setError("");
    setCanRetry(false);
    setNotice("");
    voice.stop();
    voice.cancelSpeech();
    try {
      const parentId = promptParent(version);
      const response = await fetch("/api/artsy/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: text,
          ...(parentId ? { parent_id: parentId } : {}),
        }),
      });
      if (response.status >= 500)
        throw new Error("Drawing is temporarily unavailable. Your idea is still here. Try again in a moment.");
      const data = await response.json();
      if (!response.ok || !RUN_ID.test(data.id))
        throw new Error(
          data.error || "I couldn’t start this drawing. Please try again.",
        );
      rememberRun(data.id);
      setPrompt("");
      setSavedRun(null);
    } catch (error) {
      setError(error instanceof TypeError || error instanceof SyntaxError
        ? "Drawing is temporarily unavailable. Your idea is still here. Try again in a moment."
        : error instanceof Error ? error.message : "Please try again.");
      setPrompt(text);
      setCanRetry(true);
    } finally {
      submitting.current = false;
      setStarting(false);
    }
  }
  async function cancel() {
    if (!activeId || canceling) return;
    setCanceling(true);
    try {
      const response = await fetch(`/api/runs/${activeId}/cancel`, {
        method: "POST",
      });
      if (!response.ok) throw new Error();
    } catch {
      setCanceling(false);
      setError("Couldn’t stop the drawing. Please try again.");
    }
  }
  return (
    <div className="isolate flex min-h-svh flex-col">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:p-3">Skip to drawing workspace</a>
      <header className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-5 py-5 sm:px-10 sm:py-7">
        <a href="/" className="font-serif text-3xl tracking-[-0.06em] no-underline" aria-label="Artsy home">
          artsy<span className="ml-1.5 inline-block text-2xl text-coral" aria-hidden="true">✳</span>
        </a>
        <div className="flex items-center gap-1 sm:gap-2">
          {current && (
            <button
              className={iconButton}
              onClick={newDrawing}
              disabled={busy}
              aria-label="New drawing"
              title="New drawing"
            >
              <Icon name="plus" />
            </button>
          )}
          {version && version.hasImage !== false && failedImageId !== version.id && !busy && (
            <a
              className={iconButton}
              href={`/api/runs/${version.id}/image.png`}
              download="artsy-drawing.png"
              aria-label="Save image"
              title="Save image"
            >
              <Icon name="download" />
            </a>
          )}
          <button
            className={iconButton}
            ref={historyButton}
            onClick={() => {
              setHistoryAttempt((attempt) => attempt + 1);
              dialog.current?.showModal();
            }}
            aria-label="Your drawings"
            title="Your drawings"
          >
            <Icon name="history" />
          </button>
          <button
            className={iconButton}
            onClick={voice.toggleSound}
            aria-label={voice.enabled ? "Mute Artsy" : "Unmute Artsy"}
            aria-pressed={voice.enabled}
            title={voice.enabled ? "Mute Artsy" : "Unmute Artsy"}
          >
            <Icon name={voice.enabled ? "sound" : "mute"} />
          </button>
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className={`mx-auto flex w-full max-w-5xl min-w-0 flex-1 flex-col items-center justify-center px-5 pb-10 sm:px-7 ${current || busy ? "pt-5" : "pt-10 sm:pt-14"}`}>
        {!current && !busy && (
          <section className="text-center">
            <h1 className="mb-4 font-serif text-[clamp(2.75rem,7vw,4.5rem)] leading-[1.1] font-normal tracking-[-0.045em]">What should we draw?</h1>
            <p className="text-base leading-relaxed text-muted sm:text-lg">Describe a drawing, then make it your own.</p>
          </section>
        )}
        {(current || busy) && (
          <section
            className="flex w-full min-w-0 flex-col items-center"
            aria-label="Your drawing"
            aria-busy={busy}
          >
            {version && !busy && (
              <div className="grid w-full place-items-center">
                {version.failed ? (
                  <div className="flex min-h-48 w-full flex-col items-center justify-center rounded-2xl border border-dashed border-line bg-surface/60 p-6 text-center" role="status">
                    <Icon name="image" />
                    <h2 className="mt-4 text-lg font-medium">Drawing failed</h2>
                    <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted">Your prompt is saved below. You can edit it, then retry.</p>
                  </div>
                ) : (
                  <Artwork
                    key={version.id}
                    version={version}
                    hasAlternatives={current!.versions.length > 1}
                    onUnavailable={setFailedImageId}
                    onMissing={setMissingVersionId}
                    onStartAgain={() => {
                      const idea = current!.title;
                      newDrawing();
                      setPrompt(idea);
                      requestAnimationFrame(() => input.current?.focus());
                    }}
                  />
                )}
              </div>
            )}
            {busy && (
              <Activity
                key={activeId || "starting"}
                run={run}
                busy
                status={canceling || run?.state === "stopping"
                  ? "Stopping…"
                  : connectionError || (starting ? "Getting ready…" : undefined)}
              />
            )}
          </section>
        )}
        <section className={`flex w-full max-w-xl flex-col items-center hidden:hidden ${current || busy ? "mt-6" : "mt-8 sm:mt-10"}`} aria-label="Talk to Artsy" hidden={!!version && missingVersionId === version.id}>
          <p className={voice.listening ? "mb-4 text-center text-sm leading-relaxed wrap-anywhere" : "sr-only"} role="status" aria-live="polite">
            {voice.listening ? voice.transcript || "Listening…" : busy || error ? "" : notice}
          </p>
          {error && (
            <div className="mb-4 w-full rounded-xl border border-coral/25 bg-surface p-4 text-center text-sm leading-relaxed text-coral [&>p]:mb-3" role="alert">
              <p>{error}</p>
              {canRetry && !version?.failed && (
                <button
                  className={textButton}
                  disabled={busy || !prompt.trim()}
                  onClick={() => void submit(prompt)}
                >
                  Try again
                </button>
              )}
            </div>
          )}
          {busy ? (
            <button
              className={textButton}
              disabled={canceling || starting}
              onClick={() => void cancel()}
            >
              {canceling ? "Stopping…" : "Stop drawing"}
            </button>
          ) : (
            <form
              className="grid w-full grid-cols-[1fr_auto_auto] items-end gap-2 rounded-2xl border border-line bg-surface p-3 shadow-sm transition-shadow focus-within:border-coral focus-within:ring-1 focus-within:ring-coral/30 sm:p-4"
              onSubmit={(event) => {
                event.preventDefault();
                void submit(prompt);
              }}
            >
              <label className={version ? "col-span-full px-1 font-medium" : "sr-only"} htmlFor="prompt">
                {version?.failed ? "Edit your prompt" : canRevise ? "Revise this drawing" : "What would you like Artsy to draw or change?"}
              </label>
              {canRevise && versionIndex !== undefined && (
                <p id="prompt-version" className="col-span-full px-1 text-sm text-muted">Selected version {versionIndex + 1}</p>
              )}
              <textarea
                ref={input}
                id="prompt"
                maxLength={4000}
                rows={2}
                className="col-span-full max-h-52 min-h-16 w-full min-w-0 resize-y bg-transparent px-1 py-1 text-base leading-relaxed placeholder:text-muted focus-visible:outline-none"
                aria-describedby={canRevise ? "prompt-version prompt-hint" : "prompt-hint"}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={
                  canRevise
                    ? "What would you like to change?"
                    : "A sleepy cat in a sunny window…"
                }
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    void submit(prompt);
                  }
                }}
              />
              {voice.supported && (
                <button
                  type="button"
                  className={`${iconButton} col-start-2 border border-line ${voice.listening ? "bg-coral text-surface hover:bg-coral" : "hover:bg-ink/8"}`}
                  onClick={voice.start}
                  disabled={!ready}
                  aria-label={voice.listening ? "Finish dictating" : "Dictate an idea"}
                  aria-pressed={voice.listening}
                  title={voice.listening ? "Finish dictating" : "Dictate an idea"}
                >
                  {voice.listening ? <span className="size-4 rounded-xs bg-current" /> : <Icon name="mic" />}
                </button>
              )}
              <button
                type="submit"
                className={`col-start-3 inline-flex h-11 shrink-0 items-center justify-center rounded-full bg-ink text-surface transition-colors hover:bg-coral disabled:hover:bg-ink ${version?.failed ? "gap-2 px-4 text-sm font-medium" : "w-11"}`}
                disabled={!prompt.trim() || !ready || voice.listening}
                aria-label={version?.failed ? undefined : "Send to Artsy"}
              >
                {version?.failed && "Retry prompt"}
                <Icon name="arrow" />
              </button>
            </form>
          )}
          {!busy && <p id="prompt-hint" className="mt-3 text-center text-xs leading-relaxed text-muted">Enter to send · Shift + Enter for a new line</p>}
        </section>
        {!busy && current && version && !version.failed && (
          <div className="mt-5 flex w-full max-w-2xl min-w-0 flex-col items-start gap-3 text-sm text-muted">
            <p className="max-w-full leading-relaxed wrap-anywhere">{version.prompt}</p>
            {current.versions.length > 1 && (
              <nav className="mt-1 flex w-full snap-x gap-3 overflow-x-auto px-1 pt-1 pb-4" aria-label="Drawing versions">
                {current.versions.map((v, i) => (
                  <button type="button" key={v.id} className="w-40 shrink-0 snap-start rounded-xl border border-line bg-surface p-2 text-left transition-colors hover:border-coral aria-current:border-ink aria-current:ring-1 aria-current:ring-ink"
                    aria-current={v.id === version.id ? "true" : undefined}
                    aria-label={`Version ${i + 1}: ${v.prompt}`}
                    onClick={() => {
                      const selected = { ...current, selectedId: v.id };
                      setCurrent(selected);
                      setDrawings((items) => items.map((drawing) => drawing.id === selected.id ? selected : drawing));
                      setNotice("");
                      setError("");
                      setCanRetry(false);
                      setFailedImageId("");
                      setMissingVersionId("");
                    }}>
                    <HistoryThumbnail id={v.id} hasImage={v.hasImage} wide />
                    <span className="mt-3 block text-sm font-medium text-ink">Version {i + 1}{v.id === version.id ? " · Selected" : ""}</span>
                    <span className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted">{v.prompt}</span>
                  </button>
                ))}
              </nav>
            )}
          </div>
        )}
        {!busy && savedRun && (
          <details className="mt-5 w-full max-w-2xl rounded-xl border border-line px-4 py-3 text-sm text-muted">
            <summary className="py-1 font-medium text-ink">How it came together</summary>
            <Activity key={savedRun.id} run={savedRun} busy={false} />
          </details>
        )}
        {!current && !busy && (
          <section className="mt-9 w-full max-w-2xl sm:mt-12" aria-labelledby="examples-heading">
            <h2 id="examples-heading" className="mb-2 text-sm font-semibold">Start with a little inspiration</h2>
            <p className="mb-4 text-sm leading-relaxed text-muted">Made with Artsy. Pick an idea, then make it yours.</p>
            <div className="grid grid-cols-1 gap-3 min-[480px]:grid-cols-3">
              {examples.map((example) => (
                <button type="button" className="flex items-center overflow-hidden rounded-xl border border-line bg-surface text-left transition-colors hover:border-coral min-[480px]:flex-col min-[480px]:items-stretch" key={example.image}
                  onClick={() => { setPrompt(example.prompt); input.current?.focus(); }}>
                  <img src={`/examples/${example.image}.png`} alt="" className="aspect-[8/5] w-28 shrink-0 object-contain min-[480px]:w-full" width={1920} height={1200} />
                  <span className="flex flex-1 items-center justify-between gap-2 p-3 text-sm leading-relaxed">{example.title}<span className="text-coral" aria-hidden="true">↗</span></span>
                </button>
              ))}
            </div>
          </section>
        )}
      </main>
      <dialog
        className="m-auto mr-0 h-svh max-h-svh w-96 max-w-[92vw] overflow-y-auto border-0 bg-paper p-5 text-ink shadow-2xl backdrop:bg-ink/30 backdrop:backdrop-blur-sm sm:p-6"
        ref={dialog}
        aria-labelledby="history-title"
        onClick={(event) => {
          if (event.target === event.currentTarget) closeHistory();
        }}
      >
        <div className="mb-6 flex items-center justify-between gap-3">
          <h2 id="history-title" className="font-serif text-3xl font-normal tracking-tight">Your drawings</h2>
          <button
            className={iconButton}
            onClick={closeHistory}
            aria-label="Close your drawings"
          >
            <Icon name="close" />
          </button>
        </div>
        {drawings.length ? (
          <nav aria-label="Saved drawings">
            {drawings.map((drawing) => {
              const thumbnail =
                drawing.versions.find((v) => v.id === drawing.selectedId) ||
                drawing.versions.at(-1)!;
              return (
                <button
                  key={drawing.id}
                  className="mb-2 flex w-full items-center gap-3 rounded-xl p-3 text-left transition-colors hover:bg-ink/5 aria-current:bg-ink/8"
                  disabled={busy}
                  aria-current={drawing.id === current?.id ? "true" : undefined}
                  onClick={() => openDrawing(drawing)}
                >
                  <HistoryThumbnail
                    key={`${thumbnail.id}-${historyAttempt}`}
                    id={thumbnail.id}
                    hasImage={thumbnail.hasImage}
                  />
                  <span className="min-w-0">
                    <strong className="line-clamp-2 text-sm leading-relaxed font-medium wrap-anywhere">{drawing.title}</strong>
                    <small className="mt-1 block text-xs text-muted">
                      {thumbnail.failed ? "Drawing failed" : `${drawing.versions.length} ${drawing.versions.length === 1 ? "drawing" : "versions"}`}
                    </small>
                  </span>
                </button>
              );
            })}
          </nav>
        ) : (
          <p className="rounded-xl border border-dashed border-line p-6 text-sm leading-relaxed text-muted">
            Your drawings will appear here after you make one.
          </p>
        )}
        <p className="mt-7 border-t border-line pt-4 text-xs leading-relaxed text-muted">
          History is saved in this browser.
        </p>
      </dialog>
    </div>
  );
}

function HistoryThumbnail({ id, hasImage, wide = false }: { id: string; hasImage?: boolean; wide?: boolean }) {
  const [failed, setFailed] = useState(false);
  return (
    <span className={`grid shrink-0 place-items-center overflow-hidden rounded-lg border border-line bg-surface text-xs text-muted ${wide ? "aspect-[8/5] w-full" : "size-16"}`} aria-hidden="true">
      {failed || hasImage === false
        ? <span>No preview</span>
        : <img src={`/api/runs/${id}/image.png`} alt="" className="size-full object-contain" loading="lazy" onError={() => setFailed(true)} />}
    </span>
  );
}
