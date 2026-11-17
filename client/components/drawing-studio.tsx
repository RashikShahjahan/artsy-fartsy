"use client";
import { useEffect, useRef, useState } from "react";
import {
  ACTIVE,
  STORAGE,
  completedDrawing,
  readDrawings,
} from "@/lib/sketchbook";
import { RUN_ID, type Drawing, type Run } from "@/lib/types";
import { useRun } from "@/hooks/use-run";
import { useVoice } from "@/hooks/use-voice";
import { Activity } from "./activity";
import { Icon } from "./icon";
export function DrawingStudio() {
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [current, setCurrent] = useState<Drawing | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [ready, setReady] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [failedImageId, setFailedImageId] = useState("");
  const [imageAttempt, setImageAttempt] = useState(0);
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
  const busy = starting || !!activeId;
  const version =
    current?.versions.find((v) => v.id === current.selectedId) ||
    current?.versions.at(-1);
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
    const drawing = completedDrawing(run, current, drawings);
    setSavedRun(run);
    rememberRun(null);
    setCanceling(false);
    if (drawing) {
      setCurrent(drawing);
      setDrawings((items) =>
        [drawing, ...items.filter((d) => d.id !== drawing.id)].slice(0, 50),
      );
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
    setReady(true);
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
    setSavedRun(null);
    setError("");
    setCanRetry(false);
    setNotice("");
    setPrompt("");
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
    if (busy || submitting.current || voice.listening || !text) return;
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
      const response = await fetch("/api/artsy/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: text,
          ...(version?.editable !== false && version?.id
            ? { parent_id: version.id }
            : {}),
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
    <div className="app-shell">
      <header className="topbar">
        <a href="/" className="wordmark" aria-label="Artsy home">
          artsy<span>✳</span>
        </a>
        <div className="topbar-actions">
          {current && (
            <button
              className="icon-button"
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
              className="icon-button"
              href={`/api/runs/${version.id}/image.png`}
              download="artsy-drawing.png"
              aria-label="Save image"
              title="Save image"
            >
              <Icon name="download" />
            </a>
          )}
          <button
            className="icon-button"
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
            className="icon-button"
            onClick={voice.toggleSound}
            aria-label={voice.enabled ? "Mute Artsy" : "Unmute Artsy"}
            aria-pressed={voice.enabled}
            title={voice.enabled ? "Mute Artsy" : "Unmute Artsy"}
          >
            <Icon name={voice.enabled ? "sound" : "mute"} />
          </button>
        </div>
      </header>
      <main className={current || busy ? "workspace" : "welcome-workspace"}>
        {!current && !busy && (
          <section className="welcome">
            <h1>What should we draw?</h1>
            <p>Describe a drawing, then make it your own.</p>
          </section>
        )}
        {(current || busy) && (
          <section
            className="drawing-space"
            aria-label="Your drawing"
            aria-busy={busy}
          >
            {version && !busy && (
              <div className="artwork-frame">
                {version.hasImage === false || failedImageId === version.id ? (
                  <div className="artwork-unavailable" role="status">
                    <Icon name="image" />
                    <h2>{version.hasImage === false ? "No image for this version" : "Drawing unavailable"}</h2>
                    <p>{version.hasImage === false
                      ? "Try another version or describe what to draw next."
                      : "We couldn’t load this drawing. You can try loading it again."}</p>
                    {version.hasImage !== false && (
                      <button className="text-button" onClick={() => {
                        setImageAttempt((attempt) => attempt + 1);
                        setFailedImageId("");
                      }}>Retry loading</button>
                    )}
                  </div>
                ) : (
                  <img
                    key={`${version.id}-${imageAttempt}`}
                    className="drawing-image"
                    src={`/api/runs/${version.id}/image.png?attempt=${imageAttempt}`}
                    alt={version.prompt}
                    onError={() => setFailedImageId(version.id)}
                  />
                )}
              </div>
            )}
            {busy && <Activity key={activeId || "starting"} run={run} busy />}
            {!busy && version && (
              <div className="drawing-caption">
                <p>{version.prompt}</p>
                {current!.versions.length > 1 && (
                  <select
                    aria-label="Drawing version"
                    value={version.id}
                    onChange={(event) => {
                      const selected = {
                        ...current!,
                        selectedId: event.target.value,
                      };
                      setCurrent(selected);
                      setDrawings((items) =>
                        items.map((drawing) =>
                          drawing.id === selected.id ? selected : drawing,
                        ),
                      );
                      setNotice("");
                      setError("");
                      setCanRetry(false);
                      setFailedImageId("");
                    }}
                  >
                    {current!.versions.map((v, i) => (
                      <option value={v.id} key={v.id}>
                        Version {i + 1} — {v.prompt}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
            {!busy && savedRun && (
              <details className="recorded-steps">
                <summary>How it came together</summary>
                <Activity key={savedRun.id} run={savedRun} busy={false} />
              </details>
            )}
          </section>
        )}
        <section className="voice-dock" aria-label="Talk to Artsy">
          <p className="voice-status" role="status" aria-live="polite">
            {voice.listening
              ? voice.transcript || "Listening…"
              : busy
                ? connectionError || (canceling ? "Stopping…" : "Drawing…")
                : error || (version && (failedImageId === version.id || version.hasImage === false))
                  ? ""
                  : notice || (current ? "Anything you’d like to change?" : "")}
          </p>
          {error && (
            <div className="input-error" role="alert">
              <p>{error}</p>
              {canRetry && (
                <button
                  className="text-button"
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
              className="text-button"
              disabled={canceling || starting}
              onClick={() => void cancel()}
            >
              {canceling ? "Stopping…" : "Stop drawing"}
            </button>
          ) : (
            <form
              className="prompt-form"
              onSubmit={(event) => {
                event.preventDefault();
                void submit(prompt);
              }}
            >
              <label className="sr-only" htmlFor="prompt">
                What would you like Artsy to draw or change?
              </label>
              <textarea
                ref={input}
                id="prompt"
                maxLength={4000}
                rows={2}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={
                  current
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
                  className={`composer-mic ${voice.listening ? "listening" : ""}`}
                  onClick={voice.start}
                  disabled={!ready}
                  aria-label={voice.listening ? "Finish dictating" : "Dictate an idea"}
                  aria-pressed={voice.listening}
                  title={voice.listening ? "Finish dictating" : "Dictate an idea"}
                >
                  {voice.listening ? <span className="stop-square" /> : <Icon name="mic" />}
                </button>
              )}
              <button
                type="submit"
                className="send-button"
                disabled={!prompt.trim() || !ready || voice.listening}
                aria-label="Send to Artsy"
              >
                <Icon name="arrow" />
              </button>
            </form>
          )}
        </section>
      </main>
      <dialog
        className="history-dialog"
        ref={dialog}
        onClick={(event) => {
          if (event.target === event.currentTarget) closeHistory();
        }}
      >
        <div className="history-header">
          <h2>Your drawings</h2>
          <button
            className="icon-button"
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
                  className="history-item"
                  disabled={busy}
                  aria-current={drawing.id === current?.id ? "true" : undefined}
                  onClick={() => openDrawing(drawing)}
                >
                  <HistoryThumbnail
                    key={`${thumbnail.id}-${historyAttempt}`}
                    id={thumbnail.id}
                    hasImage={thumbnail.hasImage}
                  />
                  <span className="history-item-copy">
                    <strong>{drawing.title}</strong>
                    <small>
                      {drawing.versions.length}{" "}
                      {drawing.versions.length === 1 ? "drawing" : "versions"}
                    </small>
                  </span>
                </button>
              );
            })}
          </nav>
        ) : (
          <p className="empty-history">
            Your drawings will appear here after you make one.
          </p>
        )}
        <p className="history-note">
          History is saved in this browser.
        </p>
      </dialog>
    </div>
  );
}

function HistoryThumbnail({ id, hasImage }: { id: string; hasImage?: boolean }) {
  const [failed, setFailed] = useState(false);
  return (
    <span className="history-thumbnail" aria-hidden="true">
      {failed || hasImage === false
        ? <span>No preview</span>
        : <img src={`/api/runs/${id}/image.png`} alt="" loading="lazy" onError={() => setFailed(true)} />}
    </span>
  );
}
