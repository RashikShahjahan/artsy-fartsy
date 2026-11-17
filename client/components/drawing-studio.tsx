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
  const [typed, setTyped] = useState(false);
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
    (text) => void submit(text),
    (message) => {
      setError(message);
      setTyped(true);
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
      setTyped(true);
    }
  }
  const { run, connectionError } = useRun(activeId, finish, () => {
    rememberRun(null);
    setCanceling(false);
    setError("This drawing is unavailable. Please try again.");
    setTyped(true);
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
    setSavedRun(null);
    setError("");
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
    setNotice("");
  }
  async function submit(text: string) {
    text = text.trim();
    if (busy || submitting.current || !text) return;
    if (text.length > 4000) {
      setError("Please keep your idea under 4,000 characters.");
      return;
    }
    submitting.current = true;
    setStarting(true);
    setError("");
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
      const data = await response.json();
      if (!response.ok || !RUN_ID.test(data.id))
        throw new Error(
          data.error || "I couldn’t start this drawing. Please try again.",
        );
      rememberRun(data.id);
      setPrompt("");
      setSavedRun(null);
      setTyped(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Please try again.");
      setPrompt(text);
      setTyped(true);
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
          {version && version.hasImage !== false && !busy && (
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
            onClick={() => dialog.current?.showModal()}
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
            <p className="eyebrow">A LITTLE SPACE FOR YOUR IMAGINATION</p>
            <h1>Hi, I’m Artsy.</h1>
            <p>What should we draw?</p>
          </section>
        )}
        {(current || busy) && (
          <section
            className="drawing-space"
            aria-label="Your drawing"
            aria-busy={busy}
          >
            {version && version.hasImage !== false && !busy && (
              <img
                className="drawing-image"
                src={`/api/runs/${version.id}/image.png`}
                alt={version.prompt}
                onError={() =>
                  setError(
                    "This image couldn’t be loaded. The drawing service may be unavailable.",
                  )
                }
              />
            )}
            {busy && <Activity key={activeId || "starting"} run={run} busy />}
            {!busy && version && (
              <div className="drawing-caption">
                <p>{current?.title}</p>
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
                    }}
                  >
                    {current!.versions.map((v, i) => (
                      <option value={v.id} key={v.id}>
                        Version {i + 1}
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
                : notice || (current ? "Anything you’d like to change?" : "")}
          </p>
          {error && (
            <p className="input-error" role="alert">
              {error}
            </p>
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
            <>
              {typed || !voice.supported ? (
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
                  <button
                    type="submit"
                    className="send-button"
                    disabled={!prompt.trim() || !ready}
                    aria-label="Send to Artsy"
                  >
                    <Icon name="arrow" />
                  </button>
                </form>
              ) : (
                <button
                  className={`mic-button ${voice.listening ? "listening" : ""}`}
                  onClick={voice.start}
                  disabled={!ready}
                  aria-label={
                    voice.listening
                      ? "Finish speaking and send your idea"
                      : "Start talking to Artsy"
                  }
                  aria-pressed={voice.listening}
                >
                  {voice.listening ? (
                    <span className="stop-square" />
                  ) : (
                    <Icon name="mic" />
                  )}
                </button>
              )}
              {voice.supported && (
                <button
                  className="type-toggle"
                  onClick={() => {
                    voice.stop();
                    setTyped(!typed);
                    if (!typed) setTimeout(() => input.current?.focus(), 0);
                  }}
                >
                  <Icon name={typed ? "mic" : "keyboard"} />
                  {typed ? "Use your voice" : "Or type an idea"}
                </button>
              )}
            </>
          )}
        </section>
      </main>
      <footer>Made of imagination. Yours and a little of ours.</footer>
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
            {drawings.map((drawing) => (
              <button
                key={drawing.id}
                className="history-item"
                disabled={busy}
                aria-current={drawing.id === current?.id ? "true" : undefined}
                onClick={() => openDrawing(drawing)}
              >
                <strong>{drawing.title}</strong>
                <small>
                  {drawing.versions.length}{" "}
                  {drawing.versions.length === 1 ? "drawing" : "versions"}
                </small>
              </button>
            ))}
          </nav>
        ) : (
          <p className="empty-history">
            Your ideas will find a home here.
            <br />
            Make your first drawing to get started.
          </p>
        )}
        <p className="history-note">
          Saved in this browser. Artwork stays on your drawing service.
        </p>
      </dialog>
    </div>
  );
}
