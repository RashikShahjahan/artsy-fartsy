"use client";
import { useCallback, useEffect, useRef, useState } from "react";
interface Recognition {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult:
    | ((event: {
        results: {
          isFinal: boolean;
          [index: number]: { transcript: string };
        }[];
      }) => void)
    | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
type VoiceWindow = Window & {
  SpeechRecognition?: new () => Recognition;
  webkitSpeechRecognition?: new () => Recognition;
};
export function useVoice(
  onTranscript: (text: string) => void,
  onError: (text: string) => void,
) {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [enabled, setEnabled] = useState(true);
  const recognition = useRef<Recognition | null>(null);
  const callbacks = useRef({ onTranscript, onError });
  useEffect(() => {
    callbacks.current = { onTranscript, onError };
  });
  const cancelSpeech = useCallback(() => {
    window.speechSynthesis?.cancel();
  }, []);
  const stop = useCallback(() => {
    const session = recognition.current;
    recognition.current = null;
    session?.abort();
    setListening(false);
    setTranscript("");
  }, []);
  useEffect(() => {
    const w = window as VoiceWindow;
    setSupported(!!(w.SpeechRecognition || w.webkitSpeechRecognition));
    try {
      setEnabled(localStorage.getItem("artsy.voice") !== "false");
    } catch {
      /* Optional storage. */
    }
    const visibility = () => {
      if (document.hidden) {
        cancelSpeech();
        stop();
      }
    };
    document.addEventListener("visibilitychange", visibility);
    return () => {
      recognition.current?.abort();
      cancelSpeech();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [cancelSpeech, stop]);
  async function speak(text: string) {
    cancelSpeech();
    if (!enabled || document.hidden || !window.speechSynthesis) return;
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text.slice(0, 500)));
  }

  function toggleSound() {
    cancelSpeech();
    setEnabled(!enabled);
    try {
      localStorage.setItem("artsy.voice", String(!enabled));
    } catch {
      /* Optional storage. */
    }
  }
  function start() {
    if (recognition.current) {
      recognition.current.stop();
      return;
    }
    const w = window as VoiceWindow,
      Constructor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Constructor) return;
    cancelSpeech();
    setTranscript("");
    const session = new Constructor();
    recognition.current = session;
    let final = "",
      failed = false;
    session.lang = navigator.language || "en-US";
    session.continuous = false;
    session.interimResults = true;
    session.onresult = (event) => {
      if (recognition.current !== session) return;
      final = "";
      let interim = "";
      for (let i = 0; i < event.results.length; i++) {
        if (event.results[i].isFinal)
          final += event.results[i][0].transcript + " ";
        else interim += event.results[i][0].transcript;
      }
      setTranscript((final + interim).trim());
    };
    session.onerror = (event) => {
      if (recognition.current !== session) return;
      failed = true;
      if (event.error !== "aborted")
        callbacks.current.onError(
          event.error === "not-allowed"
            ? "Allow microphone access, or type your idea below."
            : "I couldn’t hear that. Try again, or type your idea.",
        );
    };
    session.onend = () => {
      if (recognition.current !== session) return;
      recognition.current = null;
      setListening(false);
      setTranscript("");
      if (!failed && final.trim()) callbacks.current.onTranscript(final.trim());
    };
    try {
      session.start();
      setListening(true);
    } catch {
      stop();
      callbacks.current.onError(
        "The microphone couldn’t start. You can type instead.",
      );
    }
  }
  return {
    supported,
    listening,
    transcript,
    enabled,
    toggleSound,
    start,
    stop,
    speak,
    cancelSpeech,
  };
}
