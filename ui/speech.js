/* Kokoro provides the voice; Web Audio only plays the generated WAV. */
(() => {
  'use strict';
  class ArtsySpeech {
    constructor() {
      this.context = null;
      this.source = null;
      this.controller = null;
      this.generation = 0;
      this.pending = null;
      this.enabled = true;
      this.cache = new Map();
    }
    async unlock() {
      const Context = window.AudioContext || window.webkitAudioContext;
      if (!Context) return;
      this.context ||= new Context();
      try {
        if (this.context.state === 'suspended') await this.context.resume();
        if (this.pending && this.enabled) {
          const pending = this.pending;
          this.pending = null;
          this.play(pending.audio, pending.generation);
        }
      } catch { /* The next user gesture can retry audio playback. */ }
    }
    cancel() {
      this.generation++;
      this.controller?.abort();
      this.controller = null;
      this.pending = null;
      if (this.source) { try { this.source.stop(); } catch { /* Already ended. */ } }
      this.source = null;
    }
    setEnabled(enabled) {
      this.enabled = enabled;
      if (!enabled) this.cancel();
    }
    play(audio, generation) {
      if (!this.enabled || generation !== this.generation) return;
      if (!this.context || this.context.state !== 'running') {
        this.pending = {audio, generation};
        return;
      }
      const source = this.context.createBufferSource();
      source.buffer = audio;
      source.connect(this.context.destination);
      source.onended = () => { source.disconnect(); if (this.source === source) this.source = null; };
      this.source = source;
      source.start();
    }
    async speak(text) {
      this.cancel();
      if (!this.enabled) return;
      const generation = this.generation;
      const controller = new AbortController();
      this.controller = controller;
      try {
        let bytes = this.cache.get(text);
        if (!bytes) {
          const response = await fetch('/api/artsy/speech', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text.slice(0, 500)}), signal: controller.signal,
          });
          if (!response.ok) throw new Error('Kokoro speech unavailable');
          bytes = await response.arrayBuffer();
          this.cache.set(text, bytes);
          if (this.cache.size > 16) this.cache.delete(this.cache.keys().next().value);
        }
        if (generation !== this.generation || !this.enabled) return;
        const Context = window.AudioContext || window.webkitAudioContext;
        if (!Context) return;
        this.context ||= new Context();
        const audio = await this.context.decodeAudioData(bytes.slice(0));
        this.play(audio, generation);
      } catch (error) {
        if (error.name !== 'AbortError') console.warn('Artsy’s Kokoro voice is temporarily unavailable.');
      } finally {
        if (this.controller === controller) this.controller = null;
      }
    }
  }
  if (typeof module !== 'undefined') module.exports = ArtsySpeech;
  else window.ArtsySpeech = ArtsySpeech;
})();
