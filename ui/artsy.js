/* Voice-first Artsy. Model selection and turn limits are enforced by the server. */
(() => {
  'use strict';
  const ui = Object.fromEntries([...document.querySelectorAll('[id]')].map(node => [node.id.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), node]));
  const STORAGE = 'artsy.sketchbook.v1';
  const ACTIVE = 'artsy.active.v1';
  const ID = /^[a-f0-9]{32}$/;
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const speech = new window.ArtsySpeech();
  const activity = new window.ArtsyActivity(ui.agentActivity, ui.activitySteps, ui.activityStatus, approveAction);
  let drawings = [], current = null, activeRun = null, busy = false, listening = false;
  let recognition = null, pollTimer = null;
  let pollFailures = 0, lastPhase = '', previewSpoken = false, lastPrompt = '', voiceOn = true;
  let activityRequest = 0;
  const progressMessages = {
    thinking: 'Imagining your drawing…',
    drawing: 'Putting it on paper…',
    previewing: 'Taking a closer look…',
    rendering: 'Finishing up…',
    awaiting_approval: 'Waiting for your approval…',
  };

  async function approveAction(stepId) {
    if (!activeRun) return false;
    try {
      const response = await fetch(`/api/runs/${activeRun}/approve`, {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({step_id: stepId})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The action could not be approved. Please try again.');
      showError('');
      return true;
    } catch (error) { showError(error.message); return false; }
  }
  async function loadActivity(id) {
    const request = ++activityRequest;
    activity.reset(id);
    try {
      const response = await fetch(`/api/runs/${id}`, {cache: 'no-store'});
      if (!response.ok) throw new Error('Activity unavailable');
      const run = await response.json();
      if (request === activityRequest && !busy) activity.render({...run, id});
    } catch {
      if (request === activityRequest && !busy) ui.activityStatus.textContent = 'Step history is unavailable for this drawing.';
    }
  }

  function persist() {
    try { localStorage.setItem(STORAGE, JSON.stringify(drawings.slice(0, 50))); } catch { /* Artifacts stay on the server. */ }
  }
  function rememberActive(id) {
    try { if (id) localStorage.setItem(ACTIVE, id); else localStorage.removeItem(ACTIVE); } catch { /* Storage is optional. */ }
  }
  function showError(message) { ui.inputError.textContent = message; ui.inputError.hidden = !message; }
  function speak(message) {
    if (!voiceOn || listening || document.hidden) return;
    speech.speak(message);
  }
  function toggleSidebar(open) {
    ui.sidebar.hidden = !open; ui.sidebarScrim.hidden = !open;
    ui.mobileMenu.setAttribute('aria-expanded', String(open));
    ui.main.inert = open; document.querySelector('.topbar').inert = open;
    if (open) ui.closeSidebar.focus(); else ui.mobileMenu.focus();
  }
  function closeSidebar() { if (!ui.sidebar.hidden) toggleSidebar(false); }
  function renderHistory() {
    ui.history.replaceChildren(); ui.mobileMenu.hidden = !drawings.length;
    for (const drawing of drawings) {
      const button = document.createElement('button');
      button.className = 'history-item' + (current?.id === drawing.id ? ' selected' : '');
      button.disabled = busy;
      if (current?.id === drawing.id) button.setAttribute('aria-current', 'true');
      const title = document.createElement('strong'); title.textContent = drawing.title;
      const note = document.createElement('small'); note.textContent = `${drawing.versions.length} ${drawing.versions.length === 1 ? 'drawing' : 'versions'}`;
      button.append(title, note); button.onclick = () => openDrawing(drawing); ui.history.append(button);
    }
  }
  function updateVoiceLabel() {
    ui.voiceTitle.textContent = busy ? 'Drawing…' : listening ? 'Listening…' : current?.versions.length && ui.promptForm.hidden ? 'Anything you’d like to change?' : '';
  }
  function setBusy(value) {
    busy = value; ui.main.classList.toggle('is-busy', value);
    ui.micButton.disabled = value || !Recognition; ui.send.disabled = value;
    ui.newDrawing.disabled = value; ui.newDrawing.hidden = !current;
    ui.versions.disabled = value;
    ui.model.disabled = value;
    ui.executionModes.disabled = value;
    renderHistory(); updateVoiceLabel();
  }
  function showWorkspace() {
    ui.welcome.hidden = true; ui.drawingSpace.hidden = false;
    ui.main.classList.add('has-drawing'); ui.newDrawing.hidden = false;
  }
  function renderVersions(selectedId) {
    ui.versions.replaceChildren();
    ui.versions.hidden = !current || current.versions.length < 2;
    if (ui.versions.hidden) return;
    current.versions.forEach((version, index) => {
      const option = new Option(`Version ${index + 1}`, version.id);
      option.selected = version.id === selectedId; ui.versions.append(option);
    });
  }
  function showVersion(version, loadSteps = true) {
    current.selectedId = version.id;
    ui.drawingImage.src = `/api/runs/${version.id}/image.png`;
    ui.drawingImage.alt = version.prompt;
    ui.drawingImage.hidden = false; ui.workingPlaceholder.hidden = true;
    ui.download.href = ui.drawingImage.src; ui.download.hidden = false;
    renderVersions(version.id); updateVoiceLabel();
    if (loadSteps) loadActivity(version.id);
  }
  function openDrawing(drawing) {
    if (busy) return;
    stopListening(true); showError(''); current = drawing; closeSidebar();
    showWorkspace();
    ui.prompt.value = ''; ui.prompt.placeholder = 'What would you like to change?';
    showVersion(drawing.versions.find(version => version.id === drawing.selectedId) || drawing.versions.at(-1));
    renderHistory();
  }
  function newDrawing() {
    if (busy) return;
    stopListening(true); speech.cancel();
    current = null; activeRun = null; rememberActive(null); showError(''); closeSidebar();
    activityRequest++; activity.reset();
    ui.welcome.hidden = false; ui.drawingSpace.hidden = true;
    ui.main.classList.remove('has-drawing'); ui.newDrawing.hidden = true;
    ui.prompt.value = ''; ui.prompt.placeholder = 'What should we draw?';
    ui.drawingImage.hidden = true; ui.drawingImage.removeAttribute('src'); ui.download.hidden = true;
    setTypedInput(!Recognition, false); renderHistory(); updateVoiceLabel();
  }
  function setTypedInput(show, focus = true) {
    ui.promptForm.hidden = !show; ui.typeToggle.setAttribute('aria-expanded', String(show));
    ui.micButton.hidden = show || !Recognition;
    updateVoiceLabel();
    const label = show ? 'Use microphone' : 'Type an idea';
    ui.typeToggle.setAttribute('aria-label', label); ui.typeToggle.title = label;
    if (show && focus) ui.prompt.focus();
  }
  function setListening(value) {
    listening = value; ui.micButton.classList.toggle('listening', value);
    ui.micButton.setAttribute('aria-pressed', String(value));
    ui.micButton.setAttribute('aria-label', value ? 'Finish speaking and send your idea' : 'Start talking to Artsy');
    ui.micButton.querySelector('svg').toggleAttribute('hidden', value);
    ui.micButton.querySelector('.mic-stop').hidden = !value;
    updateVoiceLabel();
  }
  function stopListening(discard = false) {
    if (!recognition) return;
    const session = recognition;
    if (discard) {
      recognition = null; setListening(false); session.abort();
    } else session.stop();
  }
  function startListening() {
    if (busy) return;
    if (!Recognition) { setTypedInput(true); return; }
    if (recognition) { stopListening(); return; }
    showError(''); speech.cancel();
    const session = new Recognition();
    recognition = session;
    let finalTranscript = '', canceled = false;
    recognition.lang = navigator.language || 'en-US'; recognition.continuous = false; recognition.interimResults = true;
    recognition.onstart = () => { if (recognition === session) setListening(true); };
    recognition.onresult = event => {
      if (recognition !== session) return;
      let interim = ''; finalTranscript = '';
      for (let i = 0; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript + ' ';
        else interim += event.results[i][0].transcript;
      }
      ui.voiceTitle.textContent = (finalTranscript + interim).trim() || 'Listening…';
    };
    recognition.onerror = event => {
      if (recognition !== session) return;
      canceled = true;
      const messages = {
        'not-allowed': 'Allow microphone access, or type your idea.',
        'service-not-allowed': 'Voice isn’t available. You can type instead.',
        'audio-capture': 'No microphone found. You can type instead.',
        'no-speech': 'I didn’t catch that. Try again?',
        network: 'Voice couldn’t connect. Try again, or type instead.',
      };
      if (event.error !== 'aborted') { showError(messages[event.error] || 'I couldn’t hear that. Try again?'); setTypedInput(true, false); }
      setListening(false);
    };
    recognition.onend = () => {
      if (recognition !== session) return;
      setListening(false); recognition = null;
      if (!canceled && finalTranscript.trim()) submit(finalTranscript.trim());
    };
    try { recognition.start(); } catch { recognition = null; setListening(false); setTypedInput(true); showError('The microphone couldn’t start. You can type instead.'); }
  }
  async function submit(prompt) {
    prompt = prompt.trim();
    if (busy || !prompt) return;
    if (prompt.length > 4000) { setTypedInput(true); ui.prompt.value = prompt.slice(0, 4000); showError('Please keep your idea under 4,000 characters.'); return; }
    stopListening(true); showError(''); setBusy(true);
    let accepted = false;
    try {
      const parentId = current?.selectedId || current?.versions.at(-1)?.id;
      const response = await fetch('/api/artsy/runs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt, model: ui.model.value, execution_mode: ui.modeApproval.checked ? 'approval' : 'auto', ...(parentId ? {parent_id: parentId} : {})})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'I couldn’t start this drawing. Please try again.');
      if (!ID.test(data.id)) throw new Error('Something went wrong. Please try again.');
      accepted = true; activeRun = data.id; rememberActive(data.id); lastPrompt = prompt;
      activityRequest++; activity.reset(data.id); activity.render({id: data.id, state: 'queued'});
      if (!current?.versions.length) current = {id: data.id, title: prompt, versions: []};
      showWorkspace(); setBusy(true);
      ui.prompt.value = ''; ui.prompt.placeholder = 'What would you like to change?';
      setTypedInput(false, false); lastPhase = ''; previewSpoken = false;
      if (!parentId) { ui.drawingImage.hidden = true; ui.workingPlaceholder.hidden = false; ui.download.hidden = true; ui.versions.hidden = true; }
      speak(parentId ? 'I’ll make that change.' : 'I’ll draw that for you.');
      pollFailures = 0; poll();
    } catch (error) {
      showError(error.message); ui.prompt.value = prompt; setTypedInput(true, false);
    } finally { if (!accepted) setBusy(false); }
  }
  function applyProgress(progress) {
    if (!progress || !progressMessages[progress.phase] || progress.phase === lastPhase) return;
    lastPhase = progress.phase; ui.voiceTitle.textContent = progressMessages[progress.phase];
    if (progress.phase === 'previewing' && !previewSpoken) { speak('I’m taking a closer look.'); previewSpoken = true; }
  }
  async function poll() {
    const id = activeRun;
    if (!id) return;
    clearTimeout(pollTimer);
    try {
      const response = await fetch(`/api/runs/${id}`, {cache: 'no-store'});
      const run = await response.json();
      if (activeRun !== id) return;
      if (!response.ok) {
        if (response.status === 404) { finishFailed('This drawing is unavailable. Please try again.'); return; }
        throw new Error('Drawing service unavailable.');
      }
      pollFailures = 0; showError(''); lastPrompt = run.prompt;
      if (!current) {
        current = drawings.find(drawing => drawing.versions.some(version => version.id === run.parent_id)) || {id, title: run.prompt, versions: []};
        showWorkspace();
        if (current.versions.length) showVersion(current.versions.find(v => v.id === run.parent_id) || current.versions.at(-1), false);
        else { ui.drawingImage.hidden = true; ui.workingPlaceholder.hidden = false; }
        setBusy(true);
      }
      applyProgress(run.progress);
      ui.modeApproval.checked = run.execution_mode === 'approval'; ui.modeAuto.checked = !ui.modeApproval.checked;
      activity.render({...run, id});
      if (run.state === 'queued') ui.voiceTitle.textContent = 'Getting ready…';
      if (run.state === 'running' || run.state === 'queued') { pollTimer = setTimeout(poll, 1500); return; }
      if (run.state === 'completed' && run.result?.outcome === 'submitted' && run.artifacts?.includes('image.png')) {
        if (!current.versions.some(version => version.id === id)) current.versions.push({id, prompt: run.prompt});
        current.selectedId = id;
        drawings = [current, ...drawings.filter(drawing => drawing.id !== current.id)].slice(0, 50); persist();
        showVersion(current.versions.find(version => version.id === id), false);
        activeRun = null; rememberActive(null); setBusy(false);
        if (!Recognition) setTypedInput(true, false);
        speak('Your drawing is ready. Anything you’d like to change?');
      } else if (run.result?.outcome === 'refused') {
        finishFailed(run.result.refusalReason || run.result.reason || 'Let’s try a different idea.');
      } else finishFailed('I couldn’t finish that. Try again?');
    } catch {
      if (activeRun !== id) return;
      pollFailures++;
      showError('Connection lost. Reconnecting…');
      pollTimer = setTimeout(poll, Math.min(10000, 1500 * pollFailures));
    }
  }
  function finishFailed(message) {
    activity.fail(message);
    activeRun = null; rememberActive(null); setBusy(false);
    if (!current?.versions.length) { ui.drawingSpace.hidden = true; ui.welcome.hidden = ui.agentActivity.hidden === false; ui.main.classList.remove('has-drawing'); }
    showError(message); ui.prompt.value = lastPrompt; setTypedInput(true, false); speak(message);
  }

  ui.newDrawing.onclick = newDrawing;
  ui.model.onchange = () => {
    try { localStorage.setItem('artsy.model', ui.model.value); } catch { /* Optional preference. */ }
  };
  for (const input of [ui.modeAuto, ui.modeApproval]) input.onchange = () => {
    try { localStorage.setItem('artsy.executionMode', ui.modeApproval.checked ? 'approval' : 'auto'); } catch { /* Optional preference. */ }
  };
  ui.typeToggle.onclick = () => {
    stopListening(true);
    setTypedInput(ui.promptForm.hidden);
  };
  ui.micButton.onclick = startListening;
  ui.promptForm.onsubmit = event => { event.preventDefault(); submit(ui.prompt.value); };
  ui.prompt.onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submit(ui.prompt.value); } };
  ui.versions.onchange = () => {
    if (busy) return;
    const version = current?.versions.find(v => v.id === ui.versions.value);
    if (version) { current.selectedId = version.id; persist(); showVersion(version); }
  };
  ui.soundToggle.onclick = () => {
    voiceOn = !voiceOn; ui.soundToggle.setAttribute('aria-pressed', String(voiceOn));
    speech.setEnabled(voiceOn);
    const label = voiceOn ? 'Mute Artsy' : 'Unmute Artsy';
    ui.soundToggle.title = label; ui.soundToggle.setAttribute('aria-label', label);
    ui.soundToggle.querySelector('use').setAttribute('href', voiceOn ? '#i-volume' : '#i-mute');
    try { localStorage.setItem('artsy.voice', String(voiceOn)); } catch { /* Optional preference. */ }
    if (voiceOn) speak(current ? ui.voiceTitle.textContent : 'Hi, I’m Artsy. What should we draw?');
    else speech.cancel();
  };
  ui.mobileMenu.onclick = () => toggleSidebar(true);
  ui.sidebarScrim.onclick = closeSidebar; ui.closeSidebar.onclick = closeSidebar;
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { stopListening(true); closeSidebar(); }
    if (event.key.toLowerCase() === 'n' && !event.metaKey && !event.ctrlKey && !event.altKey && !['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(document.activeElement.tagName)) newDrawing();
  });
  ui.drawingImage.onerror = () => showError('This image couldn’t be loaded.');
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE) || '[]');
    const savedModel = localStorage.getItem('artsy.model');
    if ([...ui.model.options].some(option => option.value === savedModel)) ui.model.value = savedModel;
    if (Array.isArray(saved)) drawings = saved.filter(d => d && ID.test(d.id) && typeof d.title === 'string' && Array.isArray(d.versions) && d.versions.length && d.versions.every(v => v && ID.test(v.id) && typeof v.prompt === 'string')).slice(0, 50);
    if (localStorage.getItem('artsy.voice') === 'false') ui.soundToggle.click();
    ui.modeApproval.checked = localStorage.getItem('artsy.executionMode') === 'approval'; ui.modeAuto.checked = !ui.modeApproval.checked;
    const resume = localStorage.getItem(ACTIVE);
    if (ID.test(resume || '')) activeRun = resume;
  } catch { /* Start clean when browser storage is unavailable. */ }
  renderHistory(); updateVoiceLabel();
  if (!Recognition) { ui.micButton.hidden = true; ui.typeToggle.hidden = true; setTypedInput(true, false); }
  document.addEventListener('pointerdown', () => speech.unlock(), {capture: true});
  document.addEventListener('keydown', () => speech.unlock(), {capture: true});
  document.addEventListener('visibilitychange', () => { if (document.hidden) speech.cancel(); });
  if (!(window.AudioContext || window.webkitAudioContext)) { voiceOn = false; speech.setEnabled(false); ui.soundToggle.hidden = true; }
  if (activeRun) { setBusy(true); poll(); }
  else speak('Hi, I’m Artsy. What should we draw?');
})();
