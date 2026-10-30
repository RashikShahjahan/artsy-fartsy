"use strict";

// One launch form, three explicit allowlisted workflows. Values are never
// interpolated into shell commands; the server builds the argv itself.
const fields = {
  eval: [
    ["model", "Evaluation model", "text", "", "e.g. Qwen/Qwen3.5-4B", "Must match the connected inference endpoint."],
    ["dataset", "Prompt dataset (optional)", "text", "", "Defaults to the frozen evaluation benchmark.", ""],
    ["limit", "Limit samples", "int", "", "All prompts", ""],
    ["max_samples", "Concurrent samples", "int", "10", "", ""],
    ["epochs", "Evaluation epochs", "int", "1", "", ""],
    ["max_connections", "Max model connections", "int", "10", "", "", true],
    ["max_turns", "Max turns per episode", "int", "", "CLI default", "", true],
    ["execution_only", "Execution-only (skip judges)", "bool", false, "", "For renderer smoke tests only.", true],
    ["dry_run", "Dry run (no model calls)", "bool", false, "", "", true],
  ],
  sft: [
    ["dataset", "Accepted training JSONL", "text", "datasets/sft-v2/accepted.jsonl", "", "Required. Use SFT candidates to export approved data."],
    ["model", "Base model (optional)", "text", "", "CLI default", "Qwen3.5 / Qwen3.8 vision model."],
    ["epochs", "Epochs", "int", "1", "", ""],
    ["batch_size", "Batch size", "int", "8", "", ""],
    ["learning_rate", "Learning rate", "float", "0.0002", "", "", true],
    ["lora_rank", "LoRA rank", "int", "32", "", "", true],
    ["max_steps", "Maximum updates", "int", "", "Unlimited", "", true],
    ["save_every", "Checkpoint every N updates", "int", "50", "", "", true],
    ["allow_unreviewed", "Allow pending visual reviews", "bool", false, "", "", true],
    ["dry_run", "Dry run (no model calls)", "bool", false, "", "", true],
  ],
  rl: [
    ["dataset", "RL prompt dataset", "text", "datasets/grpo-v2/prompts.jsonl", "", ""],
    ["init_checkpoint", "Initial SFT checkpoint", "text", "", "CLI default", "Path to final.json from an SFT run."],
    ["continue_from", "Continue from previous RL run (optional)", "text", "", "checkpoints/previous-run", "Forks an existing completed run with optimizer state.", true],
    ["additional_steps", "Additional updates when continuing", "int", "", "Required only for continuation", "", true],
    ["max_steps", "Maximum optimizer updates", "int", "", "All updates", ""],
    ["batch_size", "Prompt groups per update", "int", "2", "", ""],
    ["group_size", "Rollouts per prompt", "int", "4", "", ""],
    ["max_samples", "Concurrent Docker episodes", "int", "4", "", ""],
    ["epochs", "Epochs", "int", "1", "", ""],
    ["learning_rate", "Learning rate", "float", "0.00001", "", "", true],
    ["kl_coef", "Reference KL coefficient", "float", "0.01", "", "", true],
    ["save_every", "Checkpoint every N updates", "int", "1", "", "", true],
    ["dry_run", "Dry run (no model calls)", "bool", false, "", "", true],
  ],
};
const titles = {eval: "evaluation", sft: "SFT", rl: "RL"};
const descriptions = {
  eval: "Runs the existing Inspect benchmark. Uses model calls, Docker, and judges unless execution-only is enabled.",
  sft: "Trains on approved conversation trajectories using Tinker. Output appears under checkpoints/.",
  rl: "Runs the GRPO trainer, including Docker rollouts. Results will appear in the RL monitor.",
};
const $ = id => document.getElementById(id);
const saved = {};
let selectedId = new URLSearchParams(location.hash.slice(1)).get("run");
let activeKind = "eval";
let refreshing = false;
let submitting = false;

function fieldValue(input) {
  if (input.type === "checkbox") return input.checked;
  return input.value.trim();
}
function saveFields() {
  const values = saved[activeKind] || {};
  for (const input of document.querySelectorAll("#run-fields [data-option], #advanced-fields [data-option]")) {
    values[input.dataset.option] = fieldValue(input);
  }
  saved[activeKind] = values;
}
function makeField(spec, value) {
  const [name, label, type, defaultValue, placeholder, help] = spec;
  const wrapper = document.createElement("div");
  wrapper.className = "runs-input" + (type === "bool" ? " check" : "");
  wrapper.dataset.field = name;
  const input = document.createElement("input");
  const labelEl = document.createElement("label");
  input.id = "setting-" + name;
  input.dataset.option = name;
  labelEl.htmlFor = input.id;
  labelEl.textContent = label;
  if (type === "bool") {
    input.type = "checkbox";
    input.checked = Boolean(value);
    wrapper.append(input, labelEl);
  } else {
    input.type = type === "int" || type === "float" ? "number" : "text";
    if (type === "int") input.step = "1";
    if (type === "float") input.step = "any";
    input.placeholder = placeholder || "";
    input.value = value ?? defaultValue;
    if (name === "dataset" && activeKind === "sft") input.required = true;
    wrapper.append(labelEl, input);
  }
  if (help) {
    const note = document.createElement("small");
    note.textContent = help;
    wrapper.append(note);
  }
  input.addEventListener("input", () => { saveFields(); updateContinuation(); });
  input.addEventListener("change", () => { saveFields(); updateContinuation(); });
  return wrapper;
}
function updateContinuation() {
  if (activeKind !== "rl") return;
  const continuing = Boolean((saved.rl || {}).continue_from?.trim());
  const allowed = new Set(["dataset", "continue_from", "additional_steps", "save_every", "dry_run"]);
  for (const wrapper of document.querySelectorAll("[data-field]")) {
    if (!allowed.has(wrapper.dataset.field)) {
      wrapper.hidden = continuing;
      wrapper.querySelector("input").disabled = continuing;
    }
    if (wrapper.dataset.field === "additional_steps") {
      wrapper.hidden = !continuing;
      wrapper.querySelector("input").disabled = !continuing;
      wrapper.querySelector("input").required = continuing;
    }
  }
  if (continuing) $("advanced").open = true;
}
function renderFields(kind) {
  activeKind = kind;
  const values = saved[kind] || {};
  const basic = $("run-fields"), advanced = $("advanced-fields");
  basic.replaceChildren();
  advanced.replaceChildren();
  for (const spec of fields[kind]) {
    const value = Object.hasOwn(values, spec[0]) ? values[spec[0]] : spec[3];
    (spec[6] ? advanced : basic).append(makeField(spec, value));
  }
  $("kind-note").textContent = descriptions[kind];
  $("launch-button").textContent = "Launch " + titles[kind];
  $("advanced").open = false;
  updateContinuation();
}
function collectSettings() {
  const settings = {};
  const continuing = activeKind === "rl" && Boolean((saved.rl || {}).continue_from?.trim());
  const permitted = new Set(["dataset", "continue_from", "additional_steps", "save_every", "dry_run"]);
  for (const input of document.querySelectorAll("[data-option]")) {
    if (continuing && !permitted.has(input.dataset.option)) continue;
    const value = fieldValue(input);
    if (input.type === "checkbox") settings[input.dataset.option] = value;
    else if (value !== "") {
      settings[input.dataset.option] =
        input.type === "number" ? Number(value) : value;
    }
  }
  return settings;
}
async function request(url, opts = {}) {
  const response = await fetch(url, {cache: "no-store", ...opts});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Request failed (" + response.status + ")");
  return result;
}
function message(text, isError = false) {
  $("launch-status").textContent = text;
  $("launch-status").className = isError ? "error" : "";
}
function relativeDate(date) {
  if (!date) return "";
  const value = new Date(date);
  return Number.isNaN(value.getTime()) ? date : value.toLocaleString();
}
function renderHistory(rows) {
  const root = $("runs-history");
  root.replaceChildren();
  if (!rows.length) {
    const paragraph = document.createElement("p");
    paragraph.className = "note";
    paragraph.textContent = "No experiment runs yet.";
    root.append(paragraph);
    return;
  }
  for (const row of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-current", String(row.id === selectedId));
    const name = document.createElement("span");
    name.className = "runs-entry-name";
    name.textContent = (row.kind || "").toUpperCase() + " · " + (row.output || row.id);
    const meta = document.createElement("small");
    meta.className = "runs-entry-meta";
    meta.textContent = relativeDate(row.created_at);
    name.append(meta);
    const badge = document.createElement("span");
    badge.className = "runs-entry-state";
    badge.textContent = row.state;
    button.append(name, badge);
    button.addEventListener("click", () => {
      selectedId = row.id;
      location.hash = "run=" + row.id;
      refresh().catch(error => message(error.message, true));
    });
    root.append(button);
  }
}
function renderDetail(data) {
  $("detail-title").textContent = data.kind.toUpperCase() + " run";
  $("detail-state").textContent = data.state;
  $("detail-meta").textContent = "Output: " + data.output + " · Started: " + relativeDate(data.created_at) +
    (data.finished_at ? " · Finished: " + relativeDate(data.finished_at) : "") +
    (data.exit_code !== undefined ? " · Exit: " + data.exit_code : "") +
    (data.error ? " · " + data.error : "");
  $("run-command").textContent = data.command.map(value => JSON.stringify(value)).join(" ");
  $("run-console").textContent = data.console || "No output yet.";
  $("run-updated").textContent = "Updated " + new Date().toLocaleTimeString();
  $("cancel-run").hidden = !data.can_cancel;
  $("monitor-link").hidden = data.kind !== "rl";
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const rows = (await request("/api/launches")).runs;
    if (!selectedId && rows.length) selectedId = rows[0].id;
    renderHistory(rows);
    if (selectedId) {
      try {
        renderDetail(await request("/api/launches/" + encodeURIComponent(selectedId)));
      } catch (error) {
        $("detail-state").textContent = error.message;
      }
    }
  } finally {
    refreshing = false;
  }
}
$("run-kind").addEventListener("change", event => {
  saveFields();
  renderFields(event.target.value);
  message("");
});
$("launch-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (submitting) return;
  saveFields();
  const kind = activeKind;
  const settings = collectSettings();
  if (!settings.dry_run && !window.confirm("Launch " + titles[kind] + "? This may consume model credits and compute.")) return;
  submitting = true;
  $("launch-button").disabled = true;
  message("Starting run…");
  try {
    const run = await request("/api/launches", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({kind, settings}),
    });
    selectedId = run.id;
    location.hash = "run=" + run.id;
    message("Run launched. Output: " + run.output);
    await refresh();
  } catch (error) {
    message(error.message, true);
  } finally {
    submitting = false;
    $("launch-button").disabled = false;
  }
});
$("cancel-run").addEventListener("click", async () => {
  if (!selectedId || !window.confirm("Stop this run and its subprocesses?")) return;
  try {
    await request("/api/launches/" + encodeURIComponent(selectedId) + "/cancel", {method: "POST"});
    await refresh();
  } catch (error) { message(error.message, true); }
});
$("refresh-runs").addEventListener("click", () => refresh().catch(error => message(error.message, true)));
renderFields(activeKind);
refresh().catch(error => message(error.message, true));
window.setInterval(() => {
  if (!document.hidden) refresh().catch(error => message(error.message, true));
}, 5000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh().catch(error => message(error.message, true));
});
