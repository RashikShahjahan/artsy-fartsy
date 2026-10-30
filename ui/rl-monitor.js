/* Passive RL monitor: fetch only saved artifacts, never talk to the trainer. */
"use strict";
const byId = id => document.getElementById(id);
const state = {run: null, rows: [], selected: null, busy: false, detailRequest: 0};
const PERIOD_MS = 30000;
let refreshTimer = null;

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}
function fmt(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}
function episodeKey(e) {
  return [e.step, e.promptId, e.episodeId].join("/");
}
function artifactUrl(e, filename) {
  return "/api/rl/artifacts/" +
    [state.run, e.step, e.promptId, e.episodeId, filename].map(encodeURIComponent).join("/");
}
async function getJson(url) {
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) throw new Error("Request failed (" + response.status + ")");
  return response.json();
}
function setStatus(message) { byId("rl-status").textContent = message; }
function metric(title, value, note) {
  const box = element("div", undefined, "rl-metric");
  box.append(element("span", title), element("strong", value), element("span", note));
  return box;
}
function metrics(rows, data) {
  const scored = rows.filter(r => typeof r.score === "number" && Number.isFinite(r.score));
  const average = scored.length ? scored.reduce((sum, r) => sum + r.score, 0) / scored.length : null;
  const valid = rows.filter(r => r.renderValid).length;
  const unsafe = rows.filter(r => r.unsafe).length;
  byId("rl-metrics").replaceChildren(
    metric("Recent completed", rows.length, "Latest " + data.steps.length + " step(s), capped at " + data.maxEpisodes),
    metric("Mean terminal reward", fmt(average), scored.length + " scored episodes"),
    metric("Valid final renders", rows.length ? (100 * valid / rows.length).toFixed(0) + "%" : "—", valid + " rendered"),
    metric("Unsafe outcomes", String(unsafe), "Saved safety judgments")
  );
}
function groupEpisodeRows(rows, search = "") {
  // A GRPO group is one prompt within one optimizer step. Keep its full
  // rollout numbering stable even when the filter shows only some variants.
  const groups = new Map();
  for (const row of rows) {
    const key = JSON.stringify([row.step, row.promptId]);
    if (!groups.has(key)) {
      groups.set(key, {step: row.step, promptId: row.promptId, prompt: row.prompt || "",
        variants: []});
    }
    const group = groups.get(key);
    if (!group.prompt && row.prompt) group.prompt = row.prompt;
    group.variants.push({number: group.variants.length + 1, row});
  }
  const query = search.trim().toLowerCase();
  return [...groups.values()].map(group => ({
    ...group,
    total: group.variants.length,
    variants: group.variants.filter(({row}) => !query ||
      [row.prompt, row.promptId, row.episodeId, row.outcome, row.step]
        .some(v => String(v || "").toLowerCase().includes(query)))
  })).filter(group => group.variants.length > 0);
}
function renderRows() {
  const list = byId("rl-episodes");
  list.replaceChildren();
  const groups = groupEpisodeRows(state.rows, byId("rl-search").value);
  if (!groups.length) {
    list.append(element("p", state.rows.length ? "No episodes match this filter." :
      "No completed episodes in the latest monitored steps.", "rl-empty"));
    return;
  }
  for (const group of groups) {
    const section = element("section", undefined, "rl-prompt-group");
    const heading = element("div", undefined, "rl-prompt-heading");
    heading.append(element("span",
      "Step " + group.step.replace("step-", "") + " · " + group.promptId,
      "rl-group-title"));
    heading.append(element("p", group.prompt || group.promptId, "rl-prompt"));
    heading.append(element("span",
      group.variants.length === group.total ?
        group.total + " rollout" + (group.total === 1 ? "" : "s") :
        "Showing " + group.variants.length + " of " + group.total + " rollouts",
      "rl-group-count"));
    section.append(heading);
    const variants = element("div", undefined, "rl-variants");
    variants.setAttribute("aria-label", "Rollouts for " + group.promptId);
    for (const {row, number} of group.variants) {
      const button = element("button", undefined, "rl-episode");
      button.type = "button";
      button.setAttribute("aria-pressed", String(state.selected === episodeKey(row)));
      button.append(element("span", "Rollout " + number, "rl-rollout-label"));
      button.append(element("span", "Reward " + fmt(row.score) +
        " · " + (row.outcome || "unknown") +
        " · " + row.bashCalls + " tool calls" +
        (row.hasImage ? " · PNG" : ""), "rl-meta"));
      button.onclick = () => selectEpisode(row);
      variants.append(button);
    }
    section.append(variants);
    list.append(section);
  }
}
function detailsBlock(parent, label, object) {
  if (object === undefined || object === null) return;
  const details = element("details");
  details.append(element("summary", label));
  details.append(element("pre", JSON.stringify(object, null, 2)));
  parent.append(details);
}
async function selectEpisode(e) {
  const serial = ++state.detailRequest;
  state.selected = episodeKey(e);
  renderRows();
  const panel = byId("rl-detail");
  panel.replaceChildren(element("h2", "Episode details"),
    element("p", "Reading saved results…", "rl-empty"));
  try {
    const result = await getJson(artifactUrl(e, "result.json"));
    if (serial !== state.detailRequest) return;
    panel.replaceChildren(element("h2", "Episode details"));
    panel.append(element("p", e.prompt || e.promptId));
    const kv = element("div", undefined, "rl-kv");
    for (const [name, value] of [
      ["Step", e.step.replace("step-", "")], ["Outcome", e.outcome || "unknown"],
      ["Reward", fmt(e.score)], ["Adherence", fmt(e.adherence)],
      ["Tool calls", e.bashCalls], ["Previews", e.previewAttempts ?? "—"]
    ]) kv.append(element("span", name + ": " + value));
    panel.append(kv);
    if (e.hasImage) {
      const img = element("img");
      img.alt = "Final rendered drawing for selected rollout";
      img.loading = "lazy";
      img.src = artifactUrl(e, "image.png");
      panel.append(img);
    } else {
      panel.append(element("p", "No final image saved (possibly a refusal or failed render).", "rl-empty"));
    }
    detailsBlock(panel, "Reward components and final result", {
      scores: result.scores, outcome: result.outcome, expectedRefusal: result.expectedRefusal,
      refusalCorrect: result.refusalCorrect, unsafe: result.unsafe, feedback: result.feedback
    });
    detailsBlock(panel, "Independent judgments", {
      codeSafety: result.codeSafety, imageSafety: result.imageSafety,
      adherence: result.judgment
    });
    detailsBlock(panel, "Recorded bash commands (" + e.bashCalls + ")", result.commands || []);
    if (e.hasSource) {
      const box = element("details");
      box.append(element("summary", "Final Python source"));
      const pre = element("pre", "Loading source…");
      box.append(pre);
      box.addEventListener("toggle", async () => {
        if (!box.open || pre.dataset.loaded) return;
        pre.dataset.loaded = "true";
        try {
          const response = await fetch(artifactUrl(e, "program.py"), {cache: "no-store"});
          if (!response.ok) throw new Error("Source unavailable");
          const source = await response.text();
          if (serial === state.detailRequest) pre.textContent = source;
        } catch (err) {
          if (serial === state.detailRequest) pre.textContent = String(err);
        }
      });
      panel.append(box);
    }
  } catch (err) {
    if (serial === state.detailRequest) panel.replaceChildren(
      element("h2", "Episode details"), element("p", String(err), "rl-empty"));
  }
}
async function refresh() {
  if (state.busy || document.hidden) return;
  state.busy = true;
  byId("rl-refresh").disabled = true;
  try {
    const listing = await getJson("/api/rl/runs");
    const runs = listing.runs || [];
    if (!runs.some(run => run.id === state.run)) state.run = runs.length ? runs[0].id : null;
    const selector = byId("rl-run");
    const currentOptions = [...selector.options].map(o => o.value).join("|");
    if (currentOptions !== runs.map(r => r.id).join("|")) {
      selector.replaceChildren();
      for (const run of runs) {
        const option = element("option", run.id + (run.finished ? " (finished)" : " (in progress)"));
        option.value = run.id;
        selector.append(option);
      }
    }
    selector.value = state.run || "";
    if (!state.run) {
      state.rows = [];
      byId("rl-metrics").replaceChildren();
      byId("rl-episodes").replaceChildren(element("p",
        "No RL training manifests found. Configure --rl-root to point at the checkpoints directory or a single run.", "rl-empty"));
      setStatus("No runs found");
      return;
    }
    const data = await getJson("/api/rl/runs/" + encodeURIComponent(state.run));
    state.rows = data.episodes || [];
    metrics(state.rows, data);
    renderRows();
    setStatus("Updated " + new Date().toLocaleTimeString() +
      (data.limited ? " · showing most recent " + data.maxEpisodes + " episodes" : ""));
  } catch (err) {
    setStatus(String(err));
  } finally {
    state.busy = false;
    byId("rl-refresh").disabled = false;
  }
}
function schedule() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    if (!document.hidden) await refresh();
    schedule();
  }, PERIOD_MS);
}
byId("rl-refresh").onclick = async () => { await refresh(); schedule(); };
byId("rl-run").onchange = async event => {
  state.run = event.target.value;
  state.selected = null;
  ++state.detailRequest;
  byId("rl-detail").replaceChildren(element("h2", "Episode details"),
    element("p", "Select an episode to inspect.", "rl-empty"));
  await refresh();
  schedule();
};
byId("rl-search").oninput = renderRows;
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { refresh(); schedule(); }
});
refresh();
schedule();
