/* Tests for grouping GRPO samples without changing the trainer or API. */
"use strict";
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "rl-monitor.js"), "utf8");

function createNode(tag = "div") {
  return {
    tag, children: [], attributes: {}, textContent: "", className: "", value: "",
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    setAttribute(name, value) { this.attributes[name] = value; }
  };
}
function fixture(fetchImpl = () => { throw new Error("Unexpected network request"); }) {
  const nodes = new Map();
  for (const name of ["rl-episodes", "rl-search", "rl-run", "rl-refresh", "rl-detail",
    "rl-prompt-search", "rl-prompt-category", "rl-prompts-status", "rl-prompt-total",
    "rl-prompt-list", "rl-prompt-pager", "rl-prompt-prev", "rl-prompt-next", "rl-prompt-page"]) {
    nodes.set(name, createNode());
  }
  const document = {
    hidden: true, // No network traffic or timers from the browser startup.
    createElement: tag => createNode(tag),
    getElementById: id => nodes.get(id),
    addEventListener() {}
  };
  const result = vm.runInNewContext(script + "\n;({state, groupEpisodeRows, renderRows, renderPrompts, loadPrompts})", {
    document,
    clearTimeout() {},
    setTimeout() {},
    fetch: fetchImpl
  });
  return {nodes, ...result};
}
function row(step, promptId, episodeId, values = {}) {
  return {
    step, promptId, episodeId, prompt: "Draw a blue circle",
    score: 0.75, adherence: 0.75, outcome: "submitted",
    bashCalls: 2, hasImage: true, ...values
  };
}
function allNodes(node, predicate) {
  return [node, ...node.children.flatMap(child => allNodes(child, predicate))].filter(predicate);
}

test("one prompt header per step and prompt, with each rollout preserved", () => {
  const {nodes, state, groupEpisodeRows, renderRows} = fixture();
  const rows = [
    row("step-000001", "one", "a"),
    row("step-000001", "one", "b", {score: 0.25}),
    row("step-000001", "one", "c", {score: 0.0, outcome: "refused"}),
    row("step-000001", "one", "d", {score: -1}),
    row("step-000001", "two", "e", {prompt: "Draw a green square"}),
    row("step-000002", "one", "f")
  ];
  state.rows = rows;
  const groups = groupEpisodeRows(rows);
  assert.equal(groups.length, 3);
  assert.deepEqual(Array.from(groups, g => g.variants.length), [4, 1, 1]);
  renderRows();
  const list = nodes.get("rl-episodes");
  assert.equal(allNodes(list, node => node.className === "rl-prompt-group").length, 3);
  const promptElements = allNodes(list, node => node.className === "rl-prompt");
  assert.equal(promptElements.filter(node => node.textContent === "Draw a blue circle").length, 2);
  assert.equal(promptElements.filter(node => node.textContent === "Draw a green square").length, 1);
  const buttons = allNodes(list, node => node.className === "rl-episode");
  assert.equal(buttons.length, 6);
  assert.ok(buttons.every(button => typeof button.onclick === "function"));
  const labels = allNodes(list, node => node.className === "rl-rollout-label");
  assert.deepEqual(Array.from(labels, node => node.textContent).slice(0, 4),
    ["Rollout 1", "Rollout 2", "Rollout 3", "Rollout 4"]);
});

test("filtered rollouts retain original numbering and show prompt only once", () => {
  const {nodes, state, groupEpisodeRows, renderRows} = fixture();
  state.rows = [
    row("step-000001", "circle", "a", {outcome: "submitted"}),
    row("step-000001", "circle", "b", {outcome: "refused"}),
    row("step-000001", "circle", "c", {outcome: "submitted"})
  ];
  nodes.get("rl-search").value = "refused";
  const groups = groupEpisodeRows(state.rows, "refused");
  assert.equal(groups.length, 1);
  assert.equal(groups[0].total, 3);
  assert.equal(groups[0].variants[0].number, 2);
  renderRows();
  const list = nodes.get("rl-episodes");
  assert.equal(allNodes(list, node => node.className === "rl-prompt").length, 1);
  assert.deepEqual(Array.from(allNodes(list, node => node.className === "rl-rollout-label"),
    node => node.textContent), ["Rollout 2"]);
  assert.ok(allNodes(list, node => node.textContent === "Showing 1 of 3 rollouts").length === 1);
});

test("empty and unmatched searches remain informative", () => {
  const {nodes, state, renderRows} = fixture();
  state.rows = [];
  renderRows();
  assert.match(nodes.get("rl-episodes").children[0].textContent, /No completed episodes/);
  state.rows = [row("step-000001", "circle", "a")];
  nodes.get("rl-search").value = "no-match";
  renderRows();
  assert.match(nodes.get("rl-episodes").children[0].textContent, /No episodes match/);
});


test("each rollout displays a lazy image and both scores without repeating prompts", () => {
  const {nodes, state, renderRows} = fixture();
  state.run = "grpo-v4";
  state.rows = [
    row("step-000001", "circle", "a", {score: 0.5, adherence: 0.8}),
    row("step-000001", "circle", "b", {score: 1, adherence: 1}),
    row("step-000001", "circle", "c", {score: -1, adherence: 0}),
    row("step-000001", "circle", "d", {score: 0, adherence: 0}),
  ];
  renderRows();
  const list = nodes.get("rl-episodes");
  assert.equal(allNodes(list, node => node.className === "rl-prompt").length, 1);
  assert.equal(allNodes(list, node => node.className === "rl-episode").length, 4);
  const images = allNodes(list, node => node.tag === "img");
  assert.equal(images.length, 4);
  images.forEach((image, i) => {
    assert.equal(image.loading, "lazy");
    assert.equal(image.decoding, "async");
    assert.equal(image.alt, "Final image for rollout " + (i + 1));
  });
  assert.equal(images[0].src,
    "/api/rl/artifacts/grpo-v4/step-000001/circle/a/image.png");
  const rewards = allNodes(list, node => node.className === "rl-reward");
  assert.deepEqual(Array.from(rewards, node => node.textContent),
    ["Reward 0.500", "Reward 1.000", "Reward -1.000", "Reward 0.000"]);
  const adherence = allNodes(list, node => node.className === "rl-adherence");
  assert.deepEqual(Array.from(adherence, node => node.textContent),
    ["Adherence 0.800", "Adherence 1.000", "Adherence 0.000", "Adherence 0.000"]);
});

test("refusals and failed renders show placeholders without requesting an image", () => {
  const {nodes, state, renderRows} = fixture();
  state.run = "grpo-v4";
  state.rows = [
    row("step-000001", "circle", "a", {outcome: "refused", hasImage: false, score: 1}),
    row("step-000001", "circle", "b", {outcome: "incomplete", hasImage: false, score: 0})
  ];
  renderRows();
  const list = nodes.get("rl-episodes");
  assert.equal(allNodes(list, node => node.tag === "img").length, 0);
  assert.deepEqual(Array.from(allNodes(list, node => node.className === "rl-no-image"),
    node => node.textContent), ["Refused — no image", "No final image"]);
  assert.equal(allNodes(list, node => node.className === "rl-episode").length, 2);
});

test("unchanged poll data and tile selections keep thumbnail elements mounted", async () => {
  const {nodes, state, renderRows} = fixture(async () => ({
    ok: true, json: async () => ({scores: {reward: 0.75}})
  }));
  state.run = "grpo-v4";
  state.rows = [row("step-000001", "circle", "a")];
  renderRows();
  const gallery = nodes.get("rl-episodes");
  const image = allNodes(gallery, node => node.tag === "img")[0];
  renderRows(); // A 30s poll returned identical results.
  assert.equal(allNodes(gallery, node => node.tag === "img")[0], image);
  const button = allNodes(gallery, node => node.className === "rl-episode")[0];
  await button.onclick(); // Selecting details must not rebuild the gallery.
  assert.equal(allNodes(gallery, node => node.tag === "img")[0], image);
  assert.equal(button.attributes["aria-pressed"], "true");
});

test("all prompts are searchable and paginated independently of the rollout cap", () => {
  const {nodes, state, renderPrompts} = fixture();
  state.prompts = Array.from({length: 103}, (_, index) => ({
    index: index + 1,
    id: "prompt-" + index,
    prompt: index === 102 ? "A unique needle at the end <script>alert(1)</script>" :
      "Draw numbered picture " + index,
    category: index % 2 ? "benign_boundary" : "drawing",
    difficulty: "medium", family: "grids", skills: ["composition"]
  }));
  renderPrompts();
  const list = nodes.get("rl-prompt-list");
  assert.equal(allNodes(list, n => n.className === "rl-training-prompt").length, 50);
  assert.equal(nodes.get("rl-prompt-total").textContent, "(103)");
  assert.equal(nodes.get("rl-prompt-page").textContent, "Page 1 of 3");
  assert.equal(nodes.get("rl-prompt-prev").disabled, true);
  nodes.get("rl-prompt-next").onclick();
  assert.equal(nodes.get("rl-prompt-page").textContent, "Page 2 of 3");
  assert.equal(allNodes(list, n => n.className === "rl-training-prompt").length, 50);
  nodes.get("rl-prompt-next").onclick();
  assert.equal(allNodes(list, n => n.className === "rl-training-prompt").length, 3);
  nodes.get("rl-prompt-search").value = "needle";
  nodes.get("rl-prompt-search").oninput();
  assert.equal(allNodes(list, n => n.className === "rl-training-prompt").length, 1);
  assert.equal(nodes.get("rl-prompt-page").textContent, "Page 1 of 1");
  assert.match(allNodes(list, n => n.className === "rl-prompt-full")[0].textContent,
    /<script>alert\(1\)<\/script>/); // Treated only as text, never as HTML.
  nodes.get("rl-prompt-category").value = "benign_boundary";
  nodes.get("rl-prompt-category").onchange();
  assert.match(nodes.get("rl-prompts-status").textContent, /No training prompts match/);
});

test("the full prompt endpoint is loaded for the selected run", async () => {
  const urls = [];
  const {state, nodes, loadPrompts} = fixture(async url => {
    urls.push(url);
    return {ok: true, json: async () => ({
      run: "grpo-v4", available: true, prompts: [
        {index: 1, id: "a", prompt: "Draw a circle", category: "drawing", skills: []},
        {index: 2, id: "b", prompt: "Refuse harmful code", category: "malicious_code", skills: []}
      ]
    })};
  });
  state.run = "grpo-v4";
  await loadPrompts("grpo-v4");
  assert.deepEqual(urls, ["/api/rl/runs/grpo-v4/prompts"]);
  assert.equal(state.prompts.length, 2);
  assert.equal(nodes.get("rl-prompt-category").children.length, 3);
  assert.equal(allNodes(nodes.get("rl-prompt-list"),
    n => n.className === "rl-training-prompt").length, 2);
  state.run = "other";
  await loadPrompts("other");
  assert.deepEqual(urls, ["/api/rl/runs/grpo-v4/prompts", "/api/rl/runs/other/prompts"]);
});

test("unavailable prompt data shows a dataset repair instruction instead of rollout-only data", async () => {
  const {state, nodes, loadPrompts} = fixture(async () => ({
    ok: true, json: async () => ({
      available: false, total: 100, prompts: [],
      message: "Start with --rl-dataset PATH"
    })
  }));
  state.run = "grpo-v4";
  await loadPrompts("grpo-v4");
  assert.equal(nodes.get("rl-prompt-list").children.length, 0);
  assert.match(nodes.get("rl-prompts-status").textContent, /--rl-dataset/);
});
