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
function fixture() {
  const nodes = new Map();
  for (const name of ["rl-episodes", "rl-search", "rl-run", "rl-refresh", "rl-detail"]) {
    nodes.set(name, createNode());
  }
  const document = {
    hidden: true, // No network traffic or timers from the browser startup.
    createElement: tag => createNode(tag),
    getElementById: id => nodes.get(id),
    addEventListener() {}
  };
  const result = vm.runInNewContext(script + "\n;({state, groupEpisodeRows, renderRows})", {
    document,
    clearTimeout() {},
    setTimeout() {},
    fetch() { throw new Error("Grouping must not fetch data"); }
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
