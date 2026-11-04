/* Saved evaluations only: browsing never invokes a model. */
(function () {
  "use strict";
  let roles = ["base", "sft"];
  const titles = {base: "Base", sft: "SFT", rank32: "LoRA · rank 32", rank64: "LoRA · rank 64", rank128: "LoRA · rank 128"}, pageSize = 8;
  const title = role => titles[role] || role;
  let data = null, page = 0, loadVersion = 0;
  const $ = id => document.getElementById("comparison-" + id);
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const both = pair => roles.every(role => pair.members[role].image_url);
  const scored = pair => roles.every(role => pair.members[role].scores?.adherence_score != null);
  const adherence = pair => both(pair) && scored(pair) ? roles.map(role => pair.members[role].scores.adherence_score) : null;
  const gain = pair => adherence(pair) ? Math.max(...adherence(pair)) - Math.min(...adherence(pair)) : null;
  const mean = (pairs, role, field) => pairs.length ? (pairs.reduce((sum, pair) => sum + pair.members[role].scores[field], 0) / pairs.length).toFixed(2) : "—";
  async function request(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not load saved comparison results. Try Refresh results.");
    return response.json();
  }
  function summary() {
    if (data.pending) {
      $("summary").replaceChildren(...roles.map(role => {
        const card = el("div", undefined, "comparison-stat");
        card.append(el("span", title(role)), el("strong", data.progress[role] + " / " + data.total), el("small", "Final outcomes saved · awaiting joint image scoring"));
        return card;
      }));
      return;
    }
    const paired = data.pairs.filter(pair => both(pair) && scored(pair));
    const stats = roles.map(role => {
      const wins = paired.filter(pair => pair.preferred === role).length;
      const available = data.pairs.filter(pair => pair.members[role].image_url).length;
      return [title(role), mean(paired, role, "adherence_score") + " / 4",
        "Adherence · visual quality " + mean(paired, role, "visual_quality_score") + " / 4 · preferred " + wins + "/" + paired.length + " · " + available + "/" + data.pairs.length + " images"];
    });
    $("summary").replaceChildren(...stats.map(([title, value, note]) => {
      const card = el("div", undefined, "comparison-stat");
      card.append(el("span", title), el("strong", value), el("small", note));
      return card;
    }));
  }
  function output(pair, role) {
    const member = pair.members[role], card = el("section", undefined, "comparison-output");
    card.setAttribute("aria-label", title(role) + " output");
    const heading = el("div", undefined, "comparison-model");
    heading.append(el("h3", title(role)));
    if (pair.preferred === role) heading.append(el("span", "Luna’s pick"));
    card.append(heading, el("small", data.models[role], "comparison-model-name"));
    if (member.image_url) {
      const button = el("button", undefined, "comparison-image"); button.type = "button";
      button.setAttribute("aria-label", "Enlarge " + title(role) + " image: " + pair.prompt);
      const img = el("img"); img.src = member.image_url; img.alt = title(role) + ": " + pair.prompt; img.loading = "lazy";
      button.append(img);
      img.addEventListener("error", () => { button.replaceWith(el("div", "Saved image unavailable. Refresh results to try again.", "comparison-missing")); });
      button.addEventListener("click", () => ArtsyUI.openImage({src: member.image_url, alt: img.alt, returnFocus: button}));
      card.append(button);
    } else {
      const missing = el("div", undefined, "comparison-missing");
      missing.append(el("strong", "No final image"), el("small", "Outcome: " + member.outcome));
      card.append(missing);
    }
    const scores = el("div", undefined, "comparison-scores");
    for (const [field, title] of [["adherence_score", "Prompt adherence"], ["visual_quality_score", "Visual quality"]]) {
      const value = member.scores?.[field];
      const score = el("div", undefined, "comparison-score");
      score.append(el("b", value == null ? "—" : value + " / 4"), el("span", title));
      scores.append(score);
    }
    card.append(scores, el("p", member.scores?.comments || "Awaiting a saved judgment.", "comparison-comments"));
    return card;
  }
  function render() {
    if (!data) return;
    const search = $("search").value.trim().toLowerCase(), filter = $("filter").value;
    const pairs = data.pairs.filter(pair => {
      const text = [pair.id, pair.prompt, pair.comparison, ...roles.map(role => pair.members[role].scores?.comments)].join(" ").toLowerCase();
      const matches = filter === "all" || filter === "both" && both(pair) || filter === "missing" && !both(pair) || pair.preferred === filter;
      return matches && (!search || text.includes(search));
    });
    const sort = $("sort").value;
    if (sort !== "original") pairs.sort((a, b) => {
      const ga = gain(a), gb = gain(b);
      if (ga === null || gb === null) return ga === gb ? 0 : ga === null ? 1 : -1;
      return sort === "gain" ? gb - ga : Math.min(...adherence(a)) - Math.min(...adherence(b));
    });
    const pages = Math.max(1, Math.ceil(pairs.length / pageSize)); page = Math.min(page, pages - 1);
    $("count").textContent = pairs.length + " of " + data.pairs.length + " prompts";
    $("page").textContent = "Page " + (page + 1) + " of " + pages;
    $("prev").disabled = page === 0; $("next").disabled = page === pages - 1;
    const nodes = pairs.slice(page * pageSize, (page + 1) * pageSize).map(pair => {
      const section = el("article", undefined, "comparison-pair");
      const header = el("header", undefined, "comparison-prompt");
      header.append(el("h2", pair.prompt), el("small", pair.id));
      const columns = el("div", undefined, "comparison-columns"); columns.append(...roles.map(role => output(pair, role)));
      columns.style.setProperty("--comparison-models", roles.length);
      const verdict = el("footer", undefined, "comparison-verdict");
      const preferred = pair.preferred ? ({tie: "Tie", neither: "No output"}[pair.preferred] || title(pair.preferred)) : "Awaiting judgment";
      verdict.append(el("strong", "Preferred: " + preferred), el("p", pair.comparison || "Judgment not yet saved."));
      if (pair.labels) verdict.append(el("small", "Labels in Luna’s comments: " + Object.entries(pair.labels).map(([label, role]) => label + " = " + title(role)).join(" · ")));
      section.append(header, columns, verdict);
      return section;
    });
    $("pairs").replaceChildren(...(nodes.length ? nodes : [el("p", "No prompts match these filters.", "comparison-empty")]));
  }
  async function load() {
    const version = ++loadVersion, id = $("run").value;
    data = null; $("browser").hidden = true; $("method").hidden = true; $("summary").replaceChildren();
    if (!id) return;
    $("status").textContent = "Loading saved judgments…";
    try {
      const result = await request("/api/comparisons/" + encodeURIComponent(id));
      if (version !== loadVersion) return;
      data = result; roles = Object.keys(data.models); page = 0;
      if (data.pending) {
        summary();
        $("status").textContent = data.error ? "Pipeline failed: " + data.error : "Generating eval outputs for all models. Luna will score all outputs together automatically when generation completes. Refresh results for progress.";
        history.replaceState(null, "", "#comparison=" + encodeURIComponent(id));
        return;
      }
      const options = [["all", "All prompts"], ["both", "All final images"], ...roles.map(role => [role, title(role) + " preferred"]), ["tie", "Ties"], ["missing", "Missing outputs"]];
      $("filter").replaceChildren(...options.map(([value, label]) => { const option = el("option", label); option.value = value; return option; }));
      summary(); render();
      $("browser").hidden = false; $("method").hidden = false;
      const missing = data.pairs.filter(pair => !both(pair)).length;
      $("status").textContent = "Judge: " + data.judge + " · " + data.pairs.filter(pair => pair.preferred !== null).length + "/" + data.pairs.length + " reviewed · " + missing + " prompts with missing outputs · " + roles.length + " models";
      history.replaceState(null, "", "#comparison=" + encodeURIComponent(id));
    } catch (error) {
      if (version === loadVersion) $("status").textContent = error.message;
    }
  }
  async function init() {
    $("run").addEventListener("change", load);
    $("refresh").addEventListener("click", refresh);
    for (const id of ["search", "filter", "sort"]) $(id).addEventListener(id === "search" ? "input" : "change", () => { page = 0; render(); });
    for (const [id, delta] of [["prev", -1], ["next", 1]]) $(id).addEventListener("click", () => { page += delta; render(); $("browser").scrollIntoView({block: "start"}); });
    await refresh();
  }
  async function refresh() {
    $("refresh").disabled = true;
    try {
      const selected = $("run").value || new URLSearchParams(location.hash.slice(1)).get("comparison");
      const list = await request("/api/comparisons");
      $("run").replaceChildren(...list.comparisons.map(item => {
        const option = el("option", item.id.replaceAll("-", " ")); option.value = item.id; return option;
      }));
      if (list.comparisons.some(item => item.id === selected)) $("run").value = selected;
      $("run").disabled = !list.comparisons.length;
      if (list.comparisons.length) await load();
      else { data = null; $("summary").replaceChildren(); $("browser").hidden = true; $("method").hidden = true; $("status").textContent = "No paired judgments saved yet. Completed paired evaluations will appear here."; }
    } catch (error) { $("status").textContent = error.message; }
    finally { $("refresh").disabled = false; }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {both, scored, gain, mean};
  else init();
})();
