// The editor page: watch the cut, fix its cuts by hand.
// What plays is the cut as it is now, put together in the browser from small copies of its
// files (player.js), so an edit can be watched the moment it is made. The exact render is one
// button away, for when what will be delivered has to be seen to the frame. Only while a cut is
// being dragged does the viewer show the source, because that is the one moment the picture
// outside the cut matters. Every change goes to the server as the same edit the AI would make.
"use strict";

(() => {
  const num = Number;
  const LANE = { base: 70, other: 46, captions: 34 };
  const MIN_CLIP = 0.2;
  const POLL_MS = 3000;

  let E = null; // the open editor's state; null when no editor is open

  const fmt = (seconds) => {
    const s = Math.max(0, seconds || 0);
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
  };
  const fps = () => (E?.project ? E.project.fps_num / E.project.fps_den : 30);
  const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
  const timecode = (seconds) => {
    const rate = Math.round(fps());
    const frames = Math.max(0, Math.round((seconds || 0) * fps()));
    const f = frames % rate, total = Math.floor(frames / rate);
    const pad = (value) => String(value).padStart(2, "0");
    return `${pad(Math.floor(total / 3600))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}:${pad(f)}`;
  };
  const clipLength = (clip) => (num(clip.source_range.end) - num(clip.source_range.start)) / (clip.speed || 1);
  const clipEnd = (clip) => num(clip.timeline_in) + clipLength(clip);
  const assetName = (id) => E.project.assets[id]?.name || id;

  // ------------------------------------------------------------------ server

  async function loadProject() {
    E.project = await api(`/api/projects/${encodeURIComponent(E.id)}`);
    E.captions = (await api(`/api/projects/${encodeURIComponent(E.id)}/captions`)).captions;
    await loadComments();
    E.ownVersion = E.project.version;
    if (E.selectedCaption && !E.captions.some((cue) => cue.cue_id === E.selectedCaption)) E.selectedCaption = null;
    if (E.selected && !findClip(E.selected.track, E.selected.clip)) E.selected = null;
  }

  // What the user said about moments of the cut, open and closed, where each is on the cut now.
  async function loadComments() {
    try {
      const listed = await api(`/api/projects/${encodeURIComponent(E.id)}/comments`);
      const seen = JSON.stringify(listed.comments);
      const changed = seen !== E.commentsSeen;
      E.comments = listed.comments;
      E.commentsSeen = seen;
      return changed;
    } catch {
      E.comments = E.comments || [];
      return false;
    }
  }

  // Nothing changes while the cut is playing: a cut moving under the picture being watched
  // is how a viewer loses track of what they just saw.
  const playing = () => Boolean(E) && !clock().paused;

  function lockedWhilePlaying() {
    if (E.viewing) {
      toast(T.versions.viewingLocked, "error");
      return true;
    }
    if (!playing()) return false;
    const now = Date.now();
    if (!E.lockToastAt || now - E.lockToastAt > 2500) {
      E.lockToastAt = now;
      toast(T.editor.pauseFirst, "error");
    }
    return true;
  }

  async function edit(operations) {
    if (lockedWhilePlaying()) return false;
    E.busy = true;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(E.id)}/edits`, { expected_version: E.project.version, operations });
      E.ownVersion = result.new_version;
      await loadProject();
      drawAll();
      refreshPlayback();
      return true;
    } catch (error) {
      if (/version conflict/.test(error.message)) {
        toast(T.editor.conflict, "error");
        await loadProject();
        drawAll();
        refreshPlayback();
      } else {
        toast(error.message, "error");
        drawAll();
      }
      return false;
    } finally {
      E.busy = false;
    }
  }

  async function undo() {
    if (lockedWhilePlaying()) return;
    try {
      await api(`/api/projects/${encodeURIComponent(E.id)}/undo`, {});
      await loadProject();
      drawAll();
      refreshPlayback();
      toast(T.editor.undone);
    } catch {
      toast(T.editor.nothingToUndo, "error");
    }
  }

  async function requestPreview() {
    try {
      E.previewState = await api(`/api/projects/${encodeURIComponent(E.id)}/preview`, {});
    } catch (error) {
      E.previewState = { error: error.message, status: "failed" };
    }
    showPreviewState();
    pollPreview();
  }

  function pollPreview() {
    clearTimeout(E.previewTimer);
    const state = E.previewState || {};
    if (state.status === "queued" || state.status === "running") {
      E.previewTimer = setTimeout(async () => {
        if (!E) return;
        try { E.previewState = await api(`/api/projects/${encodeURIComponent(E.id)}/preview`); } catch { /* next poll */ }
        showPreviewState();
        pollPreview();
      }, 700);
    }
  }

  function showPreviewState() {
    const state = E.previewState || {};
    const chip = $("[data-preview-chip]", E.root);
    let text = "", kind = "";
    const asked = E.wantExact;
    if (asked && state.empty) { text = T.editor.previewEmpty; kind = "warn"; }
    else if (asked && state.status === "queued") { text = T.editor.previewQueued; kind = "busy"; }
    else if (asked && state.status === "running") { text = T.editor.previewMaking(Math.round((state.progress || 0) * 100)); kind = "busy"; }
    else if (asked && (state.status === "failed" || state.error)) { text = `${T.editor.previewFailed}${state.error ? `：${state.error}` : ""}`; kind = "warn"; }
    else if (E.mode === "exact") { text = T.editor.exactShown; kind = "ok"; }
    chip.className = `state ${kind}`;
    chip.hidden = !text;
    chip.innerHTML = kind === "busy"
      ? `<span class="spinner"></span>${esc(text)}<span class="bar"><i style="width:${Math.round((state.progress || 0) * 100)}%"></i></span>`
      : esc(text);
    if (asked && state.url && state.version === E.project.version && state.status !== "queued" && state.status !== "running") {
      E.wantExact = false;
      swapPreview(state.url);
      showExact();
    }
    if (asked && (state.empty || state.status === "failed" || state.error)) E.wantExact = false;
    $("[data-exact]", E.root).hidden = E.mode === "exact";
    $("[data-exact]", E.root).disabled = kind === "busy";
    $("[data-live]", E.root).hidden = E.mode !== "exact";
  }

  function swapPreview(url) {
    const video = E.video;
    const at = video.currentTime || E.time || 0;
    const playing = !video.paused;
    E.shownUrl = url;
    E.shownVersion = E.previewState.version;
    video.src = url;
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(at, video.duration || at);
      if (playing) video.play().catch(() => {});
    }, { once: true });
  }

  // ------------------------------------------------------------------ versions

  // The page the next editor opens on, when a link on the version page leads to another video.
  let keptView = null;

  // The graph is read again when the cut has moved on, or when it is older than this.
  const GRAPH_FRESH_MS = 15000;

  async function loadGraph(force) {
    const stale = !E.graph || E.graph.project.version !== E.project.version || Date.now() - E.graphAt > GRAPH_FRESH_MS;
    if (!force && !stale) return E.graph;
    E.graph = await api(`/api/projects/${encodeURIComponent(E.id)}/versions`);
    E.graphAt = Date.now();
    return E.graph;
  }

  const markIcons = (marks) => marks.map((mark) => `<span class="vmark" title="${esc(T.versions.marks[mark])}">${T.versions.markIcons[mark]}</span>`).join("");

  function nodeHtml(node, { current, branch } = {}) {
    const chosen = E.chosenVersion?.commit && node.versions.some((version) => version.commit === E.chosenVersion.commit);
    const thumb = node.thumb ? `<img class="vthumb" loading="lazy" src="${thumbUrl(node.thumb.asset_id, node.thumb.t, 180)}" alt="">` : '<span class="vthumb"></span>';
    const many = node.versions.length > 1 ? ` · ${esc(T.versions.changes(node.versions.length))}` : "";
    const length = node.seconds ? ` · ${fmt(node.seconds)}` : "";
    const side = E.compare && !branch ? (node.versions.some((version) => version.commit === E.compare.a) ? "A"
      : node.versions.some((version) => version.commit === (E.compare.b || E.graph?.nodes[0]?.commit)) ? "B" : "") : "";
    return `<li class="vnode${chosen && !E.compare ? " sel" : ""}${side ? " side" : ""}${current ? " current" : ""}" data-node="${esc(node.commit)}"${branch ? ` data-branch="${esc(branch)}"` : ""}>
      <span class="rail"><i></i></span>${thumb}${side ? `<span class="vside">${side}</span>` : ""}
      <div class="vtext"><b>${markIcons(node.marks)}${esc(node.what)}</b>
        <span>${esc(node.by)} · ${esc(node.when)}${length}${many}${current ? ` · ${esc(T.versions.now)}` : ""}</span></div></li>`;
  }

  // The version page: the chosen version plays big on the left, the tree runs down the right,
  // and what can be done with the chosen one sits under the player.
  async function drawVersions() {
    if (!E || E.view !== "versions") return;
    const panel = $("[data-versions]", E.root);
    if (!E.graph) panel.innerHTML = `<p class="hint">${esc(T.versions.loading)}</p>`;
    let graph;
    try { graph = await loadGraph(false); } catch (error) { panel.innerHTML = `<p class="hint">${esc(error.message)}</p>`; return; }
    if (!E || E.view !== "versions") return;
    const nodes = graph.nodes;
    const parent = graph.branched_from;
    // Something is always chosen on this page: the cut as it is now, until another is clicked.
    if (!E.chosenVersion && nodes.length) E.chosenVersion = { commit: nodes[0].commit, node: nodes[0] };
    let list = "";
    nodes.forEach((node, index) => {
      list += nodeHtml(node, { current: index === 0 });
      // A branch runs beside the line from the dot it started at.
      for (const branch of graph.branches.filter((item) => node.versions.some((version) => version.commit === item.from))) {
        list += `<li class="vbranch"><span class="rail"><i></i></span><div class="vlane">
          <a href="#/project/${encodeURIComponent(branch.project_id)}" class="vbranch-name">${esc(T.versions.branchOf(branch.name || ""))}</a>
          <ol>${branch.nodes.slice(0, 3).map((item, at) => nodeHtml(item, { current: at === 0, branch: branch.project_id })).join("")}</ol>
          ${branch.nodes.length > 3 ? `<span class="hint">${esc(T.versions.more(branch.nodes.length - 3))}</span>` : ""}</div></li>`;
      }
    });
    const scrolled = panel.scrollTop;
    const total = nodes.reduce((sum, node) => sum + node.versions.length, 0);
    panel.innerHTML = `<div class="label">${esc(T.versions.treeTitle(total))}</div>
      ${parent ? `<p class="hint">${esc(T.versions.branchedFrom(parent.name || "", parent.what || ""))}
        <a href="#/project/${encodeURIComponent(parent.project_id)}" data-keep-view>${esc(T.versions.openParent)}</a></p>` : ""}
      <ol class="vgraph">${list || `<p class="hint">${esc(T.versions.none)}</p>`}</ol>
      <div class="vdisk" data-disk></div>`;
    panel.scrollTop = scrolled;
    panel.querySelectorAll("[data-node]").forEach((item) => {
      item.onclick = () => {
        if (E.compare && !item.dataset.branch) { compareWith(E.compare.a, item.dataset.node); return; }
        chooseVersion(item.dataset.branch || E.id, item.dataset.node);
      };
    });
    panel.querySelectorAll("[data-keep-view], .vbranch-name").forEach((link) => {
      link.addEventListener("click", () => { keptView = "versions"; });
    });
    if (E.compare) drawChanges($("[data-vdetail]", E.root));
    else drawVersionDetail($("[data-vdetail]", E.root));
    drawDisk($("[data-disk]", panel));
  }

  // Edit or versions: one page each, the same player in both.
  function setView(view) {
    if (!E || E.view === view) return;
    clock().pause();
    E.view = view;
    const editor = $(".editor", E.root);
    editor.classList.toggle("view-versions", view === "versions");
    E.root.querySelectorAll("[data-view]").forEach((button) => button.classList.toggle("on", button.dataset.view === view));
    if (E.compare) stopCompare(false);
    if (view === "edit") {
      // Editing is always of the cut as it is now.
      if (E.viewing) viewVersion(null);
      if (E.mode === "exact") showLive();
      drawInspector();
      drawLanes();
      drawRuler();
    } else {
      if (E.mode === "exact") showLive();
      drawVersions();
    }
    requestAnimationFrame(() => { if (E) fitStage(); });
    drawTime();
  }

  // ------------------------------------------------------------------ two versions side by side

  // Provisional (roadmap §13): a change is jumped to a frame inside where it starts — clip edges
  // are kept to the millisecond, and landing exactly on one can show the black between two clips
  // that only touch.
  const INSIDE = 0.02;
  // Provisional (roadmap §13): how far the right side may drift from where the left says it should
  // be before it is put back, and how often and how patiently a side not ready to play is asked again.
  const RESYNC_SECONDS = 0.25;
  const LOAD_TRIES = 30;
  const LOAD_RETRY_MS = 1500;

  // Left is A, right is B (null: the cut as it is). One clock, the left's; the right follows it by
  // what is on screen, and only one side is heard.
  async function compareWith(a, b) {
    if (b && b === E.graph?.nodes[0]?.commit) b = null;
    if (a === b || (!b && a === E.graph?.nodes[0]?.commit)) { toast(T.compare.sameVersion, "error"); return; }
    clock().pause();
    const fresh = !E.compare;
    if (fresh) {
      E.compare = { heard: "a" };
      const view = $("[data-compare-view]", E.root);
      E.compare.players = {
        a: Player.create($('[data-side="a"] [data-cmp-stage]', view), () => drawCompareBar()),
        b: Player.create($('[data-side="b"] [data-cmp-stage]', view), () => drawCompareBar()),
      };
      E.compare.players.b.mute(true);
      $(".editor", E.root).classList.add("comparing");
      wireCompare(view);
      E.compare.frame = requestAnimationFrame(compareTick);
    } else {
      E.compare.players.a.pause();
      E.compare.players.b.pause();
    }
    Object.assign(E.compare, { a, b, data: null, picks: {} });
    drawVersions();
    try {
      const query = new URLSearchParams({ a, ...(b ? { b } : {}) });
      E.compare.data = await api(`/api/projects/${encodeURIComponent(E.id)}/versions/compare?${query}`);
    } catch (error) { toast(error.message, "error"); stopCompare(); return; }
    if (!E?.compare) return;
    const view = $("[data-compare-view]", E.root);
    $('[data-side="a"] [data-label]', view).textContent = T.compare.label(E.compare.data.a.what);
    $('[data-side="b"] [data-label]', view).textContent = b ? T.compare.label(E.compare.data.b.what) : T.compare.now;
    view.classList.toggle("stacked", E.project.width > E.project.height);
    for (const side of ["a", "b"]) {
      const stage = $(`[data-side="${side}"] [data-cmp-stage]`, view);
      stage.style.aspectRatio = `${E.project.width} / ${E.project.height}`;
    }
    loadSide("a", a);
    loadSide("b", b);
    drawCompareBar();
    redrawVersions();
  }

  async function loadSide(side, commit, tries = 0) {
    if (!E?.compare) return;
    try {
      const at = commit ? `?commit=${encodeURIComponent(commit)}` : "";
      const described = await api(`/api/projects/${encodeURIComponent(E.id)}/playback${at}`);
      if (!E?.compare || (side === "a" ? E.compare.a : E.compare.b) !== commit) return;
      E.compare.players[side].load(described);
      drawCompareBar();
      if (described.waiting.length && tries < LOAD_TRIES) setTimeout(() => loadSide(side, commit, tries + 1), LOAD_RETRY_MS * (1 + tries / 3));
    } catch { if (tries < LOAD_TRIES) setTimeout(() => loadSide(side, commit, tries + 1), LOAD_RETRY_MS * (1 + tries / 3)); }
  }

  function stopCompare(redraw = true) {
    if (!E?.compare) return;
    cancelAnimationFrame(E.compare.frame);
    E.compare.players.a.destroy();
    E.compare.players.b.destroy();
    E.compare = null;
    $(".editor", E.root).classList.remove("comparing");
    if (redraw) { drawVersions(); requestAnimationFrame(() => { if (E) fitStage(); }); }
  }

  // Where the right side should be when the left is at a second: the same footage, where it is there.
  function rightAt(seconds) {
    for (const [aStart, aEnd, bStart, bEnd] of E.compare.data?.aligned || []) {
      if (seconds >= aStart && seconds < aEnd) return bStart + (seconds - aStart) * ((bEnd - bStart) / Math.max(0.001, aEnd - aStart));
    }
    return null;
  }

  function compareTick() {
    if (!E?.compare) return;
    const { a, b } = E.compare.players;
    if (!a.paused) {
      const wanted = rightAt(a.time);
      // Only where the same footage is in both; elsewhere the right side runs on by itself.
      if (wanted !== null && Math.abs(b.time - wanted) > RESYNC_SECONDS) b.seek(wanted);
      drawCompareTime();
    }
    E.compare.frame = requestAnimationFrame(compareTick);
  }

  function seekCompare(aAt, bAt) {
    const { a, b } = E.compare.players;
    if (aAt !== null && aAt !== undefined) a.seek(aAt + INSIDE);
    const wanted = bAt !== null && bAt !== undefined ? bAt + INSIDE : rightAt(a.time);
    if (wanted !== null && wanted !== undefined) b.seek(wanted);
    drawCompareTime();
  }

  function toggleCompare() {
    const { a, b } = E.compare.players;
    if (!a.ready) return;
    if (a.paused) { a.play(); if (b.ready) b.play(); } else { a.pause(); b.pause(); }
  }

  function hear(side) {
    E.compare.heard = side;
    E.compare.players.a.mute(side !== "a");
    E.compare.players.b.mute(side !== "b");
    drawCompareBar();
  }

  function drawCompareTime() {
    const view = $("[data-compare-view]", E.root);
    const { a, b } = E.compare.players;
    const lengths = E.compare.data?.length || [a.duration, b.duration];
    $("[data-cmp-time]", view).textContent = `${fmt(a.time)} / ${fmt(b.time)} · ${T.compare.lengths(fmt(lengths[0]), fmt(lengths[1]))}`;
    const scrub = $("[data-cmp-scrub]", view);
    if (!E.compare.scrubbing) scrub.value = a.duration ? String(Math.round((a.time / a.duration) * 1000)) : "0";
  }

  function drawCompareBar() {
    if (!E?.compare) return;
    const view = $("[data-compare-view]", E.root);
    $("[data-cmp-play]", view).innerHTML = svg(E.compare.players.a.paused ? "play" : "pause", 18);
    for (const side of ["a", "b"]) {
      $(`[data-side="${side}"]`, view).classList.toggle("heard", E.compare.heard === side);
    }
    drawCompareTime();
  }

  function wireCompare(view) {
    $("[data-cmp-play]", view).onclick = toggleCompare;
    $("[data-cmp-close]", view).onclick = () => stopCompare();
    $("[data-cmp-swap]", view).onclick = () => {
      // B as it is now has no commit of its own to put on the left: the newest version stands in.
      const now = E.graph.nodes[0]?.commit;
      compareWith(E.compare.b || now, E.compare.a);
    };
    for (const side of ["a", "b"]) $(`[data-side="${side}"]`, view).onclick = () => hear(side);
    const scrub = $("[data-cmp-scrub]", view);
    scrub.oninput = () => {
      E.compare.scrubbing = true;
      E.compare.players.a.pause();
      E.compare.players.b.pause();
      seekCompare((Number(scrub.value) / 1000) * E.compare.players.a.duration, null);
    };
    scrub.onchange = () => { if (E?.compare) E.compare.scrubbing = false; };
  }

  // What changed, part by part; a click takes both sides there. Each part that differs, and the
  // video's songs and captions, can be taken from either side into a new version.
  function drawChanges(box) {
    const data = E.compare?.data;
    if (!data) { box.innerHTML = `<p class="hint">${esc(T.versions.loading)}</p>`; return; }
    const picks = E.compare.picks;
    const at = (change) => `data-a="${change.a_at ?? ""}" data-b="${change.b_at ?? ""}"`;
    const pick = (key) => `<span class="cpick" data-pick="${esc(key)}">
      <button class="${picks[key] === "a" ? "on" : ""}" data-side-pick="a">${esc(T.compare.useLeft)}</button><button class="${picks[key] === "a" ? "" : "on"}" data-side-pick="b">${esc(T.compare.useRight)}</button></span>`;
    const parts = data.parts.map((part) => `<li class="cpart${part.same ? " same" : ""}${picks[part.name] === "a" ? " left" : ""}">
      <div class="cpart-head"><b>${esc(part.name)}</b>${part.same ? `<span class="hint">${esc(T.compare.same)}</span>` : part.order ? "" : pick(part.name)}</div>
      ${part.changes.length ? `<ul>${part.changes.map((change) => `<li><button class="cchange" ${at(change)}>${esc(change.what)}</button></li>`).join("")}</ul>` : ""}</li>`).join("");
    const whole = data.whole.map((change) => `<li class="cpart${picks[change.key] === "a" ? " left" : ""}"><div class="cpart-head"><button class="cchange" ${at(change)}>${esc(change.what)}</button>
      ${["music", "captions", "caption_style", "frame"].includes(change.key) ? pick(change.key) : ""}</div></li>`).join("");
    const unchanged = data.parts.every((part) => part.same) && !data.whole.length;
    const taken = Object.keys(picks).filter((key) => picks[key] === "a");
    box.innerHTML = `<div class="cmp-head"><h2>${esc(T.compare.title)}</h2><span class="hint">${esc(T.compare.hint)}</span></div>
      ${unchanged ? `<p class="hint">${esc(T.compare.identical)}</p>` : `<ol class="cparts">${parts}${whole}</ol>
      <div class="cmerge"><span class="hint">${esc(taken.length ? T.compare.willTake(taken.map((key) => T.compare.keys[key] || key).join("、")) : T.compare.pickHint)}</span>
        <button class="btn primary" data-merge ${taken.length ? "" : "disabled"}>${esc(T.compare.merge)}</button></div>`}`;
    box.querySelectorAll("[data-pick]").forEach((group) => {
      group.querySelectorAll("[data-side-pick]").forEach((button) => {
        button.onclick = () => {
          if (button.dataset.sidePick === "a") picks[group.dataset.pick] = "a";
          else delete picks[group.dataset.pick];
          drawChanges(box);
        };
      });
    });
    const mergeButton = $("[data-merge]", box);
    if (mergeButton) mergeButton.onclick = mergeCompared;
    box.querySelectorAll("[data-a]").forEach((button) => {
      button.onclick = () => {
        const value = (raw) => (raw === "" ? null : Number(raw));
        E.compare.players.a.pause();
        E.compare.players.b.pause();
        seekCompare(value(button.dataset.a), value(button.dataset.b));
      };
    });
  }

  // The editor's own changes are written to the history just after it answers, so a version it has
  // just made can take a moment to appear: read the graph until the newest version is another one.
  async function graphAfter(head) {
    for (let tries = 0; tries < 12; tries += 1) {
      await loadGraph(true);
      if (!E || E.graph.nodes[0]?.commit !== head) return;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }

  // The parts picked from the left made into one cut with the rest of the right, as a new version.
  async function mergeCompared() {
    const { a, b, picks, data } = E.compare;
    const taken = Object.keys(picks).filter((key) => picks[key] === "a").map((key) => T.compare.keys[key] || key);
    const ok = await confirmBox(T.compare.mergeTitle, T.compare.mergeBody(taken.join("、"), data.a.what, b ? data.b.what : T.compare.now),
      T.compare.merge);
    if (!ok) return;
    let result;
    try {
      result = await api(`/api/projects/${encodeURIComponent(E.id)}/versions/merge`,
        { a, b, picks, expected_version: E.project.version });
    } catch (error) {
      toast(/version conflict/.test(error.message) ? T.editor.conflict : error.message, "error");
      return;
    }
    const head = E.graph.nodes[0]?.commit;
    stopCompare(false);
    E.viewing = null;
    E.chosenVersion = null;
    $(".editor", E.root).classList.remove("viewing-old");
    await loadProject();
    await graphAfter(head);
    drawAll();
    refreshPlayback();
    drawVersions();
    toast(result.plan ? `${T.compare.merged} ${result.plan}` : T.compare.merged);
  }

  function redrawVersions() {
    if (E?.view === "versions") drawVersions();
  }

  // What this video takes on disk, and the one button that lets its other versions go.
  async function drawDisk(box) {
    let disk;
    try { disk = await api(`/api/projects/${encodeURIComponent(E.id)}/storage`); } catch { return; }
    if (!E || !box.isConnected) return;
    const size = (megabytes) => (megabytes >= 1000 ? `${(megabytes / 1000).toFixed(1)} GB` : `${megabytes.toFixed(megabytes < 10 ? 1 : 0)} MB`);
    box.innerHTML = `<div class="label">${esc(T.versions.diskTitle)}</div>
      <p class="hint">${esc(T.versions.disk(disk.versions, disk.exported, size(disk.outputs_megabytes), size(disk.cache_megabytes)))}</p>
      <button class="btn small danger" data-trim ${disk.freeable_megabytes > 0 && E.project.name ? "" : "disabled"}>${esc(T.versions.trim(size(disk.freeable_megabytes)))}</button>
      ${E.project.name ? "" : `<p class="hint">${esc(T.versions.trimNeedsName)}</p>`}`;
    $("[data-trim]", box).onclick = async () => {
      const name = await confirmTyped(T.versions.trimTitle, T.versions.trimBody(disk.other_outputs.length), E.project.name || "", T.versions.trimAction);
      if (name === null) return;
      try {
        const done = await api(`/api/projects/${encodeURIComponent(E.id)}/storage/trim`, { name });
        toast(T.versions.trimmed(size(done.freed_megabytes)));
        await loadGraph(true);
        redrawVersions();
      } catch (error) { toast(error.message, "error"); }
    };
  }

  function findNode(projectId, commit) {
    const graph = E.graph;
    const nodes = projectId === E.id ? graph.nodes : graph.branches.find((branch) => branch.project_id === projectId)?.nodes || [];
    return nodes.find((node) => node.versions.some((version) => version.commit === commit));
  }

  function chooseVersion(projectId, commit) {
    if (projectId !== E.id) { keptView = "versions"; location.hash = `#/project/${encodeURIComponent(projectId)}`; return; }
    const node = findNode(projectId, commit);
    E.chosenVersion = { commit, node };
    // The newest version is the cut as it is; anything older is watched in its place.
    viewVersion(commit === E.graph.nodes[0]?.commit ? null : { commit, what: node.versions.find((version) => version.commit === commit).what });
  }

  function viewVersion(version) {
    clock().pause();
    E.viewing = version;
    if (!version && E.chosenVersion?.commit !== E.graph?.nodes[0]?.commit) E.chosenVersion = null;
    $(".editor", E.root).classList.toggle("viewing-old", Boolean(version));
    E.time = 0;
    refreshPlayback();
    redrawVersions();
  }

  function drawVersionDetail(box) {
    const chosen = E.chosenVersion;
    if (!chosen?.node) {
      box.innerHTML = `<p class="hint">${esc(T.versions.pick)}</p>`;
      return;
    }
    const { node, commit } = chosen;
    const version = node.versions.find((item) => item.commit === commit) || node.versions[0];
    const newest = commit === E.graph.nodes[0]?.commit;
    const isMarked = (mark) => commit === node.commit && node.marks.includes(mark);
    box.innerHTML = `${E.viewing ? `<div class="vviewing"><span>${esc(T.versions.viewing(E.viewing.what))}</span><button class="btn small" data-now>${esc(T.versions.backToNow)}</button></div>` : ""}
      <div class="vdetail-head"><div><h2>${markIcons(commit === node.commit ? node.marks : [])}${esc(version.what)}</h2>
      <p class="hint">${esc(node.by)} · ${esc(version.when)}${node.seconds ? ` · ${fmt(node.seconds)}` : ""}</p></div></div>
      ${node.versions.length > 1 ? `<ol class="vmembers">${node.versions.map((item) =>
        `<li class="${item.commit === commit ? "on" : ""}" data-member="${esc(item.commit)}"><span>${esc(item.when)}</span>${esc(item.what)}</li>`).join("")}</ol>` : ""}
      ${commit === node.commit && node.outputs.length ? `<div class="vouts">${node.outputs.map((output) =>
        `<button class="btn small" data-reveal-output="${esc(output.job_id)}" title="${esc(T.versions.revealOutput)}">📤 ${esc(output.name)}</button>`).join("")}</div>` : ""}
      <div class="group stack-buttons">
        ${newest ? `<p class="hint">${esc(T.versions.isNow)}</p>` : `<button class="btn primary" data-restore>${esc(T.versions.restore)}</button>`}
        <button class="btn" data-branch-from>${esc(T.versions.branch)}</button>
        <button class="btn" data-compare-start>${esc(newest ? T.compare.startNow : T.compare.start)}</button>
        ${commit === node.commit ? `<div class="vmarks">
          <button class="btn small${isMarked("starred") ? " on" : ""}" data-mark="starred">${T.versions.markIcons.starred} ${esc(T.versions.marks.starred)}</button>
          <button class="btn small${isMarked("published") ? " on" : ""}" data-mark="published">${T.versions.markIcons.published} ${esc(T.versions.marks.published)}</button></div>` : ""}
      </div>`;
    const now = $("[data-now]", box);
    if (now) now.onclick = () => viewVersion(null);
    box.querySelectorAll("[data-member]").forEach((item) => { item.onclick = () => chooseVersion(E.id, item.dataset.member); });
    box.querySelectorAll("[data-reveal-output]").forEach((button) => {
      button.onclick = () => api("/api/reveal", { job_id: button.dataset.revealOutput }).catch((error) => toast(error.message, "error"));
    });
    const restore = $("[data-restore]", box);
    if (restore) restore.onclick = () => restoreVersion(commit, version.what);
    $("[data-branch-from]", box).onclick = () => branchFrom(commit);
    $("[data-compare-start]", box).onclick = () => {
      // An older version is compared with the cut as it is; the newest with the one before it.
      const nodes = E.graph.nodes;
      if (!newest) compareWith(commit, null);
      else if (nodes[1]) compareWith(nodes[1].commit, null);
      else toast(T.compare.nothing, "error");
    };
    box.querySelectorAll("[data-mark]").forEach((button) => {
      button.onclick = async () => {
        try {
          await api(`/api/projects/${encodeURIComponent(E.id)}/versions/mark`,
            { commit, mark: button.dataset.mark, on: !button.classList.contains("on") });
          await loadGraph(true);
          E.chosenVersion = { commit, node: findNode(E.id, commit) };
          redrawVersions();
        } catch (error) { toast(error.message, "error"); }
      };
    });
  }

  async function restoreVersion(commit, what) {
    let restored;
    try {
      restored = await api(`/api/projects/${encodeURIComponent(E.id)}/versions/restore`, { commit, expected_version: E.project.version });
    } catch (error) {
      toast(/version conflict/.test(error.message) ? T.editor.conflict : error.message, "error");
      return;
    }
    const head = E.graph?.nodes[0]?.commit;
    E.viewing = null;
    E.chosenVersion = null;
    $(".editor", E.root).classList.remove("viewing-old");
    await loadProject();
    await graphAfter(head);
    drawAll();
    refreshPlayback();
    redrawVersions();
    // A plan another video shares was copied for this one, which the user should know.
    toast(restored.plan_copied ? T.versions.restoredWithCopy(what) : T.versions.restored(what));
  }

  function branchFrom(commit) {
    modal(`<h2>${esc(T.versions.branchTitle)}</h2><p>${esc(T.versions.branchBody)}</p>
      <div class="field"><input class="input" data-name maxlength="120" value="${esc(T.versions.branchName(E.project.name || ""))}"></div>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-save>${esc(T.versions.branch)}</button></div>`,
    (box, close) => {
      const name = $("[data-name]", box);
      name.focus();
      name.select();
      $("[data-cancel]", box).onclick = close;
      const save = async () => {
        if (!name.value.trim()) { toast(T.newProject.nameRequired, "error"); return; }
        try {
          const made = await api(`/api/projects/${encodeURIComponent(E.id)}/versions/branch`, { commit, name: name.value.trim() });
          close();
          await loadGraph(true);
          redrawVersions();
          toast(T.versions.branched(made.name));
        } catch (error) { toast(error.message, "error"); }
      };
      $("[data-save]", box).onclick = save;
      name.addEventListener("keydown", (event) => { if (event.key === "Enter") save(); });
    });
  }

  // ------------------------------------------------------------------ live playback, or the exact render

  // What is playing: the cut put together live, or the render asked for with the button.
  const clock = () => (E.mode === "exact" ? E.exact : E.player);

  async function refreshPlayback() {
    clearTimeout(E.describeTimer);
    try {
      const at = E.viewing ? `?commit=${encodeURIComponent(E.viewing.commit)}` : "";
      const described = await api(`/api/projects/${encodeURIComponent(E.id)}/playback${at}`);
      if (!E) return;
      E.described = described;
      E.player.load(described);
      E.describeTries = 0;
    } catch { /* the next look will try again */ }
    if (!E) return;
    // The render shown no longer is this cut: back to what is.
    if (E.mode === "exact" && E.shownVersion !== E.project.version) showLive();
    showPlaybackState();
    const described = E.described;
    const unfinished = described && (described.waiting.length || (described.gain_db === null && described.duration > 0));
    if (!described || unfinished) {
      // Looks again until the copies are made; less often the longer they take.
      E.describeTries = (E.describeTries || 0) + 1;
      E.describeTimer = setTimeout(refreshPlayback, Math.min(10000, 1000 + E.describeTries * 500));
    }
  }

  function showPlaybackState() {
    const described = E.described;
    const chip = $("[data-live-chip]", E.root);
    const notice = $("[data-notice]", E.root);
    const empty = !described || described.duration <= 0;
    notice.hidden = !(E.mode === "live" && empty);
    notice.textContent = T.editor.previewEmpty;
    let text = "", kind = "";
    if (E.mode === "live" && described && !empty) {
      const files = new Set(described.waiting.filter((item) => !item.startsWith("sound:"))).size;
      if (files) { text = T.editor.livePreparing(files); kind = "busy"; }
      else if (described.waiting.length || described.gain_db === null) { text = T.editor.liveSound; kind = "busy"; }
    }
    chip.className = `state ${kind}`;
    chip.hidden = !text;
    chip.innerHTML = kind === "busy" ? `<span class="spinner"></span>${esc(text)}` : esc(text);
    const near = described && Object.entries(described.approximate).filter(([, value]) => value).map(([key]) => T.editor.approximate[key]);
    const tag = $("[data-approximate]", E.root);
    tag.hidden = !(E.mode === "live" && near && near.length);
    tag.title = near && near.length ? T.editor.approximateTitle(near.join("、")) : "";
    setPlayIcon();
  }

  function askExact() {
    E.wantExact = true;
    requestPreview();
  }

  function showExact() {
    E.player.show(false);
    E.mode = "exact";
    $("[data-stage]", E.root).classList.add("exact");
    seek(E.time);
    showPreviewState();
    showPlaybackState();
  }

  function showLive() {
    E.video.pause();
    E.mode = "live";
    E.wantExact = false;
    $("[data-stage]", E.root).classList.remove("exact");
    E.player.show(true);
    seek(E.time);
    showPreviewState();
    showPlaybackState();
  }

  // ------------------------------------------------------------------ layout

  function lanes() {
    const tracks = E.project.tracks;
    const videos = tracks.filter((track) => track.track_type === "video");
    const audios = tracks.filter((track) => track.track_type === "audio");
    const base = videos[0];
    return [
      ...videos.slice(1).reverse().map((track) => ({ track, kind: "overlay", height: LANE.other })),
      ...(base ? [{ track: base, kind: "base", height: LANE.base }] : []),
      ...audios.map((track) => ({ track, kind: track.duck_under_speech ? "music" : "audio", height: LANE.other })),
    ];
  }

  function findClip(trackId, clipId) {
    const track = E.project.tracks.find((entry) => entry.id === trackId);
    return track?.clips.find((clip) => clip.id === clipId) || null;
  }

  function html() {
    return `<div class="editor">
      <header class="ed-top">
        <a class="icon-btn" href="#/" title="${esc(T.editor.back)}">${icon("back", 16)}</a>
        <span class="crumb">${esc(T.editor.back)} /</span><h1 data-title></h1>
        <div class="seg" data-views>
          <button type="button" data-view="edit">${esc(T.editor.views.edit)}</button>
          <button type="button" data-view="versions">${esc(T.editor.views.versions)}</button>
        </div>
        <span class="state warn" data-stale hidden>${esc(T.editor.previewStale)}</span>
        <span class="state" data-locked hidden>${esc(T.editor.playingLocked)}</span>
        <div class="grow"></div>
        <span class="state" data-live-chip hidden></span>
        <span class="state" data-preview-chip hidden></span>
        <button class="btn small edit-only" data-exact title="${esc(T.editor.exactTitle)}">${esc(T.editor.exact)}</button>
        <button class="btn small edit-only" data-live hidden>${esc(T.editor.backToLive)}</button>
        <span class="state busy" data-export-chip hidden></span>
        <button class="btn primary small edit-only" data-export>${icon("download", 15)}${esc(T.exporting.button)}</button>
      </header>
      <section class="ed-viewer">
        <div class="viewer-bar"><b>${esc(T.editor.viewerTitle)}</b><span data-format></span><span class="approx" data-approximate hidden>${esc(T.editor.approximateTag)}</span></div>
        <div class="stage-wrap" data-stage-wrap>
          <div class="stage" data-stage>
            <video data-video playsinline preload="auto"></video>
            <img class="source" data-source alt="">
            <span class="tag">${icon("film", 13)}${esc(T.editor.sourceView)}</span>
            <div class="words" data-words></div>
            <div class="notice" data-notice hidden></div>
          </div>
        </div>
        <div class="scrub-row"><input type="range" class="scrub" data-scrub min="0" max="1000" step="1" value="0" title="${esc(T.versions.scrub)}"></div>
        <div class="transport">
          <span class="timecode" data-time></span>
          <div class="controls">
            <button class="icon-btn" data-replay title="${esc(T.editor.replay)}">${svg("replay")}</button>
            <button class="icon-btn" data-step="-1" title="${esc(T.editor.previousFrame)}">${svg("prev")}</button>
            <button class="icon-btn" data-play disabled title="${esc(T.editor.play)}">${svg("play")}</button>
            <button class="icon-btn" data-step="1" title="${esc(T.editor.nextFrame)}">${svg("next")}</button>
            <button class="btn small comment-btn edit-only" data-comment-btn title="${esc(T.comments.addTitle)}">${svg("comment", 14)}${esc(T.comments.add)}</button>
          </div>
          <span class="meta" data-fps></span>
        </div>
      </section>
      <aside class="ed-inspector" data-inspector></aside>
      <section class="ed-compare" data-compare-view>
        <div class="cmp-sides">
          <div class="cmp-side" data-side="a"><div class="cmp-label"><b data-label></b><span class="heard" data-heard>${svg("speaker", 13)}</span></div><div class="cmp-stage-wrap"><div class="stage" data-cmp-stage></div></div></div>
          <div class="cmp-side" data-side="b"><div class="cmp-label"><b data-label></b><span class="heard" data-heard>${svg("speaker", 13)}</span></div><div class="cmp-stage-wrap"><div class="stage" data-cmp-stage></div></div></div>
        </div>
        <div class="cmp-bar">
          <button class="icon-btn" data-cmp-play title="${esc(T.editor.play)}">${svg("play", 18)}</button>
          <input type="range" class="scrub" data-cmp-scrub min="0" max="1000" step="1" value="0">
          <span class="cmp-time" data-cmp-time></span>
          <button class="btn small" data-cmp-swap title="${esc(T.compare.swapTitle)}">${esc(T.compare.swap)}</button>
          <button class="btn small" data-cmp-close>${esc(T.compare.close)}</button>
        </div>
      </section>
      <aside class="ed-versions" data-versions></aside>
      <section class="ed-vdetail" data-vdetail></section>
      <section class="ed-timeline">
        <div class="tl-resize" data-resize></div>
        <div class="tl-tools">
          <button class="icon-btn edit-action" data-undo title="${esc(T.editor.undo)}（Ctrl+Z）">${svg("undo")}</button>
          <span class="sep"></span>
          <button class="icon-btn edit-action" data-split title="${esc(T.editor.split)}（S）">${svg("split")}</button>
          <button class="icon-btn edit-action" data-delete title="${esc(T.editor.delete)}（Delete）">${icon("trash", 16)}</button>
          <span class="sep"></span>
          <button class="icon-btn edit-action" data-make-captions title="${esc(T.captions.make)}">${svg("captions")}</button>
          <button class="icon-btn edit-action" data-add-card title="${esc(T.card.add)}">${svg("title")}</button>
          <button class="icon-btn edit-action" data-add-music title="${esc(T.music.add)}">${icon("music", 16)}</button>
          <span class="tl-status" data-tl-status hidden></span>
          <div class="grow"></div>
          <button class="icon-btn" data-zoom="-1" title="${esc(T.editor.zoomOut)}">${svg("minus")}</button>
          <input type="range" min="0" max="100" step="1" data-zoom-range title="${esc(T.editor.zoom)}">
          <button class="icon-btn" data-zoom="1" title="${esc(T.editor.zoomIn)}">${icon("plus", 16)}</button>
          <button class="icon-btn" data-fit title="${esc(T.editor.fit)}">${svg("fit")}</button>
        </div>
        <div class="tl-body">
          <div class="tl-heads" data-heads></div>
          <div class="tl-scroll" data-scroll>
            <div class="tl-canvas" data-canvas>
              <div class="ruler" data-ruler><canvas></canvas></div>
              <div data-lanes></div>
              <div class="playhead" data-playhead></div>
            </div>
          </div>
        </div>
      </section>
    </div>`;
  }

  const EXTRA = {
    play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" stroke="none"/>',
    prev: '<path d="M18 6l-8 6 8 6M6 6v12"/>',
    next: '<path d="M6 6l8 6-8 6M18 6v12"/>',
    undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
    split: '<path d="M12 3v18"/><path d="M8 7H4v10h4M16 7h4v10h-4"/>',
    minus: '<path d="M5 12h14"/>',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    replay: '<path d="M4 12a8 8 0 1 0 2.3-5.7"/><path d="M4 4v5h5"/>',
    captions: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M7 15h4M13 15h4M7 11h10"/>',
    speaker: '<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 9a4 4 0 0 1 0 6"/>',
    comment: '<path d="M4 5h16v11H9l-5 4z"/>',
    title: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 9h8M12 9v7"/>',
  };
  const svg = (name, size = 16) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${EXTRA[name]}</svg>`;

  // ------------------------------------------------------------------ drawing

  function contentWidth() {
    const visible = E.scroll.clientWidth;
    return Math.max(visible, (E.project.duration + 6) * E.pps + 120);
  }

  function drawAll() {
    $("[data-title]", E.root).textContent = E.project.name || T.projects.untitled;
    $("[data-format]", E.root).textContent = T.editor.format(E.project.width, E.project.height);
    $("[data-fps]", E.root).textContent = `${Math.round(fps() * 100) / 100} fps`;
    fitStage();
    drawLanes();
    drawRuler();
    drawInspector();
    drawPlayhead();
    drawTime();
  }

  function fitStage() {
    const wrap = $("[data-stage-wrap]", E.root);
    const stage = $("[data-stage]", E.root);
    const aspect = E.project.width / E.project.height;
    const width = Math.max(10, wrap.clientWidth - 28), height = Math.max(10, wrap.clientHeight - 28);
    const w = Math.min(width, height * aspect);
    stage.style.width = `${w}px`;
    stage.style.height = `${w / aspect}px`;
  }

  function drawLanes(overrides = {}) {
    const container = $("[data-lanes]", E.root);
    const heads = $("[data-heads]", E.root);
    const canvas = $("[data-canvas]", E.root);
    canvas.style.width = `${contentWidth()}px`;
    const laneList = lanes();
    const videos = laneList.filter((lane) => lane.kind === "base" || lane.kind === "overlay").length;
    let audio = 0;
    const trackId = (lane, index) => (lane.kind === "base" || lane.kind === "overlay") ? `V${videos - index}` : `A${++audio}`;
    heads.innerHTML = `<div class="tl-head" style="height:${LANE.captions}px"><b>T1</b>${esc(T.captions.lane)}</div>`
      + laneList.map((lane, index) => `<div class="tl-head" style="height:${lane.height}px"><b>${trackId(lane, index)}</b>${esc(T.editor.tracks[lane.kind === "base" ? "video" : lane.kind])}</div>`).join("");
    container.innerHTML = `<div class="lane captions" style="height:${LANE.captions}px" data-captions></div>`
      + laneList.map((lane) => `<div class="lane ${lane.kind === "base" ? "base" : ""}" style="height:${lane.height}px" data-lane="${esc(lane.track.id)}"></div>`).join("");
    const captionLane = $("[data-captions]", container);
    for (const cue of E.captions || []) {
      const block = document.createElement("div");
      block.className = `cap${cue.cue_id === E.selectedCaption ? " sel" : ""}`;
      block.dataset.cue = cue.cue_id;
      block.style.left = `${cue.start * E.pps}px`;
      block.style.width = `${Math.max(3, (cue.end - cue.start) * E.pps - 1)}px`;
      block.textContent = cue.text;
      block.title = cue.text;
      captionLane.append(block);
    }
    for (const lane of laneList) {
      const element = $(`[data-lane="${CSS.escape(lane.track.id)}"]`, container);
      const placed = layoutTrack(lane.track, overrides);
      for (const { clip, start, length } of placed) element.append(clipElement(lane, clip, start, length));
    }
  }

  // Where each clip of a track sits, with any clip being dragged at its dragged length and the
  // clips after it moved along, the way the magnetic timeline will move them.
  function layoutTrack(track, overrides) {
    const sorted = [...track.clips].sort((a, b) => num(a.timeline_in) - num(b.timeline_in));
    let shift = 0;
    return sorted.map((clip) => {
      const override = overrides[`${track.id}/${clip.id}`];
      const length = override ? (override.end - override.start) / (clip.speed || 1) : clipLength(clip);
      const start = num(clip.timeline_in) + shift;
      if (override) shift += length - clipLength(clip);
      return { clip, start, length, override };
    });
  }

  function clipElement(lane, clip, start, length) {
    const asset = E.project.assets[clip.asset_id];
    const element = document.createElement("div");
    const selected = E.selected?.track === lane.track.id && E.selected?.clip === clip.id;
    const isVideo = lane.kind === "base" || lane.kind === "overlay";
    element.className = `clip ${isVideo ? "video" : "audio"} ${lane.kind === "music" ? "music" : ""} ${clip.card ? "card" : ""} ${selected ? "sel" : ""} ${clip.pinned ? "pinned" : ""} ${asset ? "" : "missing"}`;
    element.style.left = `${start * E.pps}px`;
    element.style.width = `${Math.max(2, length * E.pps)}px`;
    element.dataset.track = lane.track.id;
    element.dataset.clip = clip.id;
    const width = length * E.pps;
    if (clip.card) {
      // A title card is its words, not the frame it borrows.
      element.insertAdjacentHTML("beforeend", `<span class="label">${esc(clip.card.title)}</span>
        <span class="handle l" data-handle="l"></span><span class="handle r" data-handle="r"></span>`);
      return element;
    }
    if (isVideo && asset?.has_video && width > 24) element.append(filmstrip(clip, asset, width, lane.height - 10));
    if (asset?.has_audio && width > 24) drawWave(element, clip, asset, width, lane.height - 10, isVideo);
    element.insertAdjacentHTML("beforeend", `<span class="label">${esc(asset ? asset.name : T.editor.missingFile)} · ${length.toFixed(1)}s</span>
      <span class="handle l" data-handle="l"></span><span class="handle r" data-handle="r"></span>`);
    return element;
  }

  function filmstrip(clip, asset, width, height) {
    const strip = document.createElement("div");
    strip.className = "strip";
    const aspect = asset.width && asset.height ? asset.width / asset.height : 16 / 9;
    const tile = Math.max(24, height * aspect);
    const count = Math.min(40, Math.ceil(width / tile));
    const start = num(clip.source_range.start), span = num(clip.source_range.end) - start;
    for (let index = 0; index < count; index++) {
      const at = start + Math.min(span, ((index * tile) / width) * span);
      const image = document.createElement("img");
      image.loading = "lazy";
      image.style.width = `${tile}px`;
      image.src = thumbUrl(asset.id, at, 54);
      image.onerror = () => { image.style.visibility = "hidden"; };
      strip.append(image);
    }
    return strip;
  }

  async function peaks(assetId) {
    E.waves[assetId] ??= fetch(`/wave/${encodeURIComponent(assetId)}`).then((response) => response.json()).catch(() => ({ peaks: [] }));
    return E.waves[assetId];
  }

  async function drawWave(element, clip, asset, width, height, underPicture) {
    const canvas = document.createElement("canvas");
    canvas.className = "wave";
    const pixels = Math.min(4000, Math.ceil(width));
    canvas.width = pixels;
    canvas.height = Math.round(height * (underPicture ? 0.45 : 0.8));
    canvas.style.width = `${width}px`;
    element.prepend(canvas);
    const { rate, peaks: values } = await peaks(asset.id);
    if (!values?.length) return;
    const context = canvas.getContext("2d");
    const start = num(clip.source_range.start), span = num(clip.source_range.end) - start;
    context.fillStyle = underPicture ? "rgba(140, 220, 255, .55)" : "rgba(160, 255, 220, .75)";
    const middle = canvas.height / 2;
    for (let x = 0; x < pixels; x++) {
      const from = Math.floor((start + (x / pixels) * span) * rate);
      const to = Math.max(from + 1, Math.floor((start + ((x + 1) / pixels) * span) * rate));
      let peak = 0;
      for (let index = from; index < to && index < values.length; index++) peak = Math.max(peak, values[index]);
      const h = Math.max(1, peak * canvas.height * (underPicture ? 1 : 0.95));
      if (underPicture) context.fillRect(x, canvas.height - h, 1, h);
      else context.fillRect(x, middle - h / 2, 1, h);
    }
  }

  function drawRuler() {
    const ruler = $("[data-ruler]", E.root);
    const canvas = $("canvas", ruler);
    const width = contentWidth();
    const ratio = window.devicePixelRatio || 1;
    const theme = getComputedStyle(document.documentElement);
    canvas.width = Math.min(32000, width * ratio);
    canvas.height = 32 * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = "32px";
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    // A label every ~110px: wide enough for a timecode.
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    const step = steps.find((value) => value * E.pps >= 110) || 600;
    context.fillStyle = theme.getPropertyValue("--muted").trim();
    context.strokeStyle = theme.getPropertyValue("--line-2").trim();
    context.font = `10px ${theme.getPropertyValue("--mono")}`;
    for (let t = 0; t * E.pps < width; t += step / 5) {
      const x = Math.round(t * E.pps) + 0.5;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      context.beginPath();
      context.moveTo(x, major ? 2 : 12);
      context.lineTo(x, 17);
      context.stroke();
      if (major) context.fillText(timecode(t), x + 4, 11);
    }
    ruler.querySelectorAll(".marker").forEach((marker) => marker.remove());
    for (const marker of E.project.markers || []) {
      const element = document.createElement("span");
      element.className = "marker";
      element.style.left = `${num(marker.timeline_in) * E.pps}px`;
      element.textContent = marker.name;
      ruler.append(element);
    }
    drawCommentMarks();
  }

  // A flag where each comment is on the cut now, and a bar under a stretch; a closed one faded.
  function drawCommentMarks() {
    const ruler = $("[data-ruler]", E.root);
    ruler.querySelectorAll(".cmark, .crange").forEach((element) => element.remove());
    for (const comment of E.comments || []) {
      if (comment.gone) continue;
      const state = `${comment.status}${comment.id === E.focusComment ? " on" : ""}`;
      if (comment.end !== null && comment.end !== undefined) {
        const bar = document.createElement("span");
        bar.className = `crange ${state}`;
        bar.dataset.comment = comment.id;
        bar.style.left = `${comment.start * E.pps}px`;
        bar.style.width = `${Math.max(2, (comment.end - comment.start) * E.pps)}px`;
        bar.title = comment.text;
        ruler.append(bar);
      }
      const flag = document.createElement("span");
      flag.className = `cmark ${state}`;
      flag.dataset.comment = comment.id;
      flag.style.left = `${comment.start * E.pps}px`;
      flag.title = `${fmt(comment.start)} ${comment.text}`;
      ruler.append(flag);
    }
    if (E.picking) {
      const bar = document.createElement("span");
      bar.className = "crange picking";
      const from = Math.min(E.picking.from, E.picking.to), to = Math.max(E.picking.from, E.picking.to);
      bar.style.left = `${from * E.pps}px`;
      bar.style.width = `${Math.max(2, (to - from) * E.pps)}px`;
      ruler.append(bar);
    }
  }

  function drawPlayhead() {
    $("[data-playhead]", E.root).style.left = `${(E.time || 0) * E.pps}px`;
  }

  const shownLength = () => (E.mode === "live" && E.described ? E.described.duration : E.project.duration);

  function drawTime() {
    const length = shownLength();
    $("[data-time]", E.root).innerHTML = `${timecode(E.time)} <span>/ ${timecode(length)}</span>`;
    const scrub = $("[data-scrub]", E.root);
    if (scrub && !E.scrubbing) scrub.value = length > 0 ? String(Math.round((E.time / length) * 1000)) : "0";
  }

  // ------------------------------------------------------------------ inspector

  // Tabs as an NLE inspector has them; a tab that means nothing for the selection is greyed out.
  function drawInspector() {
    const root = $("[data-inspector]", E.root);
    const clip = E.selected && findClip(E.selected.track, E.selected.clip);
    const track = clip && E.project.tracks.find((entry) => entry.id === E.selected.track);
    const context = E.selectedCaption ? ["captions"] : !clip ? [] : track.track_type === "audio" ? ["sound"]
      : (clip.card || E.project.assets[clip.asset_id]?.has_audio === false ? ["picture"] : ["picture", "sound"]);
    const available = [...context, "comments"];
    if (!available.includes(E.tab)) E.tab = context[0] || null;
    const waiting = (E.comments || []).filter((comment) => comment.status === "open").length;
    root.innerHTML = `<div class="tabs">${["picture", "sound", "captions", "comments"].map((tab) =>
      `<button type="button" data-tab="${tab}" class="${tab === E.tab ? "on" : ""}" ${available.includes(tab) ? "" : "disabled"}>${esc(T.editor.tabs[tab])}${tab === "comments" && waiting ? `<span class="count">${waiting}</span>` : ""}</button>`).join("")}</div>
      <div class="props" data-props></div>`;
    root.querySelectorAll("[data-tab]").forEach((button) => {
      button.onclick = () => { E.tab = button.dataset.tab; drawInspector(); };
    });
    const panel = $("[data-props]", root);
    if (E.tab === "comments") return drawComments(panel);
    if (E.selectedCaption) return drawCaptionInspector(panel);
    if (clip && track.track_type === "audio") return drawSoundInspector(panel, clip, track);
    if (clip && E.tab === "sound") return drawClipSound(panel, clip);
    if (!clip) {
      panel.innerHTML = `<p class="hint">${esc(T.editor.inspectorEmpty)}</p>
        <div class="group"><div class="label">${esc(T.editor.shortcutsTitle)}</div>
        <div class="keys">${T.editor.shortcuts.map(([key, what]) => `<kbd>${esc(key)}</kbd><span>${esc(what)}</span>`).join("")}</div></div>`;
      return;
    }
    if (clip.card) return drawCardInspector(panel, clip);
    const asset = E.project.assets[clip.asset_id];
    const nudges = (side) => [-1, -0.1, 0.1, 1].map((delta) =>
      `<button data-nudge="${side}" data-by="${delta}">${delta > 0 ? "+" : ""}${delta}</button>`).join("");
    const madeBy = clip.from_plan_id ? (clip.pinned ? T.editor.adjusted : T.editor.madeByAi) : "";
    panel.innerHTML = `<h2>${esc(asset ? asset.name : T.editor.missingFile)}</h2>
      <p class="hint">${madeBy ? `<span class="badge ${clip.pinned ? "warn" : ""}">${esc(madeBy)}</span>` : ""}</p>
      <div class="group edit-action">
        <div class="label">${esc(T.editor.start)}<b>${fmt(num(clip.source_range.start))}</b></div><div class="nudges">${nudges("start")}</div>
      </div>
      <div class="group edit-action">
        <div class="label">${esc(T.editor.end)}<b>${fmt(num(clip.source_range.end))}</b></div><div class="nudges">${nudges("end")}</div>
        <p class="hint">${esc(T.editor.nudgeHint)}</p>
      </div>
      <div class="group"><div class="label">${esc(T.editor.length)}<b>${clipLength(clip).toFixed(2)} ${esc(T.editor.seconds)}</b></div></div>
      ${fitGroup(clip, asset)}
      ${motionGroup(clip, asset)}
      ${textsGroup(clip)}
      <div class="group stack-buttons edit-action">
        <button class="btn" data-split>${svg("split")}${esc(T.editor.split)}</button>
        <button class="btn danger" data-delete>${icon("trash", 16)}${esc(T.editor.delete)}</button>
        ${clip.from_plan_id && clip.pinned ? `<button class="btn" data-handback>${svg("undo")}${esc(T.editor.handBack)}</button>
          <p class="hint" style="margin:0">${esc(T.editor.handBackHint)}</p>` : ""}
      </div>`;
    panel.querySelectorAll("[data-nudge]").forEach((button) => {
      button.onclick = () => nudge(clip, E.selected.track, button.dataset.nudge, num(button.dataset.by));
    });
    $("[data-split]", panel).onclick = split;
    $("[data-delete]", panel).onclick = remove;
    wireFit(panel, clip);
    wireTexts(panel, clip);
    panel.querySelectorAll("[data-motion]").forEach((button) => {
      button.onclick = () => {
        if (button.dataset.motion === (clip.motion || "push")) return;
        edit([{ action: "set_clip_look", track_id: E.selected.track, clip_id: clip.id, motion: button.dataset.motion }]);
      };
    });
    const handBack = $("[data-handback]", panel);
    if (handBack) handBack.onclick = async () => {
      if (await edit([{ action: "set_clip_pinned", track_id: E.selected.track, clip_id: clip.id, pinned: false }])) toast(T.editor.handedBack);
    };
  }

  // ------------------------------------------------------------------ a shot of another shape

  // Whether a clip's picture is another shape than the frame it fills, and so has a choice to make.
  function shapeDiffers(clip, asset) {
    if (!asset?.width || !asset?.height || clip.layout) return false;
    return Math.abs(asset.width / asset.height - E.project.width / E.project.height) > 0.01;
  }

  function fitGroup(clip, asset) {
    if (!shapeDiffers(clip, asset)) return "";
    const mode = clip.fit?.mode || "auto";
    const choice = (value) => `<button class="${mode === value ? "on" : ""}" data-fit-mode="${value}">${esc(T.fit.modes[value])}</button>`;
    const zoom = Math.round((clip.fit?.zoom || 1) * 100);
    return `<div class="group edit-action"><div class="label">${esc(T.fit.title)}</div>
      <div class="seg fit-seg">${choice("auto")}${choice("fill")}${choice("whole")}</div>
      <p class="hint">${esc(T.fit.hints[mode])}</p>
      ${mode === "fill" ? `<div class="label">${esc(T.fit.zoom)}<b data-zoom-value>${zoom}%</b></div>
        <input type="range" min="100" max="400" step="5" value="${zoom}" data-fit-zoom>` : ""}
      ${clip.fit?.center_x != null || clip.fit?.center_y != null ? `<button class="btn small" data-fit-free>${esc(T.fit.free)}</button>` : ""}</div>`;
  }

  // ------------------------------------------------------------------ words over the footage

  // The words on a clip: a row each, the chosen one opened up to change.
  function textsGroup(clip) {
    if (clip.card) return "";
    const open = (clip.texts || []).find((text) => text.id === E.selectedText);
    const rows = (clip.texts || []).map((text) =>
      `<button class="btn small ${text.id === E.selectedText ? "primary" : ""}" data-text-pick="${esc(text.id)}">${esc(T.texts.styles[text.style])}・${esc(text.text)}</button>`).join("");
    const style = (value) => `<button class="${open?.style === value ? "on" : ""}" data-text-style="${value}">${esc(T.texts.styles[value])}</button>`;
    return `<div class="group edit-action"><div class="label">${esc(T.texts.title)}</div>
      <div class="stack-buttons">${rows || `<p class="hint" style="margin:0">${esc(T.texts.none)}</p>`}</div>
      ${open ? `<input class="card-text" data-text-words maxlength="60" value="${esc(open.text)}">
        <input class="card-text" data-text-second maxlength="60" placeholder="${esc(T.texts.second)}" value="${esc(open.second || "")}">
        <div class="seg fit-seg">${["name", "place", "headline", "free"].map(style).join("")}</div>
        <p class="hint">${esc(T.texts.hints[open.style])}</p>
        <div class="label">${esc(T.texts.when)}</div>
        <div class="nudges"><button data-text-from>${esc(T.texts.fromHere)}</button><button data-text-to>${esc(T.texts.toHere)}</button></div>
        <button class="btn small danger" data-text-remove>${icon("trash", 14)}${esc(T.texts.remove)}</button>` : ""}
      <button class="btn small" data-text-add>${icon("plus", 14)}${esc(T.texts.add)}</button></div>`;
  }

  function textEdit(clip, action, fields) {
    return edit([{ action, track_id: E.selected.track, clip_id: clip.id, ...fields }]);
  }

  function wireTexts(panel, clip) {
    panel.querySelectorAll("[data-text-pick]").forEach((button) => {
      button.onclick = () => {
        E.selectedText = E.selectedText === button.dataset.textPick ? null : button.dataset.textPick;
        const text = clip.texts.find((item) => item.id === E.selectedText);
        // Onto the words, so what is being changed is what the viewer shows.
        if (text) {
          const start = num(clip.timeline_in) + (Math.max(num(text.start), num(clip.source_range.start)) - num(clip.source_range.start)) / (clip.speed || 1);
          if (E.time < start || E.time >= clipEnd(clip)) seek(start + 0.25);
        }
        drawInspector();
      };
    });
    const add = $("[data-text-add]", panel);
    if (add) add.onclick = async () => {
      const id = `text-${Math.random().toString(36).slice(2, 7)}`;
      const from = Math.max(num(clip.timeline_in), Math.min(E.time, clipEnd(clip) - 0.5));
      const to = Math.min(clipEnd(clip), from + 3);
      if (await textEdit(clip, "add_text", { text_id: id, text: T.texts.defaultText,
        timeline_start: Number(from.toFixed(3)), timeline_end: Number(to.toFixed(3)) })) {
        E.selectedText = id;
        seek(from + 0.25);
        drawAll();
        toast(T.texts.added);
        $("[data-text-words]", E.root)?.select();
      }
    };
    const open = (clip.texts || []).find((text) => text.id === E.selectedText);
    if (!open) return;
    const commit = (input, field) => {
      input.onchange = () => {
        const value = input.value.trim();
        if (field === "text" && !value) { input.value = open.text; return; }
        if (value !== (open[field] || "")) textEdit(clip, "set_text", { text_id: open.id, [field]: value });
      };
      input.onkeydown = (event) => { if (event.key === "Enter") input.blur(); event.stopPropagation(); };
    };
    commit($("[data-text-words]", panel), "text");
    commit($("[data-text-second]", panel), "second");
    panel.querySelectorAll("[data-text-style]").forEach((button) => {
      button.onclick = () => { if (button.dataset.textStyle !== open.style) textEdit(clip, "set_text", { text_id: open.id, style: button.dataset.textStyle }); };
    });
    $("[data-text-from]", panel).onclick = () => textEdit(clip, "set_text", { text_id: open.id, timeline_start: Number(E.time.toFixed(3)) });
    $("[data-text-to]", panel).onclick = () => textEdit(clip, "set_text", { text_id: open.id, timeline_end: Number(E.time.toFixed(3)) });
    $("[data-text-remove]", panel).onclick = async () => {
      if (await textEdit(clip, "remove_text", { text_id: open.id })) { E.selectedText = null; drawInspector(); }
    };
  }

  // Dragging the chosen words on the picture puts them where they are dropped, as free words.
  function startTextDrag(event) {
    const clip = E.selected && findClip(E.selected.track, E.selected.clip);
    const text = clip?.texts?.find((item) => item.id === E.selectedText);
    if (!text || E.viewing || !clock().paused) return false;
    const event_ = E.described?.texts?.events.find((item) => item.text_id === text.id && item.clip_id === clip.id
      && E.time >= item.start && E.time < item.end);
    if (!event_) return false;
    if (!E.textHinted) { E.textHinted = true; toast(T.texts.dragHint); }
    const stage = $("[data-stage]", E.root).getBoundingClientRect();
    let moved = null;
    const move = (moveEvent) => {
      moved = { x: clamp((moveEvent.clientX - stage.left) / stage.width, 0.02, 0.98),
                y: clamp((moveEvent.clientY - stage.top) / stage.height, 0.02, 0.98) };
      $("[data-stage]", E.root).style.cursor = "grabbing";
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      $("[data-stage]", E.root).style.cursor = "";
      if (moved) textEdit(clip, "set_text", { text_id: text.id, style: "free",
        x: Math.round(moved.x * 1000) / 1000, y: Math.round(moved.y * 1000) / 1000 });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return true;
  }

  // A photo moves while it is on screen; null is the slow push the render gives it.
  function motionGroup(clip, asset) {
    if (!asset?.still) return "";
    const motion = clip.motion || "push";
    const choice = (value) => `<button class="${motion === value ? "on" : ""}" data-motion="${value}">${esc(T.motion.modes[value])}</button>`;
    return `<div class="group edit-action"><div class="label">${esc(T.motion.title)}</div>
      <div class="seg fit-seg">${Object.keys(T.motion.modes).map(choice).join("")}</div>
      <p class="hint">${esc(T.motion.hint)}</p></div>`;
  }

  function fitEdit(clip, fit) {
    return edit([{ action: "set_clip_look", track_id: E.selected.track, clip_id: clip.id,
      ...(fit ? { fit } : { clear_fit: true }) }]);
  }

  function wireFit(panel, clip) {
    panel.querySelectorAll("[data-fit-mode]").forEach((button) => {
      button.onclick = () => {
        const mode = button.dataset.fitMode;
        if (mode === (clip.fit?.mode || "auto")) return;
        fitEdit(clip, mode === "auto" ? null : { mode });
      };
    });
    const zoom = $("[data-fit-zoom]", panel);
    if (zoom) {
      zoom.oninput = () => { $("[data-zoom-value]", panel).textContent = `${zoom.value}%`; };
      zoom.onchange = () => fitEdit(clip, { ...clip.fit, mode: "fill", zoom: Number(zoom.value) / 100 });
    }
    const free = $("[data-fit-free]", panel);
    if (free) free.onclick = () => fitEdit(clip, { mode: "fill", zoom: clip.fit?.zoom || 1 });
  }

  // Dragging the picture of a cropped shot moves its crop: the shot stays where it is put,
  // no longer following a face. Only the selected clip, only while paused.
  function startCropDrag(event) {
    const clip = E.selected && findClip(E.selected.track, E.selected.clip);
    const asset = clip && E.project.assets[clip.asset_id];
    if (!clip || !shapeDiffers(clip, asset) || clip.fit?.mode === "whole" || E.viewing || !clock().paused) return false;
    if (E.time < num(clip.timeline_in) || E.time >= clipEnd(clip)) return false;
    if (clip.fit?.mode !== "fill" && !E.cropHinted) { E.cropHinted = true; toast(T.fit.dragHint); }
    const shown = E.described?.tracks.flatMap((track) => track.clips).find((item) => item.id === clip.id);
    const crop = shown?.crop;
    const into = E.time - num(clip.timeline_in);
    let centre = 0.5;
    for (const [at, value] of crop?.steps || []) if (at <= into + 1e-6) centre = value;
    const axis = crop?.axis || (asset.width / asset.height > E.project.width / E.project.height ? "x" : "y");
    const cross = crop?.cross ?? 0.5;
    const zoom = clip.fit?.zoom || 1;
    const stage = $("[data-stage]", E.root).getBoundingClientRect();
    // What share of the picture the frame shows along each axis.
    const cover = Math.max(E.project.width / asset.width, E.project.height / asset.height) * zoom;
    const seenX = E.project.width / cover / asset.width, seenY = E.project.height / cover / asset.height;
    const startX = axis === "x" ? centre : cross, startY = axis === "y" ? centre : cross;
    const at = { x: event.clientX, y: event.clientY };
    let moved = null;
    const move = (moveEvent) => {
      const x = clamp(startX - (moveEvent.clientX - at.x) / stage.width * seenX, seenX / 2, 1 - seenX / 2);
      const y = clamp(startY - (moveEvent.clientY - at.y) / stage.height * seenY, seenY / 2, 1 - seenY / 2);
      moved = { x, y };
      $("[data-stage]", E.root).style.cursor = "grabbing";
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      $("[data-stage]", E.root).style.cursor = "";
      if (moved) fitEdit(clip, { mode: "fill", zoom, center_x: Math.round(moved.x * 1000) / 1000, center_y: Math.round(moved.y * 1000) / 1000 });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return true;
  }

  function drawClipSound(panel, clip) {
    const asset = E.project.assets[clip.asset_id];
    panel.innerHTML = `<h2>${esc(asset ? asset.name : T.editor.missingFile)}</h2>${volumeGroup(clip.volume)}`;
    wireVolume(panel, (volume) => [{ action: "set_clip_audio", track_id: E.selected.track, clip_id: clip.id, volume }]);
  }

  function volumeGroup(volume, label = T.editor.volume) {
    const percent = Math.round((volume ?? 1) * 100);
    return `<div class="group edit-action"><div class="label">${esc(label)}<b data-volume-value>${percent}%</b></div>
      <div class="volume"><button class="icon-btn" data-mute title="${esc(percent ? T.editor.mute : T.editor.unmute)}">${svg("speaker", 18)}</button>
      <input type="range" min="0" max="200" step="5" value="${percent}" data-volume></div></div>`;
  }

  // The slider shows its value as it moves and changes the cut once, when it is let go.
  function wireVolume(panel, operations) {
    const slider = $("[data-volume]", panel);
    if (!slider) return;
    const value = $("[data-volume-value]", panel);
    slider.oninput = () => { value.textContent = `${slider.value}%`; };
    slider.onchange = () => edit(operations(num(slider.value) / 100));
    $("[data-mute]", panel).onclick = () => edit(operations(num(slider.value) > 0 ? 0 : 1));
  }

  function drawSoundInspector(panel, clip, track) {
    const asset = E.project.assets[clip.asset_id];
    const isMusic = Boolean(track.duck_under_speech);
    const shifts = [-5, -1, 1, 5].map((delta) => `<button data-shift="${delta}">${delta > 0 ? "+" : ""}${delta}</button>`).join("");
    panel.innerHTML = `<h2>${esc(asset ? asset.name : T.editor.missingFile)}</h2>
      ${volumeGroup(clip.volume, isMusic ? T.editor.trackVolume : T.editor.volume)}
      <div class="group edit-action">
        <div class="label">${esc(T.editor.songStart)}<b>${fmt(num(clip.source_range.start))}</b></div><div class="nudges">${shifts}</div>
        <p class="hint" style="margin:8px 0 0">${esc(T.editor.songStartHint)}</p>
      </div>
      <div class="group stack-buttons edit-action">
        <button class="btn danger" data-remove-music>${icon("trash", 16)}${esc(isMusic ? T.editor.removeMusic : T.editor.delete)}</button>
      </div>`;
    // Music is one bed however many pieces it was looped into, so it is turned up or down as one.
    const clips = isMusic ? track.clips : [clip];
    wireVolume(panel, (volume) => clips.map((each) => ({ action: "set_clip_audio", track_id: track.id, clip_id: each.id, volume })));
    panel.querySelectorAll("[data-shift]").forEach((button) => {
      button.onclick = () => {
        const length = num(clip.source_range.end) - num(clip.source_range.start);
        const limit = asset?.duration ?? Infinity;
        const start = Math.max(0, Math.min(num(clip.source_range.start) + num(button.dataset.shift), limit - length));
        edit([{ action: "trim_clip", track_id: track.id, clip_id: clip.id, ripple: false,
          new_source_range: { start: start.toFixed(3), end: (start + length).toFixed(3) } }]);
      };
    });
    $("[data-remove-music]", panel).onclick = async () => {
      E.selected = null;
      const done = await edit(clips.map((each) => ({ action: "delete_clip", track_id: track.id, clip_id: each.id, ripple: false })));
      if (done && isMusic) toast(T.editor.musicRemoved);
    };
  }

  // ------------------------------------------------------------------ comments

  // What the user wants at a moment, or over a stretch, in their words; the AI reads it next time it looks.
  function writeComment(start, end = null) {
    if (E.viewing) { toast(T.versions.viewingLocked, "error"); return; }
    clock().pause();
    const where = end === null ? fmt(start) : `${fmt(start)}–${fmt(end)}`;
    modal(`<h2>${esc(T.comments.title(where))}</h2><p class="hint">${esc(end === null ? T.comments.momentHint : T.comments.stretchHint)}</p>
      <div class="field"><textarea class="input" data-text rows="3" maxlength="1000" placeholder="${esc(T.comments.placeholder)}"></textarea></div>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-save>${esc(T.comments.save)}</button></div>`,
    (box, close) => {
      const text = $("[data-text]", box);
      text.focus();
      $("[data-cancel]", box).onclick = close;
      const save = async () => {
        if (!text.value.trim()) { text.focus(); return; }
        try {
          const made = await api(`/api/projects/${encodeURIComponent(E.id)}/comments`, { text: text.value.trim(), start, end });
          close();
          await loadComments();
          E.selected = null;
          E.selectedCaption = null;
          E.tab = "comments";
          E.focusComment = made.id;
          drawLanes();
          drawRuler();
          drawInspector();
          toast(T.comments.saved);
        } catch (error) { toast(error.message, "error"); }
      };
      $("[data-save]", box).onclick = save;
      text.addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) save(); });
    });
  }

  // The version a comment was dealt with in, shown on the version page.
  async function openVersion(prefix) {
    setView("versions");
    try { await loadGraph(false); } catch (error) { toast(error.message, "error"); return; }
    const commit = E.graph.nodes.flatMap((node) => node.versions).map((version) => version.commit)
      .find((full) => full.startsWith(prefix));
    if (!commit) { toast(T.comments.versionGone, "error"); return; }
    chooseVersion(E.id, commit);
  }

  function focusComment(id, jump) {
    const comment = (E.comments || []).find((item) => item.id === id);
    if (!comment) return;
    E.focusComment = id;
    E.selected = null;
    E.selectedCaption = null;
    E.tab = "comments";
    if (jump && !comment.gone) { clock().pause(); seek(comment.start); }
    drawLanes();
    drawCommentMarks();
    drawInspector();
    $(`[data-cid="${CSS.escape(id)}"]`, E.root)?.scrollIntoView({ block: "nearest" });
  }

  function commentHtml(comment) {
    const where = comment.end !== null && comment.end !== undefined ? `${fmt(comment.start)}–${fmt(comment.end)}` : fmt(comment.start);
    const replies = comment.replies.map((reply) => `<li class="${reply.by === T.comments.you ? "mine" : "theirs"}"><b>${esc(reply.by)}</b>${esc(reply.text)}</li>`).join("");
    return `<li class="citem ${comment.status}${comment.id === E.focusComment ? " on" : ""}" data-cid="${esc(comment.id)}">
      <div class="chead"><button class="ctime" data-jump ${comment.gone ? "disabled" : ""}>${esc(where)}</button>
        ${comment.gone ? `<span class="badge warn">${esc(T.comments.gone)}</span>` : ""}
        ${comment.status === "resolved" ? `<span class="badge">${esc(T.comments.resolvedTag)}</span>` : ""}
        ${comment.resolved_in ? `<button class="ctime" data-open-version="${esc(comment.resolved_in)}">${esc(T.comments.seeVersion)}</button>` : ""}</div>
      <p class="ctext">${esc(comment.text)}</p>
      ${replies ? `<ul class="creplies">${replies}</ul>` : ""}
      <div class="cactions">
        <button class="btn small" data-reply>${esc(T.comments.reply)}</button>
        ${comment.status === "open"
          ? `<button class="btn small" data-status="resolved">${esc(T.comments.resolve)}</button>`
          : `<button class="btn small" data-status="open">${esc(T.comments.reopen)}</button>`}
        <button class="btn small danger" data-remove>${esc(T.comments.remove)}</button>
      </div>
      <div class="creply" data-reply-box hidden><textarea class="input" rows="2" maxlength="1000" placeholder="${esc(T.comments.replyPlaceholder)}"></textarea>
        <button class="btn small primary" data-send>${esc(T.comments.send)}</button></div></li>`;
  }

  function drawComments(panel) {
    const comments = E.comments || [];
    const open = comments.filter((comment) => comment.status === "open");
    const closed = comments.filter((comment) => comment.status !== "open");
    panel.innerHTML = `<div class="group"><button class="btn small" data-new-comment>${svg("comment", 14)}${esc(T.comments.addHere(fmt(E.time)))}</button>
      <p class="hint">${esc(T.comments.how)}</p></div>
      ${open.length ? `<ol class="clist">${open.map(commentHtml).join("")}</ol>` : `<p class="hint">${esc(comments.length ? T.comments.allDone : T.comments.none)}</p>`}
      ${closed.length ? `<details class="cclosed"${closed.some((comment) => comment.id === E.focusComment) ? " open" : ""}><summary>${esc(T.comments.closed(closed.length))}</summary>
        <ol class="clist">${closed.map(commentHtml).join("")}</ol></details>` : ""}`;
    $("[data-new-comment]", panel).onclick = () => writeComment(E.time);
    const change = async (id, body) => {
      try {
        await api(`/api/projects/${encodeURIComponent(E.id)}/comments/${encodeURIComponent(id)}`, body);
        await loadComments();
        E.focusComment = id;
        drawCommentMarks();
        drawInspector();
      } catch (error) { toast(error.message, "error"); }
    };
    panel.querySelectorAll("[data-cid]").forEach((item) => {
      const id = item.dataset.cid;
      item.onclick = (event) => { if (!event.target.closest("button, textarea, summary")) focusComment(id, true); };
      $("[data-jump]", item).onclick = () => focusComment(id, true);
      const fixed = $("[data-open-version]", item);
      if (fixed) fixed.onclick = () => openVersion(fixed.dataset.openVersion);
      $("[data-reply]", item).onclick = () => {
        const box = $("[data-reply-box]", item);
        box.hidden = !box.hidden;
        if (!box.hidden) $("textarea", box).focus();
      };
      $("[data-send]", item).onclick = () => {
        const text = $("[data-reply-box] textarea", item).value.trim();
        if (text) change(id, { reply: text });
      };
      $("[data-status]", item).onclick = (event) => change(id, { status: event.currentTarget.dataset.status });
      $("[data-remove]", item).onclick = async () => {
        if (!(await confirmBox(T.comments.removeTitle, T.comments.removeBody, T.comments.remove, true))) return;
        await change(id, { delete: true });
        if (E.focusComment === id) E.focusComment = null;
      };
    });
  }

  // ------------------------------------------------------------------ captions

  function selectCaption(cueId, jump) {
    E.selectedCaption = cueId;
    if (E.tab === "comments") E.tab = null;
    E.selected = null;
    const cue = E.captions.find((entry) => entry.cue_id === cueId);
    if (jump && cue) {
      clock().pause();
      seek(cue.start + 0.01);
    }
    drawLanes();
    drawInspector();
  }

  function drawCaptionInspector(panel) {
    const index = E.captions.findIndex((cue) => cue.cue_id === E.selectedCaption);
    const cue = E.captions[index];
    panel.innerHTML = `<h2>${esc(T.captions.title)}</h2>
      <p class="hint">${fmt(cue.start)} – ${fmt(cue.end)}</p>
      <div class="group edit-action">
        <textarea class="input caption-text" data-caption rows="3">${esc(cue.text)}</textarea>
        <p class="hint" style="margin:6px 0 0">${esc(T.captions.enterHint)}</p>
      </div>
      <div class="group stack-buttons">
        <button class="btn primary edit-action" data-save>${esc(T.captions.save)}</button>
        <div class="nudges" style="grid-template-columns:1fr 1fr">
          <button data-step-caption="-1" ${index > 0 ? "" : "disabled"}>${esc(T.captions.previous)}</button>
          <button data-step-caption="1" ${index < E.captions.length - 1 ? "" : "disabled"}>${esc(T.captions.next)}</button>
        </div>
        <button class="btn danger edit-action" data-remove-caption>${icon("trash", 16)}${esc(T.captions.remove)}</button>
      </div>`;
    const box = $("[data-caption]", panel);
    const save = async () => {
      const text = box.value.trim();
      if (!text || text === cue.text) return;
      if (await edit([{ action: "edit_subtitle", cue_id: cue.cue_id, text }])) toast(T.captions.saved);
    };
    box.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); save(); }
    });
    $("[data-save]", panel).onclick = save;
    panel.querySelectorAll("[data-step-caption]").forEach((button) => {
      button.onclick = () => {
        const next = E.captions[index + num(button.dataset.stepCaption)];
        if (next) selectCaption(next.cue_id, true);
      };
    });
    $("[data-remove-caption]", panel).onclick = async () => {
      const next = E.captions[index + 1] || E.captions[index - 1];
      if (await edit([{ action: "edit_subtitle", cue_id: cue.cue_id, delete: true }])) {
        if (next) selectCaption(next.cue_id, false);
      }
    };
  }

  async function makeCaptions() {
    if (lockedWhilePlaying() || E.captionsBusy) return;
    const button = $("[data-make-captions]", E.root);
    const ask = async (transcribe) => api(`/api/projects/${encodeURIComponent(E.id)}/captions/make`,
      { expected_version: E.project.version, transcribe });
    try {
      E.captionsBusy = true;
      button.classList.add("busy");
      toolbarStatus(T.captions.making);
      // Captions replace the project's, so they are made against the version as it is now.
      await loadProject();
      let answer = await ask(false);
      if (answer.needs) {
        const notes = [T.captions.needsBody(answer.needs.length), answer.model_missing ? T.captions.modelNotice : "",
          answer.replaces ? T.captions.replaceNotice(answer.replaces) : ""].filter(Boolean).join(" ");
        if (!(await confirmBox(T.captions.needsTitle, notes, T.captions.start))) return;
        answer = await ask(true);
        if (!(await waitForTranscripts(answer.jobs, button))) return;
        answer = await ask(false);
      }
      if (answer.needs) throw new Error(T.captions.transcribeFailed);
      await loadProject();
      drawAll();
      refreshPlayback();
      toast(answer.count ? T.captions.made(answer.count) : T.captions.none, answer.count ? "ok" : "error");
    } catch (error) {
      if (/version conflict/.test(error.message)) {
        toast(T.editor.conflict, "error");
        await loadProject();
        drawAll();
      } else {
        toast(error.message, "error");
      }
    } finally {
      if (E) {
        E.captionsBusy = false;
        button.classList.remove("busy");
        toolbarStatus("");
      }
    }
  }

  function toolbarStatus(text) {
    const line = $("[data-tl-status]", E.root);
    line.innerHTML = text ? `<span class="spinner"></span>${esc(text)}` : "";
    line.hidden = !text;
  }

  async function waitForTranscripts(jobIds, button) {
    while (E) {
      const state = await api(`/api/jobs?ids=${jobIds.map(encodeURIComponent).join(",")}`);
      toolbarStatus(T.captions.transcribing(state.finished, state.total, Math.round(state.progress * 100)));
      if (state.finished === state.total) {
        if (state.failed) { toast(T.captions.transcribeFailed, "error"); return false; }
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    return false;
  }

  // ------------------------------------------------------------------ export

  const SHAPE_WORDS = { landscape: "橫式", portrait: "直式", square: "方形" };

  function exportDialog() {
    const own = shapeOf(E.project.width, E.project.height);
    const shapes = ["", ...Object.keys(SHAPE_WORDS).filter((shape) => shape !== own)];
    const hasCaptions = (E.captions || []).length > 0;
    const shapeName = (shape) => (shape ? T.exporting.otherShape(T.projects.shapes[shape]) : T.exporting.sameShape(T.projects.shapes[own]));
    modal(`<h2>${esc(T.exporting.title)}</h2>
      <div class="field"><label>${esc(T.exporting.shapeLabel)}</label>
        <div class="choices">${shapes.map((shape, index) => `<label class="choice"><input type="radio" name="shape" value="${shape}" ${index ? "" : "checked"}><span>${esc(shapeName(shape))}</span></label>`).join("")}</div></div>
      <div class="field"><label class="choice"><input type="checkbox" data-burn ${hasCaptions ? "checked" : "disabled"}>
        <span>${esc(hasCaptions ? T.exporting.captionsLabel : T.exporting.captionsNone)}</span></label></div>
      <p class="hint" data-destination></p>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button>
        <button class="btn primary" data-go>${esc(T.exporting.check)}</button></div>`, (box, close) => {
      const chosen = () => box.querySelector("input[name=shape]:checked").value;
      const showDestination = () => {
        const shape = chosen();
        $("[data-destination]", box).textContent = T.exporting.destination(`${E.project.name}${shape ? `（${SHAPE_WORDS[shape]}）` : ""}`);
      };
      box.querySelectorAll("input[name=shape]").forEach((radio) => { radio.onchange = showDestination; });
      showDestination();
      $("[data-cancel]", box).onclick = close;
      $("[data-go]", box).onclick = async () => {
        const settings = { frame: chosen(), captions: $("[data-burn]", box).checked };
        const button = $("[data-go]", box);
        button.disabled = true;
        button.innerHTML = `<span class="spinner"></span>${esc(T.exporting.checking)}`;
        try {
          const query = new URLSearchParams({ frame: settings.frame, captions: settings.captions ? "1" : "" });
          const { findings } = await api(`/api/projects/${encodeURIComponent(E.id)}/check?${query}`);
          close();
          if (findings.length) showFindings(findings, settings);
          else startExport(settings, []);
        } catch (error) {
          toast(error.message, "error");
          button.disabled = false;
          button.textContent = T.exporting.check;
        }
      };
    });
  }

  // The server's findings are written for the AI; the user gets each kind said plainly, with the
  // moments the server named turned into buttons that jump there.
  function showFindings(findings, settings) {
    const kinds = [...new Set(findings.map((finding) => finding.check))];
    const rows = kinds.map((kind) => {
      const [title, body] = T.exporting.findings[kind] || [kind, ""];
      const times = findings.filter((finding) => finding.check === kind)
        .flatMap((finding) => [...finding.message.matchAll(/\b(\d+):(\d{2})(?:\.(\d))?\b/g)])
        .map((match) => num(match[1]) * 60 + num(match[2]) + (match[3] ? num(match[3]) / 10 : 0))
        .filter((seconds, index, all) => seconds <= E.project.duration + 1 && all.indexOf(seconds) === index)
        .slice(0, 6);
      return `<div class="finding"><strong>${esc(title)}</strong><p>${esc(body)}</p>
        <div class="times">${times.map((seconds) => `<button class="btn small" data-jump="${seconds}">${esc(T.exporting.jumpTo(fmt(seconds)))}</button>`).join("")}</div></div>`;
    }).join("");
    modal(`<h2>${esc(T.exporting.foundTitle)}</h2><p>${esc(T.exporting.foundBody)}</p>
      <div class="findings">${rows}</div>
      <div class="foot"><button class="btn" data-back>${esc(T.exporting.goBack)}</button>
        <button class="btn primary" data-ahead>${esc(T.exporting.goAhead)}</button></div>`, (box, close) => {
      box.querySelectorAll("[data-jump]").forEach((button) => {
        button.onclick = () => { close(); clock().pause(); seek(num(button.dataset.jump)); followPlayhead(); };
      });
      $("[data-back]", box).onclick = close;
      // Going ahead allows exactly the kinds that were shown, nothing the user never saw.
      $("[data-ahead]", box).onclick = () => { close(); startExport(settings, kinds); };
    });
  }

  async function startExport(settings, allow) {
    try {
      E.exportState = await api(`/api/projects/${encodeURIComponent(E.id)}/export`, { ...settings, allow });
      toast(T.exporting.started);
      watchExport();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function watchExport() {
    clearTimeout(E.exportTimer);
    const state = E.exportState || {};
    showExportState();
    if (state.status === "queued" || state.status === "running") {
      E.exportTimer = setTimeout(async () => {
        if (!E) return;
        try { E.exportState = await api(`/api/projects/${encodeURIComponent(E.id)}/export`); } catch { /* next look */ }
        watchExport();
      }, 1000);
    } else if (state.status === "completed" && !state.announced) {
      state.announced = true;
      showExportDone(state);
    } else if (state.status === "failed" && !state.announced) {
      state.announced = true;
      toast(`${T.exporting.failed}${state.error ? `：${state.error}` : ""}`, "error");
    }
  }

  function showExportState() {
    const chip = $("[data-export-chip]", E.root);
    const state = E.exportState || {};
    const busy = state.status === "queued" || state.status === "running";
    chip.hidden = !busy;
    if (busy) {
      const percent = Math.round((state.progress || 0) * 100);
      chip.innerHTML = `<span class="spinner"></span>${esc(state.status === "queued" ? T.exporting.queued : T.exporting.progress(percent))}
        <span class="bar"><i style="width:${percent}%"></i></span>`;
    }
  }

  function showExportDone(state) {
    modal(`<h2>${esc(T.exporting.done(state.file))}</h2><p>${esc(state.folder)}</p>
      <div class="foot"><button class="btn" data-close>${esc(T.exporting.close)}</button>
        <button class="btn" data-play-file>${svg("play")}${esc(T.exporting.play)}</button>
        <button class="btn primary" data-reveal>${icon("folder", 16)}${esc(T.exporting.reveal)}</button></div>`, (box, close) => {
      $("[data-close]", box).onclick = close;
      $("[data-reveal]", box).onclick = () => { api("/api/reveal", { job_id: state.job_id }).catch((error) => toast(error.message, "error")); close(); };
      $("[data-play-file]", box).onclick = () => { api("/api/reveal", { job_id: state.job_id, play: true }).catch((error) => toast(error.message, "error")); close(); };
    });
  }

  // ------------------------------------------------------------------ music

  function addMusic() {
    if (lockedWhilePlaying()) return;
    const bed = E.project.tracks.find((track) => track.track_type === "audio" && track.duck_under_speech);
    const replacing = Boolean(bed?.clips.length);
    modal(`<h2>${esc(T.music.title)}</h2><p>${esc(T.music.body)}</p>
      <ul class="points">${T.music.points.map((point) => `<li>${esc(point)}</li>`).join("")}</ul>
      <p class="hint">${esc(T.music.note)}${replacing ? ` ${esc(T.music.replaces)}` : ""}</p>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button>
      <button class="btn primary" data-choose>${icon("music", 16)}${esc(T.music.choose)}</button></div>`, (box, close) => {
      $("[data-cancel]", box).onclick = close;
      $("[data-choose]", box).onclick = async () => {
        close();
        try {
          const picked = await api("/api/pick", { kind: "music" });
          if (!picked.assets.length) {
            if (picked.failed.length) toast(T.media.failed(picked.failed.join("、")), "error");
            return;
          }
          const assetId = picked.assets[0];
          const song = (await api("/api/assets")).assets.find((asset) => asset.id === assetId);
          const trackId = bed ? bed.id : "music";
          const operations = [
            ...(bed ? [] : [{ action: "add_track", track_id: trackId, track_type: "audio", duck_under_speech: true }]),
            ...(bed ? bed.clips.map((clip) => ({ action: "delete_clip", track_id: trackId, clip_id: clip.id, ripple: false })) : []),
            { action: "insert_clip", track_id: trackId, clip_id: `music-${Math.random().toString(36).slice(2, 7)}`, asset_id: assetId,
              source_range: { start: "0", end: String(song.duration) }, volume: 0.5 },
            { action: "fit_track", track_id: trackId, fade_in: "1", fade_out: "2" },
          ];
          if (await edit(operations)) toast(T.music.added);
        } catch (error) { toast(error.message, "error"); }
      };
    });
  }

  function nudge(clip, trackId, side, by) {
    const asset = E.project.assets[clip.asset_id];
    let start = num(clip.source_range.start), end = num(clip.source_range.end);
    const limit = asset?.duration ?? Infinity;
    if (side === "start") start = Math.min(Math.max(0, start + by), end - MIN_CLIP);
    else end = Math.max(Math.min(limit, end + by), start + MIN_CLIP);
    trim(trackId, clip.id, start, end);
  }

  function trim(trackId, clipId, start, end) {
    return edit([{ action: "trim_clip", track_id: trackId, clip_id: clipId, ripple: true,
      new_source_range: { start: start.toFixed(3), end: end.toFixed(3) } }]);
  }

  function clipAtPlayhead() {
    const preferred = E.selected ? [E.project.tracks.find((track) => track.id === E.selected.track)] : [];
    const base = E.project.tracks.find((track) => track.track_type === "video");
    for (const track of [...preferred, base].filter(Boolean)) {
      const clip = track.clips.find((entry) => num(entry.timeline_in) + 0.05 < E.time && E.time < clipEnd(entry) - 0.05);
      if (clip) return { track, clip };
    }
    return null;
  }

  function split() {
    if (lockedWhilePlaying()) return;
    const found = clipAtPlayhead();
    if (!found) return toast(T.editor.noClipHere, "error");
    const suffix = Math.random().toString(36).slice(2, 6);
    edit([{ action: "split_clip", track_id: found.track.id, clip_id: found.clip.id, new_clip_id: `${found.clip.id}-${suffix}`, at: E.time.toFixed(3) }]);
  }

  // ------------------------------------------------------------------ title cards

  // A card goes in front of the shot under the playhead, or at the end when there is none.
  async function addCard() {
    if (lockedWhilePlaying()) return;
    const base = E.project.tracks.find((track) => track.track_type === "video");
    const shots = (base?.clips || []).filter((clip) => !clip.card);
    if (!base || !shots.length) { toast(T.card.needShot, "error"); return; }
    const under = base.clips.find((clip) => E.time >= num(clip.timeline_in) && E.time < clipEnd(clip));
    const id = `card-${Math.random().toString(36).slice(2, 7)}`;
    const done = await edit([{ action: "add_title_card", track_id: base.id, clip_id: id, title: T.card.defaultTitle,
      ...(under ? { before_clip_id: under.id } : {}) }]);
    if (!done) return;
    E.selected = { track: base.id, clip: id };
    E.tab = "picture";
    // Onto the card, past its words' fade, so it is what the viewer shows.
    const made = findClip(base.id, id);
    if (made) seek(num(made.timeline_in) + 0.5);
    drawAll();
    toast(T.card.added);
    $("[data-card-title]", E.root)?.select();
  }

  function cardEdit(clip, fields) {
    return edit([{ action: "set_title_card", track_id: E.selected.track, clip_id: clip.id, ...fields }]);
  }

  function drawCardInspector(panel, clip) {
    const card = clip.card;
    const length = clipLength(clip);
    const choice = (value) => `<button class="${card.background === value ? "on" : ""}" data-card-bg="${value}">${esc(T.card.backgrounds[value])}</button>`;
    panel.innerHTML = `<h2>${esc(T.card.name)}</h2>
      <div class="group edit-action"><div class="label">${esc(T.card.title)}</div>
        <input class="card-text" data-card-title maxlength="60" value="${esc(card.title)}">
        <div class="label">${esc(T.card.subtitle)}</div>
        <input class="card-text" data-card-subtitle maxlength="80" value="${esc(card.subtitle || "")}"></div>
      <div class="group edit-action"><div class="label">${esc(T.card.length)}<b>${length.toFixed(1)} ${esc(T.editor.seconds)}</b></div>
        <div class="nudges">${[-1, -0.5, 0.5, 1].map((delta) => `<button data-card-len="${delta}">${delta > 0 ? "+" : ""}${delta}</button>`).join("")}</div></div>
      <div class="group edit-action"><div class="label">${esc(T.card.background)}</div>
        <div class="seg fit-seg">${choice("blur")}${choice("picture")}${choice("colour")}</div>
        ${card.background === "colour" ? `<input type="color" value="${esc(card.colour)}" data-card-colour>` : ""}
        <p class="hint">${esc(T.card.hints[card.background])}</p>
        <select data-card-photo><option value="">${esc(T.card.photo)}</option></select></div>
      <div class="group stack-buttons edit-action">
        <button class="btn danger" data-delete>${icon("trash", 16)}${esc(T.editor.delete)}</button></div>`;
    const commit = (input, field) => {
      input.onchange = () => {
        const value = input.value.trim();
        if (field === "title" && !value) { input.value = card.title; return; }
        if (value !== (card[field] || "")) cardEdit(clip, { [field]: value });
      };
      input.onkeydown = (event) => { if (event.key === "Enter") input.blur(); event.stopPropagation(); };
    };
    commit($("[data-card-title]", panel), "title");
    commit($("[data-card-subtitle]", panel), "subtitle");
    panel.querySelectorAll("[data-card-len]").forEach((button) => {
      button.onclick = () => {
        const seconds = Math.round(Math.min(10, Math.max(0.5, length + Number(button.dataset.cardLen))) * 10) / 10;
        if (Math.abs(seconds - length) > 0.01) cardEdit(clip, { seconds });
      };
    });
    panel.querySelectorAll("[data-card-bg]").forEach((button) => {
      button.onclick = () => { if (button.dataset.cardBg !== card.background) cardEdit(clip, { background: button.dataset.cardBg }); };
    });
    const colour = $("[data-card-colour]", panel);
    if (colour) colour.onchange = () => cardEdit(clip, { colour: colour.value });
    $("[data-delete]", panel).onclick = remove;
    const photos = $("[data-card-photo]", panel);
    photos.onchange = () => { if (photos.value) cardEdit(clip, { photo_asset_id: photos.value }); };
    // The library's photos, fetched when the card is looked at rather than with every project.
    api("/api/assets?library=footage").then(({ assets }) => {
      const stills = assets.filter((asset) => asset.still && !asset.missing);
      photos.insertAdjacentHTML("beforeend", stills.length
        ? stills.map((asset) => `<option value="${esc(asset.id)}">${esc(asset.name)}</option>`).join("")
        : `<option disabled>${esc(T.card.noPhoto)}</option>`);
    }).catch(() => {});
  }

  async function remove() {
    if (!E.selected || lockedWhilePlaying()) return;
    const { track, clip } = E.selected;
    E.selected = null;
    await edit([{ action: "delete_clip", track_id: track, clip_id: clip, ripple: true }]);
  }

  // ------------------------------------------------------------------ playback

  function seek(seconds) {
    E.time = Math.max(0, Math.min(seconds, shownLength()));
    clock().seek(E.time);
    drawPlayhead();
    drawTime();
  }

  function togglePlay() {
    const playback = clock();
    if (!playback.ready || E.drag) return;
    if (playback.paused) playback.play();
    else playback.pause();
  }

  function tick() {
    if (!E) return;
    if (!clock().paused && !E.drag) {
      E.time = clock().time;
      drawPlayhead();
      drawTime();
      followPlayhead();
    }
    E.frame = requestAnimationFrame(tick);
  }

  function followPlayhead() {
    const x = E.time * E.pps, left = E.scroll.scrollLeft, width = E.scroll.clientWidth;
    if (x > left + width - 40 || x < left) E.scroll.scrollLeft = Math.max(0, x - 80);
  }

  function setPlayIcon() {
    $("[data-play]", E.root).innerHTML = svg(clock().paused ? "play" : "pause", 18);
    $("[data-play]", E.root).disabled = !clock().ready;
    const locked = playing();
    $(".editor", E.root).classList.toggle("playing", locked);
    $("[data-locked]", E.root).hidden = !locked;
  }

  function replay() {
    if (!clock().ready) return;
    seek(0);
    clock().play();
  }

  // ------------------------------------------------------------------ source view while dragging a cut

  function showSource(assetId, at, keptAfter) {
    const stage = $("[data-stage]", E.root);
    stage.classList.add("source-view");
    clock().pause();
    clearTimeout(E.sourceTimer);
    E.sourceTimer = setTimeout(() => {
      $("[data-source]", E.root).src = thumbUrl(assetId, at, 360);
      showWords(assetId, at, keptAfter);
    }, 90);
  }

  function hideSource() {
    clearTimeout(E.sourceTimer);
    $("[data-stage]", E.root).classList.remove("source-view");
  }

  async function showWords(assetId, at, keptAfter) {
    const key = `${assetId}@${Math.round(at)}`;
    E.words[key] ??= api(`/api/words/${encodeURIComponent(assetId)}?at=${Math.round(at)}`).catch(() => ({ words: [], transcribed: false }));
    // Only the newest position is drawn: an older answer arriving late must not replace it.
    const asked = (E.wordsAsked = (E.wordsAsked || 0) + 1);
    const { words, transcribed } = await E.words[key];
    if (!E || asked !== E.wordsAsked) return;
    const box = $("[data-words]", E.root);
    if (!transcribed) { box.innerHTML = `<span class="cut">${esc(T.editor.notTranscribed)}</span>`; return; }
    const near = words.filter((word) => Math.abs(word.start - at) < 3);
    if (!near.length) { box.innerHTML = `<span class="cut">${esc(T.editor.noWords)}</span>`; return; }
    let edgeShown = false;
    box.innerHTML = near.map((word) => {
      const kept = keptAfter ? word.start >= at : word.end <= at;
      let edge = "";
      if (!edgeShown && word.start >= at) { edge = '<i class="edge"></i>'; edgeShown = true; }
      return `${edge}<span class="${kept ? "kept" : "cut"}">${esc(word.text)}</span>`;
    }).join("") + (edgeShown ? "" : '<i class="edge"></i>');
  }

  // ------------------------------------------------------------------ pointer: trim, move, seek

  function timeAt(event) {
    const rect = $("[data-canvas]", E.root).getBoundingClientRect();
    return Math.max(0, (event.clientX - rect.left) / E.pps);
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    const flag = event.target.closest("[data-comment]");
    if (flag && flag.closest("[data-ruler]")) {
      focusComment(flag.dataset.comment, true);
      return;
    }
    if (event.shiftKey && !event.target.closest(".clip [data-handle]")) {
      clock().pause();
      const at = timeAt(event);
      E.picking = { from: at, to: at };
      E.drag = { kind: "range" };
      drawCommentMarks();
      return;
    }
    const caption = event.target.closest(".cap");
    if (caption) {
      selectCaption(caption.dataset.cue, true);
      return;
    }
    const clipElement = event.target.closest(".clip");
    const handle = event.target.closest("[data-handle]");
    if (!clipElement) {
      E.drag = { kind: "seek" };
      clock().pause();
      seek(timeAt(event));
    } else {
      const trackId = clipElement.dataset.track, clipId = clipElement.dataset.clip;
      const clip = findClip(trackId, clipId);
      if (E.selected?.clip !== clipId) E.selectedText = null;
      E.selected = { track: trackId, clip: clipId };
      if (E.tab === "comments") E.tab = null;
      E.selectedCaption = null;
      if (lockedWhilePlaying()) {
        drawLanes();
        drawInspector();
        return;
      }
      const track = E.project.tracks.find((entry) => entry.id === trackId);
      const base = track === E.project.tracks.find((entry) => entry.track_type === "video");
      E.drag = {
        kind: handle ? "trim" : "move", side: handle?.dataset.handle, trackId, clip, base,
        x: event.clientX, start: num(clip.source_range.start), end: num(clip.source_range.end), moved: false,
      };
      drawLanes();
      drawInspector();
    }
    E.scroll.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    const drag = E.drag;
    if (!drag) return;
    // The button came up somewhere the release was never heard — outside the window, say.
    // Finish where the cut last was rather than follow a mouse nobody is dragging.
    if (event.buttons === 0) { onPointerUp(); return; }
    if (drag.kind === "seek") { seek(timeAt(event)); return; }
    if (drag.kind === "range") {
      E.picking.to = Math.max(0, Math.min(timeAt(event), E.project.duration));
      drawCommentMarks();
      return;
    }
    const dx = event.clientX - drag.x;
    if (!drag.moved && Math.abs(dx) < 4) return;
    drag.moved = true;
    const clip = drag.clip, speed = clip.speed || 1;
    if (drag.kind === "trim") {
      const asset = E.project.assets[clip.asset_id];
      const limit = asset?.duration ?? Infinity;
      const by = (dx / E.pps) * speed;
      let start = drag.start, end = drag.end;
      if (drag.side === "l") start = Math.min(Math.max(0, drag.start + by), drag.end - MIN_CLIP);
      else end = Math.max(Math.min(limit, drag.end + by), drag.start + MIN_CLIP);
      drag.newStart = start;
      drag.newEnd = end;
      drawLanes({ [`${drag.trackId}/${clip.id}`]: { start, end } });
      const element = $(`.clip[data-track="${CSS.escape(drag.trackId)}"][data-clip="${CSS.escape(clip.id)}"]`, E.root);
      element?.classList.add("dragging");
      const delta = drag.side === "l" ? drag.start - start : end - drag.end;
      const edgeX = drag.side === "l" ? num(clip.timeline_in) * E.pps : (num(clip.timeline_in) + (end - start) / speed) * E.pps;
      showDelta(edgeX, delta);
      if (asset) showSource(asset.id, drag.side === "l" ? start : end, drag.side === "l");
    } else if (drag.base) {
      drawDropLine(timeAt(event), drag);
    } else {
      const element = $(`.clip[data-track="${CSS.escape(drag.trackId)}"][data-clip="${CSS.escape(clip.id)}"]`, E.root);
      drag.newIn = Math.max(0, num(clip.timeline_in) + dx / E.pps);
      if (element) { element.style.left = `${drag.newIn * E.pps}px`; element.classList.add("dragging"); }
    }
  }

  function showDelta(x, delta) {
    let label = $("[data-delta]", E.root);
    if (!label) {
      label = document.createElement("span");
      label.className = "delta";
      label.dataset.delta = "";
      $("[data-lanes]", E.root).append(label);
    }
    label.style.left = `${Math.max(36, x)}px`;
    label.style.top = "2px";
    label.textContent = T.editor.trimDelta(delta);
  }

  function drawDropLine(at, drag) {
    const track = E.project.tracks.find((entry) => entry.id === drag.trackId);
    const others = track.clips.filter((clip) => clip.id !== drag.clip.id).sort((a, b) => num(a.timeline_in) - num(b.timeline_in));
    const before = others.find((clip) => at < (num(clip.timeline_in) + clipEnd(clip)) / 2) || null;
    drag.before = before ? before.id : null;
    let line = $(".drop-line", E.root);
    if (!line) {
      line = document.createElement("div");
      line.className = "drop-line";
      $(`[data-lane="${CSS.escape(drag.trackId)}"]`, E.root).append(line);
    }
    const x = before ? num(before.timeline_in) : Math.max(0, ...others.map(clipEnd));
    line.style.left = `${x * E.pps}px`;
  }

  async function onPointerUp() {
    const drag = E.drag;
    E.drag = null;
    if (drag?.kind === "range") {
      const from = Math.min(E.picking.from, E.picking.to), to = Math.max(E.picking.from, E.picking.to);
      E.picking = null;
      drawCommentMarks();
      if (to - from >= 0.2) writeComment(from, to);
      return;
    }
    if (!drag || drag.kind === "seek" || !drag.moved) {
      hideSource();
      return;
    }
    hideSource();
    if (drag.kind === "trim") {
      const changed = Math.abs((drag.newStart ?? drag.start) - drag.start) > 0.01 || Math.abs((drag.newEnd ?? drag.end) - drag.end) > 0.01;
      if (changed) await trim(drag.trackId, drag.clip.id, drag.newStart ?? drag.start, drag.newEnd ?? drag.end);
      else drawLanes();
    } else if (drag.base) {
      const track = E.project.tracks.find((entry) => entry.id === drag.trackId);
      const sorted = [...track.clips].sort((a, b) => num(a.timeline_in) - num(b.timeline_in));
      const next = sorted[sorted.findIndex((clip) => clip.id === drag.clip.id) + 1];
      if ((next?.id ?? null) === drag.before) drawLanes();
      else await edit([{ action: "reorder_clip", track_id: drag.trackId, clip_id: drag.clip.id, ...(drag.before ? { before_clip_id: drag.before } : {}) }]);
    } else if (drag.newIn !== undefined) {
      await edit([{ action: "move_clip", track_id: drag.trackId, clip_id: drag.clip.id, new_timeline_in: drag.newIn.toFixed(3) }]);
    }
  }

  // ------------------------------------------------------------------ zoom and keys

  function setZoom(pps, anchorTime = E.time) {
    const before = anchorTime * E.pps - E.scroll.scrollLeft;
    E.pps = Math.max(2, Math.min(400, pps));
    $("[data-zoom-range]", E.root).value = Math.round((Math.log(E.pps / 2) / Math.log(200)) * 100);
    drawLanes();
    drawRuler();
    drawPlayhead();
    E.scroll.scrollLeft = Math.max(0, anchorTime * E.pps - before);
  }

  function fitZoom() {
    setZoom((E.scroll.clientWidth - 60) / Math.max(5, E.project.duration), 0);
    E.scroll.scrollLeft = 0;
  }

  function cancelDrag() {
    E.drag = null;
    hideSource();
    drawLanes();
  }

  function onKey(event) {
    if (!E || event.target.closest?.("input, textarea") || $("#modal-root").children.length) return;
    if (event.key === "Escape" && E.drag) { cancelDrag(); return; }
    const step = event.shiftKey ? 1 : 1 / 30;
    if (E.compare) {
      if (event.key === " ") { event.preventDefault(); toggleCompare(); }
      return;
    }
    if (event.key === " ") { event.preventDefault(); togglePlay(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); clock().pause(); seek(E.time - step); }
    else if (event.key === "ArrowRight") { event.preventDefault(); clock().pause(); seek(E.time + step); }
    else if (event.key === "Home") seek(0);
    else if (event.key === "End") seek(shownLength());
    else if (E.view === "versions") return;
    else if (event.key.toLowerCase() === "c" && !event.ctrlKey && !event.metaKey) { event.preventDefault(); writeComment(E.time); }
    else if (event.key.toLowerCase() === "s" && !event.ctrlKey) split();
    else if (event.key.toLowerCase() === "b" && event.ctrlKey) { event.preventDefault(); split(); }
    else if (event.key === "Delete" || event.key === "Backspace") remove();
    else if (event.key.toLowerCase() === "z" && event.ctrlKey) { event.preventDefault(); undo(); }
  }

  // ------------------------------------------------------------------ the AI editing at the same time

  async function watchForChanges() {
    if (!E) return;
    if (!E.drag && !E.busy && document.visibilityState === "visible") {
      try {
        const { version } = await api(`/api/projects/${encodeURIComponent(E.id)}/version`);
        if (E && await loadComments()) {
          drawCommentMarks();
          if (E.view === "edit") drawInspector();
        }
        if (E && version !== E.project.version && version !== E.ownVersion) {
          await loadProject();
          drawAll();
          refreshPlayback();
          if (E.view === "versions") { await loadGraph(true); redrawVersions(); }
          toast(T.editor.aiUpdated);
        }
      } catch { /* the next look will try again */ }
    }
    if (E) E.watchTimer = setTimeout(watchForChanges, POLL_MS);
  }

  // ------------------------------------------------------------------ open and close

  function close() {
    if (!E) return;
    cancelAnimationFrame(E.frame);
    clearTimeout(E.previewTimer);
    clearTimeout(E.describeTimer);
    clearTimeout(E.watchTimer);
    stopCompare(false);
    E.player.destroy();
    clearTimeout(E.exportTimer);
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", E.onResize);
    E.video.pause();
    E.video.removeAttribute("src");
    document.body.classList.remove("editing");
    E = null;
  }

  async function open(page, id) {
    close();
    document.body.classList.add("editing");
    page.innerHTML = html();
    E = { id, root: page, pps: 40, time: 0, waves: {}, words: {}, selected: null, view: "edit" };
    E.video = $("[data-video]", page);
    E.scroll = $("[data-scroll]", page);
    E.mode = "live";
    E.player = Player.create($("[data-stage]", page), () => { if (E) setPlayIcon(); });
    const video = E.video;
    E.exact = {
      get ready() { return Boolean(E.shownUrl); },
      get paused() { return video.paused; },
      get time() { return video.currentTime; },
      play() { video.play().catch(() => {}); },
      pause() { video.pause(); },
      seek(seconds) { if (E.shownUrl && Math.abs(video.currentTime - seconds) > 0.01) video.currentTime = seconds; },
    };
    await loadProject();
    if (!E.project.name) renameProject(E.project, async () => { await loadProject(); drawAll(); }, T.project.needsName);

    $("[data-play]", page).onclick = togglePlay;
    $("[data-stage]", page).addEventListener("pointerdown", (event) => {
      if (event.button === 0 && E.view === "edit" && (startTextDrag(event) || startCropDrag(event))) event.preventDefault();
    });
    $("[data-comment-btn]", page).onclick = () => writeComment(E.time);
    $("[data-replay]", page).onclick = replay;
    page.querySelectorAll("[data-step]").forEach((button) => {
      button.onclick = () => { clock().pause(); seek(E.time + num(button.dataset.step) / 30); };
    });
    E.video.addEventListener("play", setPlayIcon);
    E.video.addEventListener("pause", setPlayIcon);
    E.video.addEventListener("ended", setPlayIcon);
    $("[data-undo]", page).onclick = undo;
    $("[data-split]", page).onclick = split;
    $("[data-delete]", page).onclick = remove;
    $("[data-make-captions]", page).onclick = makeCaptions;
    $("[data-add-music]", page).onclick = addMusic;
    $("[data-add-card]", page).onclick = addCard;
    $("[data-export]", page).onclick = exportDialog;
    $("[data-exact]", page).onclick = askExact;
    $("[data-live]", page).onclick = showLive;
    page.querySelectorAll("[data-zoom]").forEach((button) => { button.onclick = () => setZoom(E.pps * (num(button.dataset.zoom) > 0 ? 1.5 : 1 / 1.5)); });
    $("[data-zoom-range]", page).oninput = (event) => setZoom(2 * Math.pow(200, num(event.target.value) / 100));
    $("[data-fit]", page).onclick = fitZoom;
    page.querySelectorAll("[data-view]").forEach((button) => { button.onclick = () => setView(button.dataset.view); });
    const scrub = $("[data-scrub]", page);
    scrub.addEventListener("input", () => {
      E.scrubbing = true;
      clock().pause();
      seek((num(scrub.value) / 1000) * shownLength());
    });
    scrub.addEventListener("change", () => { if (E) E.scrubbing = false; });
    E.scroll.addEventListener("pointerdown", onPointerDown);
    E.scroll.addEventListener("pointermove", onPointerMove);
    E.scroll.addEventListener("pointerup", onPointerUp);
    E.scroll.addEventListener("lostpointercapture", () => { if (E?.drag) onPointerUp(); });
    E.scroll.addEventListener("wheel", (event) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      setZoom(E.pps * (event.deltaY < 0 ? 1.2 : 1 / 1.2), timeAt(event));
    }, { passive: false });
    $("[data-resize]", page).addEventListener("pointerdown", (event) => {
      const editor = $(".editor", page), startY = event.clientY;
      const startHeight = $(".ed-timeline", page).getBoundingClientRect().height;
      const move = (moveEvent) => {
        editor.style.setProperty("--timeline-height", `${Math.max(180, Math.min(window.innerHeight - 260, startHeight - (moveEvent.clientY - startY)))}px`);
        fitStage();
      };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    document.addEventListener("keydown", onKey);
    E.onResize = () => { if (E) { fitStage(); drawLanes(); drawRuler(); } };
    window.addEventListener("resize", E.onResize);

    drawAll();
    fitZoom();
    setPlayIcon();
    refreshPlayback();
    $('[data-view="edit"]', page).classList.add("on");
    // Opening a branch, or the project a branch came from, from the version page stays on it.
    if (keptView) { setView(keptView); keptView = null; }
    api(`/api/projects/${encodeURIComponent(id)}/export`).then((state) => {
      if (!E || !state.job_id) return;
      // One already finished was announced when it finished; only a running one is followed.
      E.exportState = { ...state, announced: state.status !== "queued" && state.status !== "running" };
      watchExport();
    }).catch(() => {});
    E.frame = requestAnimationFrame(tick);
    E.watchTimer = setTimeout(watchForChanges, POLL_MS);
  }

  window.Editor = { open, close };
})();
