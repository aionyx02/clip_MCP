// The editor page: watch the finished cut, fix its cuts by hand.
// What plays is the rendered preview, so what is seen is what will be delivered. Only while a
// cut is being dragged does the viewer show the source, because that is the one moment the
// picture outside the cut matters. Every change goes to the server as the same edit the AI
// would make, then the preview is rendered again for the parts that changed.
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
  const clipLength = (clip) => (num(clip.source_range.end) - num(clip.source_range.start)) / (clip.speed || 1);
  const clipEnd = (clip) => num(clip.timeline_in) + clipLength(clip);
  const assetName = (id) => E.project.assets[id]?.name || id;

  // ------------------------------------------------------------------ server

  async function loadProject() {
    E.project = await api(`/api/projects/${encodeURIComponent(E.id)}`);
    E.captions = (await api(`/api/projects/${encodeURIComponent(E.id)}/captions`)).captions;
    E.ownVersion = E.project.version;
    if (E.selectedCaption && !E.captions.some((cue) => cue.cue_id === E.selectedCaption)) E.selectedCaption = null;
    if (E.selected && !findClip(E.selected.track, E.selected.clip)) E.selected = null;
  }

  // Nothing changes while the cut is playing: a cut moving under the picture being watched
  // is how a viewer loses track of what they just saw.
  const playing = () => Boolean(E?.shownUrl) && !E.video.paused;

  function lockedWhilePlaying() {
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
      requestPreview();
      return true;
    } catch (error) {
      if (/version conflict/.test(error.message)) {
        toast(T.editor.conflict, "error");
        await loadProject();
        drawAll();
        requestPreview();
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
      requestPreview();
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
    const notice = $("[data-notice]", E.root);
    let text = "", kind = "";
    if (state.empty) { text = T.editor.previewEmpty; }
    else if (state.status === "queued") { text = T.editor.previewQueued; kind = "busy"; }
    else if (state.status === "running") { text = T.editor.previewMaking(Math.round((state.progress || 0) * 100)); kind = "busy"; }
    else if (state.status === "failed" || state.error) { text = T.editor.previewFailed; kind = "warn"; }
    chip.className = `chip ${kind}`;
    chip.hidden = !text;
    chip.innerHTML = kind === "busy"
      ? `<span class="spinner"></span>${esc(text)}<span class="bar"><i style="width:${Math.round((state.progress || 0) * 100)}%"></i></span>`
      : esc(text);
    if (state.url && state.url !== E.shownUrl) swapPreview(state.url);
    const stale = E.shownVersion !== undefined && E.shownVersion !== E.project.version;
    notice.hidden = !(state.empty || (!E.shownUrl && text));
    notice.textContent = state.empty ? T.editor.previewEmpty : (state.status === "failed" ? `${T.editor.previewFailed}${state.error ? `：${state.error}` : ""}` : text);
    $("[data-stale]", E.root).hidden = !(stale && kind === "busy");
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
    $("[data-play]", E.root).disabled = false;
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
        <a class="btn small" href="#/">${icon("back", 16)}${esc(T.editor.back)}</a>
        <h1 data-title></h1>
        <span class="chip warn" data-stale hidden>${esc(T.editor.previewStale)}</span>
        <span class="chip" data-locked hidden>${svg("pause", 12)}${esc(T.editor.playingLocked)}</span>
        <div class="grow"></div>
        <span class="chip" data-preview-chip hidden></span>
        <span class="chip busy" data-export-chip hidden></span>
        <button class="btn primary" data-export>${icon("download", 16)}${esc(T.exporting.button)}</button>
      </header>
      <section class="ed-viewer">
        <div class="stage-wrap" data-stage-wrap>
          <div class="stage" data-stage>
            <video data-video playsinline preload="auto"></video>
            <img class="source" data-source alt="">
            <span class="tag">${icon("film", 14)}${esc(T.editor.sourceView)}</span>
            <div class="words" data-words></div>
            <div class="notice" data-notice hidden></div>
          </div>
        </div>
        <div class="transport">
          <button class="icon-btn" data-step="-1" title="-1">${svg("prev")}</button>
          <button class="icon-btn big" data-play disabled title="${esc(T.editor.play)}">${svg("play")}</button>
          <button class="icon-btn" data-step="1" title="+1">${svg("next")}</button>
          <button class="icon-btn" data-replay title="${esc(T.editor.replay)}">${svg("replay", 18)}</button>
          <span class="time" data-time></span>
        </div>
      </section>
      <aside class="ed-inspector" data-inspector></aside>
      <section class="ed-timeline">
        <div class="tl-resize" data-resize></div>
        <div class="tl-tools">
          <button class="btn small edit-action" data-undo>${svg("undo")}${esc(T.editor.undo)}</button>
          <span class="sep"></span>
          <button class="btn small edit-action" data-split>${svg("split")}${esc(T.editor.split)}</button>
          <button class="btn small edit-action" data-delete>${icon("trash", 16)}${esc(T.editor.delete)}</button>
          <span class="sep"></span>
          <button class="btn small edit-action" data-make-captions>${svg("captions")}${esc(T.captions.make)}</button>
          <button class="btn small edit-action" data-add-music>${icon("music", 16)}${esc(T.music.add)}</button>
          <div class="grow"></div>
          <button class="icon-btn" data-zoom="-1" title="${esc(T.editor.zoomOut)}">${svg("minus")}</button>
          <input type="range" min="0" max="100" step="1" data-zoom-range>
          <button class="icon-btn" data-zoom="1" title="${esc(T.editor.zoomIn)}">${icon("plus", 16)}</button>
          <button class="btn small" data-fit>${esc(T.editor.fit)}</button>
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
    replay: '<path d="M4 12a8 8 0 1 0 2.3-5.7"/><path d="M4 4v5h5"/>',
    captions: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M7 15h4M13 15h4M7 11h10"/>',
    speaker: '<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 9a4 4 0 0 1 0 6"/>',
  };
  const svg = (name, size = 16) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${EXTRA[name]}</svg>`;

  // ------------------------------------------------------------------ drawing

  function contentWidth() {
    const visible = E.scroll.clientWidth;
    return Math.max(visible, (E.project.duration + 6) * E.pps + 120);
  }

  function drawAll() {
    $("[data-title]", E.root).textContent = E.project.name || T.projects.untitled;
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
    heads.innerHTML = `<div class="tl-head" style="height:${LANE.captions}px">${svg("captions", 14)}${esc(T.captions.lane)}</div>`
      + laneList.map((lane) => `<div class="tl-head" style="height:${lane.height}px">
      ${icon(lane.kind === "music" ? "music" : lane.kind === "audio" ? "music" : "film", 14)}${esc(T.editor.tracks[lane.kind === "base" ? "video" : lane.kind])}</div>`).join("");
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
    element.className = `clip ${isVideo ? "video" : "audio"} ${lane.kind === "music" ? "music" : ""} ${selected ? "sel" : ""} ${clip.pinned ? "pinned" : ""} ${asset ? "" : "missing"}`;
    element.style.left = `${start * E.pps}px`;
    element.style.width = `${Math.max(2, length * E.pps)}px`;
    element.dataset.track = lane.track.id;
    element.dataset.clip = clip.id;
    const width = length * E.pps;
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
    canvas.width = Math.min(32000, width * ratio);
    canvas.height = 40 * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = "40px";
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    const step = steps.find((value) => value * E.pps >= 70) || 600;
    context.fillStyle = "#8b8f98";
    context.strokeStyle = "#3a3c44";
    context.font = "11px system-ui";
    for (let t = 0; t * E.pps < width; t += step / 5) {
      const x = Math.round(t * E.pps) + 0.5;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      context.beginPath();
      context.moveTo(x, major ? 15 : 18);
      context.lineTo(x, 22);
      context.stroke();
      if (major) context.fillText(clock(t), x + 4, 12);
    }
    ruler.querySelectorAll(".marker").forEach((marker) => marker.remove());
    for (const marker of E.project.markers || []) {
      const element = document.createElement("span");
      element.className = "marker";
      element.style.left = `${num(marker.timeline_in) * E.pps}px`;
      element.textContent = marker.name;
      ruler.append(element);
    }
  }

  function drawPlayhead() {
    $("[data-playhead]", E.root).style.left = `${(E.time || 0) * E.pps}px`;
  }

  function drawTime() {
    $("[data-time]", E.root).innerHTML = `<b>${fmt(E.time)}</b> / ${fmt(E.project.duration)}`;
  }

  // ------------------------------------------------------------------ inspector

  function drawInspector() {
    const panel = $("[data-inspector]", E.root);
    if (E.selectedCaption) return drawCaptionInspector(panel);
    const clip = E.selected && findClip(E.selected.track, E.selected.clip);
    const track = clip && E.project.tracks.find((entry) => entry.id === E.selected.track);
    if (clip && track.track_type === "audio") return drawSoundInspector(panel, clip, track);
    if (!clip) {
      panel.innerHTML = `<p class="hint">${esc(T.editor.inspectorEmpty)}</p>
        <div class="group"><div class="label">${esc(T.editor.shortcutsTitle)}</div>
        <div class="keys">${T.editor.shortcuts.map(([key, what]) => `<kbd>${esc(key)}</kbd><span>${esc(what)}</span>`).join("")}</div></div>`;
      return;
    }
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
        <p class="hint" style="margin:8px 0 0">${esc(T.editor.nudgeHint)}</p>
      </div>
      <div class="group"><div class="label">${esc(T.editor.length)}<b>${clipLength(clip).toFixed(2)} ${esc(T.editor.seconds)}</b></div></div>
      ${asset?.has_audio ? volumeGroup(clip.volume) : ""}
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
    wireVolume(panel, (volume) => [{ action: "set_clip_audio", track_id: E.selected.track, clip_id: clip.id, volume }]);
    const handBack = $("[data-handback]", panel);
    if (handBack) handBack.onclick = async () => {
      if (await edit([{ action: "set_clip_pinned", track_id: E.selected.track, clip_id: clip.id, pinned: false }])) toast(T.editor.handedBack);
    };
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

  // ------------------------------------------------------------------ captions

  function selectCaption(cueId, jump) {
    E.selectedCaption = cueId;
    E.selected = null;
    const cue = E.captions.find((entry) => entry.cue_id === cueId);
    if (jump && cue) {
      E.video.pause();
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
      button.innerHTML = `<span class="spinner"></span>${esc(T.captions.making)}`;
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
      requestPreview();
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
        button.innerHTML = `${svg("captions")}${esc(T.captions.make)}`;
      }
    }
  }

  async function waitForTranscripts(jobIds, button) {
    while (E) {
      const state = await api(`/api/jobs?ids=${jobIds.map(encodeURIComponent).join(",")}`);
      button.innerHTML = `<span class="spinner"></span>${esc(T.captions.transcribing(state.finished, state.total, Math.round(state.progress * 100)))}`;
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
        button.onclick = () => { close(); E.video.pause(); seek(num(button.dataset.jump)); followPlayhead(); };
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

  async function remove() {
    if (!E.selected || lockedWhilePlaying()) return;
    const { track, clip } = E.selected;
    E.selected = null;
    await edit([{ action: "delete_clip", track_id: track, clip_id: clip, ripple: true }]);
  }

  // ------------------------------------------------------------------ playback

  function seek(seconds) {
    E.time = Math.max(0, Math.min(seconds, E.project.duration));
    if (E.shownUrl && Math.abs(E.video.currentTime - E.time) > 0.01) E.video.currentTime = E.time;
    drawPlayhead();
    drawTime();
  }

  function togglePlay() {
    if (!E.shownUrl || E.drag) return;
    if (E.video.paused) E.video.play().catch(() => {});
    else E.video.pause();
  }

  function tick() {
    if (!E) return;
    if (!E.video.paused && !E.drag) {
      E.time = E.video.currentTime;
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
    $("[data-play]", E.root).innerHTML = svg(E.video.paused ? "play" : "pause", 18);
    const locked = playing();
    $(".editor", E.root).classList.toggle("playing", locked);
    $("[data-locked]", E.root).hidden = !locked;
  }

  function replay() {
    if (!E.shownUrl) return;
    seek(0);
    E.video.play().catch(() => {});
  }

  // ------------------------------------------------------------------ source view while dragging a cut

  function showSource(assetId, at, keptAfter) {
    const stage = $("[data-stage]", E.root);
    stage.classList.add("source-view");
    E.video.pause();
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
    const caption = event.target.closest(".cap");
    if (caption) {
      selectCaption(caption.dataset.cue, true);
      return;
    }
    const clipElement = event.target.closest(".clip");
    const handle = event.target.closest("[data-handle]");
    if (!clipElement) {
      E.drag = { kind: "seek" };
      E.video.pause();
      seek(timeAt(event));
    } else {
      const trackId = clipElement.dataset.track, clipId = clipElement.dataset.clip;
      const clip = findClip(trackId, clipId);
      E.selected = { track: trackId, clip: clipId };
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
    if (event.key === " ") { event.preventDefault(); togglePlay(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); E.video.pause(); seek(E.time - step); }
    else if (event.key === "ArrowRight") { event.preventDefault(); E.video.pause(); seek(E.time + step); }
    else if (event.key === "Home") seek(0);
    else if (event.key === "End") seek(E.project.duration);
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
        if (E && version !== E.project.version && version !== E.ownVersion) {
          await loadProject();
          drawAll();
          requestPreview();
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
    clearTimeout(E.watchTimer);
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
    E = { id, root: page, pps: 40, time: 0, waves: {}, words: {}, selected: null };
    E.video = $("[data-video]", page);
    E.scroll = $("[data-scroll]", page);
    await loadProject();
    if (!E.project.name) renameProject(E.project, async () => { await loadProject(); drawAll(); }, T.project.needsName);

    $("[data-play]", page).onclick = togglePlay;
    $("[data-replay]", page).onclick = replay;
    page.querySelectorAll("[data-step]").forEach((button) => {
      button.onclick = () => { E.video.pause(); seek(E.time + num(button.dataset.step) / 30); };
    });
    E.video.addEventListener("play", setPlayIcon);
    E.video.addEventListener("pause", setPlayIcon);
    E.video.addEventListener("ended", setPlayIcon);
    $("[data-undo]", page).onclick = undo;
    $("[data-split]", page).onclick = split;
    $("[data-delete]", page).onclick = remove;
    $("[data-make-captions]", page).onclick = makeCaptions;
    $("[data-add-music]", page).onclick = addMusic;
    $("[data-export]", page).onclick = exportDialog;
    page.querySelectorAll("[data-zoom]").forEach((button) => { button.onclick = () => setZoom(E.pps * (num(button.dataset.zoom) > 0 ? 1.5 : 1 / 1.5)); });
    $("[data-zoom-range]", page).oninput = (event) => setZoom(2 * Math.pow(200, num(event.target.value) / 100));
    $("[data-fit]", page).onclick = fitZoom;
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
    requestPreview();
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
