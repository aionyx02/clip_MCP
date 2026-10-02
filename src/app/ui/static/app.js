// The editor's pages. No framework and no build step: a page is a function that fills #page.
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Line icons, drawn at 18px in the current text colour.
const ICONS = {
  film: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  disk: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  music: '<path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/>',
  check: '<path d="M5 12l5 5L20 7"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L2 18a2 2 0 0 0 1.7 3h16.6A2 2 0 0 0 22 18L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
  open: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  back: '<path d="M15 6l-6 6 6 6"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
};
const icon = (name, size = 18) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;

async function api(path, body) {
  const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error(T.common.serverGone);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || T.common.error);
  return data;
}

function toast(message, kind = "ok") {
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.innerHTML = `${icon(kind === "error" ? "alert" : "check")}<span>${esc(message)}</span>`;
  $("#toasts").append(element);
  setTimeout(() => element.remove(), kind === "error" ? 7000 : 3500);
}

function modal(html, onReady) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="backdrop"><div class="modal">${html}</div></div>`;
  const close = () => { root.innerHTML = ""; document.removeEventListener("keydown", onKey); };
  const onKey = (event) => { if (event.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  $(".backdrop", root).addEventListener("mousedown", (event) => { if (event.target.classList.contains("backdrop")) close(); });
  onReady?.($(".modal", root), close);
}

function confirmBox(title, body, action, danger = false) {
  return new Promise((resolve) => {
    modal(`<h2>${esc(title)}</h2><p>${esc(body)}</p>
      <div class="foot"><button class="btn" data-no>${esc(T.newProject.cancel)}</button>
      <button class="btn ${danger ? "danger" : "primary"}" data-yes>${esc(action)}</button></div>`, (box, close) => {
      $("[data-no]", box).onclick = () => { close(); resolve(false); };
      $("[data-yes]", box).onclick = () => { close(); resolve(true); };
    });
  });
}

const clock = (seconds) => {
  const total = Math.round(seconds || 0);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
};
const bytes = (megabytes) => megabytes >= 1000 ? `${(megabytes / 1000).toFixed(1)} GB` : `${Math.max(megabytes, 0).toFixed(megabytes < 10 ? 1 : 0)} MB`;
const shapeOf = (width, height) => width === height ? "square" : width > height ? "landscape" : "portrait";
const thumbUrl = (assetId, t, h = 180) => `/thumb/${encodeURIComponent(assetId)}?t=${t.toFixed(1)}&h=${h}`;

function head(title, subtitle, actions = "") {
  return `<div class="page-head"><div><h1>${esc(title)}</h1><p>${esc(subtitle)}</p></div><div class="actions">${actions}</div></div>`;
}

async function copyText(text, button) {
  await navigator.clipboard.writeText(text);
  const before = button.innerHTML;
  button.innerHTML = `${icon("check", 16)}${esc(T.ai.copied)}`;
  setTimeout(() => { button.innerHTML = before; }, 1600);
}

// ---------------------------------------------------------------- projects

async function projectsPage(page) {
  page.innerHTML = head(T.projects.title, T.projects.subtitle, `<button class="btn primary" data-new>${icon("plus")}${esc(T.projects.create)}</button>`)
    + `<div class="grid">${'<div class="card skeleton" style="height:200px"></div>'.repeat(4)}</div>`;
  $("[data-new]", page).onclick = newProject;
  const { projects } = await api("/api/projects");
  if (!projects.length) {
    page.querySelector(".grid").outerHTML = `<div class="empty">
      <div class="icon-ring">${icon("sparkle", 28)}</div>
      <h2>${esc(T.projects.emptyTitle)}</h2><p>${esc(T.projects.emptyBody)}</p>
      <div class="prompt" style="margin-top:8px;max-width:560px"><span>${esc(T.projects.examplePrompt)}</span>
      <button class="btn small" data-copy>${icon("copy", 16)}${esc(T.ai.copy)}</button></div></div>`;
    $("[data-copy]", page).onclick = (event) => copyText(T.projects.examplePrompt, event.currentTarget);
    return;
  }
  const cards = projects.map((project) => {
    const shape = shapeOf(project.width, project.height);
    const picture = project.thumb ? `<img loading="lazy" alt="" onerror="this.dataset.broken=1" src="${thumbUrl(project.thumb.asset_id, project.thumb.t)}">` : icon("film", 30);
    const missing = project.missing.length
      ? `<span class="badge warn" title="${esc(T.projects.missingHint + "\n" + project.missing.join("\n"))}">${icon("alert", 13)}${esc(T.projects.missing(project.missing.length))}</span>` : "";
    return `<a class="card" href="#/project/${encodeURIComponent(project.id)}">
      <div class="thumb">${picture}<span class="badge">${clock(project.duration)}</span></div>
      <div class="card-body"><div class="card-title">${esc(project.name || T.projects.untitled)}${project.name ? "" : ` <span class="badge warn">${esc(T.projects.nameIt)}</span>`}</div>
      <div class="card-meta"><span>${esc(T.projects.shapes[shape])}</span><span class="dot-sep">${esc(T.projects.clips(project.clips))}</span>${missing}</div></div></a>`;
  });
  page.querySelector(".grid").innerHTML = `<button class="new-card" data-new>${icon("plus", 26)}${esc(T.projects.create)}</button>` + cards.join("");
  page.querySelectorAll("[data-new]").forEach((button) => { button.onclick = newProject; });
}

function newProject() {
  const sizes = { landscape: [1920, 1080], portrait: [1080, 1920], square: [1080, 1080] };
  const frames = { landscape: "width:34px;height:20px", portrait: "width:18px;height:32px", square: "width:26px;height:26px" };
  const shapes = Object.keys(sizes).map((key) => `<button class="shape${key === "landscape" ? " on" : ""}" data-shape="${key}">
    <i style="${frames[key]}"></i><strong>${esc(T.projects.shapes[key])}</strong><small>${esc(T.newProject.shapeHints[key])}</small></button>`).join("");
  modal(`<h2>${esc(T.newProject.title)}</h2>
    <div class="field"><label>${esc(T.newProject.nameLabel)}</label><input class="input" data-name maxlength="120" placeholder="${esc(T.newProject.namePlaceholder)}"></div>
    <div class="field"><label>${esc(T.newProject.shapeLabel)}</label><div class="shapes">${shapes}</div></div>
    <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-create>${esc(T.newProject.create)}</button></div>`,
  (box, close) => {
    let chosen = "landscape";
    box.querySelectorAll("[data-shape]").forEach((button) => {
      button.onclick = () => {
        chosen = button.dataset.shape;
        box.querySelectorAll("[data-shape]").forEach((other) => other.classList.toggle("on", other === button));
      };
    });
    const name = $("[data-name]", box);
    name.focus();
    $("[data-cancel]", box).onclick = close;
    const create = async () => {
      if (!name.value.trim()) {
        toast(T.newProject.nameRequired, "error");
        name.focus();
        return;
      }
      try {
        const [width, height] = sizes[chosen];
        const made = await api("/api/projects", { name: name.value.trim(), width, height });
        close();
        location.hash = `#/project/${encodeURIComponent(made.id)}`;
      } catch (error) { toast(error.message, "error"); }
    };
    $("[data-create]", box).onclick = create;
    name.addEventListener("keydown", (event) => { if (event.key === "Enter") create(); });
  });
}

// ---------------------------------------------------------------- media

async function mediaPage(page) {
  page.innerHTML = head(T.media.title, T.media.subtitle,
    `<button class="btn" data-pick="folder">${icon("folder")}${esc(T.media.addFolder)}</button>
     <button class="btn primary" data-pick="files">${icon("plus")}${esc(T.media.addFiles)}</button>`)
    + `<div class="callout warn">${icon("alert")}<span>${esc(T.media.moveWarning)}</span></div><div data-list></div>`;
  page.querySelectorAll("[data-pick]").forEach((button) => {
    button.onclick = async () => {
      const label = button.innerHTML;
      page.querySelectorAll("[data-pick]").forEach((other) => { other.disabled = true; });
      button.innerHTML = `<span class="spinner"></span>${esc(T.media.picking)}`;
      try {
        const result = await api("/api/pick", { kind: button.dataset.pick });
        if (!result.chosen) toast(T.media.nothingChosen);
        else if (result.imported) toast(T.media.added(result.imported));
        if (result.failed.length) toast(T.media.failed(result.failed.join("、")), "error");
        await fillMedia(page);
      } catch (error) { toast(error.message, "error"); }
      finally {
        button.innerHTML = label;
        page.querySelectorAll("[data-pick]").forEach((other) => { other.disabled = false; });
      }
    };
  });
  await fillMedia(page);
}

async function fillMedia(page) {
  const { assets } = await api("/api/assets");
  const list = $("[data-list]", page);
  if (!assets.length) {
    list.innerHTML = `<div class="empty"><div class="icon-ring">${icon("film", 28)}</div><h2>${esc(T.media.emptyTitle)}</h2><p>${esc(T.media.emptyBody)}</p></div>`;
    return;
  }
  list.innerHTML = `<div class="grid">${assets.map((asset) => {
    const picture = asset.has_video ? `<img loading="lazy" alt="" onerror="this.dataset.broken=1" src="${thumbUrl(asset.id, Math.min(1, (asset.duration || 0) / 2))}">` : icon("music", 30);
    const kind = asset.has_video ? (asset.has_audio ? "" : `<span class="dot-sep">${esc(T.media.noSound)}</span>`) : `<span class="dot-sep">${esc(T.media.audioOnly)}</span>`;
    return `<div class="card" title="${esc(asset.path)}"><div class="thumb">${picture}
      <span class="badge">${clock(asset.duration)}</span></div>
      <div class="card-body"><div class="card-title">${esc(asset.name)}</div><div class="card-meta"><span>${asset.width ? `${asset.width}×${asset.height}` : ""}</span>${kind}</div></div></div>`;
  }).join("")}</div>`;
}

// ---------------------------------------------------------------- AI clients

async function aiPage(page) {
  page.innerHTML = head(T.ai.title, T.ai.subtitle,
    `<button class="btn" data-refresh>${icon("refresh")}${esc(T.ai.refresh)}</button>`)
    + `<div data-clients class="list">${'<div class="row skeleton" style="height:64px"></div>'.repeat(3)}</div>
       <div class="section-title">${esc(T.ai.examplesTitle)}</div>
       <div class="list">${T.ai.examples.map((text, index) => `<div class="prompt"><span>${esc(text)}</span>
         <button class="btn small" data-example="${index}">${icon("copy", 16)}${esc(T.ai.copy)}</button></div>`).join("")}</div>`;
  page.querySelectorAll("[data-example]").forEach((button) => {
    button.onclick = () => copyText(T.ai.examples[Number(button.dataset.example)], button);
  });
  const show = (clients) => {
    const installed = clients.filter((client) => client.state !== "not_installed");
    const words = { connected: T.ai.connected, not_connected: T.ai.notConnected, not_installed: T.ai.notInstalled,
      failed: T.ai.failed, confirm: T.ai.confirm };
    const none = installed.length ? "" : `<div class="callout">${icon("info")}<div><strong style="color:var(--text)">${esc(T.ai.noneTitle)}</strong><br>${esc(T.ai.noneBody)}
      <div style="margin-top:10px"><a class="btn primary small" href="https://claude.ai/download" target="_blank" rel="noopener">${icon("download", 16)}${esc(T.ai.download)}</a></div></div></div>`;
    // Installed ones first: they are the ones there is something to do about.
    const ordered = [...clients].sort((a, b) => (a.state === "not_installed") - (b.state === "not_installed"));
    $("[data-clients]", page).innerHTML = none + ordered.map((client) => {
      const note = client.state !== "not_installed" ? (T.ai.notes[client.key] || "") : "";
      const sub = client.state === "failed" ? client.detail : note;
      const action = client.state === "confirm"
        ? `<button class="btn small" data-open-client="${esc(client.key)}">${icon("open", 15)}${esc(T.ai.addToCherry)}</button>`
        : client.state === "connected"
          ? `<button class="btn small" data-client="${esc(client.key)}" data-action="disconnect">${esc(T.ai.disconnect)}</button>`
          : client.state === "not_connected" || client.state === "failed"
            ? `<button class="btn small primary" data-client="${esc(client.key)}" data-action="connect">${icon("link", 15)}${esc(T.ai.connect)}</button>`
            : "";
      return `<div class="row">${icon("sparkle")}
        <div class="grow"><strong>${esc(T.ai.names[client.key] || client.name)}</strong><div class="sub" title="${esc(sub)}">${esc(sub)}</div></div>
        ${action}<span class="pill ${client.state}">${esc(words[client.state])}</span></div>`;
    }).join("")
      + (installed.length ? `<div class="callout" style="margin:6px 0 0">${icon("info")}<span>${esc(T.ai.restartHint)}</span></div>` : "");
    // One program at a time, and only after the user has read which file changes.
    page.querySelectorAll("[data-client]").forEach((button) => {
      button.onclick = async () => {
        const client = clients.find((entry) => entry.key === button.dataset.client);
        const name = T.ai.names[client.key] || client.name;
        const where = /[\\/]/.test(client.detail) ? client.detail : T.ai.itsSettings;
        const connecting = button.dataset.action === "connect";
        const agreed = await confirmBox(connecting ? T.ai.connectTitle(name) : T.ai.disconnectTitle(name),
          connecting ? T.ai.connectBody(where) : T.ai.disconnectBody(where),
          connecting ? T.ai.connect : T.ai.disconnect, !connecting);
        if (!agreed) return;
        try {
          show((await api("/api/clients", { key: client.key, action: button.dataset.action })).clients);
          toast(connecting ? T.ai.connectedToast(name) : T.ai.disconnectedToast(name));
        } catch (error) { toast(error.message, "error"); }
      };
    });
    page.querySelectorAll("[data-open-client]").forEach((button) => {
      button.onclick = async () => {
        try {
          await api("/api/clients/open", { key: button.dataset.openClient });
          toast(T.ai.cherryOpened);
        } catch (error) { toast(error.message, "error"); }
      };
    });
  };
  const load = async () => show((await api("/api/clients")).clients);
  $("[data-refresh]", page).onclick = load;
  await load();
}

// ---------------------------------------------------------------- storage

const SWATCHES = ["#5b8def", "#7d8aa6", "#6f9f86", "#a3906a", "#8c7ca6", "#6a98a4", "#a07c7c", "#8a8a8f", "#97a06d"];

async function storagePage(page) {
  page.innerHTML = head(T.storage.title, T.storage.subtitle) + `<div data-usage><div class="row skeleton" style="height:120px"></div></div>`;
  const { items } = await api("/api/storage");
  const inWorkspace = items.filter((item) => item.key !== "outputs");
  const total = inWorkspace.reduce((sum, item) => sum + item.megabytes, 0);
  const colour = (index) => SWATCHES[index % SWATCHES.length];
  const hint = (item) => item.key === "outputs" ? T.storage.outputsHint : item.key === "legacy" ? T.storage.legacyHint
    : item.key.startsWith("model:") ? T.storage.modelHint : "";
  $("[data-usage]", page).innerHTML = `<div class="usage-total"><span style="color:var(--muted)">${esc(T.storage.total)}</span><strong>${bytes(total)}</strong></div>
    <div class="stack">${inWorkspace.map((item, index) => `<span title="${esc(item.label)}" style="width:${total ? (item.megabytes / total) * 100 : 0}%;background:${colour(index)}"></span>`).join("")}</div>
    <div class="list">${items.map((item, index) => `<div class="row">
      <span class="swatch" style="background:${item.key === "outputs" ? "transparent;border:2px solid var(--muted)" : colour(index)}"></span>
      <div class="grow"><strong>${esc(item.label)}</strong><div class="sub">${esc(hint(item))}</div></div>
      <span class="size">${bytes(item.megabytes)}</span>
      <button class="btn small" data-open="${esc(item.key)}">${icon("open", 16)}${esc(T.storage.open)}</button>
      ${item.clearable ? `<button class="btn small danger" data-clear="${esc(item.key)}" ${item.megabytes ? "" : "disabled"}>${icon("trash", 16)}${esc(T.storage.clear)}</button>` : ""}
    </div>`).join("")}</div>`;
  page.querySelectorAll("[data-open]").forEach((button) => {
    button.onclick = () => api("/api/open-folder", { folder: button.dataset.open }).catch((error) => toast(error.message, "error"));
  });
  page.querySelectorAll("[data-clear]").forEach((button) => {
    button.onclick = async () => {
      const item = items.find((entry) => entry.key === button.dataset.clear);
      if (!(await confirmBox(T.storage.confirmTitle(item.label), T.storage.confirmBody, T.storage.clear, true))) return;
      try {
        const result = await api("/api/storage", { kinds: [item.key] });
        toast(T.storage.cleared(bytes(result.freed_megabytes)));
        await storagePage(page);
      } catch (error) { toast(error.message, "error"); }
    };
  });
}

// ---------------------------------------------------------------- one project (chapter 3 fills this in)

async function projectPage(page, id) {
  await Editor.open(page, id);
}

function renameProject(project, done, prompt = "") {
  modal(`<h2>${esc(T.project.renameTitle)}</h2>${prompt ? `<p>${esc(prompt)}</p>` : ""}
    <div class="field"><input class="input" data-name maxlength="120" value="${esc(project.name || "")}" placeholder="${esc(T.newProject.namePlaceholder)}"></div>
    <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-save>${esc(T.project.save)}</button></div>`,
  (box, close) => {
    const name = $("[data-name]", box);
    name.focus();
    name.select();
    $("[data-cancel]", box).onclick = close;
    const save = async () => {
      if (!name.value.trim()) { toast(T.newProject.nameRequired, "error"); return; }
      try {
        await api(`/api/projects/${encodeURIComponent(project.id)}/edits`,
          { expected_version: project.version, operations: [{ action: "rename_project", name: name.value.trim() }] });
        close();
        toast(T.project.renamed);
        done();
      } catch (error) { toast(error.message, "error"); }
    };
    $("[data-save]", box).onclick = save;
    name.addEventListener("keydown", (event) => { if (event.key === "Enter") save(); });
  });
}

// ---------------------------------------------------------------- shell

const PAGES = { projects: projectsPage, media: mediaPage, ai: aiPage, storage: storagePage };

async function route() {
  const [, name = "", id] = location.hash.replace(/^#/, "").split("/");
  const key = name === "project" ? "project" : (PAGES[name] ? name : "projects");
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === (key === "project" ? "projects" : key));
  });
  const page = $("#page");
  if (key !== "project") Editor.close();
  page.scrollTop = 0;
  try {
    if (key === "project") await projectPage(page, decodeURIComponent(id || ""));
    else await PAGES[key](page);
  } catch (error) {
    page.innerHTML = `<div class="empty"><div class="icon-ring">${icon("alert", 28)}</div><h2>${esc(T.common.error)}</h2>
      <p>${esc(error.message)}</p><button class="btn" data-retry>${icon("refresh")}${esc(T.common.retry)}</button></div>`;
    $("[data-retry]", page).onclick = route;
  }
}

// A newer version, offered in the sidebar; nothing is downloaded until the user chooses to update.
async function checkForUpdate() {
  let found;
  try { found = await api("/api/update"); } catch { return; }
  if (!found.available) return;
  const item = $("#update");
  item.innerHTML = `${icon("download")}<span>${esc(T.update.badge(found.version))}</span>`;
  item.hidden = false;
  item.onclick = () => modal(`<h2>${esc(T.update.title(found.version))}</h2><p>${esc(T.update.body)}</p>
    ${found.notes ? `<div class="field"><label>${esc(T.update.notes)}</label><div class="notes-box">${esc(found.notes)}</div></div>` : ""}
    <div class="foot"><button class="btn" data-later>${esc(T.update.later)}</button><button class="btn primary" data-now>${esc(T.update.now)}</button></div>`,
  (box, close) => {
    $("[data-later]", box).onclick = close;
    $("[data-now]", box).onclick = async () => {
      try {
        await api("/api/update", {});
        close();
        document.body.innerHTML = `<div class="updating"><span class="spinner"></span>${esc(T.update.installing)}</div>`;
      } catch (error) {
        toast(error.message === "busy" ? T.update.busy : error.message, "error");
      }
    };
  });
}

function start() {
  document.title = T.appName;
  $("#brand-name").textContent = T.appName;
  const navIcons = { projects: "film", media: "folder", ai: "sparkle", storage: "disk" };
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.innerHTML = `${icon(navIcons[item.dataset.page])}<span>${esc(T.nav[item.dataset.page])}</span>`;
  });

  // A dropped file would have to be copied; say why the buttons are the way in instead.
  const drop = $("#drop");
  $(".drop-card", drop).innerHTML = `${icon("alert", 30)}<h2>${esc(T.media.dropTitle)}</h2><p>${esc(T.media.dropBody)}</p>`;
  let depth = 0;
  window.addEventListener("dragenter", (event) => { if (event.dataTransfer?.types.includes("Files")) { depth++; drop.hidden = false; } });
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; drop.hidden = true; } });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    setTimeout(() => { drop.hidden = true; }, 2600);
  });

  checkForUpdate();

  // The server stops once the window stops checking in.
  setInterval(() => fetch("/api/ping").catch(() => {}), 10000);
  window.addEventListener("hashchange", route);
  route();
}

start();
