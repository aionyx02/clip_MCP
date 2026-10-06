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
  stop: '<rect x="7" y="7" width="10" height="10" rx="1.5"/>',
  back: '<path d="M15 6l-6 6 6 6"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
  folderPlus: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 10v6M9 13h6"/>',
  moveTo: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M10 13h6M13 10l3 3-3 3"/>',
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
  if (!response.ok) {
    // What the server sent back stays with the error, for a refusal that says why (which projects, say).
    const error = new Error(data.error || T.common.error);
    error.data = data;
    error.status = response.status;
    throw error;
  }
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

// ---------------------------------------------------------------- folders
// Folders exist only in clip-mcp: filing something away never moves a file on disk,
// so no project can lose its footage to a tidy-up. Projects and the library each have
// their own, shown the same way: a trail above, folders first, then what is filed here.

const folderLink = (kind, id) => `#/${kind === "assets" ? "media" : "projects"}${id ? `/${encodeURIComponent(id)}` : ""}`;

function folderTrail(folders, id) {
  const byId = new Map(folders.map((folder) => [folder.id, folder]));
  const trail = [];
  for (let at = byId.get(id); at; at = byId.get(at.parent_id)) trail.unshift(at);
  return trail;
}

// The folder and everything inside it: where it cannot be moved to.
function folderAndBelow(folders, id) {
  const found = new Set([id]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const folder of folders) {
      if (folder.parent_id && found.has(folder.parent_id) && !found.has(folder.id)) { found.add(folder.id); grew = true; }
    }
  }
  return found;
}

function askFolderName(title, value = "") {
  return new Promise((resolve) => {
    modal(`<h2>${esc(title)}</h2>
      <div class="field"><label>${esc(T.folders.nameLabel)}</label><input class="input" data-name maxlength="80" value="${esc(value)}" placeholder="${esc(T.folders.namePlaceholder)}"></div>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-save>${esc(T.folders.save)}</button></div>`,
    (box, close) => {
      const name = $("[data-name]", box);
      name.focus();
      name.select();
      $("[data-cancel]", box).onclick = () => { close(); resolve(null); };
      const save = () => {
        if (!name.value.trim()) { toast(T.folders.nameRequired, "error"); return; }
        close();
        resolve(name.value.trim());
      };
      $("[data-save]", box).onclick = save;
      name.addEventListener("keydown", (event) => { if (event.key === "Enter") save(); });
    });
  });
}

// Resolves to the chosen folder's ID, "" for the top, or undefined when the user backs out.
function chooseFolder(folders, title, current, blocked = new Set()) {
  return new Promise((resolve) => {
    const rows = [];
    const walk = (parent, depth) => folders.filter((folder) => (folder.parent_id || null) === parent).forEach((folder) => {
      rows.push(`<button data-to="${esc(folder.id)}" style="padding-left:${10 + depth * 18}px" ${blocked.has(folder.id) ? "disabled" : ""}>${icon("folder")}<span>${esc(folder.name)}</span></button>`);
      walk(folder.id, depth + 1);
    });
    walk(null, 1);
    let chosen = current || "";
    modal(`<h2>${esc(title)}</h2>
      <div class="folder-list"><button data-to="">${icon("folder")}<span>${esc(T.folders.top)}</span></button>${rows.join("")}</div>
      <div class="foot"><button class="btn" data-cancel>${esc(T.newProject.cancel)}</button><button class="btn primary" data-go>${esc(T.folders.move)}</button></div>`,
    (box, close) => {
      const mark = () => box.querySelectorAll("[data-to]").forEach((row) => row.classList.toggle("on", row.dataset.to === chosen));
      const go = () => { close(); resolve(chosen); };
      box.querySelectorAll("[data-to]").forEach((row) => {
        row.onclick = () => { chosen = row.dataset.to; mark(); };
        row.ondblclick = () => { chosen = row.dataset.to; go(); };
      });
      mark();
      $("[data-cancel]", box).onclick = () => { close(); resolve(undefined); };
      $("[data-go]", box).onclick = go;
    });
  });
}

// One page of things in folders. `config` says what the things are:
//   kind       "assets" or "projects", which folders and which things
//   load()     the things, each with `id`, `name` and `folder_id`
//   head(here) the page's heading, with its buttons
//   wire(page, refresh) hooks up the heading's own buttons
//   notes      shown under the heading
//   lead       the first tile in the grid (the new-project tile)
//   card(item, extras) one thing as a card, with `extras` (its select box and tools) inside it
//   opens      a card is a link, so a click opens it unless things are being chosen
//   tools(item) more buttons for one thing's card, beside moving it
//   remove(ids, items) takes things out for good; resolves to whether it did
//   removeLabel what the button that does it says (「移除」 for files, 「刪除」 for projects)
//   empty      what to show when there are no things and no folders at all
async function folderPage(page, folderId, config) {
  const { kind } = config;
  const chosen = new Set();
  let folders = [], items = [];
  const refresh = async () => {
    [folders, items] = await Promise.all([api(`/api/folders?kind=${kind}`).then((data) => data.folders), config.load()]);
    draw();
  };

  const fileAway = async (ids, target) => {
    const where = target ? folders.find((folder) => folder.id === target)?.name : T.folders.top;
    await api("/api/file", { kind, ids, folder_id: target || null });
    chosen.clear();
    toast(T.folders.moved(where));
    await refresh();
  };
  const moveFolder = async (id, target) => {
    if (id === target) return;
    await api("/api/folders", { action: "move", id, parent_id: target || null });
    toast(T.folders.moved(target ? folders.find((folder) => folder.id === target)?.name : T.folders.top));
    await refresh();
  };
  const named = (ids) => ids.length === 1 ? T.folders.moveOne(items.find((item) => item.id === ids[0])?.name || "") : T.folders.moveMany(ids.length);

  function draw() {
    const here = folderId && folders.some((folder) => folder.id === folderId) ? folderId : null;
    if (folderId && !here) { location.hash = folderLink(kind, ""); return; }
    const subfolders = folders.filter((folder) => (folder.parent_id || null) === here);
    const shown = items.filter((item) => (item.folder_id || null) === here);
    for (const id of [...chosen]) if (!shown.some((item) => item.id === id)) chosen.delete(id);
    const scrolled = page.scrollTop;

    if (!items.length && !folders.length) {
      page.innerHTML = config.head(here) + config.empty;
      config.wire(page, refresh);
      $("[data-new-folder]", page).onclick = makeFolder;
      return;
    }
    const trail = [{ id: "", name: T.folders.all[kind] }, ...folderTrail(folders, here)];
    const trailHtml = `<nav class="trail">${trail.map((crumb, index) => `${index ? '<span class="sep">›</span>' : ""}<a href="${folderLink(kind, crumb.id)}" data-drop="${esc(crumb.id)}" class="${index === trail.length - 1 ? "here" : ""}">${esc(crumb.name)}</a>`).join("")}</nav>`;
    const count = (id) => items.filter((item) => item.folder_id === id).length + folders.filter((folder) => folder.parent_id === id).length;
    const tiles = subfolders.map((folder) => `<a class="folder-tile" href="${folderLink(kind, folder.id)}" data-drop="${esc(folder.id)}" data-drag="folder" data-id="${esc(folder.id)}">
      ${icon("folder", 22)}<div class="grow"><div class="card-title">${esc(folder.name)}</div><div class="sub">${esc(T.folders.items(count(folder.id)))}</div></div>
      <div class="card-tools"><button class="tool" data-rename="${esc(folder.id)}" title="${esc(T.folders.rename)}">${icon("edit")}</button>
      <button class="tool" data-move-folder="${esc(folder.id)}" title="${esc(T.folders.moveTo)}">${icon("moveTo")}</button>
      <button class="tool danger" data-remove-folder="${esc(folder.id)}" title="${esc(T.folders.remove)}">${icon("trash")}</button></div></a>`).join("");
    const cards = shown.map((item) => {
      const extras = `<button class="select-box" data-select="${esc(item.id)}" title="${esc(T.folders.choose)}">${icon("check")}</button>
        <div class="card-tools">${config.tools ? config.tools(item) : ""}<button class="tool" data-move-item="${esc(item.id)}" title="${esc(T.folders.moveTo)}">${icon("moveTo")}</button>
        ${config.remove ? `<button class="tool danger" data-remove-item="${esc(item.id)}" title="${esc(config.removeLabel || T.media.remove)}">${icon("trash")}</button>` : ""}</div>`;
      return config.card(item, extras);
    }).join("");
    const bar = chosen.size ? `<div class="selection-bar"><span class="grow">${esc(T.folders.chosen(chosen.size))}</span>
      <button class="btn small" data-choose-all>${esc(T.folders.chooseAll)}</button>
      <button class="btn small" data-move-chosen>${icon("moveTo", 15)}${esc(T.folders.moveTo)}</button>
      ${config.remove ? `<button class="btn small danger" data-remove-chosen>${icon("trash", 15)}${esc(config.removeLabel || T.media.remove)}</button>` : ""}
      <button class="btn small" data-clear>${esc(T.folders.clear)}</button></div>` : "";
    page.innerHTML = config.head(here) + config.notes + trailHtml
      + (tiles ? `<div class="grid folders">${tiles}</div>` : "")
      + `<div class="grid${chosen.size ? " choosing" : ""}">${here ? "" : config.lead || ""}${cards}</div>`
      + (!shown.length && !subfolders.length && here ? `<div class="callout">${icon("info")}<span>${esc(T.folders.emptyHere)}</span></div>` : "")
      + bar;
    page.scrollTop = scrolled;
    config.wire(page, refresh);
    wire(here);
  }

  async function makeFolder() {
    const name = await askFolderName(T.folders.createTitle);
    if (!name) return;
    try {
      await api("/api/folders", { action: "create", kind, name, parent_id: folderId || null });
      await refresh();
    } catch (error) { toast(error.message, "error"); }
  }

  function wire(here) {
    const act = (selector, handler) => page.querySelectorAll(selector).forEach((element) => {
      element.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try { await handler(element, event); } catch (error) { toast(error.message, "error"); }
      });
    });
    const picked = (id) => chosen.has(id) ? [...chosen] : [id];
    $("[data-new-folder]", page).onclick = makeFolder;
    act("[data-select]", (element) => {
      const id = element.dataset.select;
      chosen.has(id) ? chosen.delete(id) : chosen.add(id);
      draw();
    });
    // While things are being chosen, a click on a card chooses it rather than opening it.
    page.querySelectorAll('.card[data-drag="item"]').forEach((card) => {
      card.classList.toggle("chosen", chosen.has(card.dataset.id));
      card.addEventListener("click", (event) => {
        if (!chosen.size && config.opens && !event.ctrlKey) return;
        event.preventDefault();
        const id = card.dataset.id;
        chosen.has(id) ? chosen.delete(id) : chosen.add(id);
        draw();
      });
    });
    act("[data-choose-all]", () => {
      items.filter((item) => (item.folder_id || null) === here).forEach((item) => chosen.add(item.id));
      draw();
    });
    act("[data-clear]", () => { chosen.clear(); draw(); });
    const moveThings = async (ids) => {
      const target = await chooseFolder(folders, T.folders.moveTitle(named(ids)), here);
      if (target !== undefined) await fileAway(ids, target);
    };
    act("[data-move-item]", (element) => moveThings(picked(element.dataset.moveItem)));
    act("[data-move-chosen]", () => moveThings([...chosen]));
    const removeThings = async (ids) => {
      if (await config.remove(ids, items)) { chosen.clear(); await refresh(); }
    };
    act("[data-remove-item]", (element) => removeThings(picked(element.dataset.removeItem)));
    act("[data-remove-chosen]", () => removeThings([...chosen]));
    act("[data-rename]", async (element) => {
      const folder = folders.find((entry) => entry.id === element.dataset.rename);
      const name = await askFolderName(T.folders.renameTitle, folder.name);
      if (!name || name === folder.name) return;
      await api("/api/folders", { action: "rename", id: folder.id, name });
      await refresh();
    });
    act("[data-move-folder]", async (element) => {
      const folder = folders.find((entry) => entry.id === element.dataset.moveFolder);
      const target = await chooseFolder(folders, T.folders.moveTitle(T.folders.moveOne(folder.name)), folder.parent_id, folderAndBelow(folders, folder.id));
      if (target !== undefined) await moveFolder(folder.id, target);
    });
    act("[data-remove-folder]", async (element) => {
      const folder = folders.find((entry) => entry.id === element.dataset.removeFolder);
      if (!(await confirmBox(T.folders.removeTitle(folder.name), T.folders.removeBody, T.folders.remove, true))) return;
      await api("/api/folders", { action: "delete", id: folder.id });
      toast(T.folders.removed);
      await refresh();
    });

    // Dragging a card, or the chosen ones, or a folder onto a folder or a step of the trail.
    page.querySelectorAll("[data-drag]").forEach((element) => {
      element.draggable = true;
      element.addEventListener("dragstart", (event) => {
        const payload = element.dataset.drag === "folder"
          ? { folder: element.dataset.id } : { ids: picked(element.dataset.id) };
        event.dataTransfer.setData("application/x-clip-mcp", JSON.stringify(payload));
        event.dataTransfer.effectAllowed = "move";
        element.classList.add("dragging");
      });
      element.addEventListener("dragend", () => element.classList.remove("dragging"));
    });
    page.querySelectorAll("[data-drop]").forEach((target) => {
      target.addEventListener("dragover", (event) => {
        if (!event.dataTransfer.types.includes("application/x-clip-mcp")) return;
        event.preventDefault();
        target.classList.add("drop-on");
      });
      target.addEventListener("dragleave", () => target.classList.remove("drop-on"));
      target.addEventListener("drop", async (event) => {
        target.classList.remove("drop-on");
        const raw = event.dataTransfer.getData("application/x-clip-mcp");
        if (!raw) return;
        event.preventDefault();
        event.stopPropagation();
        const payload = JSON.parse(raw);
        try {
          if (payload.folder) await moveFolder(payload.folder, target.dataset.drop);
          else await fileAway(payload.ids, target.dataset.drop);
        } catch (error) { toast(error.message, "error"); }
      });
    });
  }

  page.innerHTML = config.head(folderId) + `<div class="grid">${'<div class="card skeleton" style="height:200px"></div>'.repeat(4)}</div>`;
  await refresh();
}

const newFolderButton = () => `<button class="btn" data-new-folder>${icon("folderPlus")}${esc(T.folders.create)}</button>`;

// ---------------------------------------------------------------- projects

async function projectsPage(page, folderId) {
  await folderPage(page, folderId, {
    kind: "projects",
    opens: true,
    load: async () => (await api("/api/projects")).projects,
    head: () => head(T.projects.title, T.projects.subtitle,
      `${newFolderButton()}<button class="btn primary" data-new>${icon("plus")}${esc(T.projects.create)}</button>`),
    wire: (root) => {
      root.querySelectorAll("[data-new]").forEach((button) => { button.onclick = newProject; });
      const copy = $("[data-copy]", root);
      if (copy) copy.onclick = (event) => copyText(T.projects.examplePrompt, event.currentTarget);
    },
    notes: "",
    lead: `<button class="new-card" data-new>${icon("plus", 26)}${esc(T.projects.create)}</button>`,
    remove: deleteProjects,
    removeLabel: T.projects.delete,
    empty: `<div class="empty">
      <div class="icon-ring">${icon("sparkle", 28)}</div>
      <h2>${esc(T.projects.emptyTitle)}</h2><p>${esc(T.projects.emptyBody)}</p>
      <div class="prompt" style="margin-top:8px;max-width:560px"><span>${esc(T.projects.examplePrompt)}</span>
      <button class="btn small" data-copy>${icon("copy", 16)}${esc(T.ai.copy)}</button></div></div>`,
    card: (project, extras) => {
      const shape = shapeOf(project.width, project.height);
      const picture = project.thumb ? `<img loading="lazy" alt="" draggable="false" onerror="this.dataset.broken=1" src="${thumbUrl(project.thumb.asset_id, project.thumb.t)}">` : icon("film", 30);
      const missing = project.missing.length
        ? `<span class="badge warn" title="${esc(T.projects.missingHint + "\n" + project.missing.join("\n"))}">${icon("alert", 13)}${esc(T.projects.missing(project.missing.length))}</span>` : "";
      return `<a class="card" href="#/project/${encodeURIComponent(project.id)}" data-drag="item" data-id="${esc(project.id)}" draggable="true">${extras}
        <div class="thumb">${picture}<span class="badge">${clock(project.duration)}</span></div>
        <div class="card-body"><div class="card-title">${esc(project.name || T.projects.untitled)}${project.name ? "" : ` <span class="badge warn">${esc(T.projects.nameIt)}</span>`}</div>
        <div class="card-meta"><span>${esc(T.projects.shapes[shape])}</span><span class="dot-sep">${esc(T.projects.clips(project.clips))}</span>${missing}</div></div></a>`;
    },
  });
}

// Deleting a project is the user's to do, here, and never the AI's: it has no tool for it.
async function deleteProjects(ids, projects) {
  const named = projects.filter((project) => ids.includes(project.id));
  const what = named.length === 1 ? T.projects.deleteOne(named[0].name || T.projects.untitled) : T.projects.deleteMany(named.length);
  if (!(await confirmBox(T.projects.deleteTitle(what), T.projects.deleteBody, T.projects.delete, true))) return false;
  try {
    const result = await api("/api/projects/delete", { ids });
    toast(T.projects.deleted(result.deleted.length));
    return true;
  } catch (error) {
    toast(error.status === 409 && error.data?.busy ? T.projects.deleteBusy(error.data.busy) : error.message, "error");
    return false;
  }
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

async function mediaPage(page, folderId) {
  await folderPage(page, folderId, {
    kind: "assets",
    opens: false,
    load: async () => (await api("/api/assets?all=1")).assets,
    head: () => head(T.media.title, T.media.subtitle,
      `${newFolderButton()}<button class="btn" data-pick="folder">${icon("folder")}${esc(T.media.addFolder)}</button>
       <button class="btn primary" data-pick="files">${icon("plus")}${esc(T.media.addFiles)}</button>`),
    wire: (root, refresh) => { wireImport(root, refresh, folderId); wireNotes(root, refresh); },
    tools: (asset) => `<button class="tool" data-notes="${esc(asset.id)}" title="${esc(T.media.notes)}">${icon("edit")}</button>`,
    notes: `<div class="callout">${icon("info")}<span>${esc(T.media.libraryOnly)}</span></div>
      <div class="callout warn">${icon("alert")}<span>${esc(T.media.moveWarning)}</span></div>`,
    empty: `<div class="callout">${icon("info")}<span>${esc(T.media.libraryOnly)}</span></div>
      <div class="empty"><div class="icon-ring">${icon("film", 28)}</div><h2>${esc(T.media.emptyTitle)}</h2><p>${esc(T.media.emptyBody)}</p></div>`,
    card: (asset, extras) => {
      const picture = asset.has_video && !asset.missing
        ? `<img loading="lazy" alt="" draggable="false" onerror="this.dataset.broken=1" src="${thumbUrl(asset.id, Math.min(1, (asset.duration || 0) / 2))}">`
        : icon(asset.has_video ? "film" : "music", 30);
      const kind = asset.has_video ? (asset.has_audio ? "" : `<span class="dot-sep">${esc(T.media.noSound)}</span>`) : `<span class="dot-sep">${esc(T.media.audioOnly)}</span>`;
      const gone = asset.missing ? `<span class="badge warn" title="${esc(T.media.goneHint)}">${icon("alert", 13)}${esc(T.media.gone)}</span>` : "";
      const fast = asset.fast_transcript ? `<span class="badge" title="${esc(T.media.fastTranscriptHint)}">${esc(T.media.fastTranscript)}</span>` : "";
      return `<div class="card${asset.missing ? " gone" : ""}" title="${esc(asset.path)}" data-drag="item" data-id="${esc(asset.id)}">${extras}<div class="thumb">${picture}
        <span class="badge">${clock(asset.duration)}</span></div>
        <div class="card-body"><div class="card-title">${esc(asset.name)}</div><div class="card-meta"><span>${asset.width ? `${asset.width}×${asset.height}` : ""}</span>${kind}${fast}${gone}</div>
        ${asset.notes ? `<div class="card-notes" title="${esc(asset.notes)}">${esc(asset.notes)}</div>` : ""}</div></div>`;
    },
    remove: removeAssets,
  });
}

// Notes on how a file may be used: the AI reads them wherever it reads about the file.
function wireNotes(page, refresh) {
  page.querySelectorAll("[data-notes]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const assets = (await api("/api/assets?all=1")).assets;
      const asset = assets.find((item) => item.id === button.dataset.notes);
      if (!asset) return;
      modal(`<h2>${esc(T.media.notesTitle(asset.name))}</h2><p>${esc(T.media.notesBody)}</p>
        <textarea class="notes-input" rows="4" data-text placeholder="${esc(T.media.notesPlaceholder)}">${esc(asset.notes || "")}</textarea>
        <div class="foot"><button class="btn" data-no>${esc(T.newProject.cancel)}</button><button class="btn primary" data-yes>${esc(T.media.notesSave)}</button></div>`,
      (box, close) => {
        $("[data-text]", box).focus();
        $("[data-no]", box).onclick = close;
        $("[data-yes]", box).onclick = async () => {
          try {
            await api("/api/assets/notes", { asset_id: asset.id, notes: $("[data-text]", box).value });
            close();
            toast(T.media.notesSaved);
            await refresh();
          } catch (error) { toast(error.message, "error"); }
        };
      });
    });
  });
}

// Files chosen while a folder is open are filed in it.
function wireImport(page, refresh, folderId) {
  page.querySelectorAll("[data-pick]").forEach((button) => {
    button.onclick = async () => {
      const label = button.innerHTML;
      page.querySelectorAll("[data-pick]").forEach((other) => { other.disabled = true; });
      button.innerHTML = `<span class="spinner"></span>${esc(T.media.picking)}`;
      try {
        const result = await api("/api/pick", { kind: button.dataset.pick, folder_id: folderId || null });
        if (!result.chosen) toast(T.media.nothingChosen);
        else if (result.imported) toast(T.media.added(result.imported));
        if (result.failed.length) toast(T.media.failed(result.failed.join("、")), "error");
        await refresh();
      } catch (error) {
        toast(error.message, "error");
        button.innerHTML = label;
        page.querySelectorAll("[data-pick]").forEach((other) => { other.disabled = false; });
      }
    };
  });
}

// Takes files out of the library, and their originals to the recycle bin if the user ticks it.
// A file still on some project's timeline is refused, with the projects named.
function removeAssets(ids, assets) {
  const named = assets.filter((asset) => ids.includes(asset.id));
  const what = named.length === 1 ? T.media.removeOne(named[0].name) : T.media.removeMany(named.length);
  const present = named.some((asset) => !asset.missing);
  return new Promise((resolve) => {
    modal(`<h2>${esc(T.media.removeTitle(what))}</h2><p>${esc(T.media.removeBody)}</p>
      ${present ? `<label class="check-line"><input type="checkbox" data-recycle><span>${esc(T.media.recycleToo)}<br><small style="color:var(--muted)">${esc(T.media.recycleHint)}</small></span></label>` : ""}
      <div class="foot"><button class="btn" data-no>${esc(T.newProject.cancel)}</button><button class="btn danger" data-yes>${icon("trash", 15)}${esc(T.media.remove)}</button></div>`,
    (box, close) => {
      $("[data-no]", box).onclick = () => { close(); resolve(false); };
      $("[data-yes]", box).onclick = async () => {
        const recycle = Boolean($("[data-recycle]", box)?.checked);
        close();
        try {
          const result = await api("/api/assets/delete", { ids, recycle });
          toast(T.media.removed(result.removed.length));
          if (result.not_recycled?.length) toast(T.media.notRecycled(result.not_recycled.join("、")), "error");
          resolve(true);
        } catch (error) {
          if (error.status === 409 && error.data) showInUse(error.data);
          else toast(error.message, "error");
          resolve(false);
        }
      };
    });
  });
}

function showInUse(data) {
  const used = data.in_use.length ? `<p>${esc(T.media.inUseBody)}</p><ul class="used-list">${data.in_use.map((item) =>
    `<li>${esc(item.name)}${item.projects.length ? `<br><small>${esc(item.projects.join("、"))}</small>` : ""}
      ${item.plans?.length ? `<br><small>${esc(T.media.inPlans(item.plans.join("、")))}</small>` : ""}</li>`).join("")}</ul>` : "";
  const busy = data.busy.length ? `<p>${esc(T.media.busyBody(data.busy.join("、")))}</p>` : "";
  modal(`<h2>${esc(T.media.inUseTitle)}</h2>${used}${busy}
    <div class="foot"><button class="btn primary" data-ok>${esc(T.common.close)}</button></div>`, (box, close) => {
    $("[data-ok]", box).onclick = close;
  });
}

// ---------------------------------------------------------------- AI clients

async function aiPage(page) {
  page.innerHTML = head(T.ai.title, T.ai.subtitle,
    `<button class="btn" data-refresh>${icon("refresh")}${esc(T.ai.refresh)}</button>`)
    + `<div class="callout">${icon("info")}<div><strong style="color:var(--text)">${esc(T.ai.dataTitle)}</strong><br>${esc(T.ai.dataBody)}</div></div>
       <div data-clients class="list">${'<div class="row skeleton" style="height:64px"></div>'.repeat(3)}</div>
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
    else await PAGES[key](page, id ? decodeURIComponent(id) : null);
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

// ---------------------------------------------------------------- work going on

// Every job, whoever started it — mostly the AI, in the background — floating over the
// bottom of the sidebar, above 「有新版」 and 「儲存空間」 so neither is ever covered.
// Provisional (roadmap §13): how often the panel asks what is going on.
const JOB_POLL_MS = 2500;
const SHOWN_JOBS = 5;

function placeJobs(panel) {
  const update = $("#update");
  const below = update && !update.hidden ? update : $('.nav-item[data-page="storage"]');
  const top = below ? below.getBoundingClientRect().top : window.innerHeight;
  panel.style.bottom = `${Math.max(8, window.innerHeight - top + 8)}px`;
}

const jobKind = (job) => job.kind === "analyze" ? T.jobs.analyze : job.preview ? T.jobs.preview : T.jobs.render;

function jobLine(job) {
  let state;
  if (job.status === "queued") state = job.ahead ? T.jobs.queuedBehind(job.ahead) : T.jobs.queued;
  else {
    state = T.jobs.progress(Math.round(job.progress * 100));
    if (job.remaining_seconds) state += ` · ${T.jobs.remaining(Math.max(1, Math.round(job.remaining_seconds / 60)))}`;
  }
  return `<div class="job"><div class="job-head"><span class="job-kind">${esc(jobKind(job))}</span>
      <span class="job-name" title="${esc(job.name)}">${esc(job.name || T.jobs.unnamed)}</span>
      <button class="tool" data-stop="${esc(job.job_id)}" title="${esc(T.jobs.stop)}">${icon("stop", 14)}</button></div>
    <div class="job-bar"><i style="width:${Math.round(job.progress * 100)}%"></i></div>
    <div class="job-state">${esc(state)}</div></div>`;
}

// The cards on the library and projects pages say what is happening to them, too.
function markCards(jobs) {
  document.querySelectorAll(".job-badge").forEach((badge) => badge.remove());
  for (const job of jobs) {
    const id = job.kind === "analyze" ? job.asset_id : job.project_id;
    const meta = id && document.querySelector(`.card[data-id="${CSS.escape(id)}"] .card-meta`);
    if (!meta) continue;
    const percent = Math.round(job.progress * 100);
    const text = job.status === "queued" ? T.jobs.queued
      : job.kind === "analyze" ? T.jobs.badgeAnalyzing(percent) : T.jobs.badgeRendering(percent);
    meta.insertAdjacentHTML("beforeend", `<span class="badge job-badge"><span class="spinner"></span>${esc(text)}</span>`);
  }
}

function announce(job) {
  if (job.status === "completed") {
    const element = document.createElement("div");
    element.className = "toast ok";
    element.innerHTML = `${icon("check")}<span>${esc(T.jobs.done(jobKind(job), job.name || T.jobs.unnamed))}</span>`
      + (job.delivered ? `<button class="btn small" data-reveal>${esc(T.jobs.reveal)}</button>` : "");
    $("#toasts").append(element);
    $("[data-reveal]", element)?.addEventListener("click", () => api("/api/reveal", { job_id: job.job_id }).catch(() => {}));
    setTimeout(() => element.remove(), 9000);
  } else if (job.status === "failed") {
    toast(T.jobs.failed(jobKind(job), job.name || T.jobs.unnamed), "error");
  }
}

function watchJobs() {
  const panel = document.createElement("div");
  panel.className = "jobs-float";
  panel.hidden = true;
  document.body.append(panel);
  let told = null;
  // Asked even while the window is in the background, so a job that ends then is still announced.
  const poll = async () => {
    let found;
    try { found = await api("/api/jobs/active"); } catch { return; }
    // What ended before this page opened was not news to anybody here.
    if (told === null) told = new Set(found.ended.map((job) => job.job_id));
    for (const job of found.ended) {
      if (!told.has(job.job_id)) { told.add(job.job_id); announce(job); }
    }
    markCards(found.jobs);
    panel.hidden = !found.jobs.length;
    if (!found.jobs.length) return;
    const more = found.jobs.length - SHOWN_JOBS;
    panel.innerHTML = `<div class="jobs-title">${esc(T.jobs.title(found.jobs.length))}</div>`
      + found.jobs.slice(0, SHOWN_JOBS).map(jobLine).join("")
      + (more > 0 ? `<div class="job-state">${esc(T.jobs.more(more))}</div>` : "");
    placeJobs(panel);
    panel.querySelectorAll("[data-stop]").forEach((button) => {
      button.onclick = async () => {
        const job = found.jobs.find((item) => item.job_id === button.dataset.stop);
        if (!(await confirmBox(T.jobs.stopTitle(jobKind(job), job.name || T.jobs.unnamed), T.jobs.stopBody, T.jobs.stop, true))) return;
        try {
          await api("/api/jobs/cancel", { job_id: job.job_id });
          toast(T.jobs.stopped);
          poll();
        } catch (error) { toast(error.message, "error"); }
      };
    });
  };
  window.addEventListener("resize", () => { if (!panel.hidden) placeJobs(panel); });
  poll();
  setInterval(poll, JOB_POLL_MS);
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

  // The update link can appear after the panel has been placed; place it again then.
  checkForUpdate().then(() => { const panel = $(".jobs-float"); if (panel && !panel.hidden) placeJobs(panel); });
  watchJobs();

  // The server stops once the window stops checking in.
  setInterval(() => fetch("/api/ping").catch(() => {}), 10000);
  window.addEventListener("hashchange", route);
  route();
}

start();
