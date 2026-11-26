/** Two-tab confirmation UI.
 *  "Proposals" — review and edit proposals from anchored clusters.
 *  "Final check" — person-centric last look before anything is written to Immich
 *  (the only write path; development safety = no write-scoped API key configured).
 *  Behavioral source: branch prototype/ui-two-tabs (ticket 02 verdict).
 */

const API_BASE = "";
let state = {
  view: "proposals", // "proposals" | "final"
  clusters: [],
  activeDay: null,
  activeKey: null, // "window:cluster" — cluster numbers repeat across windows
  evidence: [],
  evidenceLoading: false,
  assignment: null,
  people: [],
  configKnobs: null,
  immichUrl: "",
  showLabeled: false, // queue hides fully-labeled and decided groups unless asked
  selected: new Set(), // proposal-card selection keys "asset:appearance"
  expanded: null, // card key currently showing the full-photo overlay
  final: null, // /api/final-check payload
  finalPerson: null,
  finalSelected: new Set(), // final-check item keys, default OFF
  finalNames: {}, // retargeted write-as names per item key
  jobActive: false,
  jobWindows: 0,
  jobLast: null,
  coreDown: false,
};
let evidenceToken = 0; // discards stale fetches when the selection moves on

function keyOf(row) {
  return `${row.window}:${row.cluster}`;
}
function akey(e) {
  return `${e.asset_id}:${e.appearance}`;
}
function activeRow() {
  return state.clusters.find(c => keyOf(c) === state.activeKey) || null;
}
// Fully-labeled groups and decided (confirmed/skipped) clusters are hidden
// from the queue unless the operator asks for them.
function visibleClusters() {
  return state.clusters.filter(
    c => state.showLabeled || (!c.fully_labeled && c.status === "open")
  );
}

/* ---------- API ---------- */
async function fetchJson(path) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`);
  } catch (err) {
    noteCoreDown();
    throw err;
  }
  if (!res.ok) {
    if (res.status === 502) noteCoreDown(); // UI proxies /api/* to the core
    throw new Error(`${res.status} ${res.statusText}`);
  }
  noteCoreUp();
  return res.json();
}

async function postJson(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function noteCoreDown() {
  if (state.coreDown) return;
  state.coreDown = true;
  render();
}
function noteCoreUp() {
  if (!state.coreDown) return;
  state.coreDown = false;
  render();
}

// The core reports its own Immich base_url, which is often a localhost
// address (correct for the core, useless in a browser on another machine).
// Rewrite localhost to the host the browser actually used to reach this UI,
// so "open in Immich" links work when the page is opened via the LAN name.
// A real hostname (e.g. an explicit [ui] immich_url override) is kept as-is.
function resolveImmichUrl(raw) {
  const url = (raw ?? "").replace(/\/$/, "");
  if (!url) return "";
  try {
    const u = new URL(url);
    if (["localhost", "127.0.0.1", "[::1]", "::1"].includes(u.hostname) && window.location.hostname) {
      u.hostname = window.location.hostname;
    }
    return u.toString().replace(/\/$/, "");
  } catch {
    return url;
  }
}

async function loadInitialState() {
  const data = await fetchJson("/api/state");
  state = {
    ...state,
    ...data,
    configKnobs: data.config_knobs ?? null,
    // A UI on another machine cannot use the core's own Immich URL (it may
    // be a localhost address); the UI server can inject a reachable one.
    immichUrl: resolveImmichUrl(window.__REID_UI__?.immichUrl ?? data.immich_url),
    uiCommit: window.__REID_UI__?.commit ?? null,
    coreCommit: data.core_commit ?? null,
  };
  state.activeKey = state.clusters.length ? keyOf(state.clusters[0]) : null;
  state.activeDay = state.clusters.length ? state.clusters[0].window : null;
  render();
}

async function loadEvidence(cluster, window) {
  const token = ++evidenceToken;
  try {
    const data = await fetchJson(`/api/evidence?cluster=${cluster}&window=${encodeURIComponent(window)}`);
    if (token !== evidenceToken) return;
    state.evidence = data;
    state.evidenceLoading = false;
    renderEvidence();
    const row = activeRow();
    if (row) {
      state.assignment = buildAssignment(row);
      renderPane();
    }
  } catch (err) {
    if (token !== evidenceToken) return;
    state.evidenceLoading = false;
    toast(`failed to load evidence: ${err.message}`);
  }
}

async function loadPeople() {
  const data = await fetchJson("/api/people");
  state.people = data.people || [];
}

async function loadFinal() {
  render();
  try {
    state.final = await fetchJson("/api/final-check");
    if (!state.final.persons.some(p => p.name === state.finalPerson)) {
      state.finalPerson = state.final.persons[0]?.name ?? null;
    }
    // Written items and items retargeted away from their person drop out.
    const live = new Set(state.final.persons.flatMap(p => p.items.map(akey)));
    for (const key of [...state.finalSelected]) if (!live.has(key)) state.finalSelected.delete(key);
    for (const key of Object.keys(state.finalNames)) if (!live.has(key)) delete state.finalNames[key];
  } catch (err) {
    state.final = { persons: [], error: err.message };
  }
  render();
}

/* ---------- small helpers ---------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m]));
}
function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.getElementById("toasthost").replaceChildren(t);
  setTimeout(() => t.remove(), 3200);
}
function assetLabel(assetId) {
  const card = state.evidence.find(e => e.asset_id === assetId)
    || state.final?.persons.flatMap(p => p.items).find(e => e.asset_id === assetId);
  return (card && card.asset_name) || `${assetId.slice(0, 8)}…`;
}
function statusChip(status) {
  const map = {
    will_write: ["ok", "will write"],
    already_labeled: ["acc", "already labeled"],
    conflict_skip: ["bad", "label conflict"],
    no_head_skip: ["warn", "no face found"],
    ask_for_name: ["", "needs a name"],
    rejected: ["bad", "excluded"],
  };
  return map[status] || ["", status];
}
function cropImg(e, w, h, withContext) {
  if (!e.asset_id) return "";
  const extra = withContext ? "&expand=0.6&outline=1&neighbours=1" : "";
  return `<img class="crop" loading="lazy" width="${w}" height="${h}"
    src="/api/crop?asset_id=${encodeURIComponent(e.asset_id)}&appearance=${e.appearance}${extra}"
    alt="person crop">`;
}
function renderCrops(crops, w, h) {
  return crops.map(c => cropImg(c, w, h)).join("");
}
function pausedByRun() {
  if (!state.jobActive) return false;
  toast("A pipeline run is in progress — writes are paused until it finishes.");
  return true;
}

/* ---------- top bar / banner ---------- */
function bannerHTML() {
  if (state.coreDown) {
    return `<div class="banner down">Can't reach the server — is the core running? Retrying…</div>`;
  }
  if (state.jobActive) {
    const n = state.jobWindows;
    return `<div class="banner run"><span id="jobProgressText">Pipeline running — ${n} ${n === 1 ? "day" : "days"} done</span> · writes are paused <button id="cancelRun">Cancel run</button></div>`;
  }
  return "";
}

function lastRunChipHTML() {
  const run = state.jobLast;
  if (state.jobActive || !run || run.state === "done") return "";
  const label =
    run.state === "cancelled" ? "last run cancelled" :
    run.state === "interrupted" ? "last run interrupted" : "last run failed";
  return `<span class="chip bad" title="${esc(run.error || label)}">${label}</span>`;
}

function tabsHTML() {
  return `
    <div class="tabs">
      <button class="tab ${state.view === "proposals" ? "sel" : ""}" data-view="proposals">Proposals</button>
      <button class="tab ${state.view === "final" ? "sel" : ""}" data-view="final">Final check</button>
    </div>`;
}

function topbarHTML() {
  const isP = state.view === "proposals";
  return `
  <div class="top">
    ${tabsHTML()}
    ${isP ? `<span class="chip acc">${visibleClusters().filter(c => c.status === "open").length} open</span>` : `<span class="chip acc">${(state.final?.persons ?? []).reduce((n, p) => n + p.count, 0)} appearances</span>`}
    ${lastRunChipHTML()}
    <span class="spacer"></span>
    <span class="ver mono muted" title="running code: ui ${esc(state.uiCommit ?? "?")} · core ${esc(state.coreCommit ?? "?")}">ui ${esc(state.uiCommit ?? "?")} · core ${esc(state.coreCommit ?? "?")}</span>
    ${isP ? `
      <label title="Groups that are fully labeled or already decided have nothing pending — show them here too">
        <input type="checkbox" id="showLabeledToggle" ${state.showLabeled ? "checked" : ""}> show decided</label>
      <button id="gearBtn" title="Settings">⚙</button>` : ""}
    <button id="runNow" ${state.jobActive ? "disabled" : ""} title="Run the pipeline now: look for new photos and regroup everything">${state.jobActive ? "Running…" : "Run now"}</button>
    <div class="cfgpop" id="cfgPop">${configPopoverHTML()}</div>
  </div>`;
}

function configPopoverHTML() {
  const k = state.configKnobs;
  if (!k) return `<div class="muted">settings unavailable</div>`;
  const rows = [
    ["windowDays", "Days grouped together", k.window_days, 1, 30, 1],
    ["distanceThreshold", "How alike faces must be to group", k.distance_threshold, 0.2, 1, 0.01],
    ["confidenceFloor", "Minimum detection confidence", k.confidence_floor, 0, 1, 0.01],
    ["minBoxHeight", "Minimum person size (pixels)", k.min_box_height, 40, 300, 10],
    ["faceOverlap", "How much a face must sit inside its person", k.face_overlap, 0.1, 1, 0.05],
  ];
  return rows.map(([key, label, value, min, max, step]) =>
    `<label>${label} <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-cfg="${key}"> <span class="mono" data-cfg-val="${key}">${value}</span></label>`
  ).join("")
    + `<button class="primary" id="cfgSave" style="width:100%;margin-top:8px">Save settings</button>`
    + `<div class="muted">Changes apply on the next run.</div>`;
}

/* ---------- render shell ---------- */
function render() {
  const root = document.getElementById("app");
  root.classList.toggle("final", state.view === "final");
  root.innerHTML = `
    ${bannerHTML()}
    ${topbarHTML()}
    ${state.view === "proposals" ? proposalsHTML() : finalHTML()}
  `;
  wireTopEvents();
  if (state.view === "proposals") {
    wireProposalEvents();
    renderEvidence();
    renderPane();
  } else {
    wireFinalEvents();
  }
}

function wireTopEvents() {
  document.querySelectorAll(".tab[data-view]").forEach(el =>
    el.addEventListener("click", () => setView(el.dataset.view))
  );
  document.getElementById("gearBtn")?.addEventListener("click", () =>
    document.getElementById("cfgPop")?.classList.toggle("open")
  );
  document.querySelectorAll("#cfgPop input[data-cfg]").forEach(inp => {
    inp.addEventListener("input", () => {
      const key = inp.dataset.cfg;
      const val = document.querySelector(`[data-cfg-val="${key}"]`);
      if (val) val.textContent = inp.value;
    });
  });
  document.getElementById("cfgSave")?.addEventListener("click", saveConfig);
  document.getElementById("showLabeledToggle")?.addEventListener("change", e => {
    state.showLabeled = e.target.checked;
    try { localStorage.setItem("reid-show-labeled", state.showLabeled ? "1" : "0"); } catch { /* private mode */ }
    if (!state.showLabeled) {
      const active = activeRow();
      if (active && (active.fully_labeled || active.status !== "open")) {
        const first = visibleClusters()[0];
        if (first) {
          state.activeKey = keyOf(first);
          state.activeDay = first.window;
          state.evidence = [];
          state.evidenceLoading = true;
          loadEvidence(first.cluster, first.window);
          state.assignment = buildAssignment(first);
        }
      }
    }
    render();
  });
  document.getElementById("runNow")?.addEventListener("click", runNow);
  document.getElementById("cancelRun")?.addEventListener("click", cancelRun);
}

function setView(view) {
  state.view = view;
  if (view === "final") loadFinal();
  else render();
}

/* ---------- Proposals tab ---------- */
function proposalsHTML() {
  return `
    <div class="days">${daysHTML()}</div>
    <div class="clusters">${clustersHTML()}</div>
    <div class="main">${mainHTML()}</div>
    <div class="pane">${paneHTML()}</div>`;
}

function daysHTML() {
  const groups = new Map();
  for (const c of visibleClusters()) {
    if (!groups.has(c.window)) groups.set(c.window, []);
    groups.get(c.window).push(c);
  }
  const days = [...groups.keys()].sort((a, b) => b.localeCompare(a));
  if (state.activeDay === null && days.length) state.activeDay = days[0];
  const rows = days.map(day => {
    const clusterRows = groups.get(day);
    const openCount = clusterRows.filter(r => r.status === "open").length;
    return `
      <div class="dayrow ${day === state.activeDay ? "sel" : ""}" data-day="${day}">
        <div class="dayname">${day}</div>
        <span class="chip ${openCount ? "acc" : ""}">${openCount}/${clusterRows.length}</span>
      </div>`;
  });
  return rows.join("") || `<div class="empty muted">no clusters yet — run the batch</div>`;
}

function clustersHTML() {
  const rows = visibleClusters().filter(c => c.window === state.activeDay);
  if (!rows.length) return `<div class="empty muted">select a day</div>`;
  return rows.map(c => `
    <div class="item ${keyOf(c) === state.activeKey ? "sel" : ""} ${c.status !== "open" ? "done" : ""}" data-key="${keyOf(c)}">
      <div class="item-row">
        <div class="strip">${renderCrops(c.crops || [], 26, 36)}</div>
        <div class="item-info">
          <div class="item-name">${c.proposed_name ? esc(c.proposed_name) : '<span class="muted">unnamed</span>'}</div>
          <div class="item-meta mono">group ${c.cluster} · ${(c.confidence * 100).toFixed(0)}% · ${c.appearances} photos · ${c.status === "confirmed" ? "confirmed" : c.status === "skipped" ? "skipped" : ""}</div>
        </div>
      </div>
    </div>`).join("");
}

function mainHTML() {
  const c = activeRow();
  if (!c) {
    return `<div class="main-header"><span class="cluster-id mono">—</span><h2 class="muted">Select a cluster</h2></div><div class="evgrid"></div>`;
  }
  return `
    <div class="main-header">
      <span class="cluster-id mono">${c.window} / ${c.cluster}</span>
      <h2>${c.proposed_name ? esc(c.proposed_name) : '<span class="muted">unnamed cluster</span>'}</h2>
    </div>
    <div class="muted cluster-anchor">${c.anchor || "no Immich label found"} · group confidence ${(c.confidence * 100).toFixed(0)}%</div>
    <div class="evgrid" id="evgrid"></div>`;
}

/* The evidence card: clickable photo, full-photo overlay with all sibling
 * outlines, Immich / ReID / You rows, selection checkbox, per-card actions. */
function cardHTML(e) {
  const [chipClass, chipText] = statusChip(e.status);
  const sel = state.selected.has(akey(e));
  const link = state.immichUrl ? `${state.immichUrl}/photos/${encodeURIComponent(e.asset_id)}` : null;
  const youRow = e.pin ? `pinned: ${esc(e.pin)}`
    : e.ignored ? "ignored — never labeled"
    : e.rejected_name ? `excluded — not “${esc(e.rejected_name)}”`
    : e.dismissed ? "cluster dismissed"
    : null;
  const rejectable = e.proposed_name && e.status !== "rejected" && !e.ignored && e.status !== "already_labeled";
  const wrongLabelable = e.status === "already_labeled" && e.immich_label && !e.ignored;
  return `
  <div class="evcard ${sel ? "selected" : ""} ${e.ignored ? "evignored" : ""}" data-asset="${e.asset_id}" data-appearance="${e.appearance}">
    ${state.expanded === akey(e) ? fullPhotoHTML(e) : `
    <div class="cropwrap">
      ${link ? `<a href="${link}" target="_blank" rel="noopener" title="open in Immich">${cropImg(e, 150, 170, true)}</a>` : cropImg(e, 150, 170, true)}
      <button class="photobtn" data-expand="${akey(e)}" title="show the whole photo with every detected person outlined">full photo</button>
    </div>`}
    <label class="selrow" title="select for bulk actions (Ignore / Pin as… / Not this person)">
      <input type="checkbox" data-sel="${akey(e)}" ${sel ? "checked" : ""}> select</label>
    <div class="inforow"><b>Immich:</b><span>${e.immich_label ? esc(e.immich_label) : '<span class="muted">—</span>'}</span></div>
    <div class="inforow"><b>ReID:</b><span>${e.proposed_name ? esc(e.proposed_name) : '<span class="muted">unnamed cluster</span>'} <span class="chip ${chipClass}">${esc(chipText)}</span></span></div>
    <div class="inforow"><b>You:</b><span>${youRow ? youRow : '<span class="muted">—</span>'}</span></div>
    <div class="date-conf">${esc(e.date)} · ${(e.confidence * 100).toFixed(0)}%</div>
    ${link ? `<div class="url mono"><a href="${link}" target="_blank" rel="noopener">${esc(link)}</a></div>` : ""}
    <div class="chips">
      ${e.sibling_count ? `<span class="chip" title="other detected persons in this photo — expand to see them outlined">${e.sibling_count} other${e.sibling_count === 1 ? "" : "s"} in photo</span>` : ""}
      ${e.conflict_names?.length >= 2 ? `<span class="chip warn">conflict: ${esc(e.conflict_names.join(" / "))}</span>` : ""}
    </div>
    <div class="cardbtns">
      ${e.ignored
        ? `<button data-unignore="${akey(e)}">Undo ignore</button>`
        : `<button data-ignore="${akey(e)}" data-reason="not_a_person" title="Not a person — excluded from clustering, never labeled">Not a person</button><button data-ignore="${akey(e)}" data-reason="ignore" title="Ignore this person — excluded from clustering, never labeled">Ignore this person</button>`}
      ${e.pin
        ? `<button data-unpin="${akey(e)}">Unpin</button><button data-assign="${akey(e)}">Reassign person…</button>`
        : `<button data-assign="${akey(e)}" title="Pin as a person — proposed as that person on the next batch run">Pin as… for next run</button>`}
      ${rejectable ? `<button data-reject="${akey(e)}" data-name="${esc(e.proposed_name)}">Not this person</button>` : ""}
      ${e.status === "rejected" ? `<button data-unreject="${akey(e)}" data-name="${esc(e.rejected_name ?? "")}">Undo exclusion</button>` : ""}
      ${wrongLabelable ? `<button data-wronglabel="${akey(e)}" data-name="${esc(e.immich_label)}">Wrong label</button>` : ""}
    </div>
    ${e.conflict_names?.length >= 2 && !e.ignored ? `
    <div class="resolve-row">
      <span class="muted">resolve:</span>
      ${e.conflict_names.map(name => `<button class="b-resolve ${e.pin === name ? "resolved" : ""}" data-resolve="${akey(e)}" data-name="${esc(name)}">${esc(name)}</button>`).join("")}
    </div>` : ""}
  </div>`;
}

/* Full photo with every stored appearance outlined; the card's own appearance
 * highlighted. Boxes are pixel-space; the SVG viewBox follows the loaded
 * image's natural size, so they map 1:1. */
function fullPhotoHTML(e) {
  const rects = (e.siblings || []).map(s => {
    const self = s.appearance === e.appearance;
    return `<rect class="pbox ${self ? "self" : ""}" x="${s.box[0]}" y="${s.box[1]}"
      width="${s.box[2] - s.box[0]}" height="${s.box[3] - s.box[1]}"/>
      ${self ? `<text class="ptag" x="${s.box[0]}" y="${Math.max(s.box[1] - 4, 10)}">this appearance</text>` : ""}`;
  }).join("");
  return `
  <div class="fullwrap">
    <img class="fullphoto" src="/api/crop?full=1&asset_id=${encodeURIComponent(e.asset_id)}" alt="full photo"
      onload="fitOverlay(this)">
    <svg class="overlay" onload="fitOverlay(this.previousElementSibling)">${rects}</svg>
    <button class="photobtn" data-expand="${akey(e)}">crop</button>
  </div>`;
}
function fitOverlay(img) {
  const svg = img?.parentElement?.querySelector("svg.overlay");
  if (svg && img.naturalWidth) svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
}

function renderEvidence() {
  const grid = document.getElementById("evgrid");
  if (!grid) return;
  if (state.evidenceLoading) {
    grid.innerHTML = `<div class="evloading">loading photos…</div>`;
    return;
  }
  grid.innerHTML = state.evidence.map(cardHTML).join("");
  // Crop fade-in, same as before.
  grid.querySelectorAll("img.crop, img.fullphoto").forEach(img => {
    const done = () => img.classList.add("loaded");
    if (img.complete) done();
    else {
      img.addEventListener("load", done, { once: true });
      img.addEventListener("error", done, { once: true });
    }
  });
  wireCardEvents();
}

function paneHTML() {
  const a = state.assignment;
  if (!a) return `<div class="name-display">—</div><div class="cluster-meta muted">Select a cluster</div>`;
  const hasName = a.proposed_name != null;
  const selCount = state.selected.size;
  return `
    <div class="name-display">${hasName ? esc(a.proposed_name) : '<span class="muted">—</span>'}</div>
    <div class="cluster-meta muted">${a.total_appearances} appearances · ${a.writable} to write · ${a.conflicted} label conflicts · ${a.labeled} already labeled</div>
    ${!hasName ? `<button class="b-name primary" style="width:100%;margin-bottom:10px">Name this cluster</button>` : ""}
    <div class="name-prompt">
      <label class="muted">Assign to</label>
      <input type="text" id="nameInput" list="peopleList" value="${hasName ? esc(a.proposed_name) : ""}" ${hasName ? "" : 'placeholder="Type or pick an existing person…"'} ${state.activeKey === null ? "disabled" : ""}>
      <datalist id="peopleList">${state.people.map(p => `<option value="${esc(p)}">`).join("")}</datalist>
      <div class="hint">existing person → attaches · new name → creates the person on write. Naming pins the cluster's appearances to that person on the next run.</div>
    </div>
    <div class="actions">
      ${state.activeKey !== null ? `
        <button class="b-ok primary" ${state.jobActive ? "disabled" : ""}>Confirm proposal</button>
        <button class="b-skip">Skip cluster</button>` : `<div class="chip">Select a cluster</div>`}
    </div>
    ${a.cluster_status !== "open" && state.activeKey !== null ? `
      <button class="b-reopen" data-key="${a.window}:${a.cluster}" style="width:100%;margin-top:8px">Reopen cluster</button>` : ""}
    <hr>
    <div class="bulkbar">
      <b>${selCount} selected</b>
      <div class="bulkrow">
        <button data-bulk="selall" ${(!state.evidence.length || state.selected.size === state.evidence.length || state.jobActive) ? "disabled" : ""}>Select all</button>
        <button data-bulk="selnone" ${(!state.selected.size || state.jobActive) ? "disabled" : ""}>Deselect all</button>
      </div>
      <div class="bulkrow">
        <button data-bulk="ignore" ${!selCount || state.jobActive ? "disabled" : ""}>Ignore selected</button>
        <button data-bulk="unignore" ${!selCount || state.jobActive ? "disabled" : ""}>Undo ignore</button>
        <button data-bulk="reject" ${!selCount || state.jobActive ? "disabled" : ""}>Not this person</button>
      </div>
      <div class="bulkrow">
        <button data-bulk="pin" ${!selCount || state.jobActive ? "disabled" : ""}>Pin selected as… for next run</button>
      </div>
      <div class="hint">Bulk actions apply to the visible cards only. “Pin as X” takes effect on the next pipeline run; reassigning in Final check writes as X now.</div>
    </div>
    <hr>
    <div class="footer muted">⚙ Settings (top right) adjust how photos are grouped — changes apply on the next run.</div>`;
}

function renderPane() {
  const pane = document.querySelector(".pane");
  if (pane) {
    pane.innerHTML = paneHTML();
    wirePaneEvents();
  }
}

function buildAssignment(clusterRow) {
  const w = state.evidence.filter(e => e.cluster === clusterRow.cluster && e.window === clusterRow.window);
  if (!w.length && clusterRow.appearances != null) {
    return {
      cluster: clusterRow.cluster, window: clusterRow.window, cluster_status: clusterRow.status,
      proposed_name: clusterRow.proposed_name, anchor: clusterRow.anchor,
      total_appearances: clusterRow.appearances, writable: clusterRow.writable,
      conflicted: clusterRow.conflicted, labeled: clusterRow.labeled,
    };
  }
  return {
    cluster: clusterRow.cluster, window: clusterRow.window, cluster_status: clusterRow.status,
    proposed_name: clusterRow.proposed_name, anchor: clusterRow.anchor,
    total_appearances: w.length,
    writable: w.filter(e => e.status === "will_write").length,
    conflicted: w.filter(e => e.status === "conflict_skip").length,
    labeled: w.filter(e => e.status === "already_labeled").length,
  };
}

/* ---------- Proposals events ---------- */
function wireProposalEvents() {
  document.querySelectorAll(".days .dayrow").forEach(el =>
    el.addEventListener("click", () => selectDay(el.dataset.day))
  );
  document.querySelectorAll(".clusters .item[data-key]").forEach(el =>
    el.addEventListener("click", () => selectCluster(el.dataset.key))
  );
}

function selectDay(day) {
  state.activeDay = day;
  const rows = visibleClusters().filter(c => c.window === day);
  const first = rows.find(c => c.status === "open") || rows[0];
  if (first) selectCluster(keyOf(first));
  else render();
}

function selectCluster(key) {
  state.activeKey = key;
  const row = activeRow();
  if (row) {
    state.activeDay = row.window;
    state.evidence = [];
    state.evidenceLoading = true;
    loadEvidence(row.cluster, row.window);
    state.assignment = buildAssignment(row);
  }
  render();
}

function wireCardEvents() {
  document.querySelectorAll("[data-sel]").forEach(el =>
    el.addEventListener("change", () => {
      const k = el.dataset.sel;
      if (el.checked) state.selected.add(k);
      else state.selected.delete(k);
      renderEvidence();
      renderPane();
    })
  );
  document.querySelectorAll("[data-expand]").forEach(el =>
    el.addEventListener("click", () => {
      const k = el.dataset.expand;
      state.expanded = state.expanded === k ? null : k;
      renderEvidence();
    })
  );
  document.querySelectorAll("[data-ignore]").forEach(el =>
    el.addEventListener("click", () => ignoreAppearance(el.dataset.ignore, el.dataset.reason)));
  document.querySelectorAll("[data-unignore]").forEach(el =>
    el.addEventListener("click", () => unignoreAppearance(el.dataset.unignore)));
  document.querySelectorAll("[data-assign]").forEach(el =>
    el.addEventListener("click", () => assignPrompt(el.dataset.assign)));
  document.querySelectorAll("[data-unpin]").forEach(el =>
    el.addEventListener("click", () => unpinAppearance(el.dataset.unpin)));
  document.querySelectorAll("[data-reject]").forEach(el =>
    el.addEventListener("click", () => rejectAppearance(el.dataset.reject, el.dataset.name)));
  document.querySelectorAll("[data-unreject]").forEach(el =>
    el.addEventListener("click", () => unrejectAppearance(el.dataset.unreject, el.dataset.name)));
  document.querySelectorAll("[data-wronglabel]").forEach(el =>
    el.addEventListener("click", () => markWrongLabel(el.dataset.wronglabel, el.dataset.name)));
  document.querySelectorAll("[data-resolve]").forEach(el =>
    el.addEventListener("click", () => pinAppearance(el.dataset.resolve, el.dataset.name)));
}

function wirePaneEvents() {
  const nameInput = document.getElementById("nameInput");
  if (nameInput) {
    nameInput.addEventListener("change", () => {
      const v = nameInput.value.trim();
      if (v && state.assignment) {
        state.assignment.proposed_name = v;
        const row = activeRow();
        if (row) row.proposed_name = v;
        toast(`Group named “${v}” (${state.people.includes(v) ? "existing person" : "new person"}).`);
        render();
      }
    });
  }
  document.querySelector(".b-ok")?.addEventListener("click", confirmCluster);
  document.querySelector(".b-skip")?.addEventListener("click", () => {
    const c = activeRow();
    if (c) dismissCluster(c.cluster, c.window, true);
  });
  document.querySelector(".b-name")?.addEventListener("click", () => namePrompt());
  document.querySelector(".b-reopen")?.addEventListener("click", e => {
    const [window, cluster] = e.target.dataset.key.split(":");
    dismissCluster(Number(cluster), window, false);
    confirmClusterState(Number(cluster), window, false);
  });
  document.querySelectorAll("[data-bulk]").forEach(btn =>
    btn.addEventListener("click", () => bulkAction(btn.dataset.bulk)));
}

async function bulkAction(action) {
  if (action === "selall") {
    state.evidence.forEach(e => state.selected.add(akey(e)));
    renderEvidence();
    renderPane();
    return;
  }
  if (action === "selnone") {
    state.selected.clear();
    renderEvidence();
    renderPane();
    return;
  }
  if (pausedByRun() || !state.selected.size) return;
  const items = [...state.selected].map(k => {
    const [asset_id, appearance] = k.split(":");
    return { asset_id, appearance: parseInt(appearance, 10) };
  });
  let name = null;
  let reason = "ignore";
  if (action === "pin") {
    name = await nameModal("Pin selected as which person?", "Applies on the next pipeline run.");
    if (!name) return;
  }
  if (action === "reject") {
    const row = activeRow();
    name = row?.proposed_name;
    if (!name) { toast("name the cluster first — then 'not this person' means something"); return; }
  }
  if (action === "ignore") {
    reason = await nameModal(
      "Ignore as what?",
      "Not a person → excluded as a false detection · Ignore this person → excluded but kept as a person record."
    );
    if (!reason) return;
    if (reason.toLowerCase().includes("not a person")) reason = "not_a_person";
    else reason = "ignore";
  }
  const path = { ignore: "/api/bulk-ignore", unignore: "/api/bulk-unignore", pin: "/api/bulk-pin", reject: "/api/bulk-reject" }[action];
  try {
    const data = await postJson(path, { items, ...(name ? { name } : {}), ...(action === "ignore" ? { reason } : {}) });
    if (data.error) { toast(`bulk ${action} failed: ${data.error}`); return; }
    toast(`bulk ${action}: ${data.count} appearance(s) — ${action === "pin" ? `pinned as “${name}” for the next run` : action === "reject" ? `marked not “${name}”` : action === "ignore" ? `ignored as ${reason === "not_a_person" ? "not a person" : "ignore"}` : action}`);
    state.selected.clear();
    await loadInitialState();
    const row = activeRow();
    if (row) { state.evidence = []; state.evidenceLoading = true; loadEvidence(row.cluster, row.window); state.assignment = buildAssignment(row); }
    render();
  } catch (err) {
    toast(`bulk ${action} failed: ${err.message}`);
  }
}

/* ---------- Final check tab ---------- */
function finalHTML() {
  const persons = state.final?.persons ?? [];
  const person = persons.find(p => p.name === state.finalPerson) || persons[0];
  return `
    <div class="persons">${personsHTML(persons)}</div>
    <div class="finalmain">${finalMainHTML(person)}</div>`;
}

function personsHTML(persons) {
  if (!persons.length) return `<div class="empty muted">nothing to check — confirm proposals in the Proposals tab or run the batch</div>`;
  return persons.map(p => `
    <div class="personrow ${p.name === state.finalPerson ? "sel" : ""}" data-person="${esc(p.name)}">
      <b>${esc(p.name)}</b>
      <span class="chip acc">${p.count}</span>
    </div>`).join("");
}

function finalMainHTML(person) {
  if (state.final?.error) return `<div class="panerr">final check failed to load: ${esc(state.final.error)}</div>`;
  if (!person) return `<div class="empty muted">select a person</div>`;
  const selected = person.items.filter(i => state.finalSelected.has(akey(i)) && !i.written);
  const totalSelected = (state.final?.persons ?? []).reduce(
    (n, p) => n + p.items.filter(i => state.finalSelected.has(akey(i)) && !i.written).length, 0);
  const srcChip = { proposal: ["acc", "ReID proposal"], pin: ["acc", "your pin"], conflict: ["bad", "conflict"], written: ["ok", "in Immich"] };
  return `
    <div class="writebar">
      <button id="writePerson" class="primary" ${selected.length && !state.jobActive ? "" : "disabled"}>
        Write ${selected.length} selected as “${esc(person.name)}” to Immich</button>
      <button id="selAll">Select all</button>
      <button id="selNone">Deselect all</button>
      <button id="retargetSel" ${selected.length && !state.jobActive ? "" : "disabled"} title="reassign the selected appearances to another person — the write targets that person now">Write selected as…</button>
      <span class="spacer"></span>
      <button id="writeAll" class="primary" ${totalSelected && !state.jobActive ? "" : "disabled"}>
        Write all selected across people (${totalSelected})</button>
    </div>
    <div class="main" style="grid-row:auto;height:auto;overflow:visible;padding:12px">
      <h2 style="margin:4px 0">${esc(person.name)}</h2>
      <div class="hint" style="margin-bottom:10px">Every appearance assigned to ${esc(person.name)} — ReID proposals, your pins, conflicts. Checkboxes default OFF; checked items are written to Immich on confirm.</div>
      <div class="evgrid">${person.items.map(finalCardHTML).join("")}</div>
    </div>`;
}

function finalCardHTML(i) {
  const key = akey(i);
  const on = state.finalSelected.has(key);
  const link = state.immichUrl ? `${state.immichUrl}/photos/${encodeURIComponent(i.asset_id)}` : null;
  const [cc, ct] = { proposal: ["acc", "ReID proposal"], pin: ["acc", "your pin"], conflict: ["bad", "conflict"], written: ["ok", "in Immich"] }[i.source] || ["", i.source];
  const writeAs = state.finalNames[key] ?? i.name;
  return `
  <div class="evcard ${on ? "selected" : ""}" data-key="${key}">
    <div class="cropwrap">
      ${link ? `<a href="${link}" target="_blank" rel="noopener" title="open in Immich">${cropImg(i, 150, 140, false)}</a>` : cropImg(i, 150, 140, false)}
    </div>
    <label class="selrow" title="include in the write to Immich (default off)">
      <input type="checkbox" data-fsel="${key}" ${on ? "checked" : ""}${i.written ? " disabled title=\"already in Immich\"" : ""}> write to Immich</label>
    <div class="inforow"><b>Write as:</b><span><b>${esc(writeAs)}</b>${state.finalNames[key] ? ' <span class="chip warn">retargeted</span>' : ""}</span></div>
    <div class="inforow"><b>Source:</b><span><span class="chip ${cc}">${ct}</span>${i.written ? ' <span class="chip ok">written</span>' : ""}</span></div>
    <div class="date-conf mono muted">${esc(i.asset_name || i.asset_id)} · ${esc(i.window)} · group ${i.cluster}</div>
    <div class="cardbtns">
      <button data-fassign="${key}">Write as…</button>
    </div>
  </div>`;
}

function wireFinalEvents() {
  document.querySelectorAll(".personrow[data-person]").forEach(el =>
    el.addEventListener("click", () => { state.finalPerson = el.dataset.person; render(); }));
  document.querySelectorAll("[data-fsel]").forEach(el =>
    el.addEventListener("change", () => {
      const k = el.dataset.fsel;
      if (el.checked) state.finalSelected.add(k);
      else state.finalSelected.delete(k);
      render();
    }));
  document.querySelectorAll("[data-fassign]").forEach(el =>
    el.addEventListener("click", async () => {
      const name = await nameModal("Write this appearance as which person?", "The write targets this person NOW (the pin also persists for the next run).");
      if (!name) return;
      const key = el.dataset.fassign;
      const data = await postJson("/api/retarget", { items: [keyItem(key)], name });
      if (data.error) { toast(`retarget failed: ${data.error}`); return; }
      state.finalNames[key] = name;
      toast(`“${assetLabel(key.split(":")[0])}” will be WRITTEN as “${name}” now — not just pinned`);
      loadFinal();
    }));
  document.getElementById("selAll")?.addEventListener("click", () => {
    const person = state.final.persons.find(p => p.name === state.finalPerson);
    person?.items.forEach(i => state.finalSelected.add(akey(i)));
    render();
  });
  document.getElementById("selNone")?.addEventListener("click", () => {
    const person = state.final.persons.find(p => p.name === state.finalPerson);
    person?.items.forEach(i => state.finalSelected.delete(akey(i)));
    render();
  });
  document.getElementById("writePerson")?.addEventListener("click", () => writeSelected(state.finalPerson));
  document.getElementById("writeAll")?.addEventListener("click", writeAllSelected);
  document.getElementById("retargetSel")?.addEventListener("click", async () => {
    const person = state.final.persons.find(p => p.name === state.finalPerson);
    const items = person?.items.filter(i => state.finalSelected.has(akey(i))).map(i => ({ asset_id: i.asset_id, appearance: i.appearance })) ?? [];
    const name = await nameModal("Write all selected as which person?", "The writes target this person NOW; the pins persist for the next run.");
    if (!name) return;
    const data = await postJson("/api/retarget", { items, name });
    if (data.error) { toast(`retarget failed: ${data.error}`); return; }
    for (const i of items) state.finalNames[`${i.asset_id}:${i.appearance}`] = name;
    toast(`${data.count} selected will be WRITTEN as “${name}” now`);
    loadFinal();
  });
}

function keyItem(key) {
  const [asset_id, appearance] = key.split(":");
  return { asset_id, appearance: parseInt(appearance, 10) };
}

async function writeSelected(personName) {
  if (pausedByRun()) return;
  const person = state.final?.persons.find(p => p.name === personName) ?? state.final?.persons[0] ?? null;
  if (!person) return;
  const writes = (person.items ?? [])
    .filter(i => state.finalSelected.has(akey(i)) && !i.written)
    .map(i => ({ asset_id: i.asset_id, appearance: i.appearance, name: state.finalNames[akey(i)] ?? personName }));
  if (!writes.length) {
    const hasSelected = person.items.some(i => state.finalSelected.has(akey(i)));
    toast(hasSelected ? "all selected are already written to Immich" : "nothing selected");
    return;
  }
  try {
    const data = await postJson("/api/write-faces", { writes });
    if (data.error) { toast(`write failed: ${data.error}`); return; }
    const failed = data.results.filter(r => r.status && r.status !== "written").length;
    toast(`wrote ${data.written} faces to Immich as “${personName}”` + (failed ? ` — ${failed} skipped (see server log)` : ""));
    await loadInitialState();
    await loadFinal();
  } catch (err) {
    toast(`write failed: ${err.message}`);
  }
}

async function writeAllSelected() {
  if (pausedByRun()) return;
  const writes = (state.final?.persons ?? []).flatMap(p =>
    p.items
      .filter(i => state.finalSelected.has(akey(i)) && !i.written)
      .map(i => ({ asset_id: i.asset_id, appearance: i.appearance, name: state.finalNames[akey(i)] ?? p.name }))
  );
  if (!writes.length) { toast("nothing selected"); return; }
  try {
    const data = await postJson("/api/write-faces", { writes });
    if (data.error) { toast(`write failed: ${data.error}`); return; }
    const failed = data.results.filter(r => r.status && r.status !== "written").length;
    toast(`wrote ${data.written} faces to Immich across all selected people` + (failed ? ` — ${failed} skipped` : ""));
    await loadInitialState();
    await loadFinal();
  } catch (err) {
    toast(`write failed: ${err.message}`);
  }
}

/* ---------- cluster-level actions ---------- */
async function confirmCluster() {
  if (pausedByRun()) return;
  const c = activeRow();
  if (!c) return;
  let name = state.assignment?.proposed_name ?? c.proposed_name;
  if (!name) {
    name = await nameModal("Name this cluster first", "The name becomes the proposal; the write happens in Final check.");
    if (!name) return;
  }
  try {
    const data = await postJson("/api/confirm-cluster", { window: c.window, cluster: c.cluster, name, confirmed: true });
    if (data.error) { toast(`confirm failed: ${data.error}`); return; }
    toast(`cluster confirmed as “${name}” — its appearances are in Final check; the write happens there`);
    await loadInitialState();
    const first = visibleClusters()[0];
    if (first) selectCluster(keyOf(first));
    else render();
  } catch (err) {
    toast(`confirm failed: ${err.message}`);
  }
}

async function confirmClusterState(cluster, window, confirmed) {
  try {
    await postJson("/api/confirm-cluster", { window, cluster, confirmed });
  } catch { /* reopen path: best effort */ }
}

async function dismissCluster(cluster, window, dismissed) {
  try {
    const data = await postJson("/api/skip", { cluster, window, dismissed });
    if (data.error) { toast(`skip failed: ${data.error}`); return; }
    toast(dismissed ? "cluster dismissed — it stays out of the queue" : "cluster reopened");
    await loadInitialState();
    render();
  } catch (err) {
    toast(`skip failed: ${err.message}`);
  }
}

/* ---------- per-appearance actions (single) ---------- */
async function assignPrompt(key) {
  const name = await nameModal(
    "Pin as which person?",
    "This appearance will be proposed as that person on the next batch run, and appears in Final check now."
  );
  if (!name) return;
  await pinAppearance(key, name);
}
async function ignoreAppearance(key, reason = "ignore") {
  const item = keyItem(key);
  const data = await postJson("/api/ignore", { ...item, reason });
  if (data.error) { toast(`ignore failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) card.ignored = true;
  toast(
    reason === "not_a_person"
      ? `marked “${assetLabel(item.asset_id)}” as not a person — excluded from clustering`
      : `ignored “${assetLabel(item.asset_id)}” — excluded from clustering, never labeled`
  );
  renderEvidence(); renderPane();
}
async function unignoreAppearance(key) {
  const item = keyItem(key);
  const data = await postJson("/api/unignore", item);
  if (data.error) { toast(`unignore failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) card.ignored = false;
  toast("ignore removed — re-evaluated on the next batch run");
  renderEvidence(); renderPane();
}
async function pinAppearance(key, name) {
  const data = await postJson("/api/pin", { ...keyItem(key), name });
  if (data.error) { toast(`pin failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) card.pin = name;
  toast(`pinned as “${name}” — proposed as ${name} on the NEXT batch run, then written via Final check`);
  renderEvidence(); renderPane();
  if (state.view === "final") await loadFinal();
}
async function unpinAppearance(key) {
  const data = await postJson("/api/unpin", keyItem(key));
  if (data.error) { toast(`unpin failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) card.pin = null;
  toast("pin removed — normal anchoring applies from the next batch run on");
  renderEvidence(); renderPane();
}
async function rejectAppearance(key, name) {
  if (!name) { toast("name the cluster first — then 'not this person' means something"); return; }
  const data = await postJson("/api/reject", { ...keyItem(key), name });
  if (data.error) { toast(`reject failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) { card.status = "rejected"; card.rejected_name = name; }
  toast(`“${assetLabel(key.split(":")[0])}” will not be proposed as “${name}”`);
  renderEvidence(); renderPane();
}
async function unrejectAppearance(key, name) {
  if (!name) { toast("the excluded name is unknown here — undo it from the cluster that excluded it"); return; }
  const data = await postJson("/api/unreject", { ...keyItem(key), name });
  if (data.error) { toast(`undo failed: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) { card.status = "will_write"; card.rejected_name = null; }
  toast("exclusion removed — re-evaluated on the next batch run");
  renderEvidence(); renderPane();
}
async function markWrongLabel(key, name) {
  const data = await postJson("/api/reject", { ...keyItem(key), name });
  if (data.error) { toast(`failed to record: ${data.error}`); return; }
  const card = state.evidence.find(e => akey(e) === key);
  if (card) card.immich_label = null;
  toast(`recorded: “${name}” on this appearance is treated as wrong from the next batch run on`);
  renderEvidence();
}

/* ---------- modal name prompt ---------- */
function nameModal(title, hint) {
  return new Promise(resolve => {
    const host = document.getElementById("modalhost");
    host.innerHTML = `<div class="modal"><div class="inner">
      <h3>${esc(title)}</h3>
      <div class="muted" style="margin-bottom:10px">${esc(hint ?? "")}</div>
      <input id="nm" list="nml" placeholder="Type or pick an existing person…" style="width:100%">
      <datalist id="nml">${state.people.map(p => `<option value="${esc(p)}">`).join("")}</datalist>
      <div class="modal-actions">
        <button id="nmc">Cancel</button><button id="nmo" class="primary">OK</button>
      </div>
    </div></div>`;
    const inp = host.querySelector("#nm");
    inp.focus();
    const done = v => { host.innerHTML = ""; resolve(v); };
    host.querySelector("#nmc").onclick = () => done(null);
    host.querySelector("#nmo").onclick = () => done(inp.value.trim() || null);
    inp.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); done(inp.value.trim() || null); }
    });
  });
}
function namePrompt() {
  nameModal("Name this cluster", "Existing person → attaches · new name → creates the person on write.").then(v => {
    if (v && state.assignment) {
      state.assignment.proposed_name = v;
      const row = activeRow();
      if (row) row.proposed_name = v;
      toast(`Group named “${v}”.`);
      render();
    }
  });
}

/* ---------- pipeline runs ---------- */
async function pollJobs() {
  let data;
  try {
    data = await fetchJson("/api/jobs");
  } catch {
    return;
  }
  const wasActive = state.jobActive;
  const wasWindows = state.jobWindows;
  const prevLastState = state.jobLast?.state ?? null;
  const run = data.last_run || null;
  state.jobActive = !!data.active;
  state.jobLast = run;
  state.jobWindows = state.jobActive ? (run?.windows_done ?? 0) : 0;
  if (wasActive && !state.jobActive) {
    toast(run?.state === "done" ? "Pipeline run finished — the queue is up to date." : "Pipeline run stopped.");
    await refreshAfterRun();
    return;
  }
  if (!wasActive && state.jobActive) { render(); return; }
  if (state.jobActive && wasWindows !== state.jobWindows) {
    const el = document.getElementById("jobProgressText");
    if (el) {
      const n = state.jobWindows;
      el.textContent = `Pipeline running — ${n} ${n === 1 ? "day" : "days"} done`;
    }
    return;
  }
  if (!state.jobActive && prevLastState !== (run?.state ?? null)) render();
}

async function refreshAfterRun() {
  try {
    await loadInitialState();
    if (state.view === "final") await loadFinal();
  } catch (err) {
    toast(`could not refresh after the run: ${err.message}`);
  }
}

async function runNow() {
  try {
    const data = await postJson("/api/jobs", { trigger: "manual" });
    if (data.error) { toast(data.error); return; }
    toast("Pipeline started — this can take a while.");
    await pollJobs();
  } catch (err) {
    toast(`could not start a run: ${err.message}`);
  }
}

async function cancelRun() {
  if (!window.confirm("Stop the pipeline run? Finished work is kept and the next run continues from there.")) return;
  try {
    const data = await postJson("/api/jobs/cancel", {});
    if (data.error) { toast(data.error); return; }
    toast("Stopping the pipeline run…");
    await pollJobs();
  } catch (err) {
    toast(`could not stop the run: ${err.message}`);
  }
}

/* ---------- config ---------- */
async function saveConfig() {
  const knobs = {};
  document.querySelectorAll("#cfgPop input[data-cfg]").forEach(inp => {
    const key = inp.dataset.cfg;
    knobs[key] = key === "windowDays" ? parseInt(inp.value, 10) : parseFloat(inp.value);
  });
  try {
    const data = await postJson("/api/config", knobs);
    if (data.error) toast(`tuning rejected: ${data.error}`);
    else {
      state.configKnobs = data.knobs;
      toast("settings saved — applies on the next run");
      render();
    }
  } catch (err) {
    toast(`tuning save failed: ${err.message}`);
  }
}

/* ---------- boot ---------- */
async function boot() {
  try {
    state.showLabeled = localStorage.getItem("reid-show-labeled") === "1";
  } catch { /* private mode */ }
  await loadInitialState();
  await loadPeople();
  setInterval(pollJobs, 3000);
  await pollJobs();
}
boot();
