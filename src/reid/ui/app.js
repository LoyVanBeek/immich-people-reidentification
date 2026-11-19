/** Confirmation UI — variant B (3-pane workspace).
 *  Primary behavioral source: prototype/confirmation-ui commit 4d529a2.
 */

const API_BASE = "";
let state = {
  clusters: [],
  activeDay: null, // which day's clusters the second pane shows
  activeKey: null, // "window:cluster" — cluster numbers repeat across windows
  evidence: [],
  assignment: null,
  stats: [],
  configKnobs: null,
  people: [],
  dryRun: true,
};
let keyHandler = null;

function keyOf(row) {
  return `${row.window}:${row.cluster}`;
}

function activeRow() {
  return state.clusters.find(c => keyOf(c) === state.activeKey) || null;
}

/* ---------- API ---------- */
async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadInitialState() {
  const data = await fetchJson("/api/state");
  state = {
    ...state,
    ...data,
    // Server serialises dataclasses snake_case; state uses camelCase.
    configKnobs: data.config_knobs ?? null,
    dryRun: data.dry_run ?? state.dryRun,
  };
  // The server builds the initial evidence/assignment for clusters[0]'s
  // exact window; the key is derived from that same row.
  state.activeKey = state.clusters.length ? keyOf(state.clusters[0]) : null;
  state.activeDay = state.clusters.length ? state.clusters[0].window : null;
  render();
}

async function loadEvidence(cluster, window) {
  const data = await fetchJson(`/api/evidence?cluster=${cluster}&window=${encodeURIComponent(window)}`);
  state.evidence = data;
  renderEvidence();
}

async function loadPeople() {
  const data = await fetchJson("/api/people");
  state.people = data.people || [];
}

/* ---------- Rendering ---------- */
// One pane's template throwing used to abort the whole render(): the page
// kept stale DOM while async image loads kept working, which looked like
// "clicks do nothing". Each pane now fails alone, visibly.
function safePane(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`${name} failed to render:`, err);
    return `<div class="panerr"><strong>${esc(name)} failed to render</strong><br>${esc(err.message)}</div>`;
  }
}

function render() {
  const root = document.getElementById("app");
  // Preserve column scroll positions across re-renders: a full innerHTML
  // swap otherwise resets them, losing your place in the review queue.
  const scrolls = [".days", ".clusters"].map(sel => {
    const el = document.querySelector(sel);
    return [sel, el ? el.scrollTop : 0];
  });
  root.innerHTML = `
    ${bannerHTML()}
    <div class="top">
      <strong>Review</strong>
      <span class="chip acc">${state.clusters.filter(c => c.status === "open").length} open</span>
      <span class="spacer"></span>
      <div class="knobs">
        <label><input type="checkbox" id="dryToggle" ${state.dryRun ? "checked" : ""}> dry-run</label>
        <button id="gearBtn" title="Tuning knobs">⚙</button>
      </div>
      <span class="muted"><kbd class="kbd">j</kbd>/<kbd class="kbd">k</kbd> move · <kbd class="kbd">c</kbd> confirm · <kbd class="kbd">s</kbd> skip</span>
      <div class="cfgpop" id="cfgPop">${configPopoverHTML()}</div>
    </div>
    <div class="days">${safePane("days pane", daysHTML)}</div>
    <div class="clusters">${safePane("clusters pane", clustersHTML)}</div>
    <div class="main">${safePane("evidence pane", mainHTML)}</div>
    <div class="pane">${safePane("actions pane", paneHTML)}</div>
  `;
  for (const [sel, top] of scrolls) {
    const el = document.querySelector(sel);
    if (el) el.scrollTop = top;
  }
  wireEvents();
  try {
    renderEvidence();
  } catch (err) {
    console.error("evidence grid failed to render:", err);
  }
  try {
    renderPane();
  } catch (err) {
    console.error("actions pane failed to render:", err);
  }
}

function bannerHTML() {
  return state.dryRun
    ? `<div class="banner dry">DRY-RUN — nothing will be written to Immich</div>`
    : `<div class="banner real">LIVE — confirmations write real faces to Immich</div>`;
}

function configPopoverHTML() {
  const k = state.configKnobs;
  if (!k) {
    return `<div class="muted">knobs unavailable</div>`;
  }
  const rows = [
    ["windowDays", "Timespan window (days)", k.window_days, 1, 30, 1],
    ["distanceThreshold", "Cluster distance threshold", k.distance_threshold, 0.2, 1, 0.01],
    ["confidenceFloor", "Detector confidence floor", k.confidence_floor, 0, 1, 0.01],
    ["minBoxHeight", "Min box height (px)", k.min_box_height, 40, 300, 10],
    ["faceOverlap", "Face→person containment", k.face_overlap, 0.1, 1, 0.05],
  ];
  return rows.map(([key, label, value, min, max, step]) =>
    `<label>${label} <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-cfg="${key}"> <span class="mono" data-cfg-val="${key}">${value}</span></label>`
  ).join("")
    + `<button class="primary" id="cfgSave" style="width:100%;margin-top:8px">Save tuning</button>`
    + `<div class="muted">Saved values apply to the next batch run.</div>`;
}

function daysHTML() {
  // Pane 1: one row per day (newest first), Finder column style.
  const groups = new Map();
  for (const c of state.clusters) {
    if (!groups.has(c.window)) groups.set(c.window, []);
    groups.get(c.window).push(c);
  }
  const days = [...groups.keys()].sort((a, b) => b.localeCompare(a));
  if (state.activeDay === null && days.length) {
    state.activeDay = days[0];
  }
  const rows = days.map(day => {
    const clusterRows = groups.get(day);
    const openCount = clusterRows.filter(r => r.status === "open").length;
    return `
      <div class="dayrow ${day === state.activeDay ? "sel" : ""}" data-day="${day}">
        <div class="dayname">${day}</div>
        <span class="chip ${openCount ? "acc" : ""}">${openCount}/${clusterRows.length}</span>
      </div>
    `;
  });
  return rows.join("") || `<div class="empty muted">no clusters yet — run the batch</div>`;
}

function clustersHTML() {
  // Pane 2: the clusters of the selected day.
  const rows = state.clusters.filter(c => c.window === state.activeDay);
  if (!rows.length) {
    return `<div class="empty muted">select a day</div>`;
  }
  return rows.map(c => `
    <div class="item ${keyOf(c) === state.activeKey ? "sel" : ""} ${c.status !== "open" ? "done" : ""}" data-key="${keyOf(c)}">
      <div class="item-row">
        <div class="strip">${renderCrops(c.crops || [], 26, 36)}</div>
        <div class="item-info">
          <div class="item-name">${c.proposed_name ? esc(c.proposed_name) : '<span class="muted">unnamed</span>'}</div>
          <div class="item-meta mono">cluster ${c.cluster} · ${(c.confidence * 100).toFixed(0)}% · ${c.appearances} ap</div>
        </div>
      </div>
    </div>
  `).join("");
}

function dryBatchesHTML() {
  const dryBatches = state.dryBatches || [];
  if (!dryBatches.length) return "";
  return `
    <div class="dry-batches">
      ${dryBatches.map(b => {
        // Server shape: {id, name, window, cluster, timestamp, promoted_at,
        // actions: [{asset_id, appearance, head_box, status, message}]}
        const counts = {};
        for (const a of b.actions || []) {
          counts[a.status] = (counts[a.status] || 0) + 1;
        }
        const summary = Object.entries(counts)
          .map(([status, n]) => `${status} ×${n}`)
          .join(" · ");
        return `
        <div class="dry-batch">
          <div class="batch-id">Dry-run batch ${b.id} — ${esc(b.window || "")} / ${b.cluster ?? "?"} → ${esc(b.name || "?")}</div>
          <div class="batch-entries">${esc(summary || "no actions")}</div>
          ${b.promoted_at ? '<span class="chip ok">promoted</span>' : `<button class="primary" data-promote="${b.id}">Promote to real write</button>`}
        </div>
      `;
      }).join("")}
    </div>
  `;
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
    <div class="muted cluster-anchor">${c.anchor || "no anchor found"} · window conf ${(c.confidence * 100).toFixed(0)}%</div>
    <div class="evgrid" id="evgrid"></div>
  `;
}

function paneHTML() {
  const a = state.assignment;
  if (!a) {
    return `<div class="name-display">—</div><div class="cluster-meta muted">Select a cluster</div>`;
  }
  const hasName = a.proposed_name != null;
  return `
    <div class="name-display">${hasName ? esc(a.proposed_name) : '<span class="muted">—</span>'}</div>
    <div class="cluster-meta muted">${a.total_appearances} appearances · ${a.writable} writable · ${a.conflicted} conflict · ${a.no_head ?? a.ask_for_name ?? 0} no-head</div>
    ${!hasName ? `<button class="b-name primary" style="width:100%;margin-bottom:10px">Name this cluster</button>` : ""}
    <div class="name-prompt">
      <label class="muted">Assign to</label>
      <input type="text" id="nameInput" list="peopleList" value="${hasName ? esc(a.proposed_name) : ""}" ${a.proposed_name ? "" : "placeholder=\"Type or pick an existing person…\""} ${state.activeKey === null ? "disabled" : ""}>
      <datalist id="peopleList">${state.people.map(p => `<option value="${esc(p)}">`).join("")}</datalist>
      <div class="hint">existing person → attach · new name → creates person on write</div>
    </div>
    <div class="actions">
      ${state.activeKey !== null ? `
        <button class="b-ok primary">
          ${state.dryRun ? "Record (dry-run)" : `Write ${a.writable || 0} faces`}
        </button>
        <button class="b-skip">Skip cluster</button>
      ` : `<div class="chip">Select a cluster</div>`}
    </div>
    ${a.cluster_status === "skipped" && state.activeKey !== null ? `
      <button class="b-reopen" data-key="${a.window}:${a.cluster}" style="width:100%;margin-top:8px">Reopen dismissed cluster</button>
    ` : ""}
    <hr>
    <div class="footer muted">⚙ Config (top right) holds the tuning knobs; store wipes stay out of this screen.</div>
    ${dryBatchesHTML()}
  `;
}

function renderEvidence() {
  const grid = document.getElementById("evgrid");
  if (!grid) return;
  const assignment = state.assignment;
  grid.innerHTML = state.evidence.map(e => {
    const [chipClass, chipText] = statusChip(e.status);
    const rejectable = assignment?.proposed_name && e.status !== "rejected" && e.status !== "already_labeled";
    const wrongLabelable = e.status === "already_labeled" && e.immich_label;
    return `
      <div class="evcard ${e.status === "rejected" ? "evrejected" : ""}" data-asset="${e.asset_id}" data-appearance="${e.appearance}">
        <div class="crop">${cropImg(e, 150, 170, e.status !== "rejected")}<div class="spinner" title="loading crop…"></div></div>
        <div class="asset-row">
          ${e.status === "will_write" ? `<label class="writemark" title="Checked: this appearance is written to Immich when you confirm. Uncheck to leave it out.">
            <input type="checkbox" data-asset="${e.asset_id}" ${e.included ? "checked" : ""}>write</label>` : ""}
          <span class="asset-name mono">${esc(e.asset_id)}</span>
        </div>
        <div class="date-conf">${esc(e.date)} · ${(e.confidence * 100).toFixed(0)}%</div>
        <div class="chips">
          <span class="chip ${chipClass}">${chipText}</span>
          ${e.immich_label ? `<span class="chip">immich: ${esc(e.immich_label)}</span>` : ""}
        </div>
        ${rejectable ? `<button class="b-reject" data-asset="${e.asset_id}" data-appearance="${e.appearance}">Not this person</button>` : ""}
        ${wrongLabelable ? `<button class="b-wronglabel" data-asset="${e.asset_id}" data-appearance="${e.appearance}" data-name="${esc(e.immich_label)}">Wrong label</button>` : ""}
      </div>
    `;
  }).join("");
  // Crops stream in from /api/crop: keep a spinner up until each image
  // actually arrives (or errors out), then fade the picture in.
  grid.querySelectorAll("img.crop").forEach(img => {
    const done = () => img.classList.add("loaded");
    if (img.complete) done();
    else {
      img.addEventListener("load", done, { once: true });
      img.addEventListener("error", done, { once: true });
    }
  });
  wireEvidenceEvents();
}

function renderPane() {
  const pane = document.querySelector(".pane");
  if (!pane) return;
  pane.innerHTML = paneHTML();
  wirePaneEvents();
}

/* ---------- Event wiring ---------- */
function wireEvents() {
  // Config popover
  document.getElementById("gearBtn")?.addEventListener("click", () =>
    document.getElementById("cfgPop")?.classList.toggle("open")
  );
  document.querySelectorAll("#cfgPop input[data-cfg]").forEach(inp => {
    inp.addEventListener("input", () => {
      const key = inp.dataset.cfg;
      document.querySelector(`[data-cfg-val="${key}"]`).textContent = inp.value;
    });
  });
  document.getElementById("cfgSave")?.addEventListener("click", saveConfig);

  // Dry-run toggle
  document.getElementById("dryToggle")?.addEventListener("change", e => {
    state.dryRun = e.target.checked;
    render(); // re-render banner and button labels
  });

  // Sidebar: day rows select a day, cluster rows select a cluster
  document.querySelectorAll(".days .dayrow").forEach(el => {
    el.addEventListener("click", () => selectDay(el.dataset.day));
  });
  document.querySelectorAll(".clusters .item[data-key]").forEach(el => {
    el.addEventListener("click", () => selectCluster(el.dataset.key));
  });

  // Dry-batch promote
  document.querySelectorAll("[data-promote]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.promote;
      await promoteBatch(id);
    });
  });

  // Keyboard navigation
  if (keyHandler) document.removeEventListener("keydown", keyHandler);
  keyHandler = e => {
    if (/input|textarea/i.test(e.target.tagName) || e.target.isContentEditable) return;
    const openClusters = state.clusters.filter(c => c.status === "open");
    const idx = openClusters.findIndex(c => keyOf(c) === state.activeKey);
    if (e.key === "j") {
      const next = openClusters[(idx + 1) % openClusters.length];
      if (next) selectCluster(keyOf(next));
    }
    if (e.key === "k") {
      const prev = openClusters[(idx - 1 + openClusters.length) % openClusters.length];
      if (prev) selectCluster(keyOf(prev));
    }
    if (e.key === "c") doConfirm(state.activeKey);
    if (e.key === "s") doSkip(state.activeKey);
  };
  document.addEventListener("keydown", keyHandler);
}

function selectDay(day) {
  state.activeDay = day;
  // Show the day's clusters AND load its first open one, so the
  // evidence/actions panes move with the click.
  const rows = state.clusters.filter(c => c.window === day);
  const first = rows.find(c => c.status === "open") || rows[0];
  if (first) selectCluster(keyOf(first));
  else render();
}

function selectCluster(key) {
  state.activeKey = key;
  const row = activeRow();
  if (row) {
    state.activeDay = row.window;
    loadEvidence(row.cluster, row.window);
    state.assignment = buildAssignment(row);
  }
  render();
}

function wireEvidenceEvents() {
  document.querySelectorAll(".evcard input[data-asset]").forEach(el => {
    el.addEventListener("change", () => {
      const asset = el.dataset.asset;
      const card = state.evidence.find(e => e.asset_id === asset);
      if (card) {
        card.included = el.checked;
        // TODO: persist inclusion in #13/15
      }
    });
  });
  document.querySelectorAll(".b-reject[data-asset]").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = state.assignment?.proposed_name;
      rejectAppearance(btn.dataset.asset, parseInt(btn.dataset.appearance, 10), name);
    });
  });
  document.querySelectorAll(".b-wronglabel[data-asset]").forEach(btn => {
    btn.addEventListener("click", () => {
      markWrongLabel(
        btn.dataset.asset,
        parseInt(btn.dataset.appearance, 10),
        btn.dataset.name
      );
    });
  });
  document.querySelectorAll(".b-reopen[data-key]").forEach(btn => {
    btn.addEventListener("click", () => {
      const c = state.clusters.find(x => keyOf(x) === btn.dataset.key);
      if (c) dismissCluster(c.cluster, c.window, false);
    });
  });
}

function wirePaneEvents() {
  const nameInput = document.getElementById("nameInput");
  if (nameInput) {
    nameInput.addEventListener("change", () => {
      const v = nameInput.value.trim();
      if (v && state.assignment && !state.assignment.proposed_name) {
        state.assignment.proposed_name = v;
        // Update cluster row too
        const row = activeRow();
        if (row) {
          row.proposed_name = v;
          toast(`${row.window} / ${row.cluster} → ${v} (${state.people.includes(v) ? "existing" : "new"})`);
        }
        render();
      }
    });
  }

  document.querySelector(".b-ok")?.addEventListener("click", () => doConfirm(state.activeKey));
  document.querySelector(".b-skip")?.addEventListener("click", () => doSkip(state.activeKey));
  document.querySelector(".b-name")?.addEventListener("click", () => namePrompt(state.activeKey));
}

/* ---------- Dry-run batch actions ---------- */
async function loadDryRunBatches() {
  try {
    const res = await fetch("/api/dry-run-batches");
    const batches = await res.json();
    state.dryBatches = batches;
    render();
  } catch (err) {
    console.warn("failed to load dry-run batches:", err);
  }
}

async function promoteBatch(id) {
  try {
    const res = await fetch(`/api/promote/${id}`, { method: "POST" });
    const data = await res.json();
    if (data.error) {
      toast(`promote failed: ${data.error}`);
      return;
    }
    toast(`promoted batch ${id}: ${data.results.filter(r => r.status === "written").length} written`);
    loadDryRunBatches();
  } catch (err) {
    toast(`promote error: ${err.message}`);
  }
}

/* ---------- Actions ---------- */
async function saveConfig() {
  const knobs = {};
  document.querySelectorAll("#cfgPop input[data-cfg]").forEach(inp => {
    const key = inp.dataset.cfg;
    // window_days is an integer knob; the rest are floats
    knobs[key] = key === "windowDays" ? parseInt(inp.value, 10) : parseFloat(inp.value);
  });
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(knobs),
    });
    const data = await res.json();
    if (data.error) {
      toast(`tuning rejected: ${data.error}`);
    } else {
      state.configKnobs = data.knobs;
      toast("tuning saved — applies to the next batch run");
      render();
    }
  } catch (err) {
    toast(`tuning save failed: ${err.message}`);
  }
}

async function rejectAppearance(assetId, appearance, name) {
  if (!name) {
    toast("name the cluster first — then 'not this person' means something");
    return;
  }
  try {
    const res = await fetch("/api/reject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, appearance, name }),
    });
    const data = await res.json();
    if (data.error) {
      toast(`reject failed: ${data.error}`);
      return;
    }
    // Update local state so the card flips to 'rejected' without a reload.
    const card = state.evidence.find(
      e => e.asset_id === assetId && String(e.appearance) === String(appearance)
    );
    if (card) card.status = "rejected";
    toast(`appearance ${appearance} of ${assetId.slice(0, 8)}… will not be proposed as “${name}”`);
    renderEvidence();
  } catch (err) {
    toast(`reject failed: ${err.message}`);
  }
}

async function markWrongLabel(assetId, appearance, name) {
  try {
    const res = await fetch("/api/reject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, appearance, name }),
    });
    const data = await res.json();
    if (data.error) {
      toast(`failed to record: ${data.error}`);
      return;
    }
    const card = state.evidence.find(
      e => e.asset_id === assetId && String(e.appearance) === String(appearance)
    );
    if (card) card.immich_label = null;
    toast(
      `recorded: “${name}” on this appearance is treated as wrong from the next batch run on. ` +
        "The label stays in Immich until you remove it there — until then, writes here will report a conflict."
    );
    renderEvidence();
  } catch (err) {
    toast(`failed to record: ${err.message}`);
  }
}

async function dismissCluster(cluster, window, dismissed) {
  try {
    const res = await fetch("/api/skip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cluster, window, dismissed }),
    });
    const data = await res.json();
    if (data.error) {
      toast(`skip failed: ${data.error}`);
      return;
    }
    const row = state.clusters.find(c => c.cluster === cluster && c.window === window);
    if (row) row.status = dismissed ? "skipped" : "open";
    toast(dismissed ? "cluster dismissed — it stays out of the queue" : "cluster reopened");
    render();
  } catch (err) {
    toast(`skip failed: ${err.message}`);
  }
}

function buildAssignment(clusterRow) {
  // Build assignment pane from cluster row + evidence
  const w = state.evidence.filter(e => e.cluster === clusterRow.cluster && e.window === clusterRow.window);
  const writable = w.filter(e => e.status === "will_write").length;
  const conflicted = w.filter(e => e.status === "conflict_skip").length;
  const no_head = w.filter(e => e.status === "no_head_skip").length;
  const labeled = w.filter(e => e.status === "already_labeled").length;
  return {
    cluster: clusterRow.cluster,
    window: clusterRow.window,
    cluster_status: clusterRow.status,
    proposed_name: clusterRow.proposed_name,
    anchor: clusterRow.anchor,
    total_appearances: w.length,
    writable, conflicted, no_head, labeled,
    existing_people: state.people,
  };
}

async function doConfirm(key) {
  if (key === null) return;
  const c = state.clusters.find(x => keyOf(x) === key);
  if (!c || c.status !== "open") return;

  if (!c.proposed_name) {
    namePrompt(key, name => {
      c.proposed_name = name;
      c.anchor = `named manually (${state.people.includes(name) ? "existing" : "new"})`;
      doConfirm(key);
    });
    return;
  }

  // For now just mark as confirmed locally
  c.status = "confirmed";
  const w = state.evidence.filter(e => e.cluster === c.cluster && e.window === c.window);
  const included = w.filter(e => e.status === "will_write" && e.included);
  const st = {
    writable: included.length,
    conflicted: w.filter(e => e.status === "conflict_skip").length,
    nohead: w.filter(e => e.status === "no_head_skip").length,
  };
  if (state.dryRun) {
    // POST /api/confirm with dry_run: true
    const body = {
      cluster: c.cluster,
      window: c.window,
      name: c.proposed_name,
      included: included.map(e => ({ asset_id: e.asset_id, appearance: e.appearance })),
      dry_run: true,
    };
    try {
      const res = await fetch("/api/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) {
        toast(`dry-run failed: ${data.error}`);
        return;
      }
      toast(`recorded (dry-run): ${data.name} ×${data.results.filter(r => r.status === "would_write").length} — batch ${data.batch_id}`);
      // Refresh dry-run batch list
      loadDryRunBatches();
    } catch (err) {
      toast(`dry-run error: ${err.message}`);
    }
  } else {
    toast(`wrote ${st.writable} faces → ${c.proposed_name}` +
      ((st.conflicted || 0) + (st.nohead || 0) ? ` (skipped ${(st.conflicted || 0) + (st.nohead || 0)})` : ""));
  }
  render();
}

function doSkip(key) {
  if (key === null) return;
  const c = state.clusters.find(x => keyOf(x) === key);
  if (!c) return;
  dismissCluster(c.cluster, c.window, true);
}

function namePrompt(key, onPick) {
  const row = state.clusters.find(x => keyOf(x) === key);
  const label = row ? `${row.window} / ${row.cluster}` : key;
  const host = document.getElementById("modalhost");
  const finish = name => {
    host.innerHTML = "";
    if (name && typeof onPick === "function") {
      onPick(name);
    } else if (name) {
      // Default (pane "Name this cluster" button): set the name in place,
      // so the confirm button becomes usable without another round-trip.
      const c = state.clusters.find(x => keyOf(x) === key);
      if (c) {
        c.proposed_name = name;
        c.anchor = `named manually (${state.people.includes(name) ? "existing" : "new"})`;
      }
      if (
        state.assignment &&
        `${state.assignment.window}:${state.assignment.cluster}` === key
      ) {
        state.assignment.proposed_name = name;
      }
      render();
    }
  };
  host.innerHTML = `<div class="modal"><div class="inner">
    <h3>Name this cluster <span class="mono muted">${esc(label)}</span></div>
    <div class="muted" style="margin-bottom:10px">${state.assignment?.total_appearances || 0} appearances · no Immich anchor found</div>
    <input id="np" list="pl" placeholder="Type or pick an existing person…" style="width:100%">
    <datalist id="pl">${state.people.map(p => `<option value="${esc(p)}">`).join("")}</datalist>
    <div class="modal-hint">Existing name → attaches to that person. New name → creates a person on write.</div>
    <div class="modal-actions">
      <button id="npc">Cancel</button><button id="npo" class="primary">OK</button>
    </div>
  </div></div>`;
  const inp = host.querySelector("#np");
  inp.focus();
  const submit = () => {
    const v = inp.value.trim();
    if (v) finish(v);
  };
  host.querySelector("#npc").onclick = () => (host.innerHTML = "");
  host.querySelector("#npo").onclick = submit;
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  });
}

function statusChip(status) {
  const map = {
    will_write: ["ok", "will write"],
    already_labeled: ["acc", "already labeled"],
    conflict_skip: ["bad", "conflict — skip"],
    no_head_skip: ["warn", "no head — skip"],
    ask_for_name: ["", "needs a name"],
    rejected: ["bad", "not in this cluster"],
  };
  return map[status] || ["", status];
}

function cropSVG(e, w = 150, h = 170) {
  // Deterministic placeholder based on asset_id + appearance
  const seed = [...(e.asset_id + e.appearance)].reduce((x, ch) => x + ch.charCodeAt(0), 0);
  const hue = seed % 360;
  // Randomize head position slightly
  const hx = 0.2 + (seed % 50) / 100;
  const hy = 0.18 + (seed % 30) / 200;
  const hr = 0.16 * (0.9 + (seed % 30) / 100);
  const isNoHead = e.status === "no_head_skip";
  const isConflict = e.status === "conflict_skip";

  return `<svg class="crop" width="${w}" height="${h}" viewBox="0 0 100 136">
    <rect width="100" height="136" fill="hsl(${hue},18%,20%)"/>
    <rect width="100" height="136" fill="url(#g${hue % 4})" opacity=".5"/>
    <defs><linearGradient id="g${hue % 4}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fff" stop-opacity=".08"/>
      <stop offset="1" stop-color="#000" stop-opacity=".25"/></linearGradient></defs>
    <circle cx="${hx * 100}" cy="${hy * 136}" r="${hr * 100}" fill="hsl(${hue},30%,55%)"/>
    <rect x="${hx * 100 - hr * 130}" y="${(hy + hr) * 136 - 4}"
      width="${hr * 260}" height="${136 - (hy + hr) * 136}"
      rx="12" fill="hsl(${hue},25%,42%)"/>
    ${!isNoHead ? `<rect x="${(hx - hr * 1.1) * 100}" y="${(hy - hr * 1.15) * 136}"
      width="${hr * 2.2 * 100}" height="${hr * 2.3 * 136}"
      fill="none" stroke="${isConflict ? "#e0a33a" : "#4f8cff"}"
      stroke-width="2" stroke-dasharray="4 3"/>` : ''}
    ${isConflict ? `<rect width="100" height="136" fill="rgba(224,82,82,.14)"/>
      <text x="50" y="128" text-anchor="middle" font-size="11" fill="#e05252">conflict</text>` : ''}
  </svg>`;
}

function renderCrops(crops, w, h) {
  return crops.map(c => cropImg(c, w, h)).join("");
}

function cropImg(e, w, h, withContext) {
  // Real person crop from the container; cropSVG is the fallback for refs
  // that carry no asset (shouldn't happen, but keeps tiles from breaking).
  // Evidence cards pass withContext: the crop grows context around the box
  // and the target person is outlined, so neighbours stay distinguishable.
  if (!e.asset_id) return cropSVG(e, w, h);
  const extra = withContext ? "&expand=0.6&outline=1" : "";
  return `<img class="crop" loading="lazy" width="${w}" height="${h}"
    src="/api/crop?asset_id=${encodeURIComponent(e.asset_id)}&appearance=${e.appearance}${extra}"
    alt="person crop">`;
}

function esc(s) {
  return String(s).replace(/[&<"]/g, m => ({ "&": "&amp;", "<": "&lt;", '"': "&quot;" }[m]));
}

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.getElementById("toasthost").replaceChildren(t);
  setTimeout(() => t.remove(), 2200);
}

/* ---------- Boot ---------- */
async function boot() {
  await loadInitialState();
  await loadDryRunBatches();
}
boot().catch(err => {
  console.error("Failed to load UI state:", err);
  document.getElementById("app").innerHTML = `<div style="padding:20px;color:var(--bad)">Failed to load: ${esc(err.message)}</div>`;
});
