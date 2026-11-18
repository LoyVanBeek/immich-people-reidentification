/** Confirmation UI — variant B (3-pane workspace).
 *  Primary behavioral source: prototype/confirmation-ui commit 4d529a2.
 */

const API_BASE = "";
let state = {
  clusters: [],
  activeCluster: null,
  evidence: [],
  assignment: null,
  stats: [],
  configKnobs: null,
  people: [],
  dryRun: true,
};
let keyHandler = null;

/* ---------- API ---------- */
async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadInitialState() {
  const data = await fetchJson("/api/state");
  state = { ...state, ...data };
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
function render() {
  const root = document.getElementById("app");
  // Preserve sidebar scroll across re-renders: a full innerHTML swap
  // otherwise resets it, losing your place in the review queue.
  const side = document.querySelector(".side");
  const sideScroll = side ? side.scrollTop : 0;
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
    <div class="side">${sidebarHTML()}</div>
    <div class="main">${mainHTML()}</div>
    <div class="pane">${paneHTML()}</div>
  `;
  if (side) side.scrollTop = sideScroll;
  wireEvents();
  renderEvidence();
  renderPane();
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

function sidebarHTML() {
  const items = state.clusters.map(c => `
    <div class="item ${c.cluster === state.activeCluster ? "sel" : ""} ${c.status !== "open" ? "done" : ""}" data-cluster="${c.cluster}" data-window="${c.window}">
      <div class="item-row">
        <div class="strip">${renderCrops(c.crops || [], 26, 36)}</div>
        <div class="item-info">
          <div class="item-name">${c.proposed_name ? esc(c.proposed_name) : '<span class="muted">unnamed</span>'}</div>
          <div class="item-meta mono">${c.window} · ${c.cluster} · ${(c.confidence * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  `).join("");

  const dryBatches = state.dryBatches || [];
  const batchesHTML = dryBatches.length ? `
    <div class="dry-batches">
      ${dryBatches.map(b => `
        <div class="dry-batch">
          <div class="batch-id">Dry-run batch ${b.id}</div>
          <div class="batch-entries">${b.entries.map(e => `${esc(e.name || e.clusterId)} ×${e.n}`).join(" · ")}</div>
          ${b.promoted ? '<span class="chip ok">promoted</span>' : `<button class="primary" data-promote="${b.id}">Promote to real write</button>`}
        </div>
      `).join("")}
    </div>
  ` : "";

  return items + batchesHTML;
}

function mainHTML() {
  const c = state.clusters.find(x => x.cluster === state.activeCluster);
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
    <div class="cluster-meta muted">${a.total_appearances} appearances · ${a.writable} writable · ${a.conflicted} conflict · ${a.no_head} no-head</div>
    ${!hasName ? `<button class="b-name primary" data-cluster="${a.cluster}" style="width:100%;margin-bottom:10px">Name this cluster</button>` : ""}
    <div class="name-prompt">
      <label class="muted">Assign to</label>
      <input type="text" id="nameInput" list="peopleList" value="${hasName ? esc(a.proposed_name) : ""}" ${a.proposed_name ? "" : "placeholder=\"Type or pick an existing person…\""} ${state.activeCluster === null ? "disabled" : ""}>
      <datalist id="peopleList">${state.people.map(p => `<option value="${esc(p)}">`).join("")}</datalist>
      <div class="hint">existing person → attach · new name → creates person on write</div>
    </div>
    <div class="actions">
      ${state.activeCluster !== null ? `
        <button class="b-ok primary" data-cluster="${a.cluster}">
          ${state.dryRun ? "Record (dry-run)" : `Write ${a.writable || 0} faces`}
        </button>
        <button class="b-skip" data-cluster="${a.cluster}">Skip cluster</button>
      ` : `<div class="chip">Select a cluster</div>`}
    </div>
    ${a.cluster_status === "skipped" && state.activeCluster !== null ? `
      <button class="b-reopen" data-cluster="${a.cluster}" style="width:100%;margin-top:8px">Reopen dismissed cluster</button>
    ` : ""}
    <hr>
    <div class="footer muted">⚙ Config (top right) holds the tuning knobs; store wipes stay out of this screen.</div>
  `;
}

function renderEvidence() {
  const grid = document.getElementById("evgrid");
  if (!grid) return;
  const assignment = state.assignment;
  grid.innerHTML = state.evidence.map(e => {
    const [chipClass, chipText] = statusChip(e.status);
    const rejectable = assignment?.proposed_name && e.status !== "rejected" && e.status !== "already_labeled";
    return `
      <div class="evcard ${e.status === "rejected" ? "evrejected" : ""}" data-asset="${e.asset_id}" data-appearance="${e.appearance}">
        <div class="crop">${cropImg(e, 150, 170, e.status !== "rejected")}</div>
        <div class="asset-row">
          ${e.status === "will_write" ? `<input type="checkbox" data-asset="${e.asset_id}" ${e.included ? "checked" : ""}>` : ""}
          <span class="asset-name mono">${esc(e.asset_id)}</span>
        </div>
        <div class="date-conf">${esc(e.date)} · ${(e.confidence * 100).toFixed(0)}%</div>
        <div class="chips">
          <span class="chip ${chipClass}">${chipText}</span>
          ${e.immich_label ? `<span class="chip">immich: ${esc(e.immich_label)}</span>` : ""}
        </div>
        ${rejectable ? `<button class="b-reject" data-asset="${e.asset_id}" data-appearance="${e.appearance}">Not this person</button>` : ""}
      </div>
    `;
  }).join("");
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

  // Sidebar cluster selection
  document.querySelectorAll(".side .item[data-cluster]").forEach(el => {
    el.addEventListener("click", () => {
      const cluster = parseInt(el.dataset.cluster, 10);
      const window = el.dataset.window;
      state.activeCluster = cluster;
      loadEvidence(cluster, window);
      // Update assignment pane
      const assignment = state.clusters.find(c => c.cluster === cluster && c.window === window);
      if (assignment) {
        state.assignment = buildAssignment(assignment);
      }
      render();
    });
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
    const idx = openClusters.findIndex(c => c.cluster === state.activeCluster);
    if (e.key === "j") {
      const next = openClusters[(idx + 1) % openClusters.length];
      if (next) { state.activeCluster = next.cluster; loadEvidence(next.cluster, next.window); state.assignment = buildAssignment(next); render(); }
    }
    if (e.key === "k") {
      const prev = openClusters[(idx - 1 + openClusters.length) % openClusters.length];
      if (prev) { state.activeCluster = prev.cluster; loadEvidence(prev.cluster, prev.window); state.assignment = buildAssignment(prev); render(); }
    }
    if (e.key === "c") doConfirm(state.activeCluster);
    if (e.key === "s") doSkip(state.activeCluster);
  };
  document.addEventListener("keydown", keyHandler);
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
  document.querySelectorAll(".b-reopen[data-cluster]").forEach(btn => {
    btn.addEventListener("click", () => {
      const c = state.clusters.find(x => x.cluster === parseInt(btn.dataset.cluster, 10));
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
        const cluster = state.clusters.find(c => c.cluster === state.activeCluster);
        if (cluster) cluster.proposed_name = v;
        toast(`${state.activeCluster} → ${v} (${state.people.includes(v) ? "existing" : "new"})`);
        render();
      }
    });
  }

  document.querySelector(".b-ok")?.addEventListener("click", () => doConfirm(state.activeCluster));
  document.querySelector(".b-skip")?.addEventListener("click", () => doSkip(state.activeCluster));
  document.querySelector(".b-name")?.addEventListener("click", () => namePrompt(state.activeCluster));
}

/* ---------- Dry-run batch actions ---------- */
async function loadDryRunBatches() {
  try {
    const res = await fetch("/api/dry-run-batches");
    const batches = await res.json();
    state.dryBatches = batches;
    render();
  } catch (err) {
    _log.warning("failed to load dry-run batches: %s", err);
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

async function doConfirm(cluster) {
  if (cluster === null) return;
  const c = state.clusters.find(x => x.cluster === cluster);
  if (!c || c.status !== "open") return;

  if (!c.proposed_name) {
    namePrompt(cluster, name => {
      c.proposed_name = name;
      c.anchor = `named manually (${state.people.includes(name) ? "existing" : "new"})`;
      doConfirm(cluster);
    });
    return;
  }

  // For now just mark as confirmed locally
  c.status = "confirmed";
  const w = state.evidence.filter(e => e.cluster === cluster && e.window === c.window);
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

function doSkip(cluster) {
  if (cluster === null) return;
  const c = state.clusters.find(x => x.cluster === cluster);
  if (!c) return;
  dismissCluster(cluster, c.window, true);
}

function namePrompt(cluster, onPick) {
  const host = document.getElementById("modalhost");
  const finish = name => {
    host.innerHTML = "";
    if (name && typeof onPick === "function") {
      onPick(name);
    } else if (name) {
      // Default (pane "Name this cluster" button): set the name in place,
      // so the confirm button becomes usable without another round-trip.
      const c = state.clusters.find(x => x.cluster === cluster);
      if (c) {
        c.proposed_name = name;
        c.anchor = `named manually (${state.people.includes(name) ? "existing" : "new"})`;
      }
      if (state.assignment && state.assignment.cluster === cluster) {
        state.assignment.proposed_name = name;
      }
      render();
    }
  };
  host.innerHTML = `<div class="modal"><div class="inner">
    <h3>Name this cluster <span class="mono muted">${cluster}</span></div>
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
