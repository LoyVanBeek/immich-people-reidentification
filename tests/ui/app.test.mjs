// UI smoke tests: run the real app.js in jsdom against a fake API.
//
// These exist because node --check only sees syntax: a runtime shape
// mismatch (e.g. app.js reading b.entries where the server sends
// b.actions) threw inside render() and froze the whole workspace while
// async image loads kept working — invisible to every Python test.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const APP_JS = readFileSync(join(ROOT, "src", "reid", "ui", "app.js"), "utf8");

// Two days sharing a cluster number — the shape that broke window-scoped
// selection — plus a REAL dry-run batch in the server's actual response
// shape (actions/promoted_at), which used to crash every re-render.
const CLUSTERS = [
  { cluster: 0, window: "2026-11-17", proposed_name: null, anchor: null, confidence: 0.9, appearances: 3, writable: 2, conflicted: 0, ask_for_name: 1, labeled: 0, status: "open", crops: [] },
  { cluster: 1, window: "2026-11-17", proposed_name: "Alice", anchor: "x", confidence: 0.8, appearances: 2, writable: 2, conflicted: 0, ask_for_name: 0, labeled: 0, status: "open", crops: [] },
  { cluster: 0, window: "2025-09-23", proposed_name: "Steffie", anchor: "x", confidence: 0.7, appearances: 2, writable: 1, conflicted: 0, ask_for_name: 0, labeled: 1, status: "open", crops: [] },
];
const DRY_BATCH = {
  id: "batch-1",
  name: "Alice",
  window: "2026-11-17",
  cluster: 0,
  timestamp: "2026-11-18T20:00:00Z",
  promoted_at: null,
  actions: [
    { asset_id: "asset-a", appearance: 0, head_box: null, status: "would_write", message: "" },
    { asset_id: "asset-b", appearance: 1, head_box: null, status: "conflict_skip", message: "existing label" },
  ],
};
const EVIDENCE = [
  { asset_id: "asset-a", appearance: 0, window: "2026-11-17", cluster: 0, date: "2026-11-17", confidence: 0.9, status: "will_write", included: true, immich_label: null },
];

function makeDom() {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="app"></div><div id="modalhost"></div><div id="toasthost"></div>
  </body></html>`, { url: "http://localhost/", runScripts: "outside-only" });
  const fetchLog = [];
  dom.window.fetch = async path => {
    fetchLog.push(path);
    let json;
    if (path === "/api/state") {
      json = {
        clusters: CLUSTERS,
        active_cluster: 0,
        evidence: [],
        assignment: {
          cluster: 0,
          window: "2026-11-17",
          proposed_name: null,
          anchor: null,
          total_appearances: 3,
          writable: 2,
          conflicted: 0,
          ask_for_name: 1,
          labeled: 0,
          existing_people: ["Alice", "Steffie"],
        },
        stats: [],
        config_knobs: { window_days: 1 },
        people: ["Alice", "Steffie"],
        dry_run: true,
      };
    } else if (path.startsWith("/api/evidence")) {
      json = EVIDENCE;
    } else if (path === "/api/dry-run-batches") {
      json = [DRY_BATCH];
    } else {
      json = { results: [] };
    }
    return { ok: true, json: async () => json };
  };
  dom.window.eval(APP_JS);
  return { dom, fetchLog, window: dom.window, document: dom.window.document };
}

const flush = async (window, ms = 20) => {
  await new Promise(r => setTimeout(r, ms));
};
const clickAll = (window, document, selector) => {
  const els = [...document.querySelectorAll(selector)];
  assert.ok(els.length, `no element matches ${selector}`);
  els.forEach(el => el.dispatchEvent(new window.Event("click", { bubbles: true })));
};
const header = document => document.querySelector(".main-header .cluster-id")?.textContent;
const selectedDay = document =>
  document.querySelector(".days .dayrow.sel")?.dataset.day;
const clusterKeys = document =>
  [...document.querySelectorAll(".clusters .item")].map(el => el.dataset.key);

async function testBootRendersWithDryRunBatchPresent() {
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  // The dry-run batch card must be present and shaped from the real API
  // (actions/promoted_at). Before the shape fix this threw inside
  // render() and left the workspace frozen on stale DOM.
  const batchCard = document.querySelector(".dry-batch");
  assert.ok(batchCard, "dry-run batch card missing");
  assert.match(batchCard.textContent, /Dry-run batch batch-1/);
  assert.match(batchCard.textContent, /would_write ×1/);
  assert.match(batchCard.textContent, /conflict_skip ×1/);
  assert.ok(batchCard.querySelector("[data-promote='batch-1']"), "promote button missing");
  assert.ok(header(document) === "2026-11-17 / 0", "initial header wrong");
}

async function testDayClickSelectsAndLoads() {
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);

  clickAll(window, document, '.days .dayrow[data-day="2025-09-23"]');
  await flush(window);

  assert.equal(selectedDay(document), "2025-09-23", "clicked day not highlighted");
  assert.deepEqual(clusterKeys(document), ["2025-09-23:0"], "cluster pane not scoped to the clicked day");
  assert.ok(header(document)?.startsWith("2025-09-23"), "evidence header not updated");
  assert.ok(
    fetchLog.some(p => p.startsWith("/api/evidence?cluster=0&window=2025-09-23")),
    "evidence fetch not window-scoped"
  );
}

async function testClusterClickSelects() {
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);

  clickAll(window, document, '.days .dayrow[data-day="2026-11-17"]');
  await flush(window);
  clickAll(window, document, '.clusters .item[data-key="2026-11-17:1"]');
  await flush(window);

  assert.equal(header(document), "2026-11-17 / 1", "cluster click did not update the header");
  const sel = document.querySelector('.clusters .item.sel');
  assert.ok(sel && sel.dataset.key === "2026-11-17:1", "clicked cluster not highlighted");

  // Evidence cards render with the loading-spinner markup.
  const spinner = document.querySelector(".evcard .crop .spinner");
  assert.ok(spinner, "spinner markup missing on evidence cards");
}

async function testPaneErrorDoesNotFreezeTheWorkspace() {
  // A throwing pane template must not abort render(): the pane shows an
  // error card and the rest of the UI keeps working. (State is
  // module-lexical, so poison a pane function instead — same effect the
  // dry-batch shape bug had inside paneHTML.)
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('daysHTML = () => { throw new Error("boom"); }');
  // Any re-render (a cluster click) must survive the poisoned pane.
  clickAll(window, document, '.clusters .item[data-key="2026-11-17:1"]');
  await flush(window);

  assert.ok(document.querySelector(".panerr"), "pane error card not shown");
  assert.match(document.querySelector(".panerr").textContent, /boom/);
  // The other panes still rendered and the click still took effect.
  assert.equal(header(document), "2026-11-17 / 1", "workspace froze after a pane error");
  const sel = document.querySelector(".clusters .item.sel");
  assert.ok(sel && sel.dataset.key === "2026-11-17:1", "cluster selection lost after a pane error");
}

const tests = [
  testBootRendersWithDryRunBatchPresent,
  testDayClickSelectsAndLoads,
  testClusterClickSelects,
  testPaneErrorDoesNotFreezeTheWorkspace,
];
for (const test of tests) {
  await test();
  console.log(`ok - ${test.name}`);
}
console.log(`\n${tests.length} UI tests passed`);
