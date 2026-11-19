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

function makeDom(opts = {}) {
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
        evidence: [
          {
            asset_id: "asset-a",
            appearance: 0,
            window: "2026-11-17",
            cluster: 0,
            date: "2026-11-17",
            confidence: 0.9,
            status: "conflict_skip",
            included: false,
            immich_label: null,
            pin: null,
          },
        ],
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
      // Evidence is per-cluster so tests can tell stale fetches apart.
      const cluster = new URLSearchParams(path.split("?")[1]).get("cluster");
      json = [{
        asset_id: `asset-c${cluster}`,
        appearance: 0,
        window: new URLSearchParams(path.split("?")[1]).get("window"),
        cluster: Number(cluster),
        date: "2026-11-17",
        confidence: 0.9,
        status: "will_write",
        included: true,
        immich_label: null,
      }];
      const delay = opts.evidenceDelayFor ? opts.evidenceDelayFor(cluster) : 0;
      if (delay) await new Promise(r => setTimeout(r, delay));
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

async function testInstantSwitchShowsLoadingState() {
  // Clicking a cluster must switch the evidence pane immediately: no stale
  // cards from the previous cluster, a visible loading state, the header
  // and action counts updated — while the fetch is still in flight.
  const { window, document } = makeDom({ evidenceDelayFor: () => 150 });
  await window.eval("boot()");
  await flush(window);

  clickAll(window, document, '.clusters .item[data-key="2026-11-17:1"]');
  await flush(window, 10); // fetch still in flight

  assert.equal(header(document), "2026-11-17 / 1", "header did not switch instantly");
  assert.ok(document.querySelector(".evloading"), "no loading state while fetching");
  assert.equal(document.querySelectorAll(".evcard").length, 0, "previous cluster's cards lingered");
  // Action counts come from the cluster row at click time (2 appearances).
  const meta = document.querySelector(".cluster-meta")?.textContent || "";
  assert.match(meta, /2 appearances/, "action counts not updated at click time");

  await flush(window, 250); // fetch resolves
  assert.ok(!document.querySelector(".evloading"), "loading state never cleared");
  const card = document.querySelector(".evcard");
  assert.ok(card && card.dataset.asset === "asset-c1", "fetched evidence not rendered");
}

async function testStaleEvidenceFetchIsDiscarded() {
  // A slow fetch for an older selection must never overwrite a newer one.
  const { window, document } = makeDom({
    evidenceDelayFor: cluster => (cluster === "0" ? 200 : 10),
  });
  await window.eval("boot()");
  await flush(window);

  clickAll(window, document, '.clusters .item[data-key="2026-11-17:0"]'); // slow (200ms)
  await flush(window, 10);
  clickAll(window, document, '.clusters .item[data-key="2026-11-17:1"]'); // fast (10ms)
  await flush(window, 300); // both resolved by now

  assert.equal(header(document), "2026-11-17 / 1", "selection moved on");
  const card = document.querySelector(".evcard");
  assert.ok(card && card.dataset.asset === "asset-c1", "stale fetch overwrote the newer selection");
  assert.ok(!document.querySelector(".evloading"), "loading state stuck after race");
}

async function testAssignButtonPinsThroughTheApi() {
  // "Assign to person…" posts the pin and the card flips to pinned —
  // nothing writes to Immich; the next batch run proposes the name.
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);

  // Conflicted-style card: assign must be available without a cluster name.
  window.eval(
    `assignPrompt("asset-a", 0);
     document.getElementById("ap").value = "Steffie";
     document.getElementById("apo").onclick();`
  );
  await flush(window, 30);

  assert.ok(
    fetchLog.includes("/api/pin"),
    "no /api/pin request made"
  );
  const card = document.querySelector('.evcard[data-asset="asset-a"]');
  assert.ok(card, "card missing after assign");
  assert.match(card.textContent, /pinned: Steffie/, "pinned chip not shown");
  assert.match(card.textContent, /Unpin/, "unpin control not shown");
  assert.ok(!document.querySelector("#ap"), "modal did not close");
}

async function testUnpinRemovesThePin() {
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);
  // Pin first, then unpin via the card button.
  window.eval(`pinAppearance("asset-a", 0, "Steffie")`);
  await flush(window, 30);
  window.eval(`unpinAppearance("asset-a", 0)`);
  await flush(window, 30);

  const unpinCall = fetchLog[fetchLog.length - 1];
  assert.equal(unpinCall, "/api/unpin");
  const card = document.querySelector('.evcard[data-asset="asset-a"]');
  assert.ok(!/pinned: Steffie/.test(card.textContent), "pin chip lingered after unpin");
}

const tests = [
  testBootRendersWithDryRunBatchPresent,
  testDayClickSelectsAndLoads,
  testClusterClickSelects,
  testPaneErrorDoesNotFreezeTheWorkspace,
  testInstantSwitchShowsLoadingState,
  testStaleEvidenceFetchIsDiscarded,
  testAssignButtonPinsThroughTheApi,
  testUnpinRemovesThePin,
];
for (const test of tests) {
  await test();
  console.log(`ok - ${test.name}`);
}
console.log(`\n${tests.length} UI tests passed`);
