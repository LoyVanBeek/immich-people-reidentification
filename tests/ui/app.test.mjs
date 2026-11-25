// UI smoke tests for the two-tab app: run the real app.js in jsdom against a
// fake API. These exist because node --check only sees syntax: a runtime shape
// mismatch between app.js and the core's JSON threw inside render() and froze
// the workspace — invisible to every Python test.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const APP_JS = readFileSync(join(ROOT, "src", "reid", "ui", "app.js"), "utf8");

const W1 = "2026-11-17";
const card = (asset, appearance, extra = {}) => ({
  asset_id: asset,
  appearance,
  window: W1,
  cluster: 1,
  date: W1,
  confidence: 0.9,
  immich_label: null,
  status: "will_write",
  included: true,
  pixel_source: "preview",
  pin: null,
  conflict_names: null,
  asset_name: `IMG_${asset}.jpg`,
  ignored: false,
  proposed_name: "Alice",
  siblings: [
    { appearance: 0, box: [10, 20, 110, 120], confidence: 0.9 },
    { appearance: 1, box: [200, 20, 300, 120], confidence: 0.8 },
  ],
  sibling_count: 1,
  rejected_name: null,
  dismissed: false,
  ...extra,
});

const CLUSTERS = [
  { cluster: 1, window: W1, proposed_name: "Alice", anchor: "Immich label: Alice", confidence: 0.9, appearances: 2, writable: 2, conflicted: 0, ask_for_name: 0, labeled: 0, status: "open", crops: [], fully_labeled: false },
  { cluster: 0, window: "2025-09-23", proposed_name: null, anchor: "needs a name", confidence: 0.7, appearances: 1, writable: 0, conflicted: 0, ask_for_name: 1, labeled: 0, status: "open", crops: [], fully_labeled: false },
];
const EVIDENCE_W1 = [card("asset-a", 0), card("asset-b", 0, { appearance: 0, pin: "Carol" })];
const FINAL = {
  persons: [
    {
      name: "Alice",
      count: 2,
      items: [
        card("asset-a", 0, { source: "proposal", written: false, name: "Alice" }),
        card("asset-b", 0, { source: "pin", written: false, name: "Alice", pin: "Alice" }),
      ],
    },
    { name: "Bob", count: 1, items: [card("asset-c", 0, { source: "conflict", written: false, name: "Bob", status: "conflict_skip", proposed_name: "Bob" })] },
  ],
};

const DOMS = [];

function makeDom() {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="app"></div><div id="modalhost"></div><div id="toasthost"></div>
  </body></html>`, { url: "http://localhost/", runScripts: "outside-only" });
  // The UI server injects this into index.html at serve time.
  dom.window.__REID_UI__ = { immichUrl: "http://immich.test", commit: "ui1234" };
  const posts = [];
  dom.window.fetch = async (path, opts) => {
    if (opts?.method === "POST") {
      posts.push({ path, body: JSON.parse(opts.body) });
      if (path === "/api/write-faces") {
        const writes = JSON.parse(opts.body).writes;
        return { ok: true, json: async () => ({ results: writes.map(w => ({ ...w, status: "written" })), written: writes.length }) };
      }
      return { ok: true, json: async () => ({ count: 1, results: [], confirmed: true, pin: "x" }) };
    }
    let json;
    if (path === "/api/state") {
      json = {
        clusters: CLUSTERS,
        active_cluster: 1,
        evidence: EVIDENCE_W1,
        assignment: { cluster: 1, window: W1, proposed_name: "Alice", anchor: "Immich label: Alice", total_appearances: 2, writable: 2, conflicted: 0, labeled: 0, existing_people: ["Alice", "Bob", "Carol"] },
        stats: [],
        config_knobs: { window_days: 1 },
        people: ["Alice", "Bob", "Carol"],
        immich_url: "http://immich.test",
        core_commit: "core567",
      };
    } else if (path.startsWith("/api/evidence")) {
      json = EVIDENCE_W1;
    } else if (path === "/api/people") {
      json = { people: ["Alice", "Bob", "Carol"] };
    } else if (path === "/api/jobs") {
      json = { active: false, last_run: null };
    } else if (path === "/api/final-check") {
      json = FINAL;
    } else {
      json = {};
    }
    return { ok: true, json: async () => json };
  };
  DOMS.push(dom);
  dom.window.eval(APP_JS); // app.js boots itself
  return { dom, posts, window: dom.window, document: dom.window.document };
}

const flush = async (ms = 30) => { await new Promise(r => setTimeout(r, ms)); };
const click = (window, el) => el.dispatchEvent(new window.Event("click", { bubbles: true }));
const change = (window, el) => el.dispatchEvent(new window.Event("change", { bubbles: true }));
const clickAll = (window, document, selector) => {
  const els = [...document.querySelectorAll(selector)];
  assert.ok(els.length, `no element matches ${selector}`);
  els.forEach(el => click(window, el));
};

async function testBootRendersTabsAndNewCard() {
  const { window, document } = makeDom();
  await flush();
  const tabs = [...document.querySelectorAll(".tab")].map(t => t.textContent);
  assert.deepEqual(tabs, ["Proposals", "Final check"]);
  // Version chip: which git commit the UI and the core are running.
  const ver = document.querySelector(".top .ver");
  assert.ok(ver, "version chip missing");
  assert.match(ver.textContent, /ui ui1234 · core core567/);
  // Proposals panes: days, clusters, evidence grid, actions pane.
  assert.ok(document.querySelector(".days .dayrow.sel"), "no day selected");
  assert.ok(document.querySelector(".clusters .item.sel"), "no cluster selected");
  const cards = [...document.querySelectorAll(".evcard")];
  assert.equal(cards.length, 2, "evidence cards not rendered");
  const first = cards[0];
  // Info rows: Immich / ReID / You.
  const rows = [...first.querySelectorAll(".inforow")].map(r => r.textContent);
  assert.ok(rows.some(r => r.includes("Immich:")), "Immich row missing");
  assert.ok(rows.some(r => r.includes("ReID:") && r.includes("Alice")), "ReID row missing");
  assert.ok(rows.some(r => r.includes("You:")), "You row missing");
  // Sibling badge + select-only checkbox (no write checkbox — ticket 02 verdict).
  assert.match(first.textContent, /1 other in photo/);
  assert.ok(first.querySelector("[data-sel]"), "select checkbox missing");
  assert.equal(first.querySelectorAll('input[type="checkbox"]').length, 1, "write checkbox must be gone");
  // Clickable crop links to Immich.
  assert.equal(first.querySelector(".cropwrap a")?.href, "http://immich.test/photos/asset-a");
  // Full-photo expand draws sibling outlines.
  click(window, first.querySelector("[data-expand]"));
  await flush();
  const overlay = document.querySelector(".evcard svg.overlay");
  assert.ok(overlay, "overlay missing after expand");
  assert.equal(overlay.querySelectorAll("rect.pbox").length, 2, "sibling boxes not drawn");
  assert.equal(overlay.querySelectorAll("rect.pbox.self").length, 1, "self not highlighted");
}

async function testBulkBarCountsSelectionAndPosts() {
  const { window, document, posts } = makeDom();
  await flush();
  const sels = [...document.querySelectorAll("[data-sel]")];
  assert.equal(sels.length, 2);
  sels.forEach(el => { el.checked = true; change(window, el); });
  await flush();
  assert.match(document.querySelector(".bulkbar b")?.textContent ?? "", /2 selected/);
  click(window, document.querySelector('[data-bulk="ignore"]'));
  await flush();
  const post = posts.find(p => p.path === "/api/bulk-ignore");
  assert.ok(post, "no /api/bulk-ignore POST");
  assert.equal(post.body.items.length, 2);
  assert.deepEqual(post.body.items[0], { asset_id: "asset-a", appearance: 0 });
}

async function testConfirmProposalPostsConfirmCluster() {
  const { window, document, posts } = makeDom();
  await flush();
  click(window, document.querySelector(".b-ok"));
  await flush();
  const post = posts.find(p => p.path === "/api/confirm-cluster");
  assert.ok(post, "no /api/confirm-cluster POST");
  assert.equal(post.body.window, W1);
  assert.equal(post.body.cluster, 1);
  assert.equal(post.body.name, "Alice");
  assert.equal(post.body.confirmed, true);
}

async function testFinalCheckTabPersonGridAndWrite() {
  const { window, document, posts } = makeDom();
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  const persons = [...document.querySelectorAll(".personrow")].map(r => r.dataset.person);
  assert.deepEqual(persons, ["Alice", "Bob"]);
  // Alice selected by default: grid shows her 2 items, checkboxes default OFF.
  let cards = [...document.querySelectorAll(".evcard")];
  assert.equal(cards.length, 2);
  assert.ok(cards.every(c => !c.querySelector("[data-fsel]").checked), "final items must default off");
  // Select one, write bar names the person, write posts per-item names.
  const cb = cards[0].querySelector("[data-fsel]");
  cb.checked = true; change(window, cb);
  await flush();
  const bar = document.querySelector("#writePerson");
  assert.match(bar.textContent, /Write 1 selected as “Alice” to Immich/);
  click(window, bar);
  await flush();
  const post = posts.find(p => p.path === "/api/write-faces");
  assert.ok(post, "no /api/write-faces POST");
  assert.deepEqual(post.body.writes, [{ asset_id: "asset-a", appearance: 0, name: "Alice" }]);
  // Switch person pane.
  click(window, document.querySelector('.personrow[data-person="Bob"]'));
  await flush();
  cards = [...document.querySelectorAll(".evcard")];
  assert.equal(cards.length, 1);
  assert.match(cards[0].textContent, /conflict/);
}

async function testCardIgnoreButtonPosts() {
  const { window, document, posts } = makeDom();
  await flush();
  click(window, document.querySelector("[data-ignore]"));
  await flush();
  const post = posts.find(p => p.path === "/api/ignore");
  assert.ok(post, "no /api/ignore POST");
  assert.equal(post.body.asset_id, "asset-a");
}

const tests = [
  testBootRendersTabsAndNewCard,
  testBulkBarCountsSelectionAndPosts,
  testConfirmProposalPostsConfirmCluster,
  testFinalCheckTabPersonGridAndWrite,
  testCardIgnoreButtonPosts,
];
let failed = 0;
for (const t of tests) {
  try {
    await t();
    console.log(`ok - ${t.name}`);
  } catch (err) {
    failed++;
    console.error(`FAIL - ${t.name}\n  ${err.message}`);
  }
}
console.log(`${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
