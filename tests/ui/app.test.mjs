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

function makeDom(testOpts = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="app"></div><div id="modalhost"></div><div id="toasthost"></div>
  </body></html>`, { url: testOpts.pageUrl || "http://localhost/", runScripts: "outside-only" });
  // The UI server injects this into index.html at serve time; uiConfig: null
  // simulates no injection (the core-reported immich_url is then used).
  if (testOpts.uiConfig !== null) {
    dom.window.__REID_UI__ = { immichUrl: testOpts.uiImmichUrl ?? "http://immich.test", commit: "ui1234" };
  }
  // Simulate a restored tab from a previous session (localStorage survives refresh).
  if (testOpts.view) {
    dom.window.localStorage.setItem("reid-view", testOpts.view);
  }
  const posts = [];
  const stateImmichUrl = testOpts.immichUrl || "http://immich.test";
  dom.window.fetch = async (path, init) => {
    if (init?.method === "POST") {
      posts.push({ path, body: JSON.parse(init.body) });
      if (path === "/api/write-faces") {
        const writes = JSON.parse(init.body).writes;
        return { ok: true, json: async () => ({ results: writes.map(w => ({ ...w, status: "written" })), written: writes.length }) };
      }
      return { ok: true, json: async () => ({ count: 1, results: [], confirmed: true, pin: "x" }) };
    }
    let json;
    if (path === "/api/state") {
      json = {
        clusters: testOpts.clusters ?? CLUSTERS,
        active_cluster: 1,
        evidence: EVIDENCE_W1,
        assignment: { cluster: 1, window: W1, proposed_name: "Alice", anchor: "Immich label: Alice", total_appearances: 2, writable: 2, conflicted: 0, labeled: 0, existing_people: ["Alice", "Bob", "Carol"] },
        stats: [],
        config_knobs: { window_days: 1 },
        people: ["Alice", "Bob", "Carol"],
        immich_url: stateImmichUrl,
        core_commit: "core567",
      };
    } else if (path.startsWith("/api/evidence")) {
      json = EVIDENCE_W1;
    } else if (path === "/api/people") {
      json = { people: ["Alice", "Bob", "Carol"] };
    } else if (path === "/api/jobs") {
      json = { active: false, last_run: null };
    } else if (path === "/api/final-check") {
      json = testOpts.finalOverride ?? FINAL;
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
  // Handle the ignore-reason modal
  const nm = document.querySelector("#nm");
  if (nm) {
    nm.value = "not a person";
    click(window, document.querySelector("#nmo"));
    await flush();
  }
  const post = posts.find(p => p.path === "/api/bulk-ignore");
  assert.ok(post, "no /api/bulk-ignore POST");
  assert.equal(post.body.items.length, 2);
  assert.deepEqual(post.body.items[0], { asset_id: "asset-a", appearance: 0 });
  assert.equal(post.body.reason, "not_a_person");
}

async function testSelectAllSelectsAllCards() {
  const { window, document } = makeDom();
  await flush();
  click(window, document.querySelector('[data-bulk="selall"]'));
  await flush();
  const sels = [...document.querySelectorAll("[data-sel]")];
  assert.equal(sels.length, 2, "expected 2 selection checkboxes");
  assert.ok(sels.every(el => el.checked), "all checkboxes should be checked after select all");
  assert.match(document.querySelector(".bulkbar b")?.textContent ?? "", /2 selected/);
  const btn = document.getElementById("writePerson") ?? document.querySelector('[data-bulk="ignore"]');
  assert.ok(btn, "a bulk action button should exist");
}

async function testSelectNoneClearsSelection() {
  const { window, document } = makeDom();
  await flush();
  // Select all first
  click(window, document.querySelector('[data-bulk="selall"]'));
  await flush();
  assert.match(document.querySelector(".bulkbar b")?.textContent ?? "", /2 selected/);
  // Deselect all
  click(window, document.querySelector('[data-bulk="selnone"]'));
  await flush();
  const sels = [...document.querySelectorAll("[data-sel]")];
  assert.ok(sels.every(el => !el.checked), "all checkboxes should be unchecked after deselect all");
  assert.match(document.querySelector(".bulkbar b")?.textContent ?? "", /0 selected/);
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
  assert.match(cards[0].textContent, /labels disagree/);
}

async function testCardIgnoreButtonPosts() {
  const { window, document, posts } = makeDom();
  await flush();
  // Click "Ignore this person" button (second [data-ignore], reason=ignore)
  const ignoreBtn = document.querySelectorAll('[data-ignore]')[1];
  assert.ok(ignoreBtn, "no 'Ignore this person' button");
  click(window, ignoreBtn);
  await flush();
  const post = posts.find(p => p.path === "/api/ignore");
  assert.ok(post, "no /api/ignore POST");
  assert.equal(post.body.asset_id, "asset-a");
  assert.equal(post.body.reason, "ignore");
}

async function testNotAPersonButtonPostsReason() {
  const { window, document, posts } = makeDom();
  await flush();
  // Click "Not a person" button (reason=not_a_person)
  const notAPersonBtn = document.querySelector('[data-reason="not_a_person"]');
  assert.ok(notAPersonBtn, "no 'Not a person' button");
  click(window, notAPersonBtn);
  await flush();
  const post = posts.find(p => p.path === "/api/ignore");
  assert.ok(post, "no /api/ignore POST");
  assert.ok(["asset-a", "asset-b"].includes(post.body.asset_id), "posted for one of the cards");
  assert.equal(post.body.reason, "not_a_person");
}

async function testLocalhostUiImmichUrlRewrittenToBrowserHost() {
  // The UI server injected a localhost URL (e.g. inferred from the core's
  // base_url). Opened via the LAN name, viewer links must point at that host.
  const { document } = makeDom({
    pageUrl: "http://192.168.1.10:8080/",
    uiImmichUrl: "http://localhost:2283",
  });
  await flush();
  const link = document.querySelector(".evcard .cropwrap a");
  assert.ok(link, "no viewer link on the card");
  assert.equal(
    link.href,
    "http://192.168.1.10:2283/photos/asset-a",
    "localhost should be rewritten to the page's host, keeping the port"
  );
}

async function testCoreReportedLocalhostRewrittenWhenNoUiOverride() {
  // No __REID_UI__ injection: the core-reported immich_url (localhost) is
  // used, and still rewritten to the browser's host.
  const { document } = makeDom({
    pageUrl: "http://immich-1.lan:8080/",
    uiConfig: null,
    immichUrl: "http://127.0.0.1:2283",
  });
  await flush();
  const link = document.querySelector(".evcard .cropwrap a");
  assert.ok(link, "no viewer link on the card");
  assert.equal(
    link.href,
    "http://immich-1.lan:2283/photos/asset-a",
    "127.0.0.1 from the core should be rewritten to the page's host"
  );
}

async function testNonLocalhostImmichUrlNotRewritten() {
  // A real hostname (explicit [ui] immich_url override for a remote UI)
  // must be kept as-is, even when the page is served from a LAN name.
  const { document } = makeDom({
    pageUrl: "http://192.168.1.10:8080/",
    uiImmichUrl: "http://immich.lan:2283",
  });
  await flush();
  const link = document.querySelector(".evcard .cropwrap a");
  assert.ok(link, "no viewer link on the card");
  assert.equal(
    link.href,
    "http://immich.lan:2283/photos/asset-a",
    "non-localhost URL must not be rewritten"
  );
}

async function testWrittenItemCheckboxDisabled() {
  // Already-written appearances show "written" chip and their write-to-Immich
  // checkbox is disabled, so they can't be selected for writing again.
  const FINAL_WITH_WRITTEN = {
    persons: [
      {
        name: "Alice",
        count: 2,
        items: [
          card("asset-a", 0, { source: "proposal", written: true, name: "Alice" }),
          card("asset-b", 0, { source: "pin", written: false, name: "Alice", pin: "Alice" }),
        ],
      },
    ],
  };
  const { window, document } = makeDom({ finalOverride: FINAL_WITH_WRITTEN });
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  click(window, document.querySelector('.personrow[data-person="Alice"]'));
  await flush();
  const cbWritten = document.querySelector('[data-fsel="asset-a:0"]');
  assert.ok(cbWritten, "written card missing");
  assert.ok(cbWritten.disabled, "written item checkbox not disabled");
  const cbUnwritten = document.querySelector('[data-fsel="asset-b:0"]');
  assert.ok(cbUnwritten, "unwritten card missing");
  assert.ok(!cbUnwritten.disabled, "unwritten item checkbox should be enabled");
}

async function testWriteButtonDisabledWhenAllSelectedAreWritten() {
  // Selecting only written items must not enable the write button.
  const FINAL_WITH_WRITTEN = {
    persons: [
      {
        name: "Alice",
        count: 2,
        items: [
          card("asset-a", 0, { source: "proposal", written: true, name: "Alice" }),
          card("asset-b", 0, { source: "pin", written: true, name: "Alice", pin: "Alice" }),
        ],
      },
    ],
  };
  const { window, document } = makeDom({ finalOverride: FINAL_WITH_WRITTEN });
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  click(window, document.querySelector('.personrow[data-person="Alice"]'));
  await flush();
  const cb1 = document.querySelector('[data-fsel="asset-a:0"]');
  const cb2 = document.querySelector('[data-fsel="asset-b:0"]');
  cb1.checked = true; click(window, cb1);
  cb2.checked = true; click(window, cb2);
  await flush();
  const btn = document.getElementById("writePerson");
  assert.ok(btn.disabled, "write button should be disabled when all selected are written");
}

async function testMixedSelectionWritesOnlyUnwritten() {
  // Selecting one written + one unwritten item posts only the unwritten one.
  const FINAL_MIXED = {
    persons: [
      {
        name: "Alice",
        count: 2,
        items: [
          card("asset-a", 0, { source: "proposal", written: true, name: "Alice" }),
          card("asset-b", 0, { source: "pin", written: false, name: "Alice", pin: "Alice" }),
        ],
      },
    ],
  };
  const { window, document, posts } = makeDom({ finalOverride: FINAL_MIXED });
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  click(window, document.querySelector('.personrow[data-person="Alice"]'));
  await flush();
  const cbUnwritten = document.querySelector('[data-fsel="asset-b:0"]');
  cbUnwritten.checked = true; change(window, cbUnwritten);
  await flush();
  const btn = document.getElementById("writePerson");
  assert.ok(btn, "writePerson button missing");
  assert.ok(!btn.disabled, "writePerson button should be enabled");
  click(window, btn);
  await flush();
  const post = posts.find(p => p.path === "/api/write-faces");
  assert.ok(post, "no /api/write-faces POST");
  assert.deepEqual(post.body.writes, [{ asset_id: "asset-b", appearance: 0, name: "Alice" }]);
  const toastEls = [...document.querySelectorAll(".toast")].map(t => t.textContent);
  assert.ok(toastEls.some(t => t.includes("labeled")), "expected a labeled toast");
}

async function testBothTabsAlwaysRendered() {
  // Learning: the 2nd tab (Final check) is always rendered, even when empty.
  const { window, document } = makeDom({ finalOverride: { persons: [] } });
  await flush();
  const tabs = [...document.querySelectorAll(".tab")].map(t => t.textContent);
  assert.deepEqual(tabs, ["Proposals", "Final check"]);
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  // Final check renders its empty state, not a missing tab.
  assert.match(document.body.textContent, /nothing to check yet/);
}

async function testFullyLabeledHiddenBehindToggle() {
  // Learning: fully-labeled groups are hidden from the queue by default,
  // behind the "show already labeled" toggle.
  const clusters = [
    ...CLUSTERS,
    { cluster: 7, window: W1, proposed_name: "Carol", anchor: "Immich label: Carol", confidence: 0.9, appearances: 2, writable: 0, conflicted: 0, ask_for_name: 0, labeled: 2, status: "open", crops: [], fully_labeled: true },
  ];
  const { window, document } = makeDom({ clusters });
  await flush();
  const keys = [...document.querySelectorAll(".clusters .item")].map(el => el.dataset.key);
  assert.ok(!keys.some(k => k.endsWith(":7")), "fully-labeled group visible by default");
  const toggle = document.getElementById("showLabeledToggle");
  assert.ok(toggle, "show-already-labeled toggle missing");
  assert.match(toggle.parentElement.textContent, /show already labeled/);
  toggle.checked = true; change(window, toggle);
  await flush();
  const keysAfter = [...document.querySelectorAll(".clusters .item")].map(el => el.dataset.key);
  assert.ok(keysAfter.some(k => k.endsWith(":7")), "fully-labeled group not shown after toggle");
}

async function testSemanticsLabelsPerTab() {
  // Learning: labels state semantics — Proposals pins for next run,
  // Final check writes now.
  const { window, document } = makeDom();
  await flush();
  assert.ok(document.querySelector('[data-assign]'), "Proposals should have a Pin as… button");
  assert.match(document.querySelector('[data-assign]').textContent, /Pin as… for next run/);
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  assert.ok(document.getElementById("writePerson"), "Final check should have a Write button");
  assert.match(document.getElementById("writePerson").textContent, /Write \d+ selected as/);
  assert.ok(document.querySelector('[data-fassign]'), "Final check cards should have Write as…");
}

async function testConfirmWithSelectionSendsItemsNoModal() {
  // Learning: "Confirm proposal" = confirm the current verdict on the
  // SELECTED photos and send those to Final check — no name popup when the
  // group is already named.
  const { window, document, posts } = makeDom();
  await flush();
  // Select the first card only.
  const sel = document.querySelector("[data-sel]");
  sel.checked = true; change(window, sel);
  await flush();
  click(window, document.querySelector(".b-ok"));
  await flush();
  // No name modal should have appeared.
  assert.ok(!document.querySelector("#nm"), "name modal should not appear for a named group");
  const post = posts.find(p => p.path === "/api/confirm-cluster");
  assert.ok(post, "no /api/confirm-cluster POST");
  assert.equal(post.body.name, "Alice", "should confirm the current proposed name");
  assert.ok(Array.isArray(post.body.items) && post.body.items.length === 1, "should scope to selected photos");
  assert.deepEqual(post.body.items[0], { asset_id: "asset-a", appearance: 0 });
}

async function testFinalCheckImagesGetLoadedClass() {
  // The Final check tab renders via render(); crop images must get the
  // "loaded" class (CSS keeps them at opacity:0 otherwise → black cards).
  const { window, document } = makeDom();
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  const imgs = [...document.querySelectorAll(".finalmain img.crop")];
  assert.ok(imgs.length > 0, "no crop images in Final check");
  for (const img of imgs) {
    img.dispatchEvent(new window.Event("load"));
  }
  assert.ok(imgs.every(img => img.classList.contains("loaded")), "crop images must get .loaded after load");
}

async function testFinalCheckCardHasOutlineCropAndFullPhotoToggle() {
  // Final check cards match the Proposals card: context crop with outlines
  // and neighbours, sibling badge, and a full-photo expand toggle.
  const { window, document } = makeDom();
  await flush();
  click(window, document.querySelector('.tab[data-view="final"]'));
  await flush();
  const card = document.querySelector(".finalmain .evcard");
  assert.ok(card, "no final-check card");
  const img = card.querySelector("img.crop");
  assert.ok(img, "no crop image");
  assert.ok(img.src.includes("outline=1") && img.src.includes("neighbours=1") && img.src.includes("expand=0.6"),
    "crop should include expand/outline/neighbours like the Proposals card, got: " + img.src);
  assert.match(card.textContent, /1 other in photo/, "sibling badge missing");
  const expandBtn = card.querySelector("[data-expand]");
  assert.ok(expandBtn, "full photo button missing in Final check");
  click(window, expandBtn);
  await flush();
  const overlay = document.querySelector(".finalmain svg.overlay");
  assert.ok(overlay, "full-photo overlay missing after expand");
  assert.ok(overlay.querySelectorAll("rect.pbox").length >= 2, "sibling boxes not drawn");
  assert.equal(overlay.querySelectorAll("rect.pbox.self").length, 1, "self not highlighted");
  // Toggle back to the crop.
  click(window, document.querySelector(".finalmain [data-expand]"));
  await flush();
  assert.ok(document.querySelector(".finalmain img.crop"), "crop should be back after collapsing");
}

async function testFinalTabRestoredAfterRefresh() {
  // Learning: a refresh should land back on the tab the user was on, with
  // Final check already loaded (not empty until clicked).
  const { window, document } = makeDom({ view: "final" });
  await flush();
  const finalTab = document.querySelector('.tab[data-view="final"]');
  assert.ok(finalTab.classList.contains("sel"), "final tab should be selected after refresh");
  const persons = [...document.querySelectorAll(".personrow")].map(el => el.dataset.person);
  assert.deepEqual(persons, ["Alice", "Bob"], "final-check persons should load on boot");
  // And switching tabs persists the choice.
  click(window, document.querySelector('.tab[data-view="proposals"]'));
  await flush();
  assert.equal(window.localStorage.getItem("reid-view"), "proposals");
}

const tests = [
  testBootRendersTabsAndNewCard,
  testBulkBarCountsSelectionAndPosts,
  testSelectAllSelectsAllCards,
  testSelectNoneClearsSelection,
  testConfirmProposalPostsConfirmCluster,
  testFinalCheckTabPersonGridAndWrite,
  testCardIgnoreButtonPosts,
  testNotAPersonButtonPostsReason,
  testLocalhostUiImmichUrlRewrittenToBrowserHost,
  testCoreReportedLocalhostRewrittenWhenNoUiOverride,
  testNonLocalhostImmichUrlNotRewritten,
  testWrittenItemCheckboxDisabled,
  testWriteButtonDisabledWhenAllSelectedAreWritten,
  testMixedSelectionWritesOnlyUnwritten,
  testBothTabsAlwaysRendered,
  testFullyLabeledHiddenBehindToggle,
  testSemanticsLabelsPerTab,
  testConfirmWithSelectionSendsItemsNoModal,
  testFinalCheckImagesGetLoadedClass,
  testFinalCheckCardHasOutlineCropAndFullPhotoToggle,
  testFinalTabRestoredAfterRefresh,
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
