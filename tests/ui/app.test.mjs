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
  { cluster: 0, window: "2026-11-18", proposed_name: null, anchor: "already labeled in Immich", confidence: 0.9, appearances: 3, writable: 0, conflicted: 0, ask_for_name: 0, labeled: 3, status: "open", crops: [], fully_labeled: true },
  { cluster: 1, window: "2026-11-17", proposed_name: "Alice", anchor: "x", confidence: 0.8, appearances: 2, writable: 2, conflicted: 0, ask_for_name: 0, labeled: 0, status: "open", crops: [] },
  { cluster: 0, window: "2025-09-23", proposed_name: "Steffie", anchor: "x", confidence: 0.7, appearances: 2, writable: 1, conflicted: 0, ask_for_name: 0, labeled: 1, status: "open", crops: [] },
];
const DRY_BATCH = {  id: "batch-1",
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

const WRITABLE_CARD = (asset, window_, cluster, included = true) => ({
  asset_id: asset,
  appearance: 0,
  window: window_,
  cluster,
  date: window_,
  confidence: 0.9,
  status: "will_write",
  included,
  immich_label: null,
  pin: null,
  asset_name: `IMG_${asset}.jpg`,
  ignored: false,
  conflict_names: null,
  proposed_name: "Steffie",
});
const PINNED_CARD = {
  asset_id: "asset-pin",
  appearance: 0,
  window: "2026-11-17",
  cluster: 9,
  date: "2026-11-17",
  confidence: 0.9,
  status: "will_write",
  included: false,
  immich_label: null,
  pin: "Steffie",
  asset_name: "IMG_pin.jpg",
  ignored: false,
  conflict_names: null,
  proposed_name: "Othercluster",
  pinned_elsewhere: true,
};
const REVIEW = {
  tabs: [
    {
      name: "Steffie",
      writable: 2,
      labeled: 1,
      pinned: 1,
      pinned_cards: [PINNED_CARD],
      writable_cards: [
        WRITABLE_CARD("asset-s1", "2026-11-17", 0),
        WRITABLE_CARD("asset-s2", "2025-09-23", 1),
        {
          asset_id: "asset-cx",
          appearance: 0,
          window: "2026-11-17",
          cluster: 3,
          date: "2026-11-17",
          confidence: 0.9,
          status: "conflict_skip",
          included: false,
          immich_label: null,
          pin: null,
          asset_name: "IMG_cx.jpg",
          ignored: false,
          conflict_names: ["Steffie", "Bob"],
          proposed_name: null,
        },
      ],
      labeled_cards: [
        {
          asset_id: "asset-s3",
          appearance: 0,
          window: "2026-11-17",
          cluster: 2,
          date: "2026-11-17",
          confidence: 0.9,
          status: "already_labeled",
          included: false,
          immich_label: "Steffie",
          pin: null,
          asset_name: "IMG_s3.jpg",
          ignored: false,
          conflict_names: null,
          proposed_name: null,
        },
      ],
    },
  ],
  pending: [],
  total_pending: 0,
  pending_all: [
    {
      batch_id: "b1",
      name: "Steffie",
      window: "2026-11-17",
      cluster: 0,
      asset_id: "asset-p1",
      appearance: 0,
      asset_name: "IMG_p1.jpg",
      status: "would_write",
      default_included: true,
    },
    {
      batch_id: "b1",
      name: "Steffie",
      window: "2026-11-17",
      cluster: 0,
      asset_id: "asset-p2",
      appearance: 0,
      asset_name: "IMG_p2.jpg",
      status: "would_write",
      default_included: true,
    },
  ],
  total_pending: 2,
  unnamed: [
    {
      window: "2026-11-17",
      cluster: 5,
      cards: [
        {
          asset_id: "asset-u1",
          appearance: 0,
          window: "2026-11-17",
          cluster: 5,
          date: "2026-11-17",
          confidence: 0.9,
          status: "ask_for_name",
          included: false,
          immich_label: null,
          pin: null,
          asset_name: "IMG_u1.jpg",
          ignored: false,
          conflict_names: null,
          proposed_name: null,
        },
      ],
    },
  ],
  total_writable: 2,
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
            asset_name: "IMG_4127.jpg",
            ignored: false,
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
        immich_url: "http://immich.test",
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
        pin: null,
        conflict_names: null,
        ...(opts.evidenceOverride || {}),
      }];
      const delay = opts.evidenceDelayFor ? opts.evidenceDelayFor(cluster) : 0;
      if (delay) await new Promise(r => setTimeout(r, delay));
    } else if (path === "/api/dry-run-batches") {
      json = [DRY_BATCH];
    } else if (path === "/api/write-review") {
      json = opts.emptyPending
        ? { ...REVIEW, pending: [] }
        : { ...REVIEW, pending: REVIEW.pending_all };
    } else if (path === "/api/confirm-all") {
      json = { groups: [], count: 2, batch_ids: ["batch-1", "batch-2"] };
    } else if (path === "/api/promote-all") {
      json = { promoted: 2, results: [] };
    } else if (path === "/api/write-pending") {
      json = { count: 1, written: 1, results: [] };
    } else if (path.startsWith("/api/write-decision")) {
      json = { included: true };
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
  assert.match(batchCard.textContent, /Approved.*Alice/);
  assert.match(batchCard.textContent, /ready ×1/);
  assert.match(batchCard.textContent, /label conflict ×1/);
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
  assert.match(meta, /2 faces/, "action counts not updated at click time");

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

async function testRejectedCardWordingAndUndo() {
  // The excluded chip names what it refuses, and Undo removes the exclusion.
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);
  // Name the cluster through the app's own name-input handler, then reject
  // the card the way the button does.
  const nameInput = document.getElementById("nameInput");
  nameInput.value = "Alice";
  nameInput.dispatchEvent(new window.Event("change", { bubbles: true }));
  await flush(window, 20);
  window.eval(`rejectAppearance("asset-a", 0, "Alice")`);
  await flush(window, 30);

  const card = document.querySelector('.evcard[data-asset="asset-a"]');
  assert.match(card.textContent, /excluded — not “Alice”/, "excluded wording missing");
  window.eval(`unrejectAppearance("asset-a", 0, "Alice")`);
  await flush(window, 30);
  assert.ok(fetchLog.includes("/api/unreject"), "no /api/unreject request");
}

async function testConflictedCardShowsFightingLabels() {
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  // Flip the card to conflicted with fighting labels the way the server
  // sends them: via the public evidence update path... state is
  // module-lexical, so render a card through the fake API instead: the
  // /api/evidence fake returns status per requested cluster; simplest is to
  // check the real server behavior in the Python suite and here assert the
  // chip renders when the field arrives — drive it through renderEvidence's
  // input by re-fetching with a conflicted payload.
  const { window: w2, document: d2 } = makeDom({
    evidenceOverride: {
      status: "conflict_skip",
      conflict_names: ["Steffie", "Bob"],
    },
  });
  await w2.eval("boot()");
  await flush(w2);
  // Click the cluster so loadEvidence fetches and renders the card.
  d2.querySelector('.clusters .item[data-key="2026-11-17:0"]').click();
  await flush(w2, 30);
  const card = d2.querySelector('.evcard[data-asset="asset-c0"]');
  assert.match(card.textContent, /conflict: Steffie \/ Bob/, "conflict chip missing");
}

async function testCardsLinkToImmichAndShowFileNames() {
  // The name line links into Immich's viewer; the visible name is the
  // original file name, the UUID demoted to the tooltip.
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);

  const card = document.querySelector('.evcard[data-asset="asset-a"]');
  const link = card.querySelector("a.asset-name");
  assert.ok(link, "no viewer link on the card");
  assert.equal(
    link.getAttribute("href"),
    "http://immich.test/photos/asset-a",
    "wrong viewer link"
  );
  assert.ok(!/asset-a/.test(link.textContent), "UUID still visible");
}

async function testIgnoreButtonIgnoresAndUndoRestores() {
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);

  window.eval(`ignoreAppearance("asset-a", 0)`);
  await flush(window, 30);
  assert.ok(fetchLog.includes("/api/ignore"), "no /api/ignore request");
  const card = document.querySelector('.evcard[data-asset="asset-a"]');
  assert.match(card.textContent, /ignored — never labeled/, "ignored chip missing");
  assert.ok(card.querySelector(".b-unignore"), "no undo-ignore control");
  assert.ok(!card.querySelector(".b-assign"), "ignored card still offers assign");

  window.eval(`unignoreAppearance("asset-a", 0)`);
  await flush(window, 30);
  assert.ok(fetchLog.includes("/api/unignore"), "no /api/unignore request");
  const restored = document.querySelector('.evcard[data-asset="asset-a"]');
  assert.ok(!/ignored — never labeled/.test(restored.textContent), "ignored chip lingered");
  assert.ok(restored.querySelector(".b-assign"), "assign button not restored");
}

async function testReviewPageTogglesAndRenders() {
  // The top-bar toggle opens the write-review page: person tabs, day
  // sub-groups, already-labeled with the green-border treatment, and the
  // global action bar. "Back to queue" returns.
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);

  assert.ok(document.getElementById("rvTabs"), "review shell missing");
  const tabs = [...document.querySelectorAll(".rvtab")].map(el => el.textContent.trim());
  assert.ok(tabs.some(t => t.includes("Steffie")), "person tab missing");
  assert.ok(tabs.some(t => t.includes("unnamed")), "unnamed tab missing");
  assert.ok(
    document.body.textContent.includes("Approve 2 writes for review"),
    "global dry-run action missing"
  );

  // Day sub-groups inside the person pane.
  window.eval('setReviewTab("Steffie");');
  await flush(window, 10);
  const days = [...document.querySelectorAll(".rv-dayname")].map(el => el.textContent);
  assert.ok(days.includes("2026-11-17") && days.includes("2025-09-23"), "day sub-groups missing");
  const labeledCard = document.querySelector(".evcard.evlabeled");
  assert.ok(labeledCard && labeledCard.dataset.asset === "asset-s3", "already-labeled card not styled");

  // Back to queue restores the 4-pane workspace.
  window.eval('setView("queue")');
  await flush(window, 10);
  assert.ok(document.querySelector(".days"), "queue did not return");
}

async function testReviewConfirmAllAndToggleAll() {
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);
  window.eval('setReviewTab("Steffie");');
  await flush(window, 10);

  // None: every writable card of the tab posts its decision.
  window.eval('toggleAllWrites(false)');
  await flush(window, 30);
  const decisions = fetchLog.filter(p => p === "/api/write-decision").length;
  assert.equal(decisions, 2, "toggle-all did not persist both decisions");
  const unchecked = [...document.querySelectorAll('.evcard input[type="checkbox"]')].every(
    el => !el.checked
  );
  assert.ok(unchecked, "checkboxes did not uncheck");

  // All: back on, then confirm-all posts once.
  window.eval('toggleAllWrites(true)');
  await flush(window, 30);
  window.eval('confirmAllWrites()');
  await flush(window, 30);
  assert.equal(fetchLog.filter(p => p === "/api/confirm-all").length, 1, "confirm-all not posted once");
}

async function testPendingTabAlwaysRenderedAndPinnedSection() {
  // The Pending writes tab is always visible (empty state included), and
  // cards pinned to a person show up on that person's tab under 'pinned'
  // — without a write checkbox (they are not writable until the next run).
  const { window, document } = makeDom({ emptyPending: true });
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);

  const pendingTab = document.querySelector('.rvtab[data-tab="__pending__"]');
  assert.ok(pendingTab, "pending tab not rendered");
  assert.match(pendingTab.textContent, /Pending writes 0/, "pending tab count wrong");
  window.eval('setReviewTab("__pending__");');
  await flush(window, 10);
  assert.match(
    document.getElementById("rvBody").textContent,
    /nothing pending — every approved write is in Immich/,
    "empty state missing"
  );

  // Steffie's tab: the pinned card is under 'pinned', without a checkbox.
  window.eval('setReviewTab("Steffie");');
  await flush(window, 10);
  const pinnedCard = document.querySelector('.evcard[data-asset="asset-pin"]');
  assert.ok(pinnedCard, "pinned card missing from the pinned section");
  assert.match(pinnedCard.textContent, /pinned: Steffie/, "pinned chip missing");
  assert.ok(!pinnedCard.querySelector("input[type=checkbox]"), "pinned card offers a write checkbox");
  assert.match(
    document.getElementById("rvBody").textContent,
    /will be proposed as Steffie on the next batch run/,
    "pinned section header missing"
  );
}

async function testResolveButtonsPinDirectly() {
  // A conflicted card offers one button per fighting label; clicking pins
  // the appearance to that person without opening the modal — and the card
  // updates in place (review page included).
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);
  window.eval('setReviewTab("Steffie");');
  await flush(window, 10);

  const buttons = [...document.querySelectorAll(".b-resolve")].map(b => b.dataset.name);
  assert.deepEqual(buttons, ["Steffie", "Bob"], "resolve buttons per fighting label missing");

  document.querySelector('.b-resolve[data-name="Bob"]').click();
  await flush(window, 30);

  assert.ok(fetchLog.includes("/api/pin"), "resolve did not post /api/pin");
  const card = document.querySelector('.evcard[data-asset="asset-cx"]');
  assert.match(card.textContent, /pinned: Bob/, "pinned chip not updated");
  const marked = document.querySelector('.b-resolve[data-name="Bob"]');
  assert.ok(marked.classList.contains("resolved"), "chosen option not highlighted");
}

async function testChipsCarryNames() {
  // Status chips say who: 'will write "X"' on writable cards, and
  // 'already labeled "X"' replacing the separate immich chip.
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);
  window.eval('setReviewTab("Steffie");');
  await flush(window, 10);

  const writable = document.querySelector('.evcard[data-asset="asset-s1"] .chip');
  assert.match(writable.textContent, /will write “Steffie”/, "writable chip lacks the name");

  const labeled = document.querySelector('.evcard[data-asset="asset-s3"]');
  const chips = [...labeled.querySelectorAll(".chip")].map(el => el.textContent);
  assert.ok(
    chips.some(t => t.includes('already labeled “Steffie”')),
    "labeled chip lacks the name"
  );
  assert.ok(
    !chips.some(t => t.startsWith("immich:")),
    "redundant immich chip still shown on an already-labeled card"
  );
}

async function testFullyLabeledClustersHiddenBehindToggle() {
  // Fully-labeled groups are hidden from the queue by default (no pending
  // work); the 'show already labeled' toggle reveals them, and j/k never
  // lands on a hidden group.
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);

  // Hidden: not in the clusters pane, not in day counts.
  const keys = [...document.querySelectorAll(".clusters .item")].map(el => el.dataset.key);
  assert.ok(!keys.includes("2026-11-18:0"), "fully-labeled cluster visible without the toggle");
  assert.ok(keys.includes("2026-11-17:0"), "normal cluster missing");

  // j/k navigation skips the hidden group: from the last visible open
  // cluster, j wraps to the first.
  window.eval('selectCluster("2026-11-17:1");');
  await flush(window, 10);
  window.document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "j", bubbles: true }));
  await flush(window, 10);
  assert.equal(
    document.querySelector(".main-header .cluster-id")?.textContent,
    "2025-09-23 / 0",
    "navigation did not skip the hidden fully-labeled cluster"
  );

  // The toggle reveals them: their day reappears in the days pane.
  window.eval(`
    const t = document.getElementById("showLabeledToggle");
    t.checked = true;
    t.dispatchEvent(new window.Event("change", { bubbles: true }));
  `);
  await flush(window, 10);
  const revealed = [...document.querySelectorAll(".days .dayrow")].map(
    el => el.dataset.day
  );
  assert.ok(revealed.includes("2026-11-18"), "toggle did not reveal fully-labeled clusters");

  // Toggling off again while viewing the hidden cluster moves the selection
  // to a visible one.
  window.eval('selectCluster("2026-11-18:0");');
  await flush(window, 10);
  window.eval(`
    const t2 = document.getElementById("showLabeledToggle");
    t2.checked = false;
    t2.dispatchEvent(new window.Event("change", { bubbles: true }));
  `);
  await flush(window, 10);
  assert.ok(
    document.querySelector(".main-header .cluster-id")?.textContent !== "2026-11-18 / 0",
    "selection stuck on a hidden cluster"
  );
}

async function testReviewTabClicksSwitchPanes() {
  // Tab clicks must switch panes: renderReviewTabs re-renders the strip on
  // every switch, so the binding has to live there (a click binding on the
  // first render silently died after the first re-render).
  const { window, document } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);

  document.querySelector('.rvtab[data-tab="__pending__"]').click();
  await flush(window, 10);
  assert.ok(
    document.getElementById("rvBody").textContent.includes("Pending writes") ||
      document.getElementById("rvWritePending"),
    "pending tab click did not switch the pane"
  );

  document.querySelector('.rvtab[data-tab="Steffie"]').click();
  await flush(window, 10);
  assert.ok(
    document.body.textContent.includes("Approve 2 writes for review"),
    "person tab click did not switch the pane"
  );
}

async function testPendingWritesTabCardsAndWrite() {
  // The Pending writes tab lists approved-but-unwritten items as
  // selectable cards; select all/none work; writing posts exactly the
  // checked items.
  const { window, document, fetchLog } = makeDom();
  await window.eval("boot()");
  await flush(window);
  window.eval('setView("review")');
  await flush(window, 30);
  window.eval('setReviewTab("__pending__");');
  await flush(window, 10);

  const cards = [...document.querySelectorAll(".pcard")];
  assert.equal(cards.length, 2, "pending cards missing");
  assert.ok(
    cards.every(c => c.classList.contains("selected")),
    "ready items not pre-checked"
  );

  // Deselect all, re-select only the second.
  window.eval('document.getElementById("rvSelNone").click();');
  await flush(window, 10);
  window.eval('togglePending("b1:asset-p2:0");');
  await flush(window, 10);

  const btn = document.getElementById("rvWritePending");
  assert.equal(btn.textContent.trim(), "Write 1 selected to Immich", "count wrong");
  btn.click();
  await flush(window, 30);

  assert.ok(fetchLog.includes("/api/write-pending"), "no /api/write-pending request");
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
  testRejectedCardWordingAndUndo,
  testConflictedCardShowsFightingLabels,
  testCardsLinkToImmichAndShowFileNames,
  testIgnoreButtonIgnoresAndUndoRestores,
  testReviewPageTogglesAndRenders,
  testReviewConfirmAllAndToggleAll,
  testResolveButtonsPinDirectly,
  testChipsCarryNames,
  testPendingTabAlwaysRenderedAndPinnedSection,
  testPendingWritesTabCardsAndWrite,
  testReviewTabClicksSwitchPanes,
  testFullyLabeledClustersHiddenBehindToggle,
];
for (const test of tests) {
  await test();
  console.log(`ok - ${test.name}`);
}
console.log(`\n${tests.length} UI tests passed`);
