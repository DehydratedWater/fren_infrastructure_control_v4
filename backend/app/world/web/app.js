/* ═══════════════════════════════════════════════════════════════════════════
   Twily Haven — observe & visit UI.
   Vanilla JS, no build step, no deps. Talks to the FastAPI world API at /api.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── tiny helpers ──────────────────────────────────────────────────────── */

const API = "/api";
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Escape arbitrary text before it touches innerHTML. Everything from the
 *  world (LLM output, NPC names, user input) must pass through this. */
function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** A scheme-safe href: only allow http(s); otherwise drop it. */
function safeHref(url) {
  try {
    const u = new URL(url, window.location.origin);
    if (u.protocol === "http:" || u.protocol === "https:") return u.href;
  } catch { /* not a URL */ }
  return null;
}

/** GET/POST JSON with graceful failure. Returns null on network/HTTP error. */
async function api(path, opts = {}) {
  try {
    const res = await fetch(API + path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      // 404 on a not-yet-initialised world is expected; signal it specially.
      if (res.status === 404) return { __notfound: true };
      return null;
    }
    return await res.json();
  } catch {
    return null;
  }
}

let toastTimer = null;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  requestAnimationFrame(() => el.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => (el.hidden = true), 220);
  }, 2800);
}

/* ── app state ─────────────────────────────────────────────────────────── */

const state = {
  world: null,           // /api/world payload
  mode: "observe",       // "observe" | "visit"
  lastEventId: 0,        // highest event id rendered
  oldestEventId: null,   // lowest id rendered (for "load earlier")
  hasMore: false,
  npcById: {},           // id -> npc (for actor names)
  busy: false,           // a turn is in flight
  pollTimer: null,
  emptyShown: false,
};

/* ── event rendering ───────────────────────────────────────────────────── */

const KIND_LABEL = {
  narration: "narration",
  action:    "twily",
  speech:    "twily says",
  npc:       "npc",
  research:  "research",
  move:      "moves",
  mood:      "mood",
  system:    "system",
  visitor:   "vis",
};

/** Friendly display name for an actor id, using the npc roster + world chars. */
function actorName(actor, kind) {
  if (!actor) return kind === "narration" ? "Narrator" : "";
  if (actor === "twily")    return state.world?.protagonist?.name || "Twily";
  if (actor === "vis")      return state.world?.visitor?.name || "Vis";
  if (actor === "narrator") return "Narrator";
  const npc = state.npcById[actor];
  return npc ? npc.name : actor;
}

/** Choose the actor colour class. */
function actorClass(actor) {
  if (actor === "twily")    return "actor-twily";
  if (actor === "vis")      return "actor-vis";
  if (actor === "narrator") return "actor-narration";
  return "actor-npc";
}

/** Build the DOM for a single event. */
function eventEl(ev) {
  const kind = KIND_LABEL.hasOwnProperty(ev.kind) ? ev.kind : "narration";
  const el = document.createElement("div");
  el.className = `event ${kind}`;
  el.dataset.id = ev.id;

  const who = esc(actorName(ev.actor, ev.kind));
  const tag = esc(KIND_LABEL[kind]);
  const turn = ev.turn != null ? `turn ${esc(ev.turn)}` : "";

  // Body: speech gets curly quotes; research shows query emphasis if present.
  let body = esc(ev.content || "");
  if (kind === "speech") {
    body = `<span class="speech-q">“</span>${body}<span class="speech-q">”</span>`;
  } else if (kind === "research") {
    body = `<div class="research-q">⌕ ${body}</div>`;
  } else if (kind === "move") {
    body = body || "…drifts somewhere new.";
  }

  el.innerHTML = `
    <div class="event-head">
      ${who ? `<span class="event-actor ${actorClass(ev.actor)}">${who}</span>` : ""}
      <span class="event-tag tag-${kind}">${tag}</span>
      <span class="event-turn">${turn}</span>
    </div>
    <div class="event-body">${body}</div>`;
  return el;
}

/** Is the stream scrolled (roughly) to the bottom? */
function atBottom(stream) {
  return stream.scrollHeight - stream.scrollTop - stream.clientHeight < 80;
}

/** Append new events (those with id > lastEventId), preserving read position. */
function appendEvents(events) {
  if (!events || !events.length) return;
  const stream = $("#stream");
  clearEmptyState();

  const wasAtBottom = atBottom(stream);
  // events arrive newest-last already; only render genuinely new ones.
  const fresh = events
    .filter((e) => e.id > state.lastEventId)
    .sort((a, b) => a.id - b.id);

  for (const ev of fresh) {
    stream.appendChild(eventEl(ev));
    state.lastEventId = Math.max(state.lastEventId, ev.id);
    if (state.oldestEventId === null || ev.id < state.oldestEventId) {
      state.oldestEventId = ev.id;
    }
  }
  if (fresh.length && wasAtBottom) {
    stream.scrollTop = stream.scrollHeight;
  }
}

/** Prepend older events fetched via "Earlier beats", keeping scroll anchored. */
function prependEvents(events) {
  if (!events || !events.length) return;
  const stream = $("#stream");
  const prevHeight = stream.scrollHeight;
  const older = events
    .filter((e) => state.oldestEventId === null || e.id < state.oldestEventId)
    .sort((a, b) => a.id - b.id);

  const frag = document.createDocumentFragment();
  for (const ev of older) {
    frag.appendChild(eventEl(ev));
    if (state.oldestEventId === null || ev.id < state.oldestEventId) {
      state.oldestEventId = ev.id;
    }
  }
  stream.insertBefore(frag, stream.firstChild);
  // keep the user's view anchored where it was
  stream.scrollTop = stream.scrollHeight - prevHeight;
}

/* ── empty / thinking states ───────────────────────────────────────────── */

function showEmptyState() {
  const stream = $("#stream");
  if (state.emptyShown) return;
  stream.innerHTML = `
    <div class="empty-state">
      <span class="em-mark">✦</span>
      <h3 class="em-title">Twily hasn't woken up here yet</h3>
      <p>Her haven is quiet for now. Press <strong>Advance a turn</strong>
         to let the first beat of her day begin.</p>
    </div>`;
  state.emptyShown = true;
}
function clearEmptyState() {
  if (!state.emptyShown) return;
  const es = $(".empty-state");
  if (es) es.remove();
  state.emptyShown = false;
}

function showThinking(msg) {
  const stream = $("#stream");
  clearEmptyState();
  removeThinking();
  const el = document.createElement("div");
  el.className = "thinking";
  el.id = "thinking-row";
  el.innerHTML = `<span>${esc(msg)}</span>
    <span class="dots"><span></span><span></span><span></span></span>`;
  stream.appendChild(el);
  stream.scrollTop = stream.scrollHeight;
}
function removeThinking() {
  const t = $("#thinking-row");
  if (t) t.remove();
}

/* ── world header + sidebar rendering ──────────────────────────────────── */

function renderWorld(world) {
  state.world = world;
  $("#world-name").textContent = world.name || "Twily Haven";
  $("#world-setting").textContent = world.setting || world.description || "";
  document.title = (world.name || "Twily Haven") + " — a life, lived softly";
  for (const n of world.npcs || []) state.npcById[n.id] = n;
}

function renderState(st) {
  if (!st) return;
  // clock
  $("#clock-time").textContent = st.clock_label || "--:--";
  const phase = st.day_phase ? st.day_phase : "";
  $("#clock-meta").textContent =
    `day ${st.day_count ?? "—"}${phase ? " · " + phase : ""}`;

  // location card
  const loc = st.location || {};
  $("#loc-name").textContent = loc.name || "Somewhere";
  $("#loc-kind").textContent = loc.kind || "";
  $("#loc-desc").textContent = loc.description || "";

  // activities chips
  const act = $("#loc-activities");
  act.innerHTML = "";
  for (const a of loc.activities || []) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = a.label || a.tag;
    if (a.description) c.title = a.description;
    act.appendChild(c);
  }

  // present npcs
  const present = $("#loc-present");
  present.innerHTML = "";
  const people = st.present_npcs || [];
  if (st.visitor_present) {
    const v = document.createElement("span");
    v.className = "person";
    v.textContent = (state.world?.visitor?.name || "Vis") + " (you)";
    present.appendChild(v);
  }
  for (const p of people) {
    const s = document.createElement("span");
    s.className = "person";
    s.textContent = p.name + (p.role ? ` · ${p.role}` : "");
    present.appendChild(s);
  }
  if (!people.length && !st.visitor_present) {
    present.innerHTML = `<span class="muted">She's alone here.</span>`;
  }

  // her state
  const ps = st.persona_state || {};
  $("#mood-value").textContent = ps.mood || "—";
  const energy = clamp(ps.energy);
  $("#energy-num").textContent = energy === null ? "—" : energy;
  $("#energy-fill").style.width = (energy === null ? 50 : energy) + "%";

  // neighbors
  renderNeighbors(st.neighbors || []);
}

function clamp(v) {
  if (v === null || v === undefined || isNaN(Number(v))) return null;
  return Math.max(0, Math.min(100, Math.round(Number(v))));
}

function renderNeighbors(neighbors) {
  const box = $("#neighbor-chips");
  box.innerHTML = "";
  if (!neighbors.length) {
    box.innerHTML = `<span class="muted">Nowhere to go from here.</span>`;
    return;
  }
  for (const n of neighbors) {
    const c = document.createElement("span");
    c.className = "chip go";
    c.textContent = n.label ? n.label : n.name;
    c.title = "Peek at " + (n.name || "");
    c.addEventListener("click", () => peekLocation(n.id, n.name));
    box.appendChild(c);
  }
}

/** Where she could wander — show a quick description in a toast. */
async function peekLocation(id, name) {
  const data = await api(`/location/${encodeURIComponent(id)}`);
  if (!data || data.__notfound) { toast(`Couldn't peek at ${name}.`); return; }
  const who = (data.present_npcs || []).map((p) => p.name).join(", ");
  toast(`${data.name}: ${(data.description || "").slice(0, 90)}` +
        (who ? ` — ${who} here.` : ""));
}

/* ── npc roster ────────────────────────────────────────────────────────── */

/** Map affinity (-100..100) to a warm→cool pip colour. */
function affinityColor(a) {
  const v = Math.max(-100, Math.min(100, a || 0));
  if (v >= 60)  return "#c2693f";   // warm terracotta
  if (v >= 20)  return "#d9a05f";
  if (v > -20)  return "#bcae97";   // neutral
  if (v > -60)  return "#8fa6b3";
  return "#6b7d8a";                  // cool
}
function affinityWord(a) {
  if (a >= 60)  return "close";
  if (a >= 20)  return "warm";
  if (a > -20)  return "neutral";
  if (a > -60)  return "cool";
  return "frosty";
}

function renderNpcs(npcs) {
  const list = $("#npc-list");
  list.innerHTML = "";
  if (!npcs || !npcs.length) {
    list.innerHTML = `<li class="muted">No one here yet.</li>`;
    return;
  }
  // sort by warmth, descending
  const sorted = [...npcs].sort((a, b) => (b.affinity || 0) - (a.affinity || 0));
  for (const n of sorted) {
    state.npcById[n.id] = state.npcById[n.id] || n; // keep names fresh
    const li = document.createElement("li");
    li.className = "npc-item";
    const aff = n.affinity || 0;
    li.innerHTML = `
      <span class="pip" style="background:${affinityColor(aff)}"></span>
      <span class="npc-meta">
        <span class="npc-name">${esc(n.name)}</span>
        ${n.role ? `<span class="npc-role"> · ${esc(n.role)}</span>` : ""}
      </span>
      <span class="affinity" title="relationship warmth">
        ${esc(affinityWord(aff))} ${aff > 0 ? "+" : ""}${esc(aff)}
      </span>`;
    list.appendChild(li);
  }
}

/* ── research log ──────────────────────────────────────────────────────── */

function renderResearch(items) {
  const box = $("#research-log");
  box.innerHTML = "";
  if (!items || !items.length) {
    box.innerHTML = `<p class="muted">Nothing looked up yet.</p>`;
    return;
  }
  for (const r of items) {
    const det = document.createElement("details");
    det.className = "research-item";
    const results = (r.results || []).map((res) => {
      const href = safeHref(res.link);
      const title = esc(res.title || res.link || "untitled");
      const snip = esc(res.snippet || "");
      const link = href
        ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${title}</a>`
        : `<span>${title}</span>`;
      return `<li>${link}${snip ? `<div class="research-snip">${snip}</div>` : ""}</li>`;
    }).join("");
    det.innerHTML = `
      <summary>${esc(r.query || "a lookup")}</summary>
      ${r.summary ? `<div class="research-summary">${esc(r.summary)}</div>` : ""}
      ${results ? `<ul class="research-results">${results}</ul>` : ""}`;
    box.appendChild(det);
  }
}

/* ── data refresh orchestration ────────────────────────────────────────── */

/** Initial full load. */
async function bootstrap() {
  const world = await api("/world");
  if (world && !world.__notfound) {
    renderWorld(world);
  } else {
    // No world package — degrade gracefully but keep the page alive.
    $("#world-setting").textContent =
      "This haven is still being dreamed into being.";
  }

  await Promise.all([refreshState(), refreshEvents(true), refreshNpcs(), refreshResearch()]);

  if (state.lastEventId === 0 && !state.emptyShown) showEmptyState();
}

async function refreshState() {
  const st = await api("/state");
  if (st && !st.__notfound) renderState(st);
}

async function refreshEvents(initial = false) {
  const data = await api("/events?limit=60");
  if (!data || data.__notfound || !data.events) return;
  state.hasMore = !!data.has_more;
  $("#load-more").hidden = !state.hasMore;
  if (initial && data.events.length === 0) {
    showEmptyState();
    return;
  }
  appendEvents(data.events);
}

async function loadEarlier() {
  if (state.oldestEventId === null) return;
  const btn = $("#load-more");
  btn.disabled = true;
  btn.textContent = "…";
  const data = await api(`/events?limit=40&before_id=${state.oldestEventId}`);
  if (data && data.events) {
    prependEvents(data.events);
    state.hasMore = !!data.has_more;
  }
  btn.hidden = !state.hasMore;
  btn.disabled = false;
  btn.textContent = "Earlier beats";
}

async function refreshNpcs() {
  const data = await api("/npcs");
  if (data && data.npcs) {
    for (const n of data.npcs) state.npcById[n.id] = n;
    renderNpcs(data.npcs);
  }
}

async function refreshResearch() {
  const data = await api("/research?limit=20");
  if (data && data.research) renderResearch(data.research);
}

/* ── turn actions ──────────────────────────────────────────────────────── */

/** Lock/unlock the action buttons + show the in-flight spinner. */
function setBusy(busy, btn) {
  state.busy = busy;
  const buttons = [$("#advance-btn"), $("#composer-send")];
  for (const b of buttons) {
    if (!b) continue;
    b.disabled = busy;
    const sp = $(".spinner", b);
    if (sp) sp.hidden = !busy || b !== btn;
  }
}

/** After any turn POST, fold in the returned state and pull fresh side-data. */
async function applyTurnResult(data) {
  if (data && data.state) renderState(data.state);
  // events/npcs/research may have changed; pull them in.
  await Promise.all([refreshEvents(), refreshNpcs(), refreshResearch()]);
}

async function advanceTurn() {
  if (state.busy) return;
  const btn = $("#advance-btn");
  setBusy(true, btn);
  showThinking("Twily is living the next moment");
  const data = await api("/turn", { method: "POST", body: "{}" });
  removeThinking();
  if (!data) {
    toast("The world didn't respond — try again in a moment.");
  } else if (data.__notfound) {
    toast("This world isn't ready yet.");
  } else {
    await applyTurnResult(data);
  }
  setBusy(false, btn);
}

async function visitorTurn(text) {
  if (state.busy) return;
  const input = text.trim();
  if (!input) return;

  const btn = $("#composer-send");
  setBusy(true, btn);

  // Optimistic: show Vis's line immediately as a local visitor event.
  clearEmptyState();
  const stream = $("#stream");
  const optimistic = eventEl({
    id: -1, turn: null, kind: "visitor", actor: "vis", content: input,
  });
  optimistic.style.opacity = ".8";
  stream.appendChild(optimistic);
  stream.scrollTop = stream.scrollHeight;

  showThinking("Twily and the others take you in");

  const data = await api("/visitor/turn", {
    method: "POST",
    body: JSON.stringify({ input }),
  });
  removeThinking();

  if (!data || data.__notfound) {
    toast("She didn't catch that — try again.");
    optimistic.style.opacity = "1"; // keep the optimistic line as a record
  } else {
    // The server will return the canonical visitor event among the events;
    // remove our optimistic stand-in so we don't double-show it.
    optimistic.remove();
    await applyTurnResult(data);
  }
  setBusy(false, btn);
}

/* ── polling loop (observe mode) ───────────────────────────────────────── */

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    if (state.busy) return;             // don't poll over an in-flight turn
    if (document.hidden) return;        // pause when tab is backgrounded
    await Promise.all([refreshState(), refreshEvents(), refreshResearch()]);
  }, 8000);
}
function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

/* ── mode switching ────────────────────────────────────────────────────── */

function setMode(mode) {
  state.mode = mode;
  const observing = mode === "observe";
  $("#tab-observe").classList.toggle("active", observing);
  $("#tab-visit").classList.toggle("active", !observing);
  $("#tab-observe").setAttribute("aria-selected", String(observing));
  $("#tab-visit").setAttribute("aria-selected", String(!observing));

  $("#composer").hidden = observing;
  $("#stream-title").textContent = observing
    ? "Her day, unfolding"
    : "You step into her world";

  if (!observing) {
    setTimeout(() => $("#composer-input").focus(), 50);
  }
}

/* ── wiring ────────────────────────────────────────────────────────────── */

function wire() {
  $("#advance-btn").addEventListener("click", advanceTurn);
  $("#load-more").addEventListener("click", loadEarlier);
  $("#tab-observe").addEventListener("click", () => setMode("observe"));
  $("#tab-visit").addEventListener("click", () => setMode("visit"));

  // composer: auto-grow + submit on Enter (Shift+Enter = newline)
  const ta = $("#composer-input");
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = ta.value;
    ta.value = "";
    ta.style.height = "auto";
    visitorTurn(text);
  });

  // refresh once when the tab regains focus (catch up on missed beats)
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.busy) {
      refreshState();
      refreshEvents();
    }
  });
}

/* ── go ────────────────────────────────────────────────────────────────── */

(async function main() {
  wire();
  await bootstrap();
  startPolling();
})();
