// Page renderers. Everything data-like (labels, colours, questions, nav) comes from the API / public config.
import { api } from "./api.js";
import { icon } from "./icons.js";
import { donut, hbars, stacked, timeline, volumeBars } from "./charts.js";
import { $, $$, duration, emotionPill, esc, greeting, num, pct, titleCase } from "./util.js";

const head = (title, sub, right = "") => `<div class="page-head"><div><h1>${title}</h1><p class="subtitle">${sub}</p></div>${right}</div>`;
const uploadBtn = `<button class="btn-dark" data-go="upload">${icon("upload")}Upload call</button>`;
const pulse = (n) => `<span class="pill"><i style="background:var(--green-500)"></i>${n}</span>`;

/* ------------------------------------------------------------------ overview */
export async function overview(root, ctx) {
  const { cfg } = ctx;
  let days = cfg.default_range_days;
  root.innerHTML = `<div class="content">
    <div class="page-head"><div><div class="eyebrow">${icon("sparkles")}Intelligence workspace</div>
    <h1>${greeting()}, ${esc(cfg.user.name)}</h1><p class="subtitle">Here’s what’s happening across your conversations.</p></div>${uploadBtn}</div>
    <div class="kpis" id="kpis"></div>
    <div class="grid-2"><div class="card card-pad" id="volume"></div><div class="card card-pad" id="intents"></div></div>
    <div class="card" id="recent"></div></div>`;
  const render = async () => {
    const d = await api.overview(days).catch(() => null);
    const k = d?.kpis, n = k?.calls_processed || 0;
    const tile = (label, ic, value, note) => `<div class="card kpi"><div class="kpi-top"><span class="kpi-label">${label}</span><span class="kpi-tile">${icon(ic)}</span></div><div class="kpi-value">${value}</div><div class="kpi-note">${note}</div></div>`;
    $("#kpis", root).innerHTML = [
      tile("Calls processed", "filecall", n ? n : "—", n ? `In the last ${days} days` : "Waiting for the first processed call"),
      tile("Average frustration", "pulse", n && k.avg_frustration != null ? num(k.avg_frustration) : "—", "Calculated from customer turns"),
      tile("Compliance score", "shield", n && k.avg_compliance != null ? `${num(k.avg_compliance, 1)}<small> / ${k.compliance_scale}</small>` : "—", "Based on configured rubric checks"),
      tile("Average handle time", "clock", n ? duration(k.avg_handle_time_s) : "—", "Duration across processed calls"),
    ].join("");
    const ranges = cfg.overview_ranges_days.map((r) => `<option value="${r}" ${r === days ? "selected" : ""}>Last ${r} days</option>`).join("");
    $("#volume", root).innerHTML = `<div class="card-head"><div><div class="card-title">Call volume</div><div class="card-sub">Processed conversations over time</div></div><select class="select" id="range">${ranges}</select></div>` +
      (n ? volumeBars(d.volume) : `<div class="empty-chart"><div class="msg">${icon("chart")}Volume appears after calls are processed</div></div>`);
    $("#intents", root).innerHTML = `<div class="card-head"><div><div class="card-title">Intent distribution</div><div class="card-sub">Top reasons customers call</div></div><button class="icon-btn" data-go="analytics" title="Open analytics">${icon("dots")}</button></div>` +
      (d?.intents?.length ? donut(d.intents, cfg) : `<div class="donut-wrap" style="justify-content:center;margin-top:49px"><svg width="104" height="104" viewBox="0 0 104 104"><circle cx="52" cy="52" r="44" fill="none" stroke="#f0f2ee" stroke-width="15"/></svg><div><div style="font-weight:600">No data yet</div><div class="card-sub">Intent labels load from the API</div></div></div>`);
    $("#recent", root).innerHTML = `<div class="card-pad" style="display:flex;justify-content:space-between;align-items:flex-start"><div><div class="card-title">Recent calls</div><div class="card-sub">Latest pipeline activity and outcomes</div></div><button class="link-btn" data-go="calls">View all</button></div>` +
      (d?.recent?.length ? callsTable(d.recent, cfg, false) : `<div class="empty-block"><div class="empty-icon">${icon("headset")}</div><div class="empty-title">No calls have been processed</div><div class="empty-text">Upload a customer-service recording to generate a transcript, insights, and graph relationships.</div><button class="btn-light" data-go="upload">Upload your first call</button></div>`);
    $("#range", root)?.addEventListener("change", (e) => { days = Number(e.target.value); render(); });
  };
  await render();
}

/* --------------------------------------------------------------------- calls */
function callsTable(items, cfg, withPager = true) {
  const rows = items.map((c) => `<tr data-call="${esc(c.call_id)}"><td><div class="call-id">${esc(c.call_id)}</div><div class="call-sub">${esc(c.issue || c.summary || "")}</div></td>
    <td>${c.intent ? pulse(esc(titleCase(c.intent))) : "—"}</td><td>${emotionPill(cfg, c.end_emotion)}</td>
    <td>${c.compliance_score != null ? `${num(c.compliance_score, 0)} / ${cfg.compliance_scale}` : "—"}</td><td>${duration(c.duration_s)}</td></tr>`).join("");
  return `<table class="table"><colgroup><col style="width:33.4%"><col style="width:16.6%"><col style="width:16.6%"><col style="width:16.7%"><col style="width:16.7%"></colgroup><thead><tr><th>Call</th><th>Intent</th><th>Emotion</th><th>Compliance</th><th>Duration</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export async function calls(root, ctx) {
  const { cfg } = ctx;
  const state = { page: 1, q: "", intent: "", outcome: "" };
  root.innerHTML = `<div class="content">${head("Calls", "Review transcripts, signals, and outcomes from every conversation.", uploadBtn)}
    <div class="card"><div class="toolbar"><div class="search">${icon("search")}<input id="q" placeholder="Search calls" /></div>
    <select class="select" id="intent" style="width:104px"><option value="">All intents</option>${cfg.intents.map((i) => `<option value="${esc(i)}">${esc(titleCase(i))}</option>`).join("")}</select>
    <select class="select" id="outcome" style="width:121px"><option value="">All outcomes</option>${cfg.outcomes.map((i) => `<option value="${esc(i)}">${esc(titleCase(i))}</option>`).join("")}</select></div>
    <div id="list"></div></div></div>`;
  const load = async () => {
    const d = await api.calls({ page: state.page, page_size: cfg.page_size, q: state.q, intent: state.intent, outcome: state.outcome });
    const empty = `<div class="empty-block"><div class="empty-icon">${icon("filecall")}</div><div class="empty-title">${state.q || state.intent || state.outcome ? "No calls match your filters" : "Your call library is empty"}</div><div class="empty-text">Processed calls will appear here with their detected intent, emotion, and outcome.</div></div>`;
    const pages = Math.max(1, Math.ceil(d.total / d.page_size));
    $("#list", root).innerHTML = d.items.length ? callsTable(d.items, cfg) + `<div class="pager"><span>${d.total} calls</span><span><button class="btn-light" id="prev" ${d.page <= 1 ? "disabled" : ""}>Previous</button> &nbsp; Page ${d.page} of ${pages} &nbsp; <button class="btn-light" id="next" ${d.page >= pages ? "disabled" : ""}>Next</button></span></div>` : callsTable([], cfg) + empty;
    $("#prev", root)?.addEventListener("click", () => { state.page--; load(); });
    $("#next", root)?.addEventListener("click", () => { state.page++; load(); });
  };
  let timer;
  $("#q", root).addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => { state.q = e.target.value; state.page = 1; load(); }, cfg.search_debounce_ms); });
  $("#intent", root).addEventListener("change", (e) => { state.intent = e.target.value; state.page = 1; load(); });
  $("#outcome", root).addEventListener("change", (e) => { state.outcome = e.target.value; state.page = 1; load(); });
  await load();
}

/* --------------------------------------------------------------- call detail */
export async function callDetail(root, ctx, id) {
  const { cfg } = ctx;
  root.innerHTML = `<div class="content"><div class="empty-block"><div class="spinner" style="margin:auto"></div></div></div>`;
  let d;
  try { d = await api.call(id); } catch (e) { root.innerHTML = `<div class="content"><div class="card empty-block"><div class="empty-title">Call not found</div><div class="empty-text">${esc(e.message)}</div><button class="btn-light" data-go="calls">Back to calls</button></div></div>`; return; }
  const t = d.transcript, m = d.modules, emo = m.emotion || { turns: [] }, intent = m.intent || {}, ext = m.extraction || {}, comp = m.compliance || {}, dyn = m.dynamics || {}, sum = m.summary || {};
  const byTurn = Object.fromEntries((emo.turns || []).map((e) => [e.turn_index, e]));
  const turns = t.turns.map((u, i) => `<div class="turn" data-s="${u.start}" data-e="${u.end}" style="border-left-color:${cfg.speaker_colors[u.role] || "#ccc"}"><div class="who">${esc(titleCase(u.role))}<small>${duration(u.start)}</small>${byTurn[i] ? emotionPill(cfg, byTurn[i].emotion) : ""}</div><p>${esc(u.text)}</p></div>`).join("");
  const checks = (comp.items || []).map((c) => `<div class="check"><span class="${c.passed ? "tick" : "fail"}">${icon(c.passed ? "check" : "x")}</span><div><div>${esc(titleCase(c.rubric_item))}</div>${c.evidence ? (c.evidence === cfg.compliance_na_marker ? `<div class="ev na">${esc(titleCase(c.evidence))}</div>` : `<div class="ev">“${esc(c.evidence)}”</div>`) : ""}</div></div>`).join("");
  const feats = Object.entries(dyn.features || {}).map(([k, v]) => `<span>${esc(cfg.dynamics_labels[k] || titleCase(k))}</span><span>${num(v, v % 1 ? 2 : 0)}</span>`).join("");
  root.innerHTML = `<div class="content"><button class="back" data-go="calls">${icon("back", 'width="16" height="16"')}Back to calls</button>
    <div class="page-head"><div><div class="eyebrow">${esc(t.dataset || "call")}</div><h1>${esc(t.call_id)}</h1><p class="subtitle">${esc(ext.issue_type ? titleCase(ext.issue_type) : "")}${ext.outcome ? ` · ${esc(titleCase(ext.outcome))}` : ""} · ${duration(t.duration_s)}${t.non_english ? ` · language: ${esc(t.language)}` : ""}</p></div></div>
    <div class="kpis"><div class="card kpi"><div class="kpi-top"><span class="kpi-label">Intent</span></div><div class="kpi-value" style="font-size:20px">${esc(titleCase(intent.top_intent || "—"))}</div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Frustration</span></div><div class="kpi-value">${num(emo.frustration_score)}</div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Compliance</span></div><div class="kpi-value">${comp.score != null ? `${num(comp.score, 0)}<small> / ${cfg.compliance_scale}</small>` : "—"}</div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Emotion shift</span></div><div class="kpi-value" style="font-size:20px">${esc(titleCase(emo.shift || "—"))}</div></div></div>
    <div class="detail-grid"><div class="card card-pad"><div class="card-title" style="margin-bottom:12px">Transcript</div><audio controls preload="none" src="${api.audioUrl(id)}"></audio><div class="transcript" id="tr">${turns}</div></div>
    <div><div class="card card-pad"><div class="card-title">Summary</div><p style="line-height:1.55;margin:10px 0 0">${esc(sum.summary || "—")}</p>
    ${ext.agent_actions?.length ? `<div class="card-sub" style="margin-top:12px">Agent actions</div><p style="margin:4px 0 0">${ext.agent_actions.map((a) => esc(a)).join(" · ")}</p>` : ""}</div>
    <div class="card card-pad section-gap"><div class="card-title">Customer emotion</div>${timeline((emo.turns || []).map((e) => ({ t: t.turns[e.turn_index].start, intensity: e.intensity, emotion: e.emotion })), cfg)}</div>
    <div class="card card-pad section-gap"><div class="card-title" style="margin-bottom:6px">Compliance checks</div>${checks || '<div class="no-data" style="padding:18px 0">Not available</div>'}</div>
    <div class="card card-pad section-gap"><div class="card-title" style="margin-bottom:10px">Conversation dynamics</div><div class="kv">${feats || "—"}</div>${dyn.flags?.length ? `<div style="margin-top:10px">${dyn.flags.map((f) => pulse(esc(titleCase(f)))).join(" ")}</div>` : ""}</div></div></div></div>`;
  const audio = $("audio", root), turnEls = $$(".turn", root);
  turnEls.forEach((el) => el.addEventListener("click", () => { audio.currentTime = parseFloat(el.dataset.s); audio.play(); }));
  audio.addEventListener("timeupdate", () => turnEls.forEach((el) => {
    const on = audio.currentTime >= el.dataset.s && audio.currentTime <= el.dataset.e;
    el.classList.toggle("on", on);
    if (on) el.scrollIntoView({ block: "nearest" });
  }));
}

/* ------------------------------------------------------------------- upload */
export async function upload(root, ctx) {
  const { cfg } = ctx;
  const stepCards = cfg.pipeline_steps.map((s, i) => `<div class="card step"><div class="no">${String(i + 1).padStart(2, "0")}</div><div class="tt">${esc(s.title)}</div><div class="ds">${esc(s.text)}</div></div>`).join("");
  root.innerHTML = `<div class="content narrow">${head("Upload &amp; process", "Turn a raw call into structured, queryable intelligence.")}
    <div class="card"><div class="dropzone" id="dz"><div class="empty-icon">${icon("upload")}</div><div class="dz-title" id="dz-title">Drop a call recording here</div><div class="dz-sub" id="dz-sub">or choose an audio file from your computer</div><button class="btn-light" id="browse">Browse files</button><input type="file" id="file" accept="${cfg.audio_extensions.join(",")}" hidden /></div>
    <div id="progress"></div><div class="upload-foot"><span class="note">${icon("shield")}PII is redacted before storage</span><button class="btn-dark" id="go" disabled>Process call ${icon("play", 'style="width:15px;height:15px"')}</button></div></div>
    <div class="steps">${stepCards}</div></div>`;
  let file = null;
  const pick = (f) => { file = f; $("#dz-title", root).textContent = f.name; $("#dz-sub", root).textContent = `${(f.size / cfg.bytes_per_mb).toFixed(1)} MB`; $("#go", root).disabled = false; };
  $("#browse", root).addEventListener("click", () => $("#file", root).click());
  $("#file", root).addEventListener("change", (e) => e.target.files[0] && pick(e.target.files[0]));
  const dz = $("#dz", root);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]));
  $("#go", root).addEventListener("click", async () => {
    const btn = $("#go", root); btn.disabled = true;
    const box = $("#progress", root);
    try {
      const { call_id } = await api.upload(file);
      const stages = cfg.pipeline_stages;
      const draw = (st, done) => {
        box.innerHTML = `<div class="progress">${stages.map((s) => { const v = st[s] || "queued"; const cls = v === "done" ? `<span class="tick">${icon("check")}</span>` : v === "running" ? '<span class="spinner"></span>' : String(v).startsWith("failed") ? '<span class="fail"></span>' : v === "skipped" ? '<span class="idle"></span>' : '<span class="idle"></span>'; return `<div class="progress-row">${cls}<span>${esc(titleCase(s))}</span><span class="st">${esc(titleCase(String(v).split(":")[0]))}</span></div>`; }).join("")}${done ? `<div class="progress-row"><button class="link-btn" data-open="${esc(call_id)}">Open call →</button></div>` : ""}</div>`;
      };
      draw({}, false);
      for (;;) {
        await new Promise((r) => setTimeout(r, cfg.poll_interval_s * 1000));
        const s = await api.status(call_id);
        draw(s.status, s.finished);
        if (s.finished) break;
      }
    } catch (e) { box.innerHTML = `<div class="progress"><div class="progress-row"><span class="fail"></span><span>${esc(e.message)}</span></div></div>`; btn.disabled = false; }
  });
}

/* ---------------------------------------------------------------- analytics */
export async function analytics(root, ctx) {
  const { cfg } = ctx;
  root.innerHTML = `<div class="content">${head("Analytics", "Understand what drives customer outcomes across your operation.", `<select class="select" disabled><option>All processed calls</option></select>`)}
    <div class="grid-eq" id="cards">${cfg.analytics_cards.map((c, i) => `<div class="card card-pad chart-card" id="card-${i}"><div class="card-title">${esc(c.title)}</div><div class="card-sub">${esc(c.subtitle)}</div><div class="body"></div></div>`).join("")}</div></div>`;
  await Promise.all(cfg.analytics_cards.map(async (c, i) => {
    const body = $(`#card-${i} .body`, root);
    try {
      const { rows } = await api.analytics(c.metric);
      body.innerHTML = rows.length ? (c.kind === "stacked" ? stacked(rows, cfg) : hbars(rows, { format: c.format })) : `<div class="empty-chart"><div class="msg">${icon("chart")}Awaiting processed data</div></div>`;
    } catch (e) { body.innerHTML = `<div class="no-data">${esc(e.message)}</div>`; }
  }));
}

/* -------------------------------------------------------------------- graph */
export async function graph(root, ctx) {
  const { cfg } = ctx;
  root.innerHTML = `<div class="content">${head("Knowledge graph", "Explore how calls, issues, actions, and outcomes connect.", `<select class="select">${cfg.graph_views.map((v) => `<option>${v.title}</option>`).join("")}</select>`)}
    <div class="card graph-card"><div class="graph-bar"><div class="search" style="flex:1">${icon("search")}<input id="find" placeholder="Find a node" /></div><div class="graph-legend">${["Call", "Issue", "Action", "Outcome"].map((t) => `<span><i style="background:${cfg.node_colors[t]}"></i>${t}</span>`).join("")}</div></div>
    <div class="graph-canvas" id="net"><div class="graph-empty" id="gempty"><span class="blob" style="left:24.5%;top:136px;width:34px;height:34px;background:#f0b9a2"></span><span class="blob" style="left:71%;top:172px;width:18px;height:18px;background:#c4b9ea"></span><span class="blob" style="left:79%;top:355px;width:24px;height:24px;background:#b3c0ec"></span><div class="box"><div class="gi">${icon("graph")}</div><div class="gt">The graph is ready for connections</div><div class="gx">Nodes and relationships appear after the pipeline writes its first call.</div></div></div></div></div></div>`;
  const data = await api.subgraph({}).catch(() => ({ nodes: [], edges: [] }));
  if (!data.nodes.length) return;
  $("#gempty", root).style.display = "none";
  const mk = (n) => ({ id: n.id, label: n.label.length > 26 ? `${n.label.slice(0, 25)}…` : n.label, title: `${n.type}: ${n.label}`, color: { background: cfg.node_colors[n.type], border: cfg.node_colors[n.type] }, font: { color: "#232523", size: 13 }, shape: "dot", size: n.type === "Issue" ? 18 : n.type === "Call" ? 8 : 13, _type: n.type, _label: n.label });
  const nodes = new vis.DataSet(data.nodes.map(mk));
  const edges = new vis.DataSet(data.edges.map((e, i) => ({ id: `e${i}`, from: e.source, to: e.target, value: e.weight, arrows: "to", color: { color: "#c5ccc6" } })));
  const net = new vis.Network($("#net", root), { nodes, edges }, { physics: { stabilization: { iterations: 150 }, barnesHut: { gravitationalConstant: -4500, springLength: 130 } }, interaction: { hover: true }, edges: { smooth: { type: "continuous" }, scaling: { min: 1, max: 6 } } });
  net.on("click", async (p) => {  // clicking an issue reveals the calls behind it
    const n = p.nodes[0] && nodes.get(p.nodes[0]);
    if (!n || n._type !== "Issue") return;
    const more = await api.subgraph({ issue: n._label, calls: true }).catch(() => null);
    more?.nodes.filter((x) => x.type === "Call" && !nodes.get(x.id)).forEach((x) => { nodes.add(mk(x)); });
    more?.edges.filter((e) => String(e.source).startsWith("call:")).forEach((e) => { const id = `${e.source}>${e.target}`; if (!edges.get(id)) edges.add({ id, from: e.source, to: e.target, arrows: "to", color: { color: "#dfe3de" } }); });
  });
  $("#find", root).addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const hit = q && nodes.get().find((n) => n._label.toLowerCase().includes(q));
    if (hit) { net.selectNodes([hit.id]); net.focus(hit.id, { scale: 1.1, animation: true }); }
  });
}

/* ---------------------------------------------------------------------- ask */
export async function ask(root, ctx) {
  const { cfg } = ctx;
  let strategy = localStorage.getItem("strategy") || cfg.default_strategy;
  if (!cfg.strategies.includes(strategy)) strategy = cfg.default_strategy;
  root.innerHTML = `<div class="content narrow">${head("Ask CallGraph", "Query your conversations in plain English, with cited answers.", `<div class="seg-control" id="seg">${cfg.strategies.map((s) => `<button data-s="${s}" class="${s === strategy ? "on" : ""}">${titleCase(s)}</button>`).join("")}</div>`)}
    <div class="card chat"><div class="chat-body" id="body"><div class="chat-empty"><div class="bot-tile">${icon("bot")}</div><h2>What would you like to know?</h2><p>I’ll search the call graph and transcripts, then cite the calls behind the answer.</p>
    <div class="examples">${cfg.example_questions.map((q) => `<button class="example" data-q="${esc(q)}"><span>${esc(q)}</span>${icon("send")}</button>`).join("")}</div></div></div>
    <div class="composer"><textarea id="q" rows="1" placeholder="Ask about call drivers, resolution paths, or compliance gaps…"></textarea><button class="send" id="send" disabled>${icon("send")}</button></div></div></div>`;
  const body = $("#body", root), input = $("#q", root), send = $("#send", root);
  $$("#seg button", root).forEach((b) => b.addEventListener("click", () => { strategy = b.dataset.s; localStorage.setItem("strategy", strategy); $$("#seg button", root).forEach((x) => x.classList.toggle("on", x === b)); }));
  input.addEventListener("input", () => { send.disabled = !input.value.trim(); input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, cfg.composer_max_px)}px`; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!send.disabled) go(input.value.trim()); } });
  send.addEventListener("click", () => go(input.value.trim()));
  $$(".example", root).forEach((b) => b.addEventListener("click", () => go(b.dataset.q)));
  async function go(q) {
    if (!q) return;
    $(".chat-empty", body)?.remove();
    input.value = ""; send.disabled = true; input.style.height = "auto";
    body.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${esc(q)}</div></div>`);
    body.insertAdjacentHTML("beforeend", `<div class="msg bot" id="pending"><div class="bubble"><span class="typing"><i></i><i></i><i></i></span> <span class="muted" id="wait">${esc(cfg.ask_progress_text)}… 0s</span></div></div>`);
    body.scrollTop = body.scrollHeight;
    const started = Date.now(), ctl = new AbortController();
    const tick = setInterval(() => { const w = $("#wait", root); if (w) w.textContent = `${cfg.ask_progress_text}… ${Math.round((Date.now() - started) / 1000)}s`; }, 1000);
    const timer = setTimeout(() => ctl.abort(), cfg.ask_timeout_s * 1000);
    try {
      const r = await api.ask(q, strategy, ctl.signal);
      const chips = (r.sources || []).map((s) => `<span class="chip" data-open="${esc(s)}">${esc(s)}</span>`).join("");
      $("#pending", root).outerHTML = `<div class="msg bot"><div class="bubble">${esc(r.answer || r.error || "No answer")}<div class="meta"><span>${esc(titleCase(r.strategy))} · ${r.latency_s.toFixed(1)}s</span>${chips ? `<span>·</span>${chips}` : ""}</div>${r.cypher ? `<details style="margin-top:8px"><summary>Show generated Cypher</summary><div class="cypher">${esc(r.cypher)}</div></details>` : ""}</div></div>`;
    } catch (e) {
      const msg = e.name === "AbortError" ? `No answer after ${cfg.ask_timeout_s}s. The language model may be busy or still loading; try again.` : `The question could not be answered: ${e.message}`;
      $("#pending", root).outerHTML = `<div class="msg bot"><div class="bubble">${esc(msg)}</div></div>`;
    } finally { clearInterval(tick); clearTimeout(timer); }
    body.scrollTop = body.scrollHeight;
  }
}

/* ---------------------------------------------------------- help / settings */
export async function help(root, ctx) {
  const { cfg } = ctx;
  root.innerHTML = `<div class="content narrow">${head("Help center", "How CallGraph works and where to find things.")}
    <div class="grid-eq">${cfg.help_topics.map((t) => `<div class="card card-pad"><div class="card-title">${esc(t.title)}</div><p class="card-sub" style="line-height:1.55;font-size:13px">${esc(t.text)}</p></div>`).join("")}</div>
    <div class="card card-pad section-gap"><div class="card-title">API reference</div><p class="card-sub">Interactive OpenAPI documentation is served by the API.</p><a class="btn-light" href="/docs" target="_blank">Open API docs</a></div></div>`;
}

export async function settings(root, ctx) {
  const { cfg } = ctx;
  const h = await api.health().catch(() => null);
  const row = (k, v) => `<span>${esc(k)}</span><span>${v}</span>`;
  const dot = (ok) => `<span class="pill"><i style="background:${ok ? "var(--online)" : "var(--offline)"}"></i>${ok ? "Healthy" : "Unavailable"}</span>`;
  root.innerHTML = `<div class="content narrow">${head("Settings", "Workspace details and system health.")}
    <div class="grid-eq"><div class="card card-pad"><div class="card-title" style="margin-bottom:12px">Workspace</div><div class="kv">${row("Name", esc(cfg.workspace.name))}${row("Plan", esc(cfg.workspace.plan))}${row("User", esc(cfg.user.name))}${row("Default strategy", esc(titleCase(localStorage.getItem("strategy") || cfg.default_strategy)))}</div></div>
    <div class="card card-pad"><div class="card-title" style="margin-bottom:12px">System health</div><div class="kv">${h ? row("Graph database", dot(h.falkordb)) + row("Language model", dot(h.llm)) + Object.entries(h.models).map(([k, v]) => row(titleCase(k), dot(v))).join("") : row("API", dot(false))}</div></div></div></div>`;
}
