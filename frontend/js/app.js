// Shell (sidebar, top bar), hash router and health indicator.
import { api } from "./api.js";
import { icon } from "./icons.js";
import * as pages from "./pages.js";
import { $, $$, esc } from "./util.js";

const ROUTES = { overview: pages.overview, upload: pages.upload, calls: pages.calls, analytics: pages.analytics, graph: pages.graph, ask: pages.ask, help: pages.help, settings: pages.settings };
let cfg;

function sidebar() {
  const item = (n) => `<a class="nav-item" data-nav="${n.id}" href="#/${n.id}">${icon(n.icon)}<span>${esc(n.title)}</span>${n.badge ? `<span class="badge-ai">${esc(n.badge)}</span>` : ""}</a>`;
  return `<aside class="sidebar"><div class="brand"><div class="brand-mark">${icon("pulse")}</div><div class="brand-name">CallGraph</div></div>
    <div class="nav-label">WORKSPACE</div><nav class="nav">${cfg.nav.main.map(item).join("")}</nav><div class="sidebar-spacer"></div>
    <nav class="nav">${cfg.nav.footer.map(item).join("")}</nav>
    <div class="sidebar-foot"><div class="workspace"><div class="workspace-mark">${esc(cfg.workspace.initial)}</div><div class="workspace-text"><div class="workspace-name">${esc(cfg.workspace.name)}</div><div class="workspace-plan">${esc(cfg.workspace.plan)}</div></div><button class="icon-btn-ghost" data-go="settings" title="Workspace settings">${icon("dots", 'width="18" height="18"')}</button></div></div></aside>`;
}

function topbar() {
  return `<header class="topbar"><div class="crumbs">CallGraph<span class="sep">/</span><b id="crumb"></b></div><div class="topbar-right">
    <div class="status" id="status"><span class="dot"></span><span id="status-text">API offline</span></div>
    <button class="icon-btn" title="Notifications">${icon("bell")}</button><div class="avatar">${esc(cfg.user.initials)}</div></div></header>`;
}

async function refreshHealth() {
  const el = $("#status");
  try { const h = await api.health(); el.classList.toggle("online", !!h.falkordb); $("#status-text").textContent = h.falkordb ? "API online" : "API degraded"; }
  catch (_) { el.classList.remove("online"); $("#status-text").textContent = "API offline"; }
}

async function route() {
  const [, page = "overview", arg] = location.hash.split("/");
  const id = ROUTES[page] ? page : "overview";
  $$(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.nav === id));
  const all = [...cfg.nav.main, ...cfg.nav.footer];
  $("#crumb").textContent = arg && id === "calls" ? decodeURIComponent(arg) : all.find((n) => n.id === id)?.title || "";
  const view = $("#view");
  view.scrollTop = 0; window.scrollTo(0, 0);
  const ctx = { cfg };
  try {
    if (id === "calls" && arg) await pages.callDetail(view, ctx, decodeURIComponent(arg));
    else await ROUTES[id](view, ctx);
  } catch (e) { view.innerHTML = `<div class="content"><div class="card empty-block"><div class="empty-title">Something went wrong</div><div class="empty-text">${esc(e.message)}</div></div></div>`; }
}

document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]"), open = e.target.closest("[data-open]"), row = e.target.closest("tr[data-call]");
  if (go) location.hash = `#/${go.dataset.go}`;
  else if (open) location.hash = `#/calls/${encodeURIComponent(open.dataset.open)}`;
  else if (row) location.hash = `#/calls/${encodeURIComponent(row.dataset.call)}`;
});

(async function boot() {
  try { cfg = await api.config(); }
  catch (e) { document.body.innerHTML = `<div style="padding:60px;font-family:sans-serif"><h2>API offline</h2><p>${esc(e.message)}</p></div>`; return; }
  document.title = "CallGraph";
  $("#app").innerHTML = `${sidebar()}<div class="main">${topbar()}<main id="view"></main></div>`;
  window.addEventListener("hashchange", route);
  refreshHealth(); setInterval(refreshHealth, cfg.health_poll_s * 1000);
  route();
})();
