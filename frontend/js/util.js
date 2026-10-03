// Formatting helpers (no domain values live here).
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const titleCase = (s) => { const t = String(s ?? "").replace(/[_-]+/g, " ").trim(); return t ? t[0].toUpperCase() + t.slice(1) : ""; };
const SEC_PER_MIN = 60;
export function duration(s) {
  if (s == null) return "—";
  const m = Math.floor(s / SEC_PER_MIN), sec = Math.round(s % SEC_PER_MIN);
  return `${m}:${String(sec).padStart(2, "0")}`;
}
export const num = (v, digits = 0) => (v == null ? "—" : Number(v).toFixed(digits));
export const pct = (v, digits = 0) => (v == null ? "—" : `${(Number(v) * 100).toFixed(digits)}%`);
export function greeting(date = new Date()) {
  const noon = 12, evening = 18, h = date.getHours();
  return h < noon ? "Good morning" : h < evening ? "Good afternoon" : "Good evening";
}
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export function palette(cfg, i) { const p = cfg.chart_palette; return p[i % p.length]; }
export function emotionPill(cfg, emotion) {
  if (!emotion) return '<span style="color:var(--faint)">—</span>';
  const c = cfg.emotion_colors?.[emotion] || "#98a29b";
  return `<span class="pill"><i style="background:${c}"></i>${esc(titleCase(emotion))}</span>`;
}
