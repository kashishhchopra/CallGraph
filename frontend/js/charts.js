// Small dependency-free SVG/HTML charts.
import { esc, titleCase, palette } from "./util.js";

export function volumeBars(series) {
  const max = Math.max(1, ...series.map((d) => d.count));
  const bars = series.map((d) => `<div class="bar ${d.count ? "" : "zero"}" style="height:${d.count ? Math.max(3, (d.count / max) * 100) : 1.5}%" title="${esc(d.date)}: ${d.count}"></div>`).join("");
  const first = series[0]?.date, last = series[series.length - 1]?.date;
  return `<div class="vol">${bars}</div><div class="vol-axis"><span>${esc(first || "")}</span><span>${esc(last || "")}</span></div>`;
}

export function donut(items, cfg, size = 118) {
  const total = items.reduce((a, b) => a + b.value, 0);
  const r = size / 2 - 10, c = 2 * Math.PI * r, cx = size / 2;
  let offset = 0;
  const arcs = items.map((it, i) => {
    const len = (it.value / total) * c;
    const seg = `<circle cx="${cx}" cy="${cx}" r="${r}" fill="none" stroke="${palette(cfg, i)}" stroke-width="16" stroke-dasharray="${len} ${c - len}" stroke-dashoffset="${-offset}" transform="rotate(-90 ${cx} ${cx})"/>`;
    offset += len;
    return seg;
  }).join("");
  const legend = items.slice(0, 8).map((it, i) => `<div class="legend-row"><span class="sw" style="background:${palette(cfg, i)}"></span><span class="nm">${esc(titleCase(it.label))}</span><span class="v">${it.value}</span></div>`).join("");
  return `<div class="donut-wrap"><svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><circle cx="${cx}" cy="${cx}" r="${r}" fill="none" stroke="#f0f2ee" stroke-width="16"/>${arcs}</svg><div class="legend">${legend}</div></div>`;
}

export function hbars(rows, { format = "number", digits = 1 } = {}) {
  const vals = rows.map((r) => Number(r.value) || 0);
  const max = format === "percent" ? 1 : Math.max(1e-9, ...vals);
  const fmt = (v) => (format === "percent" ? `${(v * 100).toFixed(0)}%` : Number.isInteger(v) ? String(v) : v.toFixed(digits));
  return `<div class="hbars">${rows.map((r, i) => `<div class="hbar-row"><span class="lb" title="${esc(titleCase(r.label))}">${esc(titleCase(r.label))}</span><div class="hbar-track"><div class="hbar-fill" style="width:${Math.min(100, (vals[i] / max) * 100)}%"></div></div><span class="vl">${fmt(vals[i])}</span></div>`).join("")}</div>`;
}

export function stacked(rows, cfg) {
  const labels = [...new Set(rows.map((r) => r.label))];
  const series = [...new Set(rows.map((r) => r.series))];
  const colorOf = (s, i) => cfg.emotion_colors?.[s] || palette(cfg, i);
  const bodies = labels.map((l) => {
    const parts = rows.filter((r) => r.label === l);
    const total = parts.reduce((a, b) => a + b.value, 0) || 1;
    const segs = parts.map((p) => `<div class="seg" title="${esc(titleCase(p.series))}: ${p.value}" style="width:${(p.value / total) * 100}%;background:${colorOf(p.series, series.indexOf(p.series))}"></div>`).join("");
    return `<div class="hbar-row"><span class="lb">${esc(titleCase(l))}</span><div class="hbar-track">${segs}</div><span class="vl">${total}</span></div>`;
  }).join("");
  const legend = series.map((s, i) => `<span><i style="background:${colorOf(s, i)}"></i>${esc(titleCase(s))}</span>`).join("");
  return `<div class="hbars">${bodies}</div><div class="chart-legend">${legend}</div>`;
}

export function timeline(points, cfg, width = 520, height = 120) {
  if (!points.length) return '<div class="no-data">No customer emotion data</div>';
  const pad = 14, maxT = Math.max(...points.map((p) => p.t)) || 1;
  const dots = points.map((p) => {
    const x = pad + (p.t / maxT) * (width - 2 * pad), y = height - pad - p.intensity * (height - 2 * pad);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${cfg.emotion_colors?.[p.emotion] || "#98a29b"}"><title>${esc(titleCase(p.emotion))} · ${(p.intensity * 100).toFixed(0)}%</title></circle>`;
  }).join("");
  return `<svg class="timeline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#e6e8e4"/><line x1="${pad}" y1="${pad}" x2="${width - pad}" y2="${pad}" stroke="#eef0ec" stroke-dasharray="3 4"/>${dots}</svg>`;
}
