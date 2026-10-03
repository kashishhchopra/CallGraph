// Thin client for the CallGraph API (same origin).
async function request(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error?.message || msg; } catch (_) { /* non-JSON error */ }
    throw new Error(msg);
  }
  return res.json();
}
const qs = (params) => {
  const p = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") p.set(k, v); });
  const s = p.toString();
  return s ? `?${s}` : "";
};
export const api = {
  config: () => request("/config/public"),
  health: () => request("/health"),
  overview: (days) => request(`/overview${qs({ days })}`),
  calls: (params) => request(`/calls${qs(params)}`),
  call: (id) => request(`/calls/${encodeURIComponent(id)}`),
  status: (id) => request(`/calls/${encodeURIComponent(id)}/status`),
  analytics: (metric) => request(`/analytics/${metric}`),
  subgraph: (params) => request(`/graph/subgraph${qs(params)}`),
  ask: (question, strategy, signal) => request("/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question, strategy }), signal }),
  upload: (file) => { const f = new FormData(); f.append("file", file); return request("/calls", { method: "POST", body: f }); },
  audioUrl: (id) => `/calls/${encodeURIComponent(id)}/audio`,
};
