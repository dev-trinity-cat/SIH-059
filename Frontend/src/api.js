/** Same-origin `/api` in production; Vite proxies `/api` in development. */
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

async function json(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () => json("/api/health"),
  modelInfo: () => json("/api/model-info"),
  availableDates: () => json("/api/available-dates"),
  predict: (target_date) =>
    json("/api/predict", {
      method: "POST",
      body: JSON.stringify({ target_date }),
    }),
};

/** Collapse a 66×57 SIC field (0–1) into a coarse 0–100 grid for the SVG map. */
export function predictionToDisplayGrid(prediction, rows = 6, cols = 8) {
  if (!Array.isArray(prediction) || !prediction.length || !prediction[0]?.length) {
    return null;
  }
  const H = prediction.length;
  const W = prediction[0].length;
  const out = [];
  for (let r = 0; r < rows; r += 1) {
    const row = [];
    for (let c = 0; c < cols; c += 1) {
      const y = Math.min(H - 1, Math.floor(((r + 0.5) * H) / rows));
      const x = Math.min(W - 1, Math.floor(((c + 0.5) * W) / cols));
      const v = prediction[y][x];
      row.push(v == null || Number.isNaN(Number(v)) ? 0 : Math.round(Number(v) * 100));
    }
    out.push(row);
  }
  return out;
}
