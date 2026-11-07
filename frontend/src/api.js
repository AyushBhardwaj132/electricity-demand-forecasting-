/**
 * api.js
 * Centralized API client for TerraWatt platform.
 * Communicates with FastAPI backend for forecasts, hierarchy snapshots,
 * objective evaluation summaries, and telemetry ingestion.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export function getApiBaseUrl() {
  return API_BASE;
}

export function getWebSocketUrl() {
  const wsProtocol = API_BASE.startsWith("https") ? "wss" : "ws";
  const host = API_BASE.replace(/^https?:\/\//, "");
  return `${wsProtocol}://${host}/ws/telemetry`;
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
  return res.json();
}

export async function getHierarchyNodes() {
  const res = await fetch(`${API_BASE}/forecast/hierarchy/nodes`);
  if (!res.ok) throw new Error(`Failed to load hierarchy nodes: HTTP ${res.status}`);
  return res.json();
}

export async function getHierarchySnapshot(date = null) {
  const url = date ? `${API_BASE}/forecast/hierarchy?date=${encodeURIComponent(date)}` : `${API_BASE}/forecast/hierarchy`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load hierarchy snapshot: HTTP ${res.status}`);
  return res.json();
}

export async function getNodeForecast(nodeId, horizon = null) {
  let url = `${API_BASE}/forecast/${encodeURIComponent(nodeId)}`;
  if (horizon) {
    url += `?horizon=${encodeURIComponent(horizon)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load forecast for ${nodeId}: HTTP ${res.status}`);
  return res.json();
}

export async function getEvaluationSummary() {
  const res = await fetch(`${API_BASE}/evaluation/summary`);
  if (!res.ok) throw new Error(`Failed to load evaluation summary: HTTP ${res.status}`);
  return res.json();
}

export async function ingestActual(nodeId, date, actualValue) {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      node_id: nodeId,
      date,
      actual_value: Number(actualValue),
    }),
  });
  if (!res.ok) throw new Error(`Failed to ingest observation: HTTP ${res.status}`);
  return res.json();
}

export async function getIngestHistory() {
  const res = await fetch(`${API_BASE}/ingest/history`);
  if (!res.ok) throw new Error(`Failed to load ingest history: HTTP ${res.status}`);
  return res.json();
}

export async function getScheduleRecommendation(nodeId, date = null, normalRate = 4000.0) {
  const payload = { node_id: nodeId, normal_rate: Number(normalRate) };
  if (date) payload.date = date;

  const res = await fetch(`${API_BASE}/schedule/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to get schedule recommendation: HTTP ${res.status}`);
  return res.json();
}

export async function getPoolingBenefit(normalRate = 4000.0) {
  const url = normalRate ? `${API_BASE}/schedule/pooling-benefit?normal_rate=${encodeURIComponent(normalRate)}` : `${API_BASE}/schedule/pooling-benefit`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to get pooling benefit: HTTP ${res.status}`);
  return res.json();
}
