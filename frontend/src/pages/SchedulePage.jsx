import React, { useState, useEffect } from "react";
import { getHierarchyNodes, getScheduleRecommendation, getPoolingBenefit } from "../api";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function SchedulePage() {
  const [nodes, setNodes] = useState([]);
  const [selectedNode, setSelectedNode] = useState("India/NR/Delhi");
  const [selectedDate, setSelectedDate] = useState("");
  const [availableDates, setAvailableDates] = useState([]);
  const [normalRate, setNormalRate] = useState(4000);
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState(null);
  const [poolingData, setPoolingData] = useState(null);
  const [error, setError] = useState(null);

  // Load nodes list on mount
  useEffect(() => {
    getHierarchyNodes()
      .then((data) => {
        if (data && data.nodes) {
          setNodes(data.nodes);
        }
      })
      .catch((err) => console.error("Failed to load nodes:", err));
  }, []);

  // Fetch recommendation
  const fetchRecommendation = async (dateOverride = null) => {
    setLoading(true);
    setError(null);
    try {
      const targetDate = dateOverride !== null ? dateOverride : (selectedDate || null);
      const rec = await getScheduleRecommendation(selectedNode, targetDate, normalRate);
      setRecommendation(rec);
      if (rec && rec.date && !selectedDate) {
        setSelectedDate(rec.date);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch pooling benefit
  const fetchPooling = async () => {
    try {
      const pool = await getPoolingBenefit(normalRate);
      setPoolingData(pool);
    } catch (err) {
      console.error("Failed to fetch pooling benefit:", err);
    }
  };

  useEffect(() => {
    if (selectedNode) {
      fetchRecommendation();
    }
    fetchPooling();
  }, [selectedNode, selectedDate, normalRate]);

  return (
    <div className="page-container schedule-page">
      <div className="page-header">
        <h2>DSM Decision Engine & Schedule Recommender</h2>
        <p className="page-description">
          Under India's <strong>CERC Deviation Settlement Mechanism (DSM) 2024</strong>, over-scheduling
          and under-scheduling carry highly asymmetric financial penalties. TerraWatt optimizes the submitted
          schedule <em>Y MW</em> to protect states against high deviation costs.
        </p>
      </div>

      {error && <ErrorState message={error} onRetry={fetchRecommendation} />}

      {/* Control Panel */}
      <div className="schedule-controls-card card">
        <div className="controls-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
          <div className="control-group">
            <label>Select Hierarchy Node / State:</label>
            <select
              value={selectedNode}
              onChange={(e) => {
                setSelectedNode(e.target.value);
              }}
              className="select-input"
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px" }}
            >
              {nodes.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Target Date (YYYY-MM-DD):</label>
            <input
              type="date"
              value={selectedDate}
              min="2025-01-01"
              max="2026-09-25"
              onChange={(e) => {
                setSelectedDate(e.target.value);
              }}
              className="date-input"
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px" }}
            />
            <span style={{ fontSize: "0.75rem", color: "#888" }}>Historical range: 2025-01-01 to 2026-09-25</span>
          </div>

          <div className="control-group">
            <label>DSM Normal Rate (NR) ₹/MWh:</label>
            <input
              type="number"
              value={normalRate}
              onChange={(e) => {
                const val = Number(e.target.value);
                setNormalRate(val);
              }}
              step="500"
              min="1000"
              max="20000"
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px" }}
            />
            <span style={{ fontSize: "0.75rem", color: "#888" }}>Default: ₹4000/MWh (₹4/kWh)</span>
          </div>

          <div className="control-group" style={{ display: "flex", alignItems: "flex-end" }}>
            <button
              onClick={() => fetchRecommendation()}
              className="btn btn-primary"
              style={{ padding: "0.6rem 1.2rem", width: "100%", cursor: "pointer" }}
            >
              Recalculate Schedule
            </button>
          </div>
        </div>
      </div>

      {loading && <LoadingState message="Optimizing schedule under DSM penalty slabs..." />}

      {/* Headline Recommender Result */}
      {recommendation && !loading && (
        <div className="recommendation-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.5rem", marginTop: "1.5rem" }}>
          {/* Card 1: The Decision */}
          <div className="card decision-card" style={{ borderLeft: "5px solid #10b981", padding: "1.5rem" }}>
            <div style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "#6b7280", letterSpacing: "1px" }}>
              Recommended Dispatch Order
            </div>
            <div style={{ fontSize: "2.4rem", fontWeight: "bold", color: "#10b981", margin: "0.5rem 0" }}>
              {recommendation.optimal_schedule_mw.toLocaleString()} <span style={{ fontSize: "1.2rem" }}>MW</span>
            </div>
            <div style={{ fontSize: "0.95rem", color: "#374151" }}>
              Target Date: <strong>{recommendation.date}</strong> | State: <strong>{recommendation.node_id}</strong>
            </div>

            <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #e5e7eb", fontSize: "0.9rem" }}>
              <div>Point Forecast: <strong>{recommendation.point_forecast_mw.toLocaleString()} MW</strong></div>
              <div>
                Protective Bias:{" "}
                <span style={{ color: recommendation.schedule_bias_mw >= 0 ? "#10b981" : "#ef4444", fontWeight: "bold" }}>
                  {recommendation.schedule_bias_mw > 0 ? "+" : ""}
                  {recommendation.schedule_bias_mw} MW ({recommendation.schedule_bias_pct}%)
                </span>
              </div>
              <div>Bias Direction: <strong style={{ textTransform: "capitalize" }}>{recommendation.bias_direction.replace("_", " ")}</strong></div>
            </div>
          </div>

          {/* Card 2: Financial Protection */}
          <div className="card cost-card" style={{ borderLeft: "5px solid #3b82f6", padding: "1.5rem" }}>
            <div style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "#6b7280", letterSpacing: "1px" }}>
              Expected Financial Settlement
            </div>
            <div style={{ fontSize: "2rem", fontWeight: "bold", color: "#3b82f6", margin: "0.5rem 0" }}>
              ₹{Math.round(recommendation.expected_penalty_inr).toLocaleString()}
            </div>
            <div style={{ fontSize: "0.9rem", color: "#6b7280" }}>
              Expected net deviation penalty at optimal schedule
            </div>

            <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #e5e7eb", fontSize: "0.9rem" }}>
              <div>Penalty at Raw Forecast: <strong>₹{Math.round(recommendation.expected_penalty_at_forecast_inr).toLocaleString()}</strong></div>
              <div style={{ color: "#10b981", fontWeight: "bold" }}>
                Biasing Savings: ₹{Math.round(recommendation.penalty_savings_inr).toLocaleString()}
              </div>
              <div style={{ fontSize: "0.8rem", color: "#6b7280", marginTop: "0.3rem" }}>
                Uncertainty (σ): ±{recommendation.forecast_std_mw} MW
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Regional Pooling Benefit Table */}
      {poolingData && poolingData.summary && (
        <div className="card" style={{ marginTop: "2rem", padding: "1.5rem" }}>
          <h3>Regional Pooling Benefit (Error Cancellation Savings)</h3>
          <p style={{ color: "#6b7280", fontSize: "0.9rem", marginBottom: "1rem" }}>
            When states coordinate their schedules regionally, individual forecasting errors cancel out.
            Below is the verified DSM penalty savings compared to each state scheduling alone.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
            <div style={{ background: "#f3f4f6", padding: "1rem", borderRadius: "6px" }}>
              <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>Standalone Penalties (No Pooling)</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "bold" }}>
                ₹{Math.round(poolingData.summary.total_standalone_penalty_inr).toLocaleString()}
              </div>
            </div>
            <div style={{ background: "#f3f4f6", padding: "1rem", borderRadius: "6px" }}>
              <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>Pooled Hierarchy Penalties</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "bold", color: "#10b981" }}>
                ₹{Math.round(poolingData.summary.total_pooled_penalty_inr).toLocaleString()}
              </div>
            </div>
            <div style={{ background: "#ecfdf5", padding: "1rem", borderRadius: "6px", border: "1px solid #a7f3d0" }}>
              <div style={{ fontSize: "0.8rem", color: "#065f46" }}>Total Saved by Coordination</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "bold", color: "#059669" }}>
                ₹{Math.round(poolingData.summary.total_savings_inr).toLocaleString()}{" "}
                <span style={{ fontSize: "0.9rem" }}>({poolingData.summary.savings_pct.toFixed(1)}%)</span>
              </div>
            </div>
          </div>

          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
            <thead>
              <tr style={{ background: "#f9fafb", textAlign: "left", borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ padding: "0.6rem" }}>Region</th>
                <th style={{ padding: "0.6rem" }}>Standalone Penalty</th>
                <th style={{ padding: "0.6rem" }}>Pooled Penalty</th>
                <th style={{ padding: "0.6rem" }}>Savings (₹)</th>
                <th style={{ padding: "0.6rem" }}>Error Cancellation Ratio</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(poolingData.by_region || {}).map(([reg, info]) => (
                <tr key={reg} style={{ borderBottom: "1px solid #e5e7eb" }}>
                  <td style={{ padding: "0.6rem", fontWeight: "bold" }}>{reg}</td>
                  <td style={{ padding: "0.6rem" }}>₹{Math.round(info.standalone_penalty_inr).toLocaleString()}</td>
                  <td style={{ padding: "0.6rem" }}>₹{Math.round(info.pooled_penalty_inr).toLocaleString()}</td>
                  <td style={{ padding: "0.6rem", color: "#10b981", fontWeight: "bold" }}>
                    ₹{Math.round(info.savings_inr).toLocaleString()} ({info.savings_pct.toFixed(1)}%)
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    {(info.error_cancellation_ratio * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

