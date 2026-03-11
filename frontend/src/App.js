import React, { useState } from "react";
import axios from "axios";

const API = "http://localhost:5000/api";

function App() {
  const [scenario,    setScenario]    = useState("real-time");
  const [recommendation, setRec]      = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);

  const getRecommendation = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API}/recommend`, {
        scenario,
        hardware: "gpu",
        time_constraint_ms: scenario === "real-time" ? 150 : null,
      });
      setRec(res.data);
    } catch (e) {
      setRec({ recommended_method: "N/A", reason: "Backend not connected yet." });
    }
    setLoading(false);
  };

  return (
    <div style={{ fontFamily: "Arial", maxWidth: 900,
                  margin: "40px auto", padding: "0 20px" }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ color: "#2c3e50", fontSize: 32 }}>
          🎨 StyleSense — NST Comparison Platform
        </h1>
        <p style={{ color: "#7f8c8d" }}>
          Team T59 | GLA University | B.Tech CSE AI/ML
        </p>
      </div>

      {/* Recommendation Engine */}
      <div style={{ background: "#f8f9fa", borderRadius: 12,
                    padding: 24, marginBottom: 30,
                    border: "1px solid #dee2e6" }}>
        <h2 style={{ color: "#2c3e50", marginTop: 0 }}>
          🤖 Get Method Recommendation
        </h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center",
                      flexWrap: "wrap" }}>
          <select
            value={scenario}
            onChange={e => setScenario(e.target.value)}
            style={{ padding: "10px 16px", borderRadius: 8,
                     border: "1px solid #ced4da", fontSize: 15 }}>
            <option value="real-time">Real-Time Mobile Filter</option>
            <option value="quality-first">High-Quality Artwork</option>
            <option value="batch">Batch Processing</option>
          </select>
          <button
            onClick={getRecommendation}
            disabled={loading}
            style={{ padding: "10px 24px", background: "#3498db",
                     color: "#fff", border: "none", borderRadius: 8,
                     fontSize: 15, cursor: "pointer" }}>
            {loading ? "Loading..." : "Get Recommendation →"}
          </button>
        </div>

        {recommendation && (
          <div style={{ marginTop: 20, padding: 16,
                        background: "#d4edda", borderRadius: 8,
                        border: "1px solid #c3e6cb" }}>
            <strong>Recommended Method:</strong>{" "}
            <span style={{ color: "#155724", fontWeight: "bold",
                           textTransform: "uppercase" }}>
              {recommendation.recommended_method}
            </span>
            <p style={{ margin: "8px 0 0", color: "#155724" }}>
              {recommendation.reason}
            </p>
          </div>
        )}
      </div>

      {/* Upload Placeholder (Week 6) */}
      <div style={{ background: "#fff3cd", borderRadius: 12,
                    padding: 24, border: "1px solid #ffc107",
                    textAlign: "center" }}>
        <h2 style={{ color: "#856404", marginTop: 0 }}>
          📁 Image Upload + Compare (Coming Week 6)
        </h2>
        <p style={{ color: "#856404" }}>
          Upload content + style image → Run both NST methods →
          View side-by-side results + metrics
        </p>
        <div style={{ display: "flex", gap: 16, justifyContent: "center",
                      flexWrap: "wrap", marginTop: 16 }}>
          {["Upload Content Image", "Upload Style Image",
            "▶ Compare Both", "📊 View Benchmarks"].map(label => (
            <button key={label}
              style={{ padding: "10px 20px", borderRadius: 8,
                       border: "2px dashed #ffc107", background: "#fff",
                       color: "#856404", cursor: "not-allowed",
                       fontSize: 14 }}
              disabled>
              {label}
            </button>
          ))}
        </div>
      </div>

      <p style={{ textAlign: "center", color: "#adb5bd",
                  marginTop: 40, fontSize: 13 }}>
        StyleSense v0.5 — Week 5 | Backend integration coming Week 6
      </p>
    </div>
  );
}

export default App;
