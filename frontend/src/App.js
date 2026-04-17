import React, { useState, useEffect } from "react";
import axios from "axios";

const API = "http://localhost:5000/api";

// ─── Small reusable components ───────────────────────────────────────────────

function StatusBadge({ status }) {
  const colors = {
    online:   { dot: "#4ade80", text: "#4ade80", label: "Backend Online" },
    offline:  { dot: "#f87171", text: "#f87171", label: "Backend Offline" },
    checking: { dot: "#fbbf24", text: "#fbbf24", label: "Checking..." },
  };
  const c = colors[status] || colors.checking;
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:6,
      background:"#1e1e2e", border:"1px solid #2a2a3a",
      borderRadius:20, padding:"4px 12px", fontSize:13
    }}>
      <span style={{
        width:8, height:8, borderRadius:"50%",
        background: c.dot,
        boxShadow: status==="online" ? `0 0 6px ${c.dot}` : "none",
        display:"inline-block"
      }}/>
      <span style={{ color: c.text }}>{c.label}</span>
    </div>
  );
}

// ─── PAGES ───────────────────────────────────────────────────────────────────

function HomePage({ navigate }) {
  return (
    <div style={{ maxWidth:800, margin:"0 auto", textAlign:"center", paddingTop:60 }}>
      <div style={{ fontSize:64, marginBottom:16 }}>🎨</div>
      <h1 style={{
        fontSize:42, fontWeight:800, marginBottom:12,
        background:"linear-gradient(135deg,#a78bfa,#60a5fa)",
        WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent"
      }}>
        StyleSense
      </h1>
      <p style={{ color:"#9999b3", fontSize:18, marginBottom:8 }}>
        Optimization-Based vs Fast Neural Style Transfer
      </p>
      <p style={{ color:"#666", fontSize:14, marginBottom:48 }}>
        Team T59 · GLA University · B.Tech CSE AI/ML
      </p>

      {/* Feature Cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16, marginBottom:48 }}>
        {[
          { icon:"⚡", title:"522x Speedup", desc:"Fast NST vs Optimization NST" },
          { icon:"🖼️", title:"Side-by-Side", desc:"Compare output quality visually" },
          { icon:"🤖", title:"Smart Recommend", desc:"Best method for your use-case" },
        ].map(f => (
          <div key={f.title} style={{
            background:"#16161e", border:"1px solid #2a2a3a",
            borderRadius:12, padding:24
          }}>
            <div style={{ fontSize:32, marginBottom:8 }}>{f.icon}</div>
            <h3 style={{ fontSize:16, marginBottom:6, color:"#e8e8f0" }}>{f.title}</h3>
            <p style={{ fontSize:13, color:"#9999b3" }}>{f.desc}</p>
          </div>
        ))}
      </div>

      <button
        onClick={() => navigate("compare")}
        style={{
          padding:"14px 40px", fontSize:16, fontWeight:700,
          background:"linear-gradient(135deg,#a78bfa,#60a5fa)",
          border:"none", borderRadius:10, color:"#fff",
          cursor:"pointer", marginRight:12
        }}>
        🚀 Start Comparing
      </button>
      <button
        onClick={() => navigate("recommend")}
        style={{
          padding:"14px 40px", fontSize:16, fontWeight:600,
          background:"#1e1e2e", border:"1px solid #2a2a3a",
          borderRadius:10, color:"#e8e8f0", cursor:"pointer"
        }}>
        🤖 Get Recommendation
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function ComparePage({ backendStatus }) {
  const [contentFile, setContentFile] = useState(null);
  const [styleFile,   setStyleFile]   = useState(null);
  const [method,      setMethod]      = useState("fast");
  const [result,      setResult]      = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);

  const handleRun = async () => {
    if (!contentFile || !styleFile) {
      setError("Please upload both content and style images!");
      return;
    }
    setLoading(true); setError(null); setResult(null);
    const formData = new FormData();
    formData.append("content_image", contentFile);
    formData.append("style_image",   styleFile);
    formData.append("method",        method);
    try {
      const res = await axios.post(`${API}/stylize`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.error || "Backend error — is Flask running?");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth:900, margin:"0 auto" }}>
      <h2 style={{ fontSize:28, fontWeight:700, marginBottom:8 }}>
        🖼️ NST Comparison
      </h2>
      <p style={{ color:"#9999b3", marginBottom:32 }}>
        Upload images → choose method → run NST → compare results
      </p>

      {backendStatus === "offline" && (
        <div style={{
          background:"#2d1b1b", border:"1px solid #f87171",
          borderRadius:10, padding:16, marginBottom:24, color:"#fca5a5"
        }}>
          ⚠️ Backend offline — start Flask: <code style={{
            background:"#1a1a2e", padding:"2px 8px", borderRadius:4
          }}>cd ~/T59_fresh && python backend/app.py</code>
        </div>
      )}

      {/* Upload Row */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:24 }}>
        {[
          { label:"Content Image", key:"content", setter:setContentFile, file:contentFile },
          { label:"Style Image",   key:"style",   setter:setStyleFile,   file:styleFile   },
        ].map(({ label, key, setter, file }) => (
          <label key={key} style={{
            display:"block", background:"#16161e",
            border: file ? "2px solid #a78bfa" : "2px dashed #2a2a3a",
            borderRadius:12, padding:32, textAlign:"center",
            cursor:"pointer", transition:"all 0.2s"
          }}>
            <input type="file" accept="image/*" style={{ display:"none" }}
              onChange={e => setter(e.target.files[0])} />
            <div style={{ fontSize:36, marginBottom:8 }}>
              {file ? "✅" : "📁"}
            </div>
            <div style={{ fontSize:14, color: file ? "#a78bfa" : "#9999b3" }}>
              {file ? file.name : `Upload ${label}`}
            </div>
          </label>
        ))}
      </div>

      {/* Method Select */}
      <div style={{
        display:"flex", gap:12, alignItems:"center", marginBottom:24,
        background:"#16161e", border:"1px solid #2a2a3a",
        borderRadius:12, padding:16
      }}>
        <span style={{ color:"#9999b3", fontSize:14 }}>Method:</span>
        {["fast","optimization"].map(m => (
          <button key={m}
            onClick={() => setMethod(m)}
            style={{
              padding:"8px 20px", borderRadius:8, border:"1px solid",
              borderColor: method===m ? "#a78bfa" : "#2a2a3a",
              background: method===m ? "#2d1f4e" : "#1e1e2e",
              color: method===m ? "#a78bfa" : "#9999b3",
              cursor:"pointer", fontWeight: method===m ? 600 : 400,
              fontSize:14
            }}>
            {m === "fast" ? "⚡ Fast NST (~160ms)" : "🎨 Optimization NST (~3s)"}
          </button>
        ))}
      </div>

      {/* Run Button */}
      <button
        onClick={handleRun}
        disabled={loading || backendStatus==="offline"}
        style={{
          width:"100%", padding:"14px", fontSize:16, fontWeight:700,
          background: loading ? "#2a2a3a" : "linear-gradient(135deg,#a78bfa,#60a5fa)",
          border:"none", borderRadius:10, color:"#fff",
          cursor: loading ? "not-allowed" : "pointer", marginBottom:24
        }}>
        {loading ? "⏳ Running NST..." : "▶ Run Style Transfer"}
      </button>

      {/* Error */}
      {error && (
        <div style={{
          background:"#2d1b1b", border:"1px solid #f87171",
          borderRadius:10, padding:16, marginBottom:24, color:"#fca5a5"
        }}>{error}</div>
      )}

      {/* Result */}
      {result && (
        <div style={{
          background:"#16161e", border:"1px solid #2a2a3a",
          borderRadius:12, padding:24
        }}>
          <h3 style={{ marginBottom:16, color:"#4ade80" }}>
            ✅ Result — {result.method?.toUpperCase()} NST
          </h3>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12, marginBottom:16 }}>
            {[
              { label:"⏱ Time",      val: result.time_seconds ? `${result.time_seconds.toFixed(3)}s` : "—" },
              { label:"📐 Size",      val: result.output_size  || "—" },
              { label:"⚡ Speedup",   val: result.speedup      ? `${result.speedup}x`  : "—" },
            ].map(stat => (
              <div key={stat.label} style={{
                background:"#1e1e2e", borderRadius:10, padding:16, textAlign:"center"
              }}>
                <div style={{ fontSize:24, fontWeight:800, color:"#a78bfa" }}>{stat.val}</div>
                <div style={{ fontSize:12, color:"#9999b3", marginTop:4 }}>{stat.label}</div>
              </div>
            ))}
          </div>
          {result.output_image_base64 && (
            <img
              src={`data:image/jpeg;base64,${result.output_image_base64}`}
              alt="NST Output"
              style={{ width:"100%", borderRadius:10, border:"1px solid #2a2a3a" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function BenchmarkPage() {
  const benchmarks = [
    { method:"Fast NST",         time:0.160,  quality:78, useCase:"Real-time / Mobile" },
    { method:"Optimization NST", time:3.2,    quality:94, useCase:"High-quality Artwork" },
  ];
  return (
    <div style={{ maxWidth:800, margin:"0 auto" }}>
      <h2 style={{ fontSize:28, fontWeight:700, marginBottom:8 }}>📊 Benchmarks</h2>
      <p style={{ color:"#9999b3", marginBottom:32 }}>RTX 3060 · 512×512 images</p>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:32 }}>
        {[
          { label:"Speedup Factor", val:"522x",  icon:"⚡", color:"#a78bfa" },
          { label:"Fast NST Time",  val:"160ms", icon:"🚀", color:"#60a5fa" },
          { label:"Opt NST Time",   val:"3.2s",  icon:"🎨", color:"#4ade80" },
          { label:"Extractor Acc.", val:"87.47%",icon:"🧠", color:"#fbbf24" },
        ].map(s => (
          <div key={s.label} style={{
            background:"#16161e", border:"1px solid #2a2a3a",
            borderRadius:12, padding:24, textAlign:"center"
          }}>
            <div style={{ fontSize:36, marginBottom:8 }}>{s.icon}</div>
            <div style={{ fontSize:32, fontWeight:800, color:s.color }}>{s.val}</div>
            <div style={{ fontSize:13, color:"#9999b3", marginTop:4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Comparison Table */}
      <div style={{
        background:"#16161e", border:"1px solid #2a2a3a",
        borderRadius:12, overflow:"hidden"
      }}>
        <table style={{ width:"100%", borderCollapse:"collapse" }}>
          <thead>
            <tr style={{ background:"#1e1e2e", borderBottom:"1px solid #2a2a3a" }}>
              {["Method","Avg Time","Quality Score","Best For"].map(h => (
                <th key={h} style={{
                  padding:"14px 20px", textAlign:"left",
                  fontSize:13, color:"#9999b3", fontWeight:600
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((b, i) => (
              <tr key={b.method} style={{
                borderBottom: i < benchmarks.length-1 ? "1px solid #2a2a3a" : "none"
              }}>
                <td style={{ padding:"16px 20px", fontWeight:600 }}>{b.method}</td>
                <td style={{ padding:"16px 20px", color:"#60a5fa" }}>{b.time}s</td>
                <td style={{ padding:"16px 20px" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                    <div style={{
                      width:80, height:8, background:"#2a2a3a", borderRadius:4, overflow:"hidden"
                    }}>
                      <div style={{
                        width:`${b.quality}%`, height:"100%",
                        background:"linear-gradient(90deg,#a78bfa,#60a5fa)", borderRadius:4
                      }}/>
                    </div>
                    <span style={{ fontSize:13, color:"#9999b3" }}>{b.quality}%</span>
                  </div>
                </td>
                <td style={{ padding:"16px 20px", color:"#9999b3", fontSize:13 }}>{b.useCase}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function RecommendPage({ backendStatus }) {
  const [scenario,       setScenario]  = useState("real-time");
  const [recommendation, setRec]       = useState(null);
  const [loading,        setLoading]   = useState(false);

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
      setRec({ recommended_method: "N/A", reason: "Backend not connected." });
    }
    setLoading(false);
  };

  const scenarios = [
    { val:"real-time",     label:"⚡ Real-Time Mobile Filter" },
    { val:"quality-first", label:"🎨 High-Quality Artwork"    },
    { val:"batch",         label:"📦 Batch Processing"        },
  ];

  return (
    <div style={{ maxWidth:700, margin:"0 auto" }}>
      <h2 style={{ fontSize:28, fontWeight:700, marginBottom:8 }}>🤖 Method Recommender</h2>
      <p style={{ color:"#9999b3", marginBottom:32 }}>
        Tell us your use-case — we'll suggest the best NST method
      </p>

      <div style={{
        background:"#16161e", border:"1px solid #2a2a3a",
        borderRadius:12, padding:24, marginBottom:24
      }}>
        <div style={{ display:"flex", flexDirection:"column", gap:12, marginBottom:20 }}>
          {scenarios.map(s => (
            <label key={s.val} style={{
              display:"flex", alignItems:"center", gap:12,
              padding:"14px 16px", borderRadius:10, cursor:"pointer",
              border:"1px solid",
              borderColor: scenario===s.val ? "#a78bfa" : "#2a2a3a",
              background: scenario===s.val ? "#2d1f4e" : "#1e1e2e",
              transition:"all 0.2s"
            }}>
              <input type="radio" value={s.val}
                checked={scenario===s.val}
                onChange={() => setScenario(s.val)}
                style={{ accentColor:"#a78bfa" }} />
              <span style={{ color: scenario===s.val ? "#e8e8f0" : "#9999b3", fontWeight: scenario===s.val ? 600 : 400 }}>
                {s.label}
              </span>
            </label>
          ))}
        </div>

        <button
          onClick={getRecommendation}
          disabled={loading || backendStatus==="offline"}
          style={{
            width:"100%", padding:"12px", fontSize:15, fontWeight:700,
            background:"linear-gradient(135deg,#a78bfa,#60a5fa)",
            border:"none", borderRadius:10, color:"#fff",
            cursor: loading ? "not-allowed" : "pointer"
          }}>
          {loading ? "Analyzing..." : "Get Recommendation →"}
        </button>
      </div>

      {recommendation && (
        <div style={{
          background:"#1a2e1a", border:"1px solid #4ade80",
          borderRadius:12, padding:24
        }}>
          <div style={{ fontSize:13, color:"#4ade80", marginBottom:8, fontWeight:600 }}>
            RECOMMENDED METHOD
          </div>
          <div style={{ fontSize:28, fontWeight:800, color:"#e8e8f0", marginBottom:12, textTransform:"uppercase" }}>
            {recommendation.recommended_method}
          </div>
          <p style={{ color:"#86efac", lineHeight:1.6 }}>{recommendation.reason}</p>
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────

export default function App() {
  const [page,          setPage]   = useState("home");
  const [backendStatus, setStatus] = useState("checking");

  useEffect(() => {
    fetch("http://localhost:5000/api/health")
      .then(r => r.json())
      .then(d => setStatus(d.status === "ok" ? "online" : "offline"))
      .catch(() => setStatus("offline"));
  }, []);

  const NAV = [
    { key:"home",      label:"🏠 Home"      },
    { key:"compare",   label:"🖼️ Compare"   },
    { key:"benchmark", label:"📊 Benchmark" },
    { key:"recommend", label:"🤖 Recommend" },
  ];

  return (
    <div style={{ fontFamily:"'Segoe UI',sans-serif", minHeight:"100vh",
                  background:"#0f0f13", color:"#e8e8f0" }}>
      {/* Navbar */}
      <nav style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"14px 32px", background:"#16161e",
        borderBottom:"1px solid #2a2a3a",
        position:"sticky", top:0, zIndex:100
      }}>
        <div style={{
          fontSize:20, fontWeight:800, cursor:"pointer",
          background:"linear-gradient(135deg,#a78bfa,#60a5fa)",
          WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent"
        }} onClick={() => setPage("home")}>
          🎨 StyleSense
        </div>
        <div style={{ display:"flex", gap:4 }}>
          {NAV.map(n => (
            <button key={n.key} onClick={() => setPage(n.key)}
              style={{
                padding:"8px 16px", borderRadius:8,
                border:"none", cursor:"pointer", fontSize:14,
                background: page===n.key ? "#2a2a3a" : "none",
                color: page===n.key ? "#a78bfa" : "#9999b3",
                fontWeight: page===n.key ? 600 : 400,
                transition:"all 0.2s"
              }}>
              {n.label}
            </button>
          ))}
        </div>
        <StatusBadge status={backendStatus} />
      </nav>

      {/* Main */}
      <main style={{ padding:"40px 32px", maxWidth:1200, margin:"0 auto" }}>
        {page === "home"      && <HomePage      navigate={setPage} />}
        {page === "compare"   && <ComparePage   backendStatus={backendStatus} />}
        {page === "benchmark" && <BenchmarkPage />}
        {page === "recommend" && <RecommendPage backendStatus={backendStatus} />}
      </main>
    </div>
  );
}
