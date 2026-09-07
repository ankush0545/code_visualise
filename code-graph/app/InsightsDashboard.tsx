"use client";

import React from "react";

interface ModelMetric {
  name: string;
  value: string;
  subtext: string;
  color: string;
}

const METRICS: ModelMetric[] = [
  { name: "Test Accuracy", value: "98.57%", subtext: "Across 16 architectural classes", color: "#38bdf8" },
  { name: "Macro F1 Score", value: "95.79%", subtext: "Unweighted average across all classes", color: "#2dd4bf" },
  { name: "Minority F1", value: "96.84%", subtext: "Weighted inverse class frequency", color: "#a78bfa" },
  { name: "Test Loss", value: "0.0291", subtext: "Cross-entropy with class re-weighting", color: "#f59e0b" },
];

const TAXONOMY = [
  { label: "backend", precision: "0.992", recall: "0.972", f1: "0.982", support: 8637, desc: "Server, business logic, runtime" },
  { label: "vendored_dependency", precision: "0.997", recall: "1.000", f1: "0.998", support: 7046, desc: "Third-party libraries & vendor code" },
  { label: "testing", precision: "1.000", recall: "1.000", f1: "1.000", support: 6092, desc: "Unit, integration, and e2e test suites" },
  { label: "configuration", precision: "0.989", recall: "0.995", f1: "0.992", support: 5798, desc: "Configs, env vars, build parameters" },
  { label: "utility", precision: "0.976", recall: "0.947", f1: "0.961", support: 2154, desc: "General-purpose helper functions" },
  { label: "ml_data", precision: "0.968", recall: "0.996", f1: "0.982", support: 2012, desc: "Data pipelines & model training scripts" },
  { label: "api", precision: "0.964", recall: "0.998", f1: "0.981", support: 1190, desc: "Endpoints, controllers, and routing" },
  { label: "mobile", precision: "0.993", recall: "0.916", f1: "0.953", support: 956, desc: "Mobile client views & platform logic" },
  { label: "example", precision: "0.994", recall: "1.000", f1: "0.997", support: 862, desc: "Samples, demos, and reference setups" },
  { label: "frontend", precision: "0.944", recall: "0.991", f1: "0.967", support: 632, desc: "UI components, client state & templates" },
  { label: "build_tooling", precision: "0.988", recall: "0.982", f1: "0.985", support: 570, desc: "Webpack, Vite, compilers & scripts" },
  { label: "infra_devops", precision: "0.960", recall: "0.992", f1: "0.976", support: 528, desc: "Docker, Kubernetes, CI/CD workflows" },
  { label: "entry_point", precision: "0.819", recall: "0.974", f1: "0.890", support: 195, desc: "main.py, index.js, CLI runners" },
  { label: "database", precision: "0.701", recall: "0.896", f1: "0.787", support: 183, desc: "ORM models, schemas & migrations" },
  { label: "auth", precision: "0.859", recall: "0.948", f1: "0.902", support: 174, desc: "OAuth, JWT, session verification" },
  { label: "documentation", precision: "0.966", recall: "0.982", f1: "0.974", support: 114, desc: "Markdown docs, guides & API references" },
];

export default function InsightsDashboard() {
  return (
    <div style={{ padding: "28px 36px", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Top Banner */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
          <span style={{ fontSize: "12px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#38bdf8", fontWeight: 600 }}>
            Machine Learning Evaluation & Architecture Metrics
          </span>
          <span className="badge-cyan" style={{ padding: "2px 8px", borderRadius: "12px", fontSize: "11px", fontWeight: 500 }}>
            GraphSAGE + CodeBERT
          </span>
        </div>
        <h1 style={{ fontSize: "26px", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.02em" }}>
          Model Performance & Graph Taxonomy
        </h1>
        <p style={{ color: "#94a3b8", fontSize: "14px", marginTop: "4px", maxWidth: "800px" }}>
          Statistical evaluation of CodeGraph AI on cross-repository source files. Models combine Tree-sitter AST structural features with GNN message-passing and CodeBERT semantic embeddings.
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "18px", marginBottom: "32px" }}>
        {METRICS.map((m) => (
          <div key={m.name} className="glass-card" style={{ padding: "20px 24px" }}>
            <div style={{ fontSize: "12px", color: "#94a3b8", fontWeight: 500, marginBottom: "6px" }}>{m.name}</div>
            <div style={{ fontSize: "32px", fontWeight: 800, color: m.color, letterSpacing: "-0.03em" }}>{m.value}</div>
            <div style={{ fontSize: "11.5px", color: "#64748b", marginTop: "6px" }}>{m.subtext}</div>
          </div>
        ))}
      </div>

      {/* Two Column Section: Pipeline Overview & Architecture Breakdown */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: "24px", marginBottom: "32px" }}>
        {/* Pipeline Architecture */}
        <div className="glass-card" style={{ padding: "24px" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 600, color: "#f1f5f9", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ color: "#38bdf8" }}>⚡</span> Unified Multi-Stage ML Pipeline
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px", fontSize: "13px" }}>
            <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.06)", borderRadius: "8px", padding: "12px 16px" }}>
              <div style={{ fontWeight: 600, color: "#38bdf8", marginBottom: "3px" }}>1. Tree-sitter AST & Call Graph Extraction</div>
              <div style={{ color: "#94a3b8" }}>Parses Python, TypeScript, JavaScript, and Java codebases to extract classes, functions, variable scopes, and cross-file import statements.</div>
            </div>
            <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.06)", borderRadius: "8px", padding: "12px 16px" }}>
              <div style={{ fontWeight: 600, color: "#2dd4bf", marginBottom: "3px" }}>2. GraphSAGE GNN Edge Prediction</div>
              <div style={{ color: "#94a3b8" }}>Uses inductive neighborhood sampling and edge existence heads to predict latent dataflow & function call relationships across candidate file pairs.</div>
            </div>
            <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.06)", borderRadius: "8px", padding: "12px 16px" }}>
              <div style={{ fontWeight: 600, color: "#a78bfa", marginBottom: "3px" }}>3. CodeBERT Fine-Tuned Semantic Classifier</div>
              <div style={{ color: "#94a3b8" }}>A 125M-parameter Transformer model fine-tuned on code syntax to assign files into 16 distinct architectural layers with calibrated confidence scores.</div>
            </div>
            <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.06)", borderRadius: "8px", padding: "12px 16px" }}>
              <div style={{ fontWeight: 600, color: "#f59e0b", marginBottom: "3px" }}>4. Interactive D3.js Force & Layer Visualizer</div>
              <div style={{ color: "#94a3b8" }}>Aggregates file dependencies into real-time modular clusters and tiered architectural flows for interactive exploration.</div>
            </div>
          </div>
        </div>

        {/* Visual Benchmark Charts from Assets */}
        <div className="glass-card" style={{ padding: "24px", display: "flex", flexDirection: "column" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 600, color: "#f1f5f9", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ color: "#a78bfa" }}>📈</span> Empirical Test Metrics
          </h3>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "16px", justifyContent: "center" }}>
            <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "8px", overflow: "hidden", border: "1px solid rgba(255, 255, 255, 0.06)" }}>
              <img
                src="/assets/per_class_metrics.png"
                alt="Per-class metrics"
                style={{ width: "100%", maxHeight: "200px", objectFit: "contain", display: "block" }}
                onError={(e) => { (e.currentTarget as HTMLElement).style.display = "none"; }}
              />
            </div>
            <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "8px", overflow: "hidden", border: "1px solid rgba(255, 255, 255, 0.06)" }}>
              <img
                src="/assets/test_results.png"
                alt="Test results summary"
                style={{ width: "100%", maxHeight: "150px", objectFit: "contain", display: "block" }}
                onError={(e) => { (e.currentTarget as HTMLElement).style.display = "none"; }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Taxonomy Table */}
      <div className="glass-card" style={{ padding: "24px", overflowX: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
          <div>
            <h3 style={{ fontSize: "17px", fontWeight: 600, color: "#f1f5f9" }}>16-Class Architectural Classification Taxonomy</h3>
            <p style={{ fontSize: "13px", color: "#94a3b8", marginTop: "2px" }}>Precision, Recall, F1 Score, and Support on the held-out evaluation dataset.</p>
          </div>
          <span className="badge-purple" style={{ padding: "4px 10px", borderRadius: "14px", fontSize: "12px", fontWeight: 500 }}>
            37,429 Evaluated Samples
          </span>
        </div>

        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.1)", color: "#94a3b8" }}>
              <th style={{ padding: "10px 14px", fontWeight: 500 }}>Class</th>
              <th style={{ padding: "10px 14px", fontWeight: 500 }}>Functional Scope</th>
              <th style={{ padding: "10px 14px", fontWeight: 500, textAlign: "right" }}>Precision</th>
              <th style={{ padding: "10px 14px", fontWeight: 500, textAlign: "right" }}>Recall</th>
              <th style={{ padding: "10px 14px", fontWeight: 500, textAlign: "right" }}>F1 Score</th>
              <th style={{ padding: "10px 14px", fontWeight: 500, textAlign: "right" }}>Samples</th>
            </tr>
          </thead>
          <tbody>
            {TAXONOMY.map((item, idx) => (
              <tr
                key={item.label}
                style={{
                  borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                  backgroundColor: idx % 2 === 0 ? "rgba(255, 255, 255, 0.01)" : "transparent"
                }}
              >
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", color: "#38bdf8", fontWeight: 600 }}>
                  {item.label}
                </td>
                <td style={{ padding: "10px 14px", color: "#94a3b8" }}>{item.desc}</td>
                <td style={{ padding: "10px 14px", textAlign: "right", color: "#cbd5e1" }}>{item.precision}</td>
                <td style={{ padding: "10px 14px", textAlign: "right", color: "#cbd5e1" }}>{item.recall}</td>
                <td style={{ padding: "10px 14px", textAlign: "right", fontWeight: 600, color: Number(item.f1) > 0.95 ? "#4ade80" : "#f59e0b" }}>
                  {item.f1}
                </td>
                <td style={{ padding: "10px 14px", textAlign: "right", color: "#64748b", fontFamily: "var(--font-mono)" }}>
                  {item.support.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
