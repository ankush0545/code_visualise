"use client";

import React, { useState } from "react";
import FileFlowExplorer from "@/app/FileFlowExplorer";
import ArchitectureExplorer from "@/app/FileDataFlowMap";
import InsightsDashboard from "@/app/InsightsDashboard";

type ViewMode = "architecture" | "dependency" | "insights";

export default function Home() {
  const [activeTab, setActiveTab] = useState<ViewMode>("architecture");

  return (
    <div className="min-h-screen flex flex-col bg-[#070a0f] text-slate-100">
      {/* Top Glass Navigation Bar */}
      <header className="sticky top-0 z-50 glass-nav border-b border-white/[0.08] px-5 py-3.5 flex items-center justify-between flex-wrap gap-4">
        {/* Brand & Status */}
        <div className="flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 via-sky-500 to-indigo-600 p-[1px] shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-[#0c1118] rounded-[7px] flex items-center justify-center">
              <svg className="w-5 h-5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="6" cy="6" r="3" />
                <circle cx="18" cy="6" r="3" />
                <circle cx="6" cy="18" r="3" />
                <circle cx="18" cy="18" r="3" />
                <path d="M6 9v6M18 9v6M9 6h6M9 18h6M8.5 8.5l7 7M15.5 8.5l-7 7" />
              </svg>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white font-sans">
                CodeGraph <span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-semibold">
                v2.0
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5">
                <span className="pulsing-dot" />
                <span className="text-slate-300 font-medium">Pipeline Active</span>
              </span>
              <span>•</span>
              <span>Tree-sitter + GraphSAGE + CodeBERT</span>
            </div>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center bg-[#0d141f] p-1 rounded-xl border border-white/[0.08]">
          <button
            onClick={() => setActiveTab("architecture")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === "architecture"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🌐</span>
            <span>Architecture Map</span>
          </button>
          <button
            onClick={() => setActiveTab("dependency")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === "dependency"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🕸️</span>
            <span>Dependency Graph</span>
          </button>
          <button
            onClick={() => setActiveTab("insights")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === "insights"
                ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>📊</span>
            <span>Model Insights</span>
          </button>
        </div>

        {/* Right Action: GitHub & Preset Badges */}
        <div className="flex items-center gap-2.5">
          <div className="hidden md:flex items-center gap-2 text-[11px] font-mono bg-white/[0.03] border border-white/[0.08] px-2.5 py-1.5 rounded-lg text-slate-300">
            <span className="text-slate-500">Repo:</span>
            <span className="text-cyan-300 font-medium truncate max-w-[150px]">code_visualise</span>
          </div>
          <a
            href="https://github.com/ankush0545/code_visualise"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.1] text-white transition-all shadow-sm"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </header>

      {/* Main Content Area based on Tab */}
      <main className="flex-1 flex flex-col p-3 md:p-5">
        {activeTab === "architecture" && (
          <div className="flex-1 flex flex-col">
            <ArchitectureExplorer />
          </div>
        )}

        {activeTab === "dependency" && (
          <div className="flex-1 flex flex-col">
            <FileFlowExplorer />
          </div>
        )}

        {activeTab === "insights" && (
          <div className="flex-1">
            <InsightsDashboard />
          </div>
        )}
      </main>

      {/* Subtle Footer */}
      <footer className="px-5 py-3 border-t border-white/[0.06] text-xs text-slate-500 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <span>CodeGraph AI © {new Date().getFullYear()}</span>
          <span>•</span>
          <span>Dual Architecture: AST Parsing + Graph Neural Network</span>
        </div>
        <div className="flex items-center gap-4 text-slate-400">
          <span className="font-mono text-[11px]">Accuracy: 98.57%</span>
          <span className="font-mono text-[11px]">Macro F1: 95.79%</span>
        </div>
      </footer>
    </div>
  );
}