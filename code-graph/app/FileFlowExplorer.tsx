"use client"

import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";

// ---------------------------------------------------------------------
// Types describing the JSON this tool expects. Both shapes mirror what a
// typical "predicted edges" / "file classification" export looks like:
//
//   edges.json  (required) — an array of:
//     {
//       source_filepath: string,
//       target_filepath: string,
//       predicted_types: string[],       // e.g. ["CALLS", "FLOWS_TO"]
//       type_probs: { [type: string]: number },
//       edge_exists_prob?: number
//     }
//
//   predictions.json (optional) — an array of:
//     { filepath: string, predicted_label?: string, confidence?: number }
//
// Nothing about the repo, folder names, or edge type vocabulary is
// hard-coded — it's all discovered from whatever JSON is loaded.
// ---------------------------------------------------------------------

interface RawEdge {
  source_filepath: string;
  target_filepath: string;
  predicted_types?: string[];
  type_probs?: Record<string, number>;
  edge_exists_prob?: number;
}

interface PredictionEntry {
  filepath: string;
  predicted_label?: string;
  confidence?: number;
}

interface FileNode {
  id: string; // display label (filename, or full path if names collide)
  path: string;
  conf: number;
}

interface FlowEdge {
  s: string;
  t: string;
  p: number; // probability of the selected edge type
  secondary: Record<string, number>; // other type probabilities, for the detail panel
}

interface ModuleData {
  rawIntraCount: number;
  nodes: FileNode[];
  edges: FlowEdge[];
}

interface SimNode extends FileNode {
  degree: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface SimLink {
  source: number;
  target: number;
  p: number;
  secondary: Record<string, number>;
}

const COLORS = {
  bg: "#0B0F14",
  panel: "#0E141B",
  border: "#1C2530",
  text: "#D8DEE9",
  bright: "#EDEFF2",
  subtext: "#8A93A3",
  muted: "#4B5563",
  chip: "#131A22",
  edge: "#E8A33D",
  edgeDim: "#1C2530",
  node: "#5EC8D8",
  nodeSel: "#E8A33D",
};

// ---------------------------------------------------------------------
// module grouping: same convention as before — a "module" is a file's
// path truncated to its first `depth` folder segments.
// ---------------------------------------------------------------------
function moduleOf(filepath: string, depth = 2): string {
  const parts = filepath.split("/");
  if (parts.length <= depth) {
    return parts.length > 1 ? parts.slice(0, -1).join("/") : parts[0];
  }
  return parts.slice(0, depth).join("/");
}

function edgeWeight(e: RawEdge, edgeType: string): number {
  if (e.type_probs && typeof e.type_probs[edgeType] === "number") return e.type_probs[edgeType];
  return e.edge_exists_prob ?? 0.5;
}

function detectEdgeTypes(edges: RawEdge[]): string[] {
  const seen = new Map<string, number>();
  edges.forEach((e) => {
    (e.predicted_types || []).forEach((t) => seen.set(t, (seen.get(t) || 0) + 1));
  });
  return Array.from(seen.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([t]) => t);
}

function buildModules(
  edges: RawEdge[],
  predictions: PredictionEntry[] | null,
  edgeType: string,
  topK: number
): { modules: Record<string, ModuleData>; order: string[]; totalTyped: number } {
  const confByPath = new Map<string, number>();
  (predictions || []).forEach((p) => confByPath.set(p.filepath, p.confidence ?? 0));

  const typed = edgeType
    ? edges.filter((e) => (e.predicted_types || []).includes(edgeType))
    : edges;

  const intraByModule = new Map<string, RawEdge[]>();
  typed.forEach((e) => {
    const s = moduleOf(e.source_filepath);
    const t = moduleOf(e.target_filepath);
    if (s === t) {
      if (!intraByModule.has(s)) intraByModule.set(s, []);
      intraByModule.get(s)!.push(e);
    }
  });

  const modules: Record<string, ModuleData> = {};
  intraByModule.forEach((lst, m) => {
    const bySrc = new Map<string, RawEdge[]>();
    lst.forEach((e) => {
      if (!bySrc.has(e.source_filepath)) bySrc.set(e.source_filepath, []);
      bySrc.get(e.source_filepath)!.push(e);
    });

    const sparse: RawEdge[] = [];
    bySrc.forEach((l2) => {
      l2.sort((a, b) => edgeWeight(b, edgeType) - edgeWeight(a, edgeType));
      sparse.push(...l2.slice(0, topK));
    });

    const nodeSet = new Set<string>();
    sparse.forEach((e) => {
      nodeSet.add(e.source_filepath);
      nodeSet.add(e.target_filepath);
    });

    const nameCounts = new Map<string, number>();
    Array.from(nodeSet).forEach((p) => {
      const name = p.split("/").pop() as string;
      nameCounts.set(name, (nameCounts.get(name) || 0) + 1);
    });
    const idFor = (p: string) => {
      const name = p.split("/").pop() as string;
      return (nameCounts.get(name) || 0) > 1 ? p : name;
    };

    const fnodes: FileNode[] = Array.from(nodeSet)
      .sort()
      .map((p) => ({ id: idFor(p), path: p, conf: confByPath.get(p) ?? 0 }));

    const fedges: FlowEdge[] = sparse.map((e) => {
      const secondary: Record<string, number> = {};
      Object.entries(e.type_probs || {}).forEach(([k, v]) => {
        if (k !== edgeType) secondary[k] = v;
      });
      return {
        s: idFor(e.source_filepath),
        t: idFor(e.target_filepath),
        p: edgeWeight(e, edgeType),
        secondary,
      };
    });

    modules[m] = { rawIntraCount: lst.length, nodes: fnodes, edges: fedges };
  });

  const order = Object.keys(modules).sort((a, b) => modules[b].edges.length - modules[a].edges.length);
  return { modules, order, totalTyped: typed.length };
}

function radiusFor(degree: number): number {
  return 6 + Math.sqrt(degree + 1) * 3.2;
}

// ---------------------------------------------------------------------
// main component
// ---------------------------------------------------------------------
// Graph data is loaded at runtime from the /api route below (see
// app/api/route.ts), which reads both JSON files server-side and
// returns them combined as { edges, predictions }. Fetching happens
// inside the component's useEffect further down, not here at module
// scope — a top-level `await fetch` here would also run during SSR,
// where there's no page origin to resolve a relative URL against.
// ---------------------------------------------------------------------

export default function FileFlowExplorer() {
  const [rawEdges, setRawEdges] = useState<RawEdge[] | null>(null);
  const [rawPredictions, setRawPredictions] = useState<PredictionEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [edgeType, setEdgeType] = useState<string>("");
  const [topK, setTopK] = useState<number>(2);
  const [moduleId, setModuleId] = useState<string>("");
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState<string>("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setLoadError(null);
      try {
        const res = await fetch("/api");
        if (!res.ok) {
          let detail = `HTTP ${res.status}`;
          try {
            const body = await res.json();
            if (body?.missingFiles?.length) {
              detail = `missing file(s): ${body.missingFiles.join(", ")}`;
            } else if (body?.error) {
              detail = body.error;
            }
          } catch {
            // body wasn't JSON — fall back to the HTTP status detail above
          }
          throw new Error(`Failed to load graph data \u2014 ${detail}`);
        }

        const data = await res.json();
        if (cancelled) return;

        const edgesJson = data.edges as RawEdge[];
        setRawEdges(edgesJson);
        const types = detectEdgeTypes(edgesJson);
        setEdgeType((prev) => prev || types.find((t) => t.toUpperCase().includes("FLOW")) || types[0] || "");

        if (data.predictions) {
          setRawPredictions(data.predictions as PredictionEntry[]);
        }
      } catch (err) {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "Failed to load JSON.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const edgeTypes = useMemo(() => (rawEdges ? detectEdgeTypes(rawEdges) : []), [rawEdges]);

  const built = useMemo(() => {
    if (!rawEdges || !edgeType) return null;
    return buildModules(rawEdges, rawPredictions, edgeType, topK);
  }, [rawEdges, rawPredictions, edgeType, topK]);

  useEffect(() => {
    if (!built || built.order.length === 0) return;

    // Delay setting state to avoid synchronous setState within effect
    // which can trigger cascading renders. Schedule via rAF and clean up.
    if (!built.modules[moduleId]) {
      const id = typeof window !== "undefined" && window.requestAnimationFrame
        ? window.requestAnimationFrame(() => setModuleId(built.order[0]))
        : // fallback to timeout in non-browser envs
          (setTimeout(() => setModuleId(built.order[0]), 0) as unknown as number);

      return () => {
        if (typeof window !== "undefined" && window.cancelAnimationFrame) {
          window.cancelAnimationFrame(id as number);
        } else {
          clearTimeout(id as number);
        }
      };
    }
    return;
  }, [built, moduleId]);

  const totalFiles = rawPredictions
    ? rawPredictions.length
    : rawEdges
    ? new Set(rawEdges.flatMap((e) => [e.source_filepath, e.target_filepath])).size
    : 0;

  if (loadError) {
    return <StatusScreen kind="error" message={loadError} />;
  }
  if (loading || !rawEdges || !built) {
    return <StatusScreen kind="loading" message="loading graph data\u2026" />;
  }

  return (
    <GraphScreen
      built={built}
      edgeTypes={edgeTypes}
      edgeType={edgeType}
      setEdgeType={setEdgeType}
      topK={topK}
      setTopK={setTopK}
      moduleId={moduleId}
      setModuleId={setModuleId}
      selected={selected}
      setSelected={setSelected}
      query={query}
      setQuery={setQuery}
      totalFiles={totalFiles}
      totalRawEdges={rawEdges.length}
    />
  );
}

function StatusScreen({ kind, message }: { kind: "loading" | "error"; message: string }) {
  return (
    <div
      style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
        background: COLORS.bg,
        color: COLORS.text,
        width: "100%",
        minHeight: "560px",
        borderRadius: "10px",
        border: `1px solid ${COLORS.border}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px",
        textAlign: "center",
      }}
    >
      <div>
        <div style={{ fontSize: "13px", color: kind === "error" ? "#E88686" : COLORS.bright, fontWeight: 600, marginBottom: "6px" }}>
          {kind === "error" ? "couldn't load data" : "loading\u2026"}
        </div>
        <div style={{ fontSize: "11.5px", color: COLORS.subtext, maxWidth: "420px" }}>
          {message}
          {kind === "error" && (
            <div style={{ marginTop: "10px" }}>
              Check that <code>Final_Output/output_predicted_edges.json</code> and
              <code>Final_Output/Prediction_Agent-Reach_codebert.json</code> both exist \u2014 the
              <code>/api</code> route reads them server-side and needs both files present.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------
// graph screen (post-upload) — same interaction model as the original
// FileDataFlowMap: force layout, drag, zoom/pan, hub-gated labels, search.
// ---------------------------------------------------------------------
function GraphScreen(props: {
  built: { modules: Record<string, ModuleData>; order: string[]; totalTyped: number };
  edgeTypes: string[];
  edgeType: string;
  setEdgeType: (t: string) => void;
  topK: number;
  setTopK: (n: number) => void;
  moduleId: string;
  setModuleId: (m: string) => void;
  selected: string | null;
  setSelected: (s: string | null) => void;
  query: string;
  setQuery: (q: string) => void;
  totalFiles: number;
  totalRawEdges: number;
}) {
  const {
    built, edgeTypes, edgeType, setEdgeType, topK, setTopK,
    moduleId, setModuleId, selected, setSelected, query, setQuery,
    totalFiles, totalRawEdges,
  } = props;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const simRef = useRef<d3.Simulation<SimNode, undefined> | null>(null);

  const [dims, setDims] = useState({ w: 900, h: 600 });
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [zoomT, setZoomT] = useState({ x: 0, y: 0, k: 1 });
  const [, forceTick] = useState(0);

  const moduleData = built.modules[moduleId] || { rawIntraCount: 0, nodes: [], edges: [] };

  useEffect(() => {
    function measure() {
      if (containerRef.current) {
        const r = containerRef.current.getBoundingClientRect();
        setDims({ w: Math.max(320, r.width), h: Math.max(420, r.height) });
      }
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const builtGraph = useMemo(() => {
    const idIndex = new Map<string, number>();
    const nodes: SimNode[] = moduleData.nodes.map((n, i) => {
      idIndex.set(n.id, i);
      return { ...n, degree: 0 };
    });
    const links: SimLink[] = [];
    moduleData.edges.forEach((e) => {
      const si = idIndex.get(e.s);
      const ti = idIndex.get(e.t);
      if (si == null || ti == null) return;
      nodes[si].degree += 1;
      nodes[ti].degree += 1;
      links.push({ source: si, target: ti, p: e.p, secondary: e.secondary });
    });
    return { nodes, links };
  }, [moduleData]);

  useEffect(() => {
    setSelected(null);
    setQuery("");
    const { nodes, links } = builtGraph;
    const spread = Math.max(1, Math.sqrt(nodes.length / 40));
    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(links as any)
          .id((_: any, i: number) => i)
          .distance(70 + 70 * spread)
          .strength((l: any) => 0.08 + l.p * 0.3)
      )
      .force("charge", d3.forceManyBody().strength(-260 * spread).distanceMax(700))
      .force("center", d3.forceCenter(dims.w / 2, dims.h / 2))
      .force("collide", d3.forceCollide<SimNode>().radius((n) => radiusFor(n.degree) + 26))
      .alpha(1)
      .alphaDecay(0.02)
      .on("tick", () => forceTick((t) => t + 1));

    simRef.current = sim;
    return () => {
      sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [builtGraph, dims.w, dims.h]);

  useEffect(() => {
    if (!svgRef.current) return;
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.25, 5])
      .on("zoom", (event) => {
        setZoomT({ x: event.transform.x, y: event.transform.y, k: event.transform.k });
      });
    zoomBehaviorRef.current = zoom;
    const sel = d3.select(svgRef.current);
    sel.call(zoom);
    return () => {
      sel.on(".zoom", null);
    };
  }, [dims.w, dims.h]);

  useEffect(() => {
    setZoomT({ x: 0, y: 0, k: 1 });
    if (svgRef.current && zoomBehaviorRef.current) {
      d3.select(svgRef.current).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
    }
  }, [moduleId]);

  function zoomBy(factor: number) {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current).transition().duration(200).call(zoomBehaviorRef.current.scaleBy, factor);
  }
  function resetZoom() {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current).transition().duration(250).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  }

  const dragBehavior = useCallback((simNode: SimNode) => {
    return d3
      .drag<SVGGElement, unknown>()
      .on("start", (event) => {
        if (!event.active && simRef.current) simRef.current.alphaTarget(0.25).restart();
        simNode.fx = simNode.x;
        simNode.fy = simNode.y;
      })
      .on("drag", (event) => {
        simNode.fx = event.x;
        simNode.fy = event.y;
      })
      .on("end", (event) => {
        if (!event.active && simRef.current) simRef.current.alphaTarget(0);
        simNode.fx = null;
        simNode.fy = null;
      });
  }, []);

  const { nodes, links } = builtGraph;

  const hubSet = useMemo(() => {
    const sorted = [...nodes].sort((a, b) => b.degree - a.degree);
    return new Set(sorted.slice(0, Math.min(14, Math.ceil(nodes.length * 0.12))).map((n) => n.id));
  }, [nodes]);

  const searchMatches = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    return nodes.filter((n) => n.id.toLowerCase().includes(q)).slice(0, 8);
  }, [query, nodes]);

  const neighborSet = useMemo(() => {
    if (!selected) return null;
    const idx = nodes.findIndex((n) => n.id === selected);
    if (idx < 0) return null;
    const s = new Set<number>([idx]);
    links.forEach((l) => {
      const si = typeof l.source === "number" ? l.source : (l.source as any).index;
      const ti = typeof l.target === "number" ? l.target : (l.target as any).index;
      if (si === idx) s.add(ti);
      if (ti === idx) s.add(si);
    });
    return s;
  }, [selected, nodes, links]);

  const selectedNode = selected ? nodes.find((n) => n.id === selected) : null;
  const inEdges = selectedNode
    ? links.filter((l) => nodes[typeof l.target === "number" ? l.target : (l.target as any).index]?.id === selected)
    : [];
  const outEdges = selectedNode
    ? links.filter((l) => nodes[typeof l.source === "number" ? l.source : (l.source as any).index]?.id === selected)
    : [];

  return (
    <div
      style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
        background: COLORS.bg,
        color: COLORS.text,
        width: "100%",
        minHeight: "680px",
        borderRadius: "10px",
        border: `1px solid ${COLORS.border}`,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* header */}
      <div
        style={{
          padding: "14px 20px",
          borderBottom: `1px solid ${COLORS.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "12px",
          background: "linear-gradient(180deg, #0E141B 0%, #0B0F14 100%)",
        }}
      >
        <div>
          <div style={{ fontSize: "11px", letterSpacing: "0.14em", color: COLORS.edge, textTransform: "uppercase" }}>
            file-level data flow \u00b7 loaded from /api
          </div>
          <div style={{ fontSize: "16px", fontWeight: 600, color: COLORS.bright, marginTop: "2px" }}>
            {totalFiles.toLocaleString()} files \u00b7 {totalRawEdges.toLocaleString()} raw edges loaded
          </div>
        </div>
      </div>

      {/* controls */}
      <div style={{ padding: "10px 20px", display: "flex", alignItems: "center", gap: "14px", borderBottom: `1px solid ${COLORS.border}`, fontSize: "12px", flexWrap: "wrap" }}>
        <span style={{ color: COLORS.subtext }}>edge type</span>
        <select value={edgeType} onChange={(e) => setEdgeType(e.target.value)} style={selectStyle}>
          {edgeTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <span style={{ color: COLORS.subtext }}>folder</span>
        <select value={moduleId} onChange={(e) => setModuleId(e.target.value)} style={selectStyle}>
          {built.order.length === 0 && <option value="">(no intra-folder edges of this type)</option>}
          {built.order.map((m) => (
            <option key={m} value={m}>{m} ({built.modules[m].edges.length} edges)</option>
          ))}
        </select>

        <span style={{ color: COLORS.subtext }}>top-K per file</span>
        <input type="range" min={1} max={6} value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: "90px" }} />
        <span style={{ color: COLORS.bright }}>{topK}</span>

        <div style={{ marginLeft: "auto", position: "relative" }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="find a file\u2026"
            style={{ background: COLORS.chip, color: COLORS.bright, border: `1px solid ${COLORS.border}`, borderRadius: "6px", padding: "5px 10px", fontFamily: "inherit", fontSize: "12px", width: "160px" }}
          />
          {searchMatches.length > 0 && (
            <div style={{ position: "absolute", top: "28px", right: 0, width: "260px", background: COLORS.chip, border: `1px solid ${COLORS.border}`, borderRadius: "6px", zIndex: 5, maxHeight: "220px", overflowY: "auto" }}>
              {searchMatches.map((n) => (
                <div
                  key={n.id}
                  onClick={() => { setSelected(n.id); setQuery(""); }}
                  style={{ padding: "6px 10px", fontSize: "11.5px", color: COLORS.text, cursor: "pointer", borderBottom: `1px solid ${COLORS.border}` }}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#1A222C")}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = "transparent")}
                >
                  {n.id}
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: "4px" }}>
          <ToolbarButton onClick={() => zoomBy(1.4)} label="+" />
          <ToolbarButton onClick={() => zoomBy(1 / 1.4)} label="\u2212" />
          <ToolbarButton onClick={resetZoom} label="reset" />
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <div ref={containerRef} style={{ flex: 1, position: "relative", minHeight: "460px" }}>
          <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block", cursor: "grab" }}>
            <defs>
              <marker id="flowArrow" viewBox="0 0 10 10" refX="17" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.edge} />
              </marker>
              <marker id="flowArrowDim" viewBox="0 0 10 10" refX="17" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.edgeDim} />
              </marker>
            </defs>
            <g transform={`translate(${zoomT.x},${zoomT.y}) scale(${zoomT.k})`}>
              <g>
                {links.map((l, i) => {
                  const si = typeof l.source === "number" ? l.source : (l.source as any).index;
                  const ti = typeof l.target === "number" ? l.target : (l.target as any).index;
                  const s = nodes[si];
                  const t = nodes[ti];
                  if (!s || !t || s.x == null || t.x == null || t.y == null || s.y == null) return null;
                  const active = !neighborSet || (neighborSet.has(si) && neighborSet.has(ti));
                  return (
                    <line
                      key={i}
                      x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                      stroke={active ? COLORS.edge : COLORS.edgeDim}
                      strokeOpacity={active ? 0.3 + l.p * 0.45 : 0.15}
                      strokeWidth={(1 + l.p * 2.2) / Math.max(1, zoomT.k * 0.6)}
                      markerEnd={active ? "url(#flowArrow)" : "url(#flowArrowDim)"}
                    />
                  );
                })}
              </g>
              <g>
                {nodes.map((n, i) => {
                  if (n.x == null || n.y == null) return null;
                  const r = radiusFor(n.degree);
                  const dim = neighborSet && !neighborSet.has(i);
                  const isSel = selected === n.id;
                  const isHov = hoveredId === n.id;
                  const showLabel = isSel || isHov || hubSet.has(n.id) || zoomT.k > 1.6;
                  const label = n.id.length > 24 ? n.id.slice(0, 22) + "\u2026" : n.id;
                  return (
                    <g
                      key={n.id}
                      transform={`translate(${n.x},${n.y})`}
                      onClick={() => setSelected(isSel ? null : n.id)}
                      onMouseEnter={() => setHoveredId(n.id)}
                      onMouseLeave={() => setHoveredId(null)}
                      style={{ cursor: "pointer" }}
                      ref={(el) => {
                        if (el && !(el as any).__dragged) {
                          d3.select(el).call(dragBehavior(n) as any);
                          (el as any).__dragged = true;
                        }
                      }}
                    >
                      <circle
                        r={r}
                        fill={isSel ? COLORS.nodeSel : hubSet.has(n.id) ? "#7FD9EC" : COLORS.node}
                        fillOpacity={dim ? 0.15 : isSel ? 0.95 : 0.7}
                        stroke={isSel ? COLORS.bright : "transparent"}
                        strokeWidth={2}
                      />
                      {showLabel && !dim && (
                        <g pointerEvents="none">
                          <rect
                            x={-(label.length * 3.1) / 2 - 4}
                            y={r + 4}
                            width={label.length * 3.1 + 8}
                            height={13}
                            rx={3}
                            fill={COLORS.bg}
                            fillOpacity={0.82}
                          />
                          <text textAnchor="middle" dy={r + 14} fontSize="9.5" fill={isSel || isHov ? COLORS.bright : "#9BA5B4"}>
                            {label}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </g>
            </g>
          </svg>
          {!selected && (
            <div style={{ position: "absolute", bottom: 14, left: 20, fontSize: "11px", color: COLORS.muted }}>
              scroll / pinch to zoom \u00b7 drag background to pan \u00b7 drag a node to move it \u00b7 click a file to trace its flow
            </div>
          )}
        </div>

        <div style={{ width: selected ? "300px" : "0px", transition: "width 0.15s ease", borderLeft: selected ? `1px solid ${COLORS.border}` : "none", overflowY: "auto", background: COLORS.panel }}>
          {selectedNode && (
            <div style={{ padding: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: COLORS.bright, wordBreak: "break-all" }}>{selectedNode.path}</div>
                <button onClick={() => setSelected(null)} style={{ background: "none", border: "none", color: COLORS.subtext, cursor: "pointer", fontSize: "16px", lineHeight: 1 }}>
                  {"\u00d7"}
                </button>
              </div>
              {selectedNode.conf > 0 && (
                <div style={{ marginTop: "10px", fontSize: "12px", color: COLORS.subtext }}>
                  {Math.round(selectedNode.conf * 100)}% classification confidence
                </div>
              )}

              <div style={{ marginTop: "16px" }}>
                <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.08em", color: COLORS.muted, marginBottom: "6px" }}>
                  flows out to ({outEdges.length})
                </div>
                <EdgeList edges={outEdges} nodes={nodes} dir="out" edgeType={edgeType} />
              </div>

              <div style={{ marginTop: "16px" }}>
                <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.08em", color: COLORS.muted, marginBottom: "6px" }}>
                  flows in from ({inEdges.length})
                </div>
                <EdgeList edges={inEdges} nodes={nodes} dir="in" edgeType={edgeType} />
              </div>

              {outEdges.length === 0 && inEdges.length === 0 && (
                <div style={{ fontSize: "11px", color: COLORS.muted, marginTop: "10px" }}>
                  no edges cleared the top-{topK} threshold for this file.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: COLORS.chip,
  color: COLORS.bright,
  border: `1px solid ${COLORS.border}`,
  borderRadius: "6px",
  padding: "5px 10px",
  fontFamily: "inherit",
  fontSize: "12px",
};

function ToolbarButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button onClick={onClick} style={{ background: COLORS.chip, color: COLORS.text, border: `1px solid ${COLORS.border}`, borderRadius: "6px", padding: "5px 9px", fontFamily: "inherit", fontSize: "11px", cursor: "pointer" }}>
      {label}
    </button>
  );
}

function EdgeList({ edges, nodes, dir, edgeType }: { edges: SimLink[]; nodes: SimNode[]; dir: "in" | "out"; edgeType: string }) {
  if (edges.length === 0) return <div style={{ fontSize: "11px", color: COLORS.muted }}>none</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {edges.map((e, i) => {
        const idx = dir === "out" ? (typeof e.target === "number" ? e.target : (e.target as any).index) : (typeof e.source === "number" ? e.source : (e.source as any).index);
        const other = nodes[idx];
        if (!other) return null;
        const secondary = Object.entries(e.secondary || {});
        return (
          <div key={i} style={{ fontSize: "11.5px", color: COLORS.text, background: COLORS.chip, borderRadius: "5px", padding: "6px 8px" }}>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{other.id}</div>
            <div style={{ color: COLORS.muted, fontSize: "10.5px", marginTop: "2px" }}>
              {edgeType} {Math.round(e.p * 100)}%
              {secondary.map(([k, v]) => ` \u00b7 ${k} ${Math.round(v * 100)}%`).join("")}
            </div>
          </div>
        );
      })}
    </div>
  );
}