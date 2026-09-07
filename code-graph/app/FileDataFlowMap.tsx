"use client"
import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";

// ---------------------------------------------------------------------
// Set these two paths to your JSON files. They're fetched at runtime
// (e.g. files served from /public in Next.js, or any reachable URL) —
// nothing to upload, just point these at your data.
//
//   EDGES_JSON_PATH (required) — array of:
//     { source_filepath, target_filepath, predicted_types: string[],
//       type_probs: { [type: string]: number }, edge_exists_prob?: number }
//
//   PREDICTIONS_JSON_PATH (optional) — array of:
//     { filepath, predicted_label?: string, confidence?: number }
//     predicted_label is used as the layer if it already says
//     "frontend" / "backend" / "database" / "api"; otherwise the layer
//     is guessed from the file path (see classifyLayer below).
// ---------------------------------------------------------------------
// Drop the two JSON files straight into your Next.js `public/data/` folder
// (create the folder if it doesn't exist) — nothing else to configure.
const EDGES_JSON_PATH = "/Data/predicted_edges.json";
const PREDICTIONS_JSON_PATH = "/Data/Prediction.json";

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

type Level = "layers" | "files" | "connections";

const LAYER_ORDER = ["frontend", "api", "backend", "database", "other"] as const;
type LayerId = (typeof LAYER_ORDER)[number];

const LAYER_META: Record<LayerId, { title: string; color: string; desc: string; pos: { x: number; y: number } }> = {
  frontend: { title: "frontend", color: "#E8779C", desc: "UI / client-rendered code", pos: { x: 0.18, y: 0.22 } },
  api: { title: "api", color: "#E8A33D", desc: "routes, controllers, endpoints", pos: { x: 0.82, y: 0.22 } },
  backend: { title: "backend", color: "#5EC8D8", desc: "server / runtime / orchestration", pos: { x: 0.82, y: 0.78 } },
  database: { title: "database", color: "#8B7FD6", desc: "models, schema, data access", pos: { x: 0.18, y: 0.78 } },
  other: { title: "other", color: "#B0B7C3", desc: "config, docs, build tooling", pos: { x: 0.5, y: 0.5 } },
};

const COLORS = {
  bg: "#0B0F14",
  panel: "#0E141B",
  border: "#1C2530",
  text: "#D8DEE9",
  bright: "#EDEFF2",
  subtext: "#8A93A3",
  muted: "#4B5563",
  chip: "#131A22",
};

const CONNECTION_PALETTE = [
  "#5EC8D8", "#E8A33D", "#8B7FD6", "#5FBF7A", "#E8779C",
  "#7FA8E8", "#D6C15E", "#E86A5E", "#6EDCC0", "#B0847F",
  "#9FCF5E", "#C77FE8", "#5EA8D6", "#E8A05E", "#7F9BD6",
];

// ---------------------------------------------------------------------
// layer classification — uses predicted_label when it already matches
// one of the four buckets, otherwise falls back to path heuristics so
// this still works on data that only ever says "backend".
// ---------------------------------------------------------------------
function classifyLayer(filepath: string, predictedLabel?: string): LayerId {
  const label = (predictedLabel || "").toLowerCase();
  if ((LAYER_ORDER as readonly string[]).includes(label) && label !== "other") return label as LayerId;

  const p = filepath.toLowerCase();
  if (/(^|\/)(api|routes?|endpoints?|controllers?)(\/|$)/.test(p) || /route|endpoint|controller/.test(p)) return "api";
  if (/(^|\/)(db|database|models?|schema|datastore|migrations?)(\/|$)/.test(p) || /\.sql$/.test(p)) return "database";
  if (/\.(jsx|tsx|vue|svelte|css|scss|html)$/.test(p) || /(^|\/)(components?|pages?|views?|ui|frontend|client|public)(\/|$)/.test(p)) return "frontend";
  if (/\.(py|go|java|rb|php|rs|cs|ts|js)$/.test(p) || /(^|\/)(server|backend|services?|core)(\/|$)/.test(p)) return "backend";
  return "other";
}

function detectEdgeTypes(edges: RawEdge[]): string[] {
  const seen = new Map<string, number>();
  edges.forEach((e) => (e.predicted_types || []).forEach((t) => seen.set(t, (seen.get(t) || 0) + 1)));
  return Array.from(seen.entries()).sort((a, b) => b[1] - a[1]).map(([t]) => t);
}
function edgeWeight(e: RawEdge, edgeType: string): number {
  if (e.type_probs && typeof e.type_probs[edgeType] === "number") return e.type_probs[edgeType];
  return e.edge_exists_prob ?? 0.5;
}
function radiusFor(degree: number): number {
  return 7 + Math.sqrt(degree + 1) * 3.2;
}

// d3-force mutates plain data objects in place, adding x/y/vx/vy/fx/fy/index at
// runtime — SimulationNodeDatum/SimulationLinkDatum already model that, so we
// extend them instead of reaching for `any`.
interface SimNode extends d3.SimulationNodeDatum {
  path: string;
  id: string;
  degree: number;
  conf: number;
}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  p: number;
}
// d3 attaches drag behavior imperatively via a ref callback; tag the element
// with a typed marker instead of casting the ref to `any`.
type DraggableEl = SVGGElement & { __dragged?: boolean };

// =======================================================================
// main component
// =======================================================================
export default function ArchitectureExplorer() {
  const [rawEdges, setRawEdges] = useState<RawEdge[] | null>(null);
  const [rawPredictions, setRawPredictions] = useState<PredictionEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [edgeType, setEdgeType] = useState<string>("");

  const [level, setLevel] = useState<Level>("layers");
  const [layer, setLayer] = useState<LayerId | null>(null);
  const [file, setFile] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setLoadError(null);
      try {
        const edgesRes = await fetch(EDGES_JSON_PATH);
        if (!edgesRes.ok) throw new Error(`${EDGES_JSON_PATH} \u2014 HTTP ${edgesRes.status}`);
        const edgesJson = (await edgesRes.json()) as RawEdge[];
        if (cancelled) return;
        setRawEdges(edgesJson);
        const types = detectEdgeTypes(edgesJson);
        setEdgeType((prev) => prev || types.find((t) => t.toUpperCase().includes("FLOW")) || types[0] || "");

        if (PREDICTIONS_JSON_PATH) {
          try {
            const predRes = await fetch(PREDICTIONS_JSON_PATH);
            if (predRes.ok) {
              const predJson = (await predRes.json()) as PredictionEntry[];
              if (!cancelled) setRawPredictions(predJson);
            }
          } catch {
            /* predictions are optional */
          }
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

  const labelByPath = useMemo(() => {
    const m = new Map<string, string>();
    (rawPredictions || []).forEach((p) => m.set(p.filepath, p.predicted_label || ""));
    return m;
  }, [rawPredictions]);

  const confByPath = useMemo(() => {
    const m = new Map<string, number>();
    (rawPredictions || []).forEach((p) => m.set(p.filepath, p.confidence ?? 0));
    return m;
  }, [rawPredictions]);

  const layerOf = useCallback((fp: string) => classifyLayer(fp, labelByPath.get(fp)), [labelByPath]);

  if (loadError) return <StatusScreen kind="error" message={loadError} />;
  if (loading || !rawEdges) return <StatusScreen kind="loading" message={`loading ${EDGES_JSON_PATH}\u2026`} />;

  return (
    <div
      style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
        background: COLORS.bg,
        color: COLORS.text,
        width: "100%",
        minHeight: "700px",
        borderRadius: "10px",
        border: `1px solid ${COLORS.border}`,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Breadcrumb
        level={level}
        layer={layer}
        file={file}
        onLayers={() => { setLevel("layers"); setLayer(null); setFile(null); }}
        onFiles={() => { setLevel("files"); setFile(null); }}
      />
      {level === "layers" && (
        <LayerLevel
          edges={rawEdges}
          layerOf={layerOf}
          edgeType={edgeType}
          onPick={(l) => { setLayer(l); setLevel("files"); }}
        />
      )}
      {level === "files" && layer && (
        <FileLevel
          edges={rawEdges}
          layerOf={layerOf}
          confByPath={confByPath}
          layer={layer}
          edgeType={edgeType}
          onPickFile={(f) => { setFile(f); setLevel("connections"); }}
        />
      )}
      {level === "connections" && file && (
        <ConnectionLevel edges={rawEdges} file={file} layerOf={layerOf} confByPath={confByPath} />
      )}
    </div>
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
        <div style={{ fontSize: "11.5px", color: COLORS.subtext, maxWidth: "420px" }}>{message}</div>
      </div>
    </div>
  );
}

function Breadcrumb({ level, layer, file, onLayers, onFiles }: {
  level: Level; layer: LayerId | null; file: string | null;
  onLayers: () => void; onFiles: () => void;
}) {
  return (
    <div style={{ padding: "12px 20px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", background: "linear-gradient(180deg, #0E141B 0%, #0B0F14 100%)" }}>
      <span onClick={onLayers} style={{ cursor: "pointer", color: level === "layers" ? COLORS.bright : "#5EC8D8", fontWeight: level === "layers" ? 600 : 400 }}>
        architecture
      </span>
      {layer && (
        <>
          <span style={{ color: COLORS.muted }}>/</span>
          <span onClick={onFiles} style={{ cursor: "pointer", color: level === "files" ? COLORS.bright : "#5EC8D8", fontWeight: level === "files" ? 600 : 400 }}>
            {LAYER_META[layer].title}
          </span>
        </>
      )}
      {file && (
        <>
          <span style={{ color: COLORS.muted }}>/</span>
          <span style={{ color: COLORS.bright, fontWeight: 600 }}>{file.split("/").pop()}</span>
        </>
      )}
    </div>
  );
}

// =======================================================================
// LEVEL 1 — how frontend / api / backend / database flow into each other
// =======================================================================
function LayerLevel({ edges, layerOf, edgeType, onPick }: {
  edges: RawEdge[]; layerOf: (fp: string) => LayerId; edgeType: string; onPick: (l: LayerId) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dims, setDims] = useState({ w: 900, h: 560 });

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

  const { counts, flows } = useMemo(() => {
    const counts: Record<string, number> = {};
    const seenFiles = new Set<string>();
    const fileLayer = new Map<string, LayerId>();
    function reg(fp: string) {
      if (seenFiles.has(fp)) return;
      seenFiles.add(fp);
      const l = layerOf(fp);
      fileLayer.set(fp, l);
      counts[l] = (counts[l] || 0) + 1;
    }
    edges.forEach((e) => { reg(e.source_filepath); reg(e.target_filepath); });

    const pairAgg = new Map<string, { ab: number; ba: number; sumP: number; count: number }>();
    const typed = edgeType ? edges.filter((e) => (e.predicted_types || []).includes(edgeType)) : edges;
    typed.forEach((e) => {
      const s = fileLayer.get(e.source_filepath)!;
      const t = fileLayer.get(e.target_filepath)!;
      if (s === t) return;
      const [a, b] = [s, t].sort();
      const key = `${a}|${b}`;
      if (!pairAgg.has(key)) pairAgg.set(key, { ab: 0, ba: 0, sumP: 0, count: 0 });
      const rec = pairAgg.get(key)!;
      rec.count += 1;
      rec.sumP += edgeWeight(e, edgeType);
      if (s === a) rec.ab += 1; else rec.ba += 1;
    });

    const flows: { s: LayerId; t: LayerId; weight: number; avgP: number }[] = [];
    pairAgg.forEach((rec, key) => {
      const [a, b] = key.split("|") as [LayerId, LayerId];
      const [s, t] = rec.ab >= rec.ba ? [a, b] : [b, a];
      flows.push({ s, t, weight: rec.count, avgP: rec.sumP / rec.count });
    });
    return { counts, flows };
  }, [edges, layerOf, edgeType]);

  const totalFiles = Object.values(counts).reduce((a, b) => a + b, 0);
  const maxWeight = Math.max(1, ...flows.map((f) => f.weight));
  const visibleLayers = LAYER_ORDER.filter((l) => l !== "other" || counts.other > 0);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div style={{ padding: "10px 20px", fontSize: "12px", color: COLORS.subtext, borderBottom: `1px solid ${COLORS.border}` }}>
        step 1 of 3 \u00b7 {totalFiles.toLocaleString()} files bucketed into frontend / api / backend / database (classification label when it matches, otherwise a path-based guess) \u00b7 click a layer to see its files
      </div>
      <div ref={containerRef} style={{ flex: 1, position: "relative", minHeight: "460px" }}>
        <svg width={dims.w} height={dims.h} style={{ display: "block" }}>
          <defs>
            {visibleLayers.map((l) => (
              <marker key={l} id={`arrow-${l}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={LAYER_META[l].color} />
              </marker>
            ))}
          </defs>
          <g>
            {flows.map((f, i) => {
              if (!visibleLayers.includes(f.s) || !visibleLayers.includes(f.t)) return null;
              const sp = LAYER_META[f.s].pos;
              const tp = LAYER_META[f.t].pos;
              const x1 = sp.x * dims.w, y1 = sp.y * dims.h;
              const x2 = tp.x * dims.w, y2 = tp.y * dims.h;
              const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
              // bow the curve slightly so opposite-direction flows between the same pair don't overlap
              const dx = y2 - y1, dy = x1 - x2;
              const norm = Math.hypot(dx, dy) || 1;
              const bow = 18;
              const cx = mx + (dx / norm) * bow;
              const cy = my + (dy / norm) * bow;
              return (
                <path
                  key={i}
                  d={`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`}
                  fill="none"
                  stroke={LAYER_META[f.s].color}
                  strokeOpacity={0.55}
                  strokeWidth={1.5 + (f.weight / maxWeight) * 6}
                  markerEnd={`url(#arrow-${f.s})`}
                />
              );
            })}
          </g>
          <g>
            {visibleLayers.map((l) => {
              const meta = LAYER_META[l];
              const cx = meta.pos.x * dims.w;
              const cy = meta.pos.y * dims.h;
              const count = counts[l] || 0;
              const empty = count === 0;
              const r = 46 + Math.min(30, Math.sqrt(count));
              return (
                <g key={l} transform={`translate(${cx},${cy})`} onClick={() => !empty && onPick(l)} style={{ cursor: empty ? "default" : "pointer" }}>
                  <circle r={r} fill={COLORS.chip} stroke={meta.color} strokeWidth={empty ? 1 : 2.5} opacity={empty ? 0.4 : 1} />
                  <text textAnchor="middle" dy={-4} fontSize="13" fontWeight={600} fill={empty ? COLORS.muted : COLORS.bright}>
                    {meta.title}
                  </text>
                  <text textAnchor="middle" dy={14} fontSize="11" fill={empty ? COLORS.muted : meta.color}>
                    {count} files
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
        <div style={{ position: "absolute", bottom: 14, left: 20, fontSize: "11px", color: COLORS.muted }}>
          arrow thickness = number of cross-layer {edgeType || ""} edges \u00b7 color = source layer
        </div>
      </div>
    </div>
  );
}

// =======================================================================
// LEVEL 2 — files inside the selected layer
// =======================================================================
function FileLevel({ edges, layerOf, confByPath, layer, edgeType, onPickFile }: {
  edges: RawEdge[]; layerOf: (fp: string) => LayerId; confByPath: Map<string, number>;
  layer: LayerId; edgeType: string; onPickFile: (f: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const [dims, setDims] = useState({ w: 900, h: 560 });
  const [zoomT, setZoomT] = useState({ x: 0, y: 0, k: 1 });
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [, tick] = useState(0);

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

  const built = useMemo((): { nodes: SimNode[]; links: SimLink[]; rawIntraCount: number } => {
    const typed = edgeType ? edges.filter((e) => (e.predicted_types || []).includes(edgeType)) : edges;
    const intra = typed.filter((e) => layerOf(e.source_filepath) === layer && layerOf(e.target_filepath) === layer);
    const bySrc = new Map<string, RawEdge[]>();
    intra.forEach((e) => {
      if (!bySrc.has(e.source_filepath)) bySrc.set(e.source_filepath, []);
      bySrc.get(e.source_filepath)!.push(e);
    });
    const sparse: RawEdge[] = [];
    bySrc.forEach((l2) => {
      l2.sort((a, b) => edgeWeight(b, edgeType) - edgeWeight(a, edgeType));
      sparse.push(...l2.slice(0, 2));
    });
    const nodePaths = Array.from(new Set(sparse.flatMap((e) => [e.source_filepath, e.target_filepath])));
    const idIndex = new Map(nodePaths.map((p, i) => [p, i]));
    const nodes: SimNode[] = nodePaths.map((p) => ({ path: p, id: p.split("/").pop() as string, degree: 0, conf: confByPath.get(p) ?? 0 }));
    const links: SimLink[] = sparse.map((e) => {
      const si = idIndex.get(e.source_filepath)!;
      const ti = idIndex.get(e.target_filepath)!;
      nodes[si].degree += 1;
      nodes[ti].degree += 1;
      return { source: si, target: ti, p: edgeWeight(e, edgeType) };
    });
    return { nodes, links, rawIntraCount: intra.length };
  }, [edges, layerOf, layer, edgeType, confByPath]);

  // `built` only changes when `layer` changes, and a layer change always
  // unmounts/remounts FileLevel from the parent (you have to pass back
  // through the layers screen to get here) — so a fresh `useState(null)` /
  // `useState("")` already starts clean on every mount. No need to reset
  // them imperatively inside the effect (which would call setState
  // synchronously during an effect body — the thing set-state-in-effect
  // warns about).
  useEffect(() => {
    const { nodes, links } = built;
    const spread = Math.max(1, Math.sqrt(nodes.length / 40));
    const sim = d3
      .forceSimulation<SimNode, SimLink>(nodes)
      .force("link", d3.forceLink<SimNode, SimLink>(links).id((_, i) => i).distance(70 + 70 * spread).strength((l) => 0.1 + l.p * 0.3))
      .force("charge", d3.forceManyBody().strength(-240 * spread).distanceMax(650))
      .force("center", d3.forceCenter(dims.w / 2, dims.h / 2))
      .force("collide", d3.forceCollide<SimNode>().radius((n) => radiusFor(n.degree) + 24))
      .alpha(1)
      .alphaDecay(0.02)
      .on("tick", () => tick((t) => t + 1));
    simRef.current = sim;
    return () => { sim.stop(); };
  }, [built, dims.w, dims.h]);

  useEffect(() => {
    if (!svgRef.current) return;
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.25, 5]).on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => setZoomT({ x: event.transform.x, y: event.transform.y, k: event.transform.k }));
    zoomBehaviorRef.current = zoom;
    const sel = d3.select(svgRef.current);
    sel.call(zoom);
    return () => { sel.on(".zoom", null); };
  }, [dims.w, dims.h]);

  const dragBehavior = useCallback((simNode: SimNode) => {
    return d3.drag<SVGGElement, unknown>()
      .on("start", (event: d3.D3DragEvent<SVGGElement, unknown, unknown>) => { if (!event.active && simRef.current) simRef.current.alphaTarget(0.25).restart(); simNode.fx = simNode.x; simNode.fy = simNode.y; })
      .on("drag", (event: d3.D3DragEvent<SVGGElement, unknown, unknown>) => { simNode.fx = event.x; simNode.fy = event.y; })
      .on("end", (event: d3.D3DragEvent<SVGGElement, unknown, unknown>) => { if (!event.active && simRef.current) simRef.current.alphaTarget(0); simNode.fx = null; simNode.fy = null; });
  }, []);

  const { nodes, links } = built;
  const meta = LAYER_META[layer];

  const hubSet = useMemo(() => {
    const sorted = [...nodes].sort((a, b) => b.degree - a.degree);
    return new Set(sorted.slice(0, Math.min(14, Math.ceil(nodes.length * 0.15))).map((n) => n.id));
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
      const si = typeof l.source === "number" ? l.source : (l.source as SimNode).index!;
      const ti = typeof l.target === "number" ? l.target : (l.target as SimNode).index!;
      if (si === idx) s.add(ti);
      if (ti === idx) s.add(si);
    });
    return s;
  }, [selected, nodes, links]);

  function zoomBy(factor: number) {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current).transition().duration(200).call(zoomBehaviorRef.current.scaleBy, factor);
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div style={{ padding: "10px 20px", display: "flex", alignItems: "center", gap: "14px", fontSize: "12px", color: COLORS.subtext, borderBottom: `1px solid ${COLORS.border}`, flexWrap: "wrap" }}>
        <span>
          step 2 of 3 \u00b7 <span style={{ color: meta.color }}>{meta.title}</span> files \u00b7 {built.rawIntraCount} raw intra-layer edges, {links.length} shown (top-2/file) \u00b7 double-click a file for its connections
        </span>
        <div style={{ marginLeft: "auto", position: "relative" }}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="find a file\u2026" style={{ background: COLORS.chip, color: COLORS.bright, border: `1px solid ${COLORS.border}`, borderRadius: "6px", padding: "5px 10px", fontFamily: "inherit", fontSize: "12px", width: "160px" }} />
          {searchMatches.length > 0 && (
            <div style={{ position: "absolute", top: "28px", right: 0, width: "240px", background: COLORS.chip, border: `1px solid ${COLORS.border}`, borderRadius: "6px", zIndex: 5 }}>
              {searchMatches.map((n) => (
                <div key={n.path} onClick={() => { onPickFile(n.path); }} style={{ padding: "6px 10px", fontSize: "11.5px", cursor: "pointer", borderBottom: `1px solid ${COLORS.border}` }}>
                  {n.id}
                </div>
              ))}
            </div>
          )}
        </div>
        <ToolbarButton onClick={() => zoomBy(1.3)} label="+" />
        <ToolbarButton onClick={() => zoomBy(1 / 1.3)} label="\u2212" />
      </div>
      <div ref={containerRef} style={{ flex: 1, position: "relative", minHeight: "460px" }}>
        <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block", cursor: "grab" }}>
          <defs>
            <marker id="fileArrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={meta.color} />
            </marker>
          </defs>
          <g transform={`translate(${zoomT.x},${zoomT.y}) scale(${zoomT.k})`}>
            <g>
              {links.map((l, i) => {
                const si = typeof l.source === "number" ? l.source : (l.source as SimNode).index!;
                const ti = typeof l.target === "number" ? l.target : (l.target as SimNode).index!;
                const s = nodes[si];
                const t = nodes[ti];
                if (!s || !t || s.x == null || t.x == null || s.y == null || t.y == null) return null;
                const active = !neighborSet || (neighborSet.has(si) && neighborSet.has(ti));
                return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={meta.color} strokeOpacity={active ? 0.3 + l.p * 0.5 : 0.12} strokeWidth={1 + l.p * 2} markerEnd="url(#fileArrow)" />;
              })}
            </g>
            <g>
              {nodes.map((n, i) => {
                if (n.x == null || n.y == null) return null;
                const r = radiusFor(n.degree);
                const dim = neighborSet && !neighborSet.has(i);
                const isSel = selected === n.id;
                const showLabel = isSel || hubSet.has(n.id) || zoomT.k > 1.6;
                return (
                  <g
                    key={n.path}
                    transform={`translate(${n.x},${n.y})`}
                    onClick={() => setSelected(isSel ? null : n.id)}
                    onDoubleClick={() => onPickFile(n.path)}
                    style={{ cursor: "pointer" }}
                    ref={(el: DraggableEl | null) => { if (el && !el.__dragged) { d3.select(el).call(dragBehavior(n)); el.__dragged = true; } }}
                  >
                    <circle r={r} fill={isSel ? meta.color : hubSet.has(n.id) ? "#FFFFFF" : meta.color} fillOpacity={dim ? 0.15 : isSel ? 1 : hubSet.has(n.id) ? 0.9 : 0.55} stroke={isSel ? COLORS.bright : "transparent"} strokeWidth={2} />
                    {showLabel && !dim && (
                      <text textAnchor="middle" dy={r + 13} fontSize="10" fill="#9BA5B4">{n.id.length > 20 ? n.id.slice(0, 18) + "\u2026" : n.id}</text>
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      </div>
    </div>
  );
}

// =======================================================================
// LEVEL 3 — inside a file: every connection colored by which OTHER file
// it goes to, so you can visually separate "everything this file talks to"
// =======================================================================
function ConnectionLevel({ edges, file, layerOf, confByPath }: {
  edges: RawEdge[]; file: string; layerOf: (fp: string) => LayerId; confByPath: Map<string, number>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dims, setDims] = useState({ w: 900, h: 560 });

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

  const { connections, colorOf } = useMemo(() => {
    const rel = edges.filter((e) => e.source_filepath === file || e.target_filepath === file);
    const others = Array.from(new Set(rel.map((e) => (e.source_filepath === file ? e.target_filepath : e.source_filepath))));
    others.sort();
    const colorOf = new Map<string, string>();
    others.forEach((p, i) => colorOf.set(p, CONNECTION_PALETTE[i % CONNECTION_PALETTE.length]));

    const connections = others.map((p) => {
      const out = rel.find((e) => e.source_filepath === file && e.target_filepath === p);
      const inc = rel.find((e) => e.target_filepath === file && e.source_filepath === p);
      return {
        path: p,
        id: p.split("/").pop() as string,
        layer: layerOf(p),
        conf: confByPath.get(p) ?? 0,
        outTypes: out?.predicted_types || [],
        outProb: out ? (out.type_probs ? Math.max(...Object.values(out.type_probs)) : out.edge_exists_prob ?? 0) : 0,
        inTypes: inc?.predicted_types || [],
        inProb: inc ? (inc.type_probs ? Math.max(...Object.values(inc.type_probs)) : inc.edge_exists_prob ?? 0) : 0,
        direction: out && inc ? "both" : out ? "out" : "in",
      };
    });
    return { connections, colorOf };
  }, [edges, file, layerOf, confByPath]);

  const cx = dims.w / 2;
  const cy = dims.h / 2;
  const R = Math.max(120, Math.min(dims.w, dims.h) / 2 - 110);
  const n = connections.length;

  return (
    <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
      <div ref={containerRef} style={{ flex: 1, position: "relative", minHeight: "460px" }}>
        <div style={{ padding: "10px 20px", fontSize: "12px", color: COLORS.subtext, borderBottom: `1px solid ${COLORS.border}` }}>
          step 3 of 3 \u00b7 {file} \u00b7 {n} connected file{n !== 1 ? "s" : ""}, each drawn in its own color
        </div>
        <svg width={dims.w} height={dims.h - 36} style={{ display: "block" }}>
          <defs>
            {connections.map((c) => (
              <marker key={c.path} id={`cm-${c.id}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={colorOf.get(c.path)} />
              </marker>
            ))}
          </defs>
          <g>
            {connections.map((c, i) => {
              const angle = (i / Math.max(1, n)) * Math.PI * 2 - Math.PI / 2;
              const x = cx + R * Math.cos(angle);
              const y = cy - 18 + R * Math.sin(angle);
              const color = colorOf.get(c.path) as string;
              return (
                <path
                  key={c.path}
                  d={`M ${cx} ${cy - 18} L ${x} ${y}`}
                  stroke={color}
                  strokeWidth={2}
                  strokeOpacity={0.75}
                  markerEnd={c.direction !== "in" ? `url(#cm-${c.id})` : undefined}
                />
              );
            })}
          </g>
          <g>
            {connections.map((c, i) => {
              const angle = (i / Math.max(1, n)) * Math.PI * 2 - Math.PI / 2;
              const x = cx + R * Math.cos(angle);
              const y = cy - 18 + R * Math.sin(angle);
              const color = colorOf.get(c.path) as string;
              return (
                <g key={c.path} transform={`translate(${x},${y})`}>
                  <circle r={9} fill={color} fillOpacity={0.9} stroke={COLORS.bright} strokeWidth={1} />
                  <text textAnchor={Math.cos(angle) > 0.1 ? "start" : Math.cos(angle) < -0.1 ? "end" : "middle"} x={Math.cos(angle) > 0.1 ? 13 : Math.cos(angle) < -0.1 ? -13 : 0} dy={4} fontSize="10" fill={color}>
                    {c.id.length > 22 ? c.id.slice(0, 20) + "\u2026" : c.id}
                  </text>
                </g>
              );
            })}
          </g>
          <g>
            <circle cx={cx} cy={cy - 18} r={22} fill={COLORS.chip} stroke={COLORS.bright} strokeWidth={2} />
            <text x={cx} y={cy - 14} textAnchor="middle" fontSize="9" fill={COLORS.bright} fontWeight={600}>
              this file
            </text>
          </g>
        </svg>
      </div>

      <div style={{ width: "300px", borderLeft: `1px solid ${COLORS.border}`, overflowY: "auto", background: COLORS.panel, padding: "16px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: COLORS.bright, wordBreak: "break-all" }}>{file}</div>
        <div style={{ fontSize: "11px", color: COLORS.subtext, marginTop: "4px" }}>layer: {layerOf(file)}</div>
        <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.08em", color: COLORS.muted, margin: "16px 0 8px" }}>
          connections ({n})
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {connections.map((c) => (
            <div key={c.path} style={{ fontSize: "11px", background: COLORS.chip, borderRadius: "5px", padding: "6px 8px", borderLeft: `3px solid ${colorOf.get(c.path)}` }}>
              <div style={{ color: COLORS.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.id}</div>
              <div style={{ color: COLORS.muted, fontSize: "10px", marginTop: "2px" }}>
                {LAYER_META[c.layer].title} \u00b7 {c.direction === "both" ? "both directions" : c.direction === "out" ? "outgoing" : "incoming"}
                {c.direction !== "in" && ` \u00b7 out ${Math.round(c.outProb * 100)}%`}
                {c.direction !== "out" && ` \u00b7 in ${Math.round(c.inProb * 100)}%`}
              </div>
            </div>
          ))}
          {n === 0 && <div style={{ fontSize: "11px", color: COLORS.muted }}>no edges reference this file.</div>}
        </div>
      </div>
    </div>
  );
}

function ToolbarButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button onClick={onClick} style={{ background: COLORS.chip, color: COLORS.text, border: `1px solid ${COLORS.border}`, borderRadius: "6px", padding: "5px 9px", fontFamily: "inherit", fontSize: "11px", cursor: "pointer" }}>
      {label}
    </button>
  );
}