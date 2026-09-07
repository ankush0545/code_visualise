import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET() {
  try {
    // 1. Primary paths (from Python pipeline outputs)
    const primaryEdgesPath = path.join(
      process.cwd(),
      "..",
      "Final_Output",
      "output_predicted_edges.json"
    );
    const primaryPredsPath = path.join(
      process.cwd(),
      "..",
      "Final_Output",
      "Prediction_codebert.json"
    );

    // 2. Standalone / Demo fallback paths (in public/Data)
    const fallbackEdgesPath = path.join(
      process.cwd(),
      "public",
      "Data",
      "predicted_edges.json"
    );
    const fallbackPredsPath = path.join(
      process.cwd(),
      "public",
      "Data",
      "Prediction.json"
    );

    let edgesData: any[] = [];
    let predsData: any[] = [];
    let dataSource = "live_pipeline";

    // Try reading primary edges
    try {
      const raw = await fs.readFile(primaryEdgesPath, "utf-8");
      edgesData = JSON.parse(raw);
    } catch {
      try {
        const raw = await fs.readFile(fallbackEdgesPath, "utf-8");
        edgesData = JSON.parse(raw);
        dataSource = "demo_preset";
      } catch (e) {
        console.warn("Could not read edges from fallback either:", e);
      }
    }

    // Try reading primary predictions
    try {
      const raw = await fs.readFile(primaryPredsPath, "utf-8");
      predsData = JSON.parse(raw);
    } catch {
      try {
        const raw = await fs.readFile(fallbackPredsPath, "utf-8");
        predsData = JSON.parse(raw);
        dataSource = "demo_preset";
      } catch (e) {
        console.warn("Could not read predictions from fallback either:", e);
      }
    }

    // Generate summary statistics for dashboard metrics
    const layerCounts: Record<string, number> = {};
    let totalConf = 0;
    predsData.forEach((p: any) => {
      const label = p.predicted_label || "other";
      layerCounts[label] = (layerCounts[label] || 0) + 1;
      totalConf += p.confidence ?? 0;
    });

    const edgeTypeCounts: Record<string, number> = {};
    edgesData.forEach((e: any) => {
      (e.predicted_types || []).forEach((t: string) => {
        edgeTypeCounts[t] = (edgeTypeCounts[t] || 0) + 1;
      });
    });

    const stats = {
      totalEdges: edgesData.length,
      totalFiles: predsData.length,
      avgConfidence: predsData.length ? Math.round((totalConf / predsData.length) * 100) : 0,
      layerCounts,
      edgeTypeCounts,
      dataSource,
      status: "ready"
    };

    return NextResponse.json({
      edges: edgesData,
      predictions: predsData,
      stats,
    });
  } catch (error) {
    console.error("Failed to load graph data:", error);
    return NextResponse.json(
      {
        error: "Failed to load graph data",
        edges: [],
        predictions: [],
        stats: { totalEdges: 0, totalFiles: 0, avgConfidence: 0, status: "error" },
      },
      { status: 200 } // Return 200 with empty data so the client renders a friendly state
    );
  }
}