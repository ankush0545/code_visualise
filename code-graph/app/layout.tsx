import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "CodeGraph AI | Intelligent Code Architecture & Graph Explorer",
  description: "Interactive software architecture visualization, Tree-sitter AST dataflow graphs, CodeBERT file-level role classification, and GraphSAGE GNN edge predictions.",
  keywords: ["CodeGraph", "Software Architecture", "GNN", "GraphSAGE", "CodeBERT", "Tree-sitter", "Code Analysis", "Next.js", "D3.js"],
  authors: [{ name: "Ankush Pal" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#070a0f] text-slate-100 min-h-screen selection:bg-cyan-500/20 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}