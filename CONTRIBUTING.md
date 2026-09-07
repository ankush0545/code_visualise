# Contributing to CodeGraph AI

Thank you for your interest in contributing to **CodeGraph AI**! We welcome contributions to improve our code analysis pipeline, extend tree-sitter language support, optimize graph neural network models, and enhance the visualization UI.

---

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/ankush0545/code_visualise.git
cd code_visualise
```

### 2. Python Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd code-graph
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the interactive web dashboard.

---

## Development Workflow

### Adding New Language Grammars
Language analyzers reside in `codegraph/treesitter_analyzers/`. To add support for a new language:
1. Ensure the language grammar is registered in `tree-sitter-language-pack`.
2. Implement an AST visitor that extracts:
   - File-level definitions (classes, interfaces, structs)
   - Function & method declarations
   - Imports and external package references
   - Function calls and variable references
3. Add test fixtures under `tests/`.

### Improving GNN & Edge Prediction
The graph neural network models reside in `Node_Classification/`:
- `feature_extraction.py`: Computes PageRank, clustering coefficients, degree centrality, and AST structural embeddings.
- `model.py`: PyTorch Geometric GraphSAGE & Edge existence/type heads.
- `predict.py`: Inference pipeline with configurable `--top-k` scoring.

### Frontend Enhancements
The web application is built with Next.js 15, React 19, TypeScript, and D3.js:
- `app/FileDataFlowMap.tsx`: Tiered architectural layer visualization (Frontend ➔ API ➔ Backend ➔ DB).
- `app/FileFlowExplorer.tsx`: D3 force-directed dependency and call network.
- `app/InsightsDashboard.tsx`: Machine learning benchmark metrics and taxonomy overview.

---

## Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(parser): add TypeScript decorator extraction support"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request on GitHub against `main`.

---

## Code Style & Standards

- **Python**: Follow PEP 8 guidelines. Type hints are encouraged where applicable.
- **TypeScript/React**: Ensure `npm run lint` and `npm run build` pass without warnings.
- **Git Hygiene**: Do not commit large model weights (>100MB) or generated binary caches.

---

## Questions or Feedback?
Feel free to open an issue or reach out directly on GitHub: [@ankush0545](https://github.com/ankush0545).
