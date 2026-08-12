"""
Trains a GNN encoder + linear classifier head on the graph built by
feature_extraction.py, to predict architecture-role labels
(database / frontend / infra / service / utility) for source files.

Run feature_extraction.py first to produce data.pt + label_meta.json.

Supports three interchangeable GNN encoders via `model_type`:
  - "sage" (GraphSAGE, default -- original behavior)
  - "gcn"  (GCNConv)
  - "gat"  (GATConv, multi-head attention)

Changes from the original version:
  - pos_weight is clamped to avoid rare classes destabilizing training.
  - Per-class thresholds are tuned on the validation set instead of a
    single fixed 0.5 cutoff, so evaluation matches the soft-label design
    (a file can genuinely have partial membership across several roles).
  - LayerNorm added between GNN layers for more stable convergence
    under class imbalance.
  - ReduceLROnPlateau scheduler added.
  - Removed unused `unknown_idx` variable.
  - Added a class-support check on train/val/test masks so a silently
    empty split for a rare class doesn't produce a misleading macro-F1.
  - NEW: GraphSAGEEncoder generalized to GNNEncoder(conv_type=...) so
    the same training loop can run GraphSAGE, GCN, or GAT.
  - NEW: checkpoints are now namespaced per model_type (finetuned_sage.pt,
    finetuned_gcn.pt, finetuned_gat.pt) so training one architecture no
    longer overwrites another's saved weights.
  - NEW: compare_architectures() trains all three back-to-back on the
    same data/masks and prints a macro-F1 comparison table.
"""

import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

from edge_label_extraction import sample_negative_edges, EDGE_TYPE_NAMES

# Seeds weight init / dropout so repeated runs on the same data.pt are
# reproducible. feature_extraction.py's stratified_split() was already
# seeded (seed=42); this was the one remaining unseeded source of
# run-to-run variance in reported metrics.
torch.manual_seed(42)


# Soft-label cutoff used consistently everywhere we need a binary "does
# this file have this role" decision for reporting purposes (pos/neg
# counts, support checks). This is NOT the classifier's decision
# threshold -- that's tuned per-class in tune_thresholds().
SOFT_LABEL_CUTOFF = 0.3

# Hard cap on BCEWithLogitsLoss's pos_weight. Without this, a class with
# e.g. 2 positives out of 300 training nodes gets pos_weight ~150, which
# lets that one class's loss dominate the gradient and destabilize the
# whole model instead of just gently upweighting it.
MAX_POS_WEIGHT = 20.0

# Same rationale as MAX_POS_WEIGHT, applied to the edge-type head: a rare
# edge type (e.g. very few FLOWS_TO edges relative to CALLS in a given
# repo) shouldn't be allowed to dominate the gradient.
MAX_EDGE_POS_WEIGHT = 20.0

VALID_MODEL_TYPES = ("sage", "gcn", "gat")


def _build_conv(conv_type, in_channels, out_channels, heads=4, is_final=False):
    """Returns a single graph-conv layer for the requested architecture.

    GAT needs special handling: intermediate layers concatenate `heads`
    attention heads, so each head must output out_channels/heads to keep
    the layer's total output width equal to out_channels (matching what
    SAGE/GCN produce in one shot). The final layer instead averages heads
    (concat=False) so the classifier head always sees a plain
    `out_channels`-wide vector regardless of architecture.
    """
    if conv_type == "sage":
        return SAGEConv(in_channels, out_channels)
    elif conv_type == "gcn":
        return GCNConv(in_channels, out_channels)
    elif conv_type == "gat":
        if is_final:
            return GATConv(in_channels, out_channels, heads=1, concat=False)
        if out_channels % heads != 0:
            raise ValueError(
                f"GAT hidden width ({out_channels}) must be divisible by "
                f"heads ({heads}) so per-head outputs concatenate back to "
                f"out_channels."
            )
        return GATConv(in_channels, out_channels // heads, heads=heads, concat=True)
    else:
        raise ValueError(
            f"Unknown conv_type '{conv_type}'. Expected one of {VALID_MODEL_TYPES}."
        )


class GNNEncoder(torch.nn.Module):
    """Generalized GraphSAGE / GCN / GAT encoder.

    Same architecture shape as the original GraphSAGEEncoder (stacked
    conv -> LayerNorm -> ReLU -> dropout, final layer left un-normalized
    for the classifier head) -- just parameterized on conv_type so all
    three GNN variants share one training loop, checkpointing scheme,
    and evaluation code instead of three near-duplicate files.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, conv_type="sage", heads=4):
        super().__init__()
        if conv_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown conv_type '{conv_type}'. Expected one of {VALID_MODEL_TYPES}."
            )
        self.conv_type = conv_type
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(_build_conv(conv_type, in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(_build_conv(conv_type, hidden_channels, hidden_channels, heads=heads))
        self.convs.append(_build_conv(conv_type, hidden_channels, out_channels, heads=heads, is_final=True))

        # One LayerNorm per non-final layer (applied after conv, before
        # relu+dropout). LayerNorm is generally more stable than
        # BatchNorm on graph data since node-batch statistics can be
        # noisy/small relative to CV/NLP batch sizes. This matters even
        # more for GCN/GAT than SAGE, since raw structural features
        # (degree counts, betweenness) are unnormalized upstream -- see
        # feature_extraction.py notes.
        for _ in range(num_layers - 1):
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class EdgeHead(torch.nn.Module):
    """Small MLP over pair features [z_u, z_v, z_u*z_v] -> per-class
    logits. Same module shape serves both heads -- existence
    (out_dim=1) and type (out_dim=len(EDGE_TYPE_NAMES)) -- just with a
    different output width, so they're two instances of this one class
    rather than two bespoke architectures.

    z_u*z_v (Hadamard product) is included alongside the raw concat
    because it gives the MLP an explicit "similarity" signal cheaply --
    standard in GNN link-prediction setups, and empirically stabilizes
    early training compared to concat alone.
    """

    def __init__(self, embed_dim, out_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, z, edge_index):
        u, v = edge_index[0], edge_index[1]
        pair = torch.cat([z[u], z[v], z[u] * z[v]], dim=-1)
        return self.net(pair)


def _binary_f1(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _multilabel_macro_f1(logits, target, num_classes, threshold=0.5):
    probs = torch.sigmoid(logits)
    f1s = []
    for c in range(num_classes):
        pred_c = (probs[:, c] >= threshold).float()
        tp = (pred_c * target[:, c]).sum().item()
        fp = (pred_c * (1 - target[:, c])).sum().item()
        fn = ((1 - pred_c) * target[:, c]).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _check_class_support(y_soft, masks, class_names, cutoff=SOFT_LABEL_CUTOFF):
    """Warns if any class has zero positive examples in any split. A
    silently empty split for a rare class produces a misleading
    support=0 row in the report and can skew macro-F1 without any
    visible error."""
    for split_name, mask in masks.items():
        binary = (y_soft[mask] >= cutoff).float()
        counts = binary.sum(dim=0)
        for c, count in zip(class_names, counts.tolist()):
            if count == 0:
                print(
                    f"WARNING: class '{c}' has 0 positive examples in "
                    f"the {split_name} split (cutoff={cutoff}). Macro-F1 "
                    f"for this split will be unreliable for this class."
                )


def modelling(data, meta, model_type="sage", hidden_channels=128, out_channels=64,
              num_layers=2, heads=4, verbose=True):
    """Trains one GNN architecture (model_type in {"sage","gcn","gat"}) on
    `data`/`meta` and returns a results dict:
        {model_type, macro_f1, tuned_thresholds, known_label_names}
    so callers (e.g. compare_architectures) can compare runs without
    re-parsing stdout.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected one of {VALID_MODEL_TYPES}."
        )

    label_names = meta["label_names"]
    label_to_idx = meta["label_to_idx"]

    # Classifier only needs to predict *known* classes -- "unknown" nodes are
    # never a training/eval target (see train_mask/val_mask/test_mask).
    known_label_names = [n for n in label_names if n != "unknown"]
    num_classes = len(known_label_names)

    # SOFT multi-label targets (built in feature_extraction.py from
    # labelFunction.label_file()'s normalized `scores` distribution).
    # Columns are in the same order as known_label_names.
    # A file can have partial membership in multiple roles at once
    # instead of being forced into a single winning class.
    y_soft = data.y_soft

    feat_dim = data.x.size(1)

    _check_class_support(
        y_soft,
        {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        known_label_names,
    )

    # Per-class positive/negative balance from *training* targets only, to
    # counter imbalance (e.g. infra has very few positive examples). This
    # is the multi-label equivalent of the old cross-entropy class_weights:
    # BCEWithLogitsLoss's pos_weight upweights the loss on rarer classes.
    # Capped at MAX_POS_WEIGHT to avoid a single rare class dominating
    # the gradient and destabilizing training.
    train_targets_soft = y_soft[data.train_mask]
    train_targets_binary = (train_targets_soft >= SOFT_LABEL_CUTOFF).float()
    pos_counts = train_targets_binary.sum(dim=0).clamp(min=1.0)
    neg_counts = (train_targets_binary.shape[0] - pos_counts).clamp(min=1.0)
    pos_weight = (neg_counts / pos_counts).clamp(max=MAX_POS_WEIGHT)

    num_clamped = (neg_counts / pos_counts > MAX_POS_WEIGHT).sum().item()
    if verbose and num_clamped > 0:
        print(
            f"NOTE: {num_clamped} class(es) hit the pos_weight cap of "
            f"{MAX_POS_WEIGHT}. This means they have very few positive "
            f"training examples -- consider collecting more data for "
            f"those classes rather than relying on loss weighting alone."
        )

    if verbose:
        print(f"\n{'='*60}\nTraining model_type='{model_type}'\n{'='*60}")
        print("Known classes:", known_label_names)
        print("Train positive counts:", pos_counts.tolist())
        print("Pos weight (for BCEWithLogitsLoss, capped):", pos_weight.tolist())

    encoder = GNNEncoder(
        in_channels=feat_dim, hidden_channels=hidden_channels,
        out_channels=out_channels, num_layers=num_layers,
        conv_type=model_type, heads=heads,
    )
    classifier = torch.nn.Linear(out_channels, num_classes)

    # Checkpoints are namespaced per model_type so training GCN doesn't
    # overwrite a previously trained SAGE (or GAT) checkpoint, and each
    # architecture can independently resume from its own best weights.
    encoder_ckpt = f"finetuned_{model_type}.pt"
    classifier_ckpt = f"finetuned_{model_type}_classifier.pt"
    if os.path.exists(encoder_ckpt) and os.path.exists(classifier_ckpt):
        try:
            encoder.load_state_dict(torch.load(encoder_ckpt))
            classifier.load_state_dict(torch.load(classifier_ckpt))
            if verbose:
                print(f"Resumed training from existing checkpoints: "
                      f"{encoder_ckpt}, {classifier_ckpt}")
        except RuntimeError as e:
            if verbose:
                print(f"WARNING: could not load existing checkpoints "
                      f"({e}). Falling back to fresh initialization.")
    else:
        if verbose:
            print("No existing checkpoints found -- starting from fresh initialization.")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3, weight_decay=5e-4,
    )

    # Decays LR once validation macro-F1 plateaus, instead of a flat
    # lr=1e-3 for the whole run. Often squeezes out extra F1 in the
    # final epochs without any architecture change.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    # BCEWithLogitsLoss = sigmoid + binary cross entropy per class,
    # independent decisions instead of softmax's forced competition.
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_step():
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)
        out = classifier(z[data.train_mask])
        loss = criterion(out, y_soft[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_probs(mask):
        encoder.eval()
        classifier.eval()
        z = encoder(data.x, data.edge_index)
        return torch.sigmoid(classifier(z[mask]))

    @torch.no_grad()
    def evaluate(mask, thresholds=None):
        """Returns macro-F1 over the given mask. `thresholds` is either
        None (uses a flat 0.5 for a quick check, e.g. during early
        stopping) or a per-class list/tensor of tuned thresholds."""
        probs = get_probs(mask)
        target_binary = (y_soft[mask] >= SOFT_LABEL_CUTOFF).float()

        if thresholds is None:
            thresholds = [0.5] * num_classes

        f1s = []
        for c in range(num_classes):
            pred_c = (probs[:, c] >= thresholds[c]).float()
            tp = (pred_c * target_binary[:, c]).sum().item()
            fp = (pred_c * (1 - target_binary[:, c])).sum().item()
            fn = ((1 - pred_c) * target_binary[:, c]).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0

    @torch.no_grad()
    def tune_thresholds(mask, steps=17):
        """Sweeps a threshold per class on `mask` and returns the
        threshold that maximizes that class's F1. Fixes the mismatch
        between soft multi-label targets and a single global 0.5 cutoff:
        rare/ambiguous classes usually want a different threshold than
        dominant ones."""
        probs = get_probs(mask)
        target_binary = (y_soft[mask] >= SOFT_LABEL_CUTOFF).float()

        best_thresholds = []
        for c in range(num_classes):
            best_f1, best_t = 0.0, 0.5
            for t in torch.linspace(0.1, 0.9, steps):
                pred = (probs[:, c] >= t).float()
                tp = (pred * target_binary[:, c]).sum().item()
                fp = (pred * (1 - target_binary[:, c])).sum().item()
                fn = ((1 - pred) * target_binary[:, c]).sum().item()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t.item()
            best_thresholds.append(best_t)
        return best_thresholds

    @torch.no_grad()
    def predict_for_report(mask, thresholds):
        probs = get_probs(mask)
        target_binary = (y_soft[mask] >= SOFT_LABEL_CUTOFF).float()
        pred_binary = torch.zeros_like(probs)
        for c in range(num_classes):
            pred_binary[:, c] = (probs[:, c] >= thresholds[c]).float()
        return pred_binary, target_binary

    def print_multilabel_report(pred_binary, target_binary, class_names, thresholds):
        col_w = max(len(c) for c in class_names) + 2
        print(f"\nMulti-label report (per-class tuned thresholds, each class scored independently)")
        print(f"{'class':<{col_w}}{'thresh':>8}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}")
        f1s = []
        for i, c in enumerate(class_names):
            tp = (pred_binary[:, i] * target_binary[:, i]).sum().item()
            fp = (pred_binary[:, i] * (1 - target_binary[:, i])).sum().item()
            fn = ((1 - pred_binary[:, i]) * target_binary[:, i]).sum().item()
            support = target_binary[:, i].sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
            print(f"{c:<{col_w}}{thresholds[i]:>8.2f}{precision:>12.3f}{recall:>12.3f}{f1:>12.3f}{support:>10.0f}")

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        print(f"\nMacro F1 (unweighted avg across classes): {macro_f1:.3f}")
        print("(Macro F1 weights every class equally, so it won't hide a class")
        print(" being ignored the way overall accuracy can with imbalanced data.")
        print(" Rows/cols here are independent per-class scores -- a file can")
        print(" contribute to multiple classes' TP/FP/FN counts at once.)")
        return macro_f1

    best_val_f1 = 0.0
    patience, patience_counter = 30, 0

    for epoch in range(1, 301):
        loss = train_step()
        # Flat 0.5 threshold here is just a cheap proxy for early-stopping
        # comparisons across epochs -- the real per-class thresholds are
        # tuned once at the end, after the best checkpoint is restored.
        val_f1 = evaluate(data.val_mask)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(encoder.state_dict(), encoder_ckpt)
            torch.save(classifier.state_dict(), classifier_ckpt)
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Macro F1: {val_f1:.4f} | LR: {current_lr:.2e}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    encoder.load_state_dict(torch.load(encoder_ckpt))
    classifier.load_state_dict(torch.load(classifier_ckpt))

    # Tune per-class thresholds on val (never on test), then apply those
    # fixed thresholds to report final test performance.
    tuned_thresholds = tune_thresholds(data.val_mask)
    if verbose:
        print("\nTuned per-class thresholds (from val set):")
        for name, t in zip(known_label_names, tuned_thresholds):
            print(f"  {name:<12} {t:.2f}")

    test_pred, test_target = predict_for_report(data.test_mask, tuned_thresholds)
    if verbose:
        print(f"\nBest Val Macro F1 (flat 0.5 threshold, for early-stopping comparability): {best_val_f1:.4f}")
    test_macro_f1 = print_multilabel_report(test_pred, test_target, known_label_names, tuned_thresholds) \
        if verbose else _silent_macro_f1(test_pred, test_target, num_classes)

    with open(f"known_label_names_{model_type}.json", "w") as f:
        json.dump(known_label_names, f)

    with open(f"tuned_thresholds_{model_type}.json", "w") as f:
        json.dump(dict(zip(known_label_names, tuned_thresholds)), f)

    return {
        "model_type": model_type,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_macro_f1,
        "tuned_thresholds": dict(zip(known_label_names, tuned_thresholds)),
        "known_label_names": known_label_names,
    }


def modelling_multitask(data, meta, neg_val_edge_index, neg_test_edge_index,
                         model_type="sage", hidden_channels=128, out_channels=64,
                         num_layers=2, heads=4,
                         w_node=1.0, w_exist=1.0, w_type=1.0, verbose=True):
    """Trains one shared GNN encoder with three heads jointly:
        - node classifier   (unchanged from modelling(): file-type roles)
        - edge existence    (does a File->File dataflow edge exist)
        - edge type         (CALLS / FLOWS_TO, multi-label)

    `neg_val_edge_index` / `neg_test_edge_index` come from
    edge_label_extraction.build_edge_labels() -- fixed negative pairs so
    existence-head evaluation is stable across epochs. Training negatives
    are instead resampled fresh every step via sample_negative_edges().

    Returns a results dict with per-task metrics, mirroring modelling()'s
    shape so callers can log/compare the same way.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected one of {VALID_MODEL_TYPES}."
        )

    label_names = meta["label_names"]
    known_label_names = [n for n in label_names if n != "unknown"]
    num_classes = len(known_label_names)
    num_edge_types = len(EDGE_TYPE_NAMES)

    y_soft = data.y_soft
    feat_dim = data.x.size(1)
    num_nodes = data.x.size(0)

    has_edge_task = data.file_edge_index.size(1) > 0
    if not has_edge_task and verbose:
        print("NOTE: no file-to-file dataflow edges for this repo -- "
              "training node task only, edge losses will be 0.")

    _check_class_support(
        y_soft,
        {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        known_label_names,
    )

    # ---- node task pos_weight (unchanged from modelling()) ----------
    train_targets_soft = y_soft[data.train_mask]
    train_targets_binary = (train_targets_soft >= SOFT_LABEL_CUTOFF).float()
    pos_counts = train_targets_binary.sum(dim=0).clamp(min=1.0)
    neg_counts = (train_targets_binary.shape[0] - pos_counts).clamp(min=1.0)
    node_pos_weight = (neg_counts / pos_counts).clamp(max=MAX_POS_WEIGHT)

    # ---- edge type pos_weight, same rationale, own cap --------------
    if has_edge_task:
        train_edge_type = data.file_edge_type[data.file_edge_train_mask]
        edge_pos_counts = train_edge_type.sum(dim=0).clamp(min=1.0)
        edge_neg_counts = (train_edge_type.shape[0] - edge_pos_counts).clamp(min=1.0)
        edge_type_pos_weight = (edge_neg_counts / edge_pos_counts).clamp(max=MAX_EDGE_POS_WEIGHT)
    else:
        edge_type_pos_weight = torch.ones(num_edge_types)

    if verbose:
        print(f"\n{'='*60}\nTraining multi-task model_type='{model_type}'\n{'='*60}")
        print("Known node classes:", known_label_names)
        print("Edge type classes:", EDGE_TYPE_NAMES)
        print("Node pos weight:", node_pos_weight.tolist())
        if has_edge_task:
            print("Edge type pos weight:", edge_type_pos_weight.tolist())

    encoder = GNNEncoder(
        in_channels=feat_dim, hidden_channels=hidden_channels,
        out_channels=out_channels, num_layers=num_layers,
        conv_type=model_type, heads=heads,
    )
    classifier = torch.nn.Linear(out_channels, num_classes)
    exist_head = EdgeHead(out_channels, 1)
    type_head = EdgeHead(out_channels, num_edge_types)

    # Namespaced per model_type, same convention as modelling(). Edge
    # heads get their own checkpoint files so they don't collide with
    # (or get accidentally loaded by) the single-task modelling() run's
    # encoder/classifier checkpoints -- those two functions should not
    # silently share weights unless you explicitly want that.
    encoder_ckpt = f"finetuned_{model_type}_mt.pt"
    classifier_ckpt = f"finetuned_{model_type}_mt_classifier.pt"
    exist_ckpt = f"finetuned_{model_type}_mt_exist.pt"
    type_ckpt = f"finetuned_{model_type}_mt_type.pt"
    ckpts = [encoder_ckpt, classifier_ckpt, exist_ckpt, type_ckpt]
    if all(os.path.exists(c) for c in ckpts):
        try:
            encoder.load_state_dict(torch.load(encoder_ckpt))
            classifier.load_state_dict(torch.load(classifier_ckpt))
            exist_head.load_state_dict(torch.load(exist_ckpt))
            type_head.load_state_dict(torch.load(type_ckpt))
            if verbose:
                print(f"Resumed multi-task training from existing checkpoints.")
        except RuntimeError as e:
            if verbose:
                print(f"WARNING: could not load existing multi-task checkpoints "
                      f"({e}). Falling back to fresh initialization.")
    else:
        if verbose:
            print("No existing multi-task checkpoints found -- starting fresh.")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters())
        + list(exist_head.parameters()) + list(type_head.parameters()),
        lr=1e-3, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    node_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=node_pos_weight)
    exist_criterion = torch.nn.BCEWithLogitsLoss()  # ~1:1 by construction (NEG_RATIO), no pos_weight needed
    type_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=edge_type_pos_weight)

    train_pos_edge_index = data.file_edge_index[:, data.file_edge_train_mask] if has_edge_task \
        else torch.zeros((2, 0), dtype=torch.long)
    all_positive_pairs = {
        (u.item(), v.item()) for u, v in zip(*data.file_edge_index)
    } if has_edge_task else set()

    def train_step():
        encoder.train(); classifier.train(); exist_head.train(); type_head.train()
        optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)

        node_out = classifier(z[data.train_mask])
        node_loss = node_criterion(node_out, y_soft[data.train_mask])

        if has_edge_task and train_pos_edge_index.size(1) > 0:
            neg_edge_index = sample_negative_edges(
                num_nodes, all_positive_pairs,
                train_pos_edge_index.size(1), seed=None,  # fresh negatives every step
            )
            exist_edge_index = torch.cat([train_pos_edge_index, neg_edge_index], dim=1)
            exist_target = torch.cat([
                torch.ones(train_pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1)),
            ])
            exist_logits = exist_head(z, exist_edge_index).squeeze(-1)
            exist_loss = exist_criterion(exist_logits, exist_target)

            type_logits = type_head(z, train_pos_edge_index)
            type_loss = type_criterion(type_logits, data.file_edge_type[data.file_edge_train_mask])
        else:
            exist_loss = torch.tensor(0.0)
            type_loss = torch.tensor(0.0)

        loss = w_node * node_loss + w_exist * exist_loss + w_type * type_loss
        loss.backward()
        optimizer.step()
        return loss.item(), node_loss.item(), float(exist_loss), float(type_loss)

    @torch.no_grad()
    def evaluate(node_mask, edge_mask, neg_edge_index):
        encoder.eval(); classifier.eval(); exist_head.eval(); type_head.eval()
        z = encoder(data.x, data.edge_index)

        node_target = (y_soft[node_mask] >= SOFT_LABEL_CUTOFF).float()
        node_f1 = _multilabel_macro_f1(classifier(z[node_mask]), node_target, num_classes)

        if not has_edge_task or edge_mask.sum() == 0:
            return node_f1, 0.0, 0.0

        pos_edges = data.file_edge_index[:, edge_mask]
        exist_edge_index = torch.cat([pos_edges, neg_edge_index], dim=1)
        exist_target = torch.cat([
            torch.ones(pos_edges.size(1)), torch.zeros(neg_edge_index.size(1)),
        ])
        exist_logits = exist_head(z, exist_edge_index).squeeze(-1)
        exist_f1 = _binary_f1(exist_logits, exist_target)

        type_logits = type_head(z, pos_edges)
        type_target = data.file_edge_type[edge_mask]
        type_f1 = _multilabel_macro_f1(type_logits, type_target, num_edge_types)

        return node_f1, exist_f1, type_f1

    best_combined = 0.0
    patience, patience_counter = 30, 0

    for epoch in range(1, 301):
        loss, node_loss, exist_loss, type_loss = train_step()
        node_f1, exist_f1, type_f1 = evaluate(data.val_mask, data.file_edge_val_mask, neg_val_edge_index)
        combined = (node_f1 + exist_f1 + type_f1) / (3 if has_edge_task else 1)
        scheduler.step(combined)

        if combined > best_combined:
            best_combined = combined
            patience_counter = 0
            torch.save(encoder.state_dict(), encoder_ckpt)
            torch.save(classifier.state_dict(), classifier_ckpt)
            torch.save(exist_head.state_dict(), exist_ckpt)
            torch.save(type_head.state_dict(), type_ckpt)
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} (node {node_loss:.4f}, "
                  f"exist {exist_loss:.4f}, type {type_loss:.4f}) | "
                  f"Val F1 -- node: {node_f1:.4f} exist: {exist_f1:.4f} type: {type_f1:.4f} | LR: {lr:.2e}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    encoder.load_state_dict(torch.load(encoder_ckpt))
    classifier.load_state_dict(torch.load(classifier_ckpt))
    exist_head.load_state_dict(torch.load(exist_ckpt))
    type_head.load_state_dict(torch.load(type_ckpt))

    test_node_f1, test_exist_f1, test_type_f1 = evaluate(
        data.test_mask, data.file_edge_test_mask, neg_test_edge_index
    )
    if verbose:
        print(f"\nTest results -- node macro F1: {test_node_f1:.4f} | "
              f"edge existence F1: {test_exist_f1:.4f} | "
              f"edge type macro F1: {test_type_f1:.4f}")

    return {
        "model_type": model_type,
        "known_label_names": known_label_names,
        "edge_type_names": EDGE_TYPE_NAMES,
        "test_node_macro_f1": test_node_f1,
        "test_edge_existence_f1": test_exist_f1,
        "test_edge_type_macro_f1": test_type_f1,
        "has_edge_task": has_edge_task,
    }


def modelling_edges(data, neg_val_edge_index, neg_test_edge_index,
                     model_type="sage", hidden_channels=128, out_channels=64,
                     num_layers=2, heads=4, w_exist=1.0, w_type=1.0, verbose=True,
                     checkpoint_path=None):
    """Edge-classification-only training: one shared GNN encoder with
    TWO heads -- edge existence (does a File->File dataflow edge exist)
    and edge type (CALLS / FLOWS_TO, multi-label). No node classifier,
    no node loss, no node evaluation at all -- this is the leaner
    counterpart to modelling_multitask() for when you only care about
    the flow/relationship between files, not per-file role labels.

    `data` only needs data.x / data.edge_index (from
    Label_extraction.build_graph_data()) plus the file_edge_* attributes
    attached by edge_label_extraction.build_edge_labels(). No y_soft and
    no train/val/test node masks are required or used.

    `neg_val_edge_index` / `neg_test_edge_index` come from
    build_edge_labels() -- fixed negative pairs so existence-head
    evaluation is stable across epochs. Training negatives are instead
    resampled fresh every step via sample_negative_edges().

    ONE MODEL ACROSS BATCHES: all three sub-modules (encoder, existence
    head, type head) are saved to and loaded from a SINGLE checkpoint
    file (`checkpoint_path`, default `finetuned_{model_type}_edges.pt`).
    If you're looping this call once per repo (each repo = one training
    batch), every call resumes from that same file and writes back to
    it -- so the model keeps accumulating updates across repos instead
    of each repo training its own separate, throwaway model. Pass an
    explicit `checkpoint_path` only if you deliberately want a
    different/fresh model lineage.

    Returns a results dict with edge-only metrics.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected one of {VALID_MODEL_TYPES}."
        )

    num_edge_types = len(EDGE_TYPE_NAMES)
    feat_dim = data.x.size(1)
    num_nodes = data.x.size(0)

    if data.file_edge_index.size(1) == 0:
        raise ValueError(
            "No file-to-file dataflow edges were found for this repo -- "
            "edge-only training has nothing to learn from. Check that "
            "edge_label_extraction.build_edge_labels() found CALLS/FLOWS_TO "
            "edges upstream before calling modelling_edges()."
        )

    # Per-class positive/negative balance from *training* edge targets
    # only, same rationale/cap as modelling_multitask()'s edge_type_pos_weight.
    train_edge_type = data.file_edge_type[data.file_edge_train_mask]
    edge_pos_counts = train_edge_type.sum(dim=0).clamp(min=1.0)
    edge_neg_counts = (train_edge_type.shape[0] - edge_pos_counts).clamp(min=1.0)
    edge_type_pos_weight = (edge_neg_counts / edge_pos_counts).clamp(max=MAX_EDGE_POS_WEIGHT)

    if verbose:
        print(f"\n{'='*60}\nTraining edge-only model_type='{model_type}'\n{'='*60}")
        print("Edge type classes:", EDGE_TYPE_NAMES)
        print("Edge type pos weight:", edge_type_pos_weight.tolist())

    encoder = GNNEncoder(
        in_channels=feat_dim, hidden_channels=hidden_channels,
        out_channels=out_channels, num_layers=num_layers,
        conv_type=model_type, heads=heads,
    )
    exist_head = EdgeHead(out_channels, 1)
    type_head = EdgeHead(out_channels, num_edge_types)

    # ONE combined checkpoint file for the whole edge-only model (encoder
    # + both heads), shared across every call/batch/repo -- not one file
    # per sub-module and not one set of files per repo. This is what
    # lets a loop over many repos keep fine-tuning the SAME model instead
    # of starting fresh (or colliding on filenames) each time.
    ckpt_path = checkpoint_path or f"finetuned_{model_type}_edges.pt"
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path)
            encoder.load_state_dict(checkpoint["encoder"])
            exist_head.load_state_dict(checkpoint["exist_head"])
            type_head.load_state_dict(checkpoint["type_head"])
            if verbose:
                print(f"Resumed edge-only training from '{ckpt_path}' -- "
                      f"continuing to fine-tune the same model.")
        except (RuntimeError, KeyError) as e:
            if verbose:
                print(f"WARNING: could not load existing checkpoint '{ckpt_path}' "
                      f"({e}). Falling back to fresh initialization.")
    else:
        if verbose:
            print(f"No existing checkpoint at '{ckpt_path}' -- starting fresh.")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(exist_head.parameters()) + list(type_head.parameters()),
        lr=1e-3, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    exist_criterion = torch.nn.BCEWithLogitsLoss()  # ~1:1 by construction (NEG_RATIO)
    type_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=edge_type_pos_weight)

    train_pos_edge_index = data.file_edge_index[:, data.file_edge_train_mask]
    all_positive_pairs = {
        (u.item(), v.item()) for u, v in zip(*data.file_edge_index)
    }

    def train_step():
        encoder.train(); exist_head.train(); type_head.train()
        optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)

        neg_edge_index = sample_negative_edges(
            num_nodes, all_positive_pairs,
            train_pos_edge_index.size(1), seed=None,  # fresh negatives every step
        )
        exist_edge_index = torch.cat([train_pos_edge_index, neg_edge_index], dim=1)
        exist_target = torch.cat([
            torch.ones(train_pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1)),
        ])
        exist_logits = exist_head(z, exist_edge_index).squeeze(-1)
        exist_loss = exist_criterion(exist_logits, exist_target)

        type_logits = type_head(z, train_pos_edge_index)
        type_loss = type_criterion(type_logits, data.file_edge_type[data.file_edge_train_mask])

        loss = w_exist * exist_loss + w_type * type_loss
        loss.backward()
        optimizer.step()
        return loss.item(), float(exist_loss), float(type_loss)

    @torch.no_grad()
    def evaluate(edge_mask, neg_edge_index):
        encoder.eval(); exist_head.eval(); type_head.eval()
        z = encoder(data.x, data.edge_index)

        if edge_mask.sum() == 0:
            return 0.0, 0.0

        pos_edges = data.file_edge_index[:, edge_mask]
        exist_edge_index = torch.cat([pos_edges, neg_edge_index], dim=1)
        exist_target = torch.cat([
            torch.ones(pos_edges.size(1)), torch.zeros(neg_edge_index.size(1)),
        ])
        exist_logits = exist_head(z, exist_edge_index).squeeze(-1)
        exist_f1 = _binary_f1(exist_logits, exist_target)

        type_logits = type_head(z, pos_edges)
        type_target = data.file_edge_type[edge_mask]
        type_f1 = _multilabel_macro_f1(type_logits, type_target, num_edge_types)

        return exist_f1, type_f1

    def save_checkpoint():
        torch.save({
            "encoder": encoder.state_dict(),
            "exist_head": exist_head.state_dict(),
            "type_head": type_head.state_dict(),
        }, ckpt_path)

    best_combined = 0.0
    patience, patience_counter = 30, 0

    for epoch in range(1, 301):
        loss, exist_loss, type_loss = train_step()
        exist_f1, type_f1 = evaluate(data.file_edge_val_mask, neg_val_edge_index)
        combined = (exist_f1 + type_f1) / 2
        scheduler.step(combined)

        if combined > best_combined:
            best_combined = combined
            patience_counter = 0
            save_checkpoint()
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} (exist {exist_loss:.4f}, "
                  f"type {type_loss:.4f}) | Val F1 -- exist: {exist_f1:.4f} type: {type_f1:.4f} | LR: {lr:.2e}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Always leave a checkpoint behind even if this batch's val F1 never
    # beat 0.0 (e.g. a very small repo) -- otherwise the next repo in the
    # loop would find no file and start over instead of continuing.
    if not os.path.exists(ckpt_path):
        save_checkpoint()

    checkpoint = torch.load(ckpt_path)
    encoder.load_state_dict(checkpoint["encoder"])
    exist_head.load_state_dict(checkpoint["exist_head"])
    type_head.load_state_dict(checkpoint["type_head"])

    test_exist_f1, test_type_f1 = evaluate(data.file_edge_test_mask, neg_test_edge_index)
    if verbose:
        print(f"\nTest results -- edge existence F1: {test_exist_f1:.4f} | "
              f"edge type macro F1: {test_type_f1:.4f}")

    return {
        "model_type": model_type,
        "edge_type_names": EDGE_TYPE_NAMES,
        "test_edge_existence_f1": test_exist_f1,
        "test_edge_type_macro_f1": test_type_f1,
        "checkpoint_path": ckpt_path,
    }



def _silent_macro_f1(pred_binary, target_binary, num_classes):
    """Same math as print_multilabel_report's macro-F1, without printing --
    used when verbose=False (e.g. inside compare_architectures' loop)."""
    f1s = []
    for i in range(num_classes):
        tp = (pred_binary[:, i] * target_binary[:, i]).sum().item()
        fp = (pred_binary[:, i] * (1 - target_binary[:, i])).sum().item()
        fn = ((1 - pred_binary[:, i]) * target_binary[:, i]).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compare_architectures(data, meta, architectures=VALID_MODEL_TYPES, **kwargs):
    """Trains each architecture in `architectures` on the same data/masks
    (each gets its own namespaced checkpoints, so runs don't interfere)
    and prints a side-by-side macro-F1 comparison table. Returns the list
    of result dicts sorted best-first by test macro-F1.

    Usage:
        results = compare_architectures(data, meta)
    """
    results = []
    for model_type in architectures:
        result = modelling(data, meta, model_type=model_type, **kwargs)
        results.append(result)

    results.sort(key=lambda r: r["test_macro_f1"], reverse=True)

    print(f"\n{'='*60}\nArchitecture comparison (test set)\n{'='*60}")
    print(f"{'model_type':<12}{'val macro F1':>16}{'test macro F1':>16}")
    for r in results:
        print(f"{r['model_type']:<12}{r['best_val_macro_f1']:>16.4f}{r['test_macro_f1']:>16.4f}")
    print(f"\nBest: {results[0]['model_type']} (test macro F1 = {results[0]['test_macro_f1']:.4f})")

    return results