"""
CodeBERT fine-tuning script for Kaggle with improved handling of imbalanced data.

Key improvements:
- Focal Loss instead of CrossEntropy (with corrected pt calculation)
- Class-balanced sampling
- More epochs with early stopping
- Higher learning rate for minority classes
- Gradient accumulation for stability
- Robust evaluation metrics
- Fixes for the silent dataloader-worker deadlock that caused a 12h hang
  with zero log output:
    * TOKENIZERS_PARALLELISM disabled BEFORE the tokenizer is created
    * dataloader_num_workers=0 (no forked worker processes)
- More frequent logging + checkpointing so a hang is visible within minutes,
  not after a 12-hour session timeout
- Auto-resume from the latest checkpoint if the kernel restarts
"""

import os

# ============================================================
# MUST run before importing/instantiating the tokenizer.
# Prevents a fork-related deadlock in HF `tokenizers` when combined with
# multi-worker DataLoaders (this was the cause of the 12h silent hang).
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "microsoft/codebert-base"
# /kaggle/working persists for the life of the session and is what gets
# saved when you commit ("Save Version"). Use this as the single source
# of truth for checkpoints/outputs — don't write to /kaggle/temp.
MODEL_OUT = "/kaggle/working/codebert-file-classifier"

MAX_LEN = 512
TRAIN_SUBSAMPLE_FRACTION = 0.60  # use 60% of training data
FOCAL_LOSS_GAMMA = 2.0  # Focus on hard examples
CLASS_WEIGHT_TEMPERATURE = 0.5  # Smooth class weights to avoid over-penalization

os.makedirs(MODEL_OUT, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  No GPU detected. Check Settings -> Accelerator -> GPU T4 x2 (and your weekly quota).")

# ============================================================
# FIND DATASET AUTOMATICALLY
# ============================================================

label_files = glob.glob("/kaggle/input/**/labels.json", recursive=True)
if len(label_files) == 0:
    raise FileNotFoundError("labels.json not found inside /kaggle/input")

LABEL_FILE = label_files[0]
DATASET_DIR = os.path.dirname(LABEL_FILE)

print("\nDataset directory:", DATASET_DIR)
print("\nFiles inside dataset:", os.listdir(DATASET_DIR))

# ============================================================
# LOAD LABELS
# ============================================================

with open(LABEL_FILE, "r") as f:
    labels = json.load(f)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("\nLabels:", labels)
print(f"Number of classes: {len(labels)}")

# ============================================================
# LOAD DATASET
# ============================================================

train_file = os.path.join(DATASET_DIR, "train.jsonl")
val_file = os.path.join(DATASET_DIR, "val.jsonl")
test_file = os.path.join(DATASET_DIR, "test.jsonl")

for path in [train_file, val_file, test_file]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

dataset = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "validation": val_file,
        "test": test_file,
    },
)

print("\nDataset splits:", dataset)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)


def tokenize(batch):
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    encoded["labels"] = [label2id[label] for label in batch["label"]]
    return encoded


# num_proc parallelizes the .map() call itself (this is a different, safe
# kind of parallelism from the DataLoader worker issue below — it runs to
# completion and exits, it doesn't stay alive during training).
NUM_PROC = max(1, (os.cpu_count() or 2) - 1)

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=NUM_PROC,
)

# ============================================================
# SUBSAMPLE TRAINING DATA
# ============================================================

full_train_size = len(dataset["train"])
subsample_size = int(full_train_size * TRAIN_SUBSAMPLE_FRACTION)

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(subsample_size))

print(
    f"\nUsing {subsample_size} / {full_train_size} training examples "
    f"({TRAIN_SUBSAMPLE_FRACTION * 100:.0f}%)"
)

# ============================================================
# CLASS WEIGHTS WITH TEMPERATURE
# ============================================================

# Vectorized instead of a Python-level list comprehension over 250K+ rows
# (this was taking ~2+ minutes previously; direct column access is fast).
train_labels = np.array(dataset["train"]["labels"])
labels_present = sorted(set(train_labels.tolist()))

class_counts = np.bincount(train_labels, minlength=len(labels))
class_frequencies = class_counts / len(train_labels)

class_weights_np = 1.0 / (class_frequencies + 1e-8)
class_weights_np = class_weights_np ** CLASS_WEIGHT_TEMPERATURE
class_weights_np = class_weights_np / class_weights_np.mean()  # Normalize to mean=1

class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

print("\nClass weights (normalized):")
for i, weight in enumerate(class_weights):
    print(f"  {id2label[i]}: {weight:.3f}")

missing_labels = [id2label[i] for i in range(len(labels)) if i not in labels_present]
if len(missing_labels) > 0:
    print("\n⚠️ Missing labels from training split (after subsampling):", missing_labels)

# Track minority classes (smallest counts) for a meaningful f1_minority metric
# below — previously this was hardcoded to indices [0:5], which had nothing
# to do with which classes were actually rare.
minority_ids = np.argsort(class_counts)[:5]
print("Minority classes tracked in f1_minority metric:", [id2label[i] for i in minority_ids])

# ============================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    NOTE: pt must be computed from the *unweighted* cross-entropy. Passing
    `weight=` into F.cross_entropy and then doing exp(-ce_loss) gives
    p_t ** weight_t, not p_t — which corrupts the (1-pt)^gamma hard-example
    term. Class weighting is applied separately, after pt is computed.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # unweighted
        pt = torch.exp(-ce_loss)  # true p_t
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ============================================================
# MODEL
# ============================================================

model = RobertaForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)
model.to(device)

# ============================================================
# CUSTOM TRAINER WITH FOCAL LOSS
# ============================================================
# We only override compute_loss(), not training_step(), so this stays
# compatible with transformers versions that pass num_items_in_batch as a
# 4th positional arg to training_step().


class FocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        self.loss_fn = FocalLoss(alpha=self.class_weights, gamma=self.gamma, reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels_)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# METRICS - Focus on macro F1 for imbalanced data
# ============================================================


def compute_metrics(eval_pred):
    logits, labels_ = eval_pred
    predictions = np.argmax(logits, axis=-1)
    per_class_f1 = f1_score(labels_, predictions, average=None, labels=list(range(len(labels))))

    return {
        "accuracy": accuracy_score(labels_, predictions),
        "f1_macro": f1_score(labels_, predictions, average="macro"),
        "f1_weighted": f1_score(labels_, predictions, average="weighted"),
        "f1_minority": per_class_f1[minority_ids].mean(),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=MODEL_OUT,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    # Frequent logging so a hang is visible within minutes, not hours.
    logging_steps=20,
    logging_strategy="steps",
    fp16=torch.cuda.is_available(),
    # 0 = no forked worker processes. This is the fix for the silent
    # deadlock: forking DataLoader workers after HF `tokenizers`/`datasets`
    # have already used internal threading can hang forever with zero
    # error output. Slightly slower per-batch, but reliable.
    dataloader_num_workers=0,
    report_to="none",
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    optim="adamw_torch",
    adam_epsilon=1e-8,
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
)

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    gamma=FOCAL_LOSS_GAMMA,
    callbacks=[early_stopping_callback],
)

# ============================================================
# SANITY-CHECK RUN (strongly recommended for the first run after any
# code change, or after the earlier 12h hang)
# ============================================================
# Uncomment the block below, run once, and confirm you see logging_steps
# output within a couple of minutes before committing to a full run.
#
# training_args.max_steps = 30
# trainer.train()
# raise SystemExit("Sanity check complete — remove max_steps and rerun for full training.")

# ============================================================
# TRAIN (with auto-resume if a checkpoint already exists)
# ============================================================

print("\n🚀 Starting training with Focal Loss...\n")

# Kaggle interactive sessions reset /kaggle/working between "Edit" sessions
# in some cases, but NOT between cells within the same running session, and
# NOT when you use "Save Version" -> "Save & Run All (Commit)", which runs
# the whole notebook top-to-bottom in the background on Kaggle's servers —
# this is the Kaggle equivalent of Colab's background execution, and it's
# available on the free tier. Use Commit for long runs instead of keeping
# the interactive tab open.
#
# This resume check protects you if a commit run gets interrupted (e.g. by
# the 12h/9h session limit) and you need to continue from a re-run.
last_checkpoint = None
if os.path.isdir(MODEL_OUT):
    checkpoints = [d for d in os.listdir(MODEL_OUT) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(
            MODEL_OUT, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        )
        print(f"Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

# ============================================================
# EVALUATE ON TEST SET
# ============================================================

print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(dataset["test"])

print("\nTest results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# ============================================================
# DETAILED CLASSIFICATION REPORT
# ============================================================

print("\n📈 Detailed classification report:")

predictions = trainer.predict(dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

report = classification_report(true_labels, pred_labels, target_names=labels, digits=3)
print(report)

# ============================================================
# SAVE MODEL
# ============================================================

trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

config = {
    "model": BASE_MODEL,
    "num_labels": len(labels),
    "labels": labels,
    "class_weights": class_weights.cpu().tolist(),
    "focal_loss_gamma": FOCAL_LOSS_GAMMA,
    "max_length": MAX_LEN,
    "training_epochs": training_args.num_train_epochs,
}

with open(os.path.join(MODEL_OUT, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Model saved to: {MODEL_OUT}")
print("\n🏁 Training complete!")