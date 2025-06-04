#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_sentence_fragment_classifier.py
"""
Fine-tune MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
to classify independent vs. fragment/dependent sentences,
using spaCy flags as numeric features concatenated to DeBERTa’s CLS embedding.

Data file expected:
Data/training_data/sentenceQC/validated_sentences_mix_orig_text_and_contextualized.tsv
with columns:
  • Text      (str) – the sentence
  • Complete  (bool / 1-0 / True-False) – True if FULL sentence

Outputs:
  • best model saved to ./sentence_frag_chkpt/
  • metrics printed to console
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import torch
import torch.nn.functional as F
import spacy

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
SEED = 42
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
DATA_PATH = Path(
    "Data/training_data/sentenceQC/validated_sentences_mix_orig_text_and_contextualized.tsv"
)
OUTPUT_DIR = Path("./sentence_frag_chkpt")
VAL_SIZE = 0.15  # 15 % for validation
TEST_SIZE = 0.15  # 15 % held-out test
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
set_seed(SEED)

# ---------------------------------------------------------------------
# 2. Load and prepare the dataset + compute spaCy flags
# ---------------------------------------------------------------------
# Load spaCy once
nlp = spacy.load("en_core_web_sm")
DEMO_PRONS = {"this", "that", "these", "those", "it", "they"}


def extract_spacy_flags(text: str):
    """
    Return (missing_verb_flag, dem_pron_flag).
      - missing_verb_flag = 1 if spaCy’s root POS is not VERB/AUX
      - dem_pron_flag    = 1 if a demonstrative pronoun appears without
                           an immediately preceding NOUN/PROPN/PRON
    """
    doc = nlp(text)
    # missing_verb_flag
    roots = [tok for tok in doc if tok.dep_ == "ROOT"]
    if roots:
        root = roots[0]
        missing_verb_flag = int(root.pos_ not in {"VERB", "AUX"})
    else:
        missing_verb_flag = 1
    # dem_pron_flag
    dem_pron_flag = 0
    for tok in doc:
        if tok.text.lower() in DEMO_PRONS and tok.pos_ == "PRON":
            left = tok.nbor(-1) if tok.i > 0 else None
            if left is None or left.pos_ not in {"NOUN", "PROPN", "PRON"}:
                dem_pron_flag = 1
                break
    return missing_verb_flag, dem_pron_flag


# Read TSV
df = pd.read_csv(DATA_PATH, sep="\t")
# Labels
df["label"] = df["Complete"].astype(int)
# Compute spaCy flags
flags = [extract_spacy_flags(txt) for txt in df["Text"].tolist()]
missing_verbs, dem_prons = zip(*flags)
df["missing_verb_flag"] = missing_verbs
df["dem_pron_flag"] = dem_prons
# Keep only needed columns
df = df[["Text", "label", "missing_verb_flag", "dem_pron_flag"]]

# stratified split -> train / val / test
df_train, df_temp = train_test_split(
    df, test_size=VAL_SIZE + TEST_SIZE, stratify=df["label"], random_state=SEED
)
rel_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
df_val, df_test = train_test_split(
    df_temp, test_size=1 - rel_val_size, stratify=df_temp["label"], random_state=SEED
)

# Build HuggingFace datasets
ds = DatasetDict(
    {
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    }
)

# ---------------------------------------------------------------------
# 3. Tokenisation (we keep flags in the dataset)
# ---------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    enc = tok(batch["Text"], truncation=True, max_length=512)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        # preserve flags and label for the model
        "missing_verb_flag": batch["missing_verb_flag"],
        "dem_pron_flag": batch["dem_pron_flag"],
        "labels": batch["label"],
    }


ds = ds.map(
    tokenize,
    batched=True,
    remove_columns=["Text", "label", "missing_verb_flag", "dem_pron_flag"],
)
data_collator = DataCollatorWithPadding(tok)


# ---------------------------------------------------------------------
# 4. Define custom model: DeBERTa encoder + two flags + classification head
# ---------------------------------------------------------------------
class FragmentClassifier(torch.nn.Module):
    def __init__(self, model_name: str, class_weights: torch.Tensor):
        super().__init__()
        # Base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # e.g. 1024
        # Classification head: takes [CLS_hidden; flag1; flag2]
        self.classifier = torch.nn.Linear(hidden_size + 2, 2)
        self.class_weights = class_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        missing_verb_flag: torch.Tensor,
        dem_pron_flag: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        # Encoder pass
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token hidden state (first token)
        pooled = outputs.last_hidden_state[:, 0, :]  # (bs, hidden)
        # Stack flags into shape (bs, 2), convert to float32
        flags = (
            torch.stack([missing_verb_flag, dem_pron_flag], dim=1)
            .float()
            .to(pooled.device)
        )
        # Concatenate
        features = torch.cat([pooled, flags], dim=1)  # (bs, hidden+2)
        logits = self.classifier(features)  # (bs, 2)

        loss = None
        if labels is not None:
            # CrossEntropyLoss expects class-weights on device
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(pooled.device)
            )
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits


# Compute class weights: w_i = total/(2 * count_i)
neg = (df["label"] == 0).sum()
pos = (df["label"] == 1).sum()
total = neg + pos
w0 = total / (2 * neg)
w1 = total / (2 * pos)
class_weights = torch.tensor([w0, w1], dtype=torch.float32)
print(f"Class weights: fragment={w0:.2f}, complete={w1:.2f}")

model = FragmentClassifier(MODEL_NAME, class_weights)

# ---------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")[
        "f1"
    ]
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------
# 6. TrainingArguments & Trainer
# ---------------------------------------------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    seed=SEED,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------
# 8. Evaluate on held-out test set
# ---------------------------------------------------------------------
# --- show first 10 mistakes --------------------------------------------------
test_labels = np.array(ds["test"]["labels"])  # <- note the final “s”
preds = np.argmax(trainer.predict(ds["test"]).predictions, axis=-1)
mis_idx = np.where(preds != test_labels)[0]

print(f"\nMis-classified examples: {len(mis_idx)}/{len(test_labels)}")
for i in mis_idx[:10]:
    print(f"- Pred {preds[i]}, Gold {test_labels[i]} :: {df_test.iloc[i]['Text']}")


# ── 9. Threshold tuning on validation set ────────────────────────────
val_logits = trainer.predict(ds["validation"]).predictions
val_labels = np.array(ds["validation"]["labels"])  # ← change here
val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()[:, 1]



prec, rec, thr = precision_recall_curve(val_labels, val_probs)
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_thr = thr[np.argmax(f1_scores)]
print(f"\nBest validation F1={f1_scores.max():.3f} at threshold={best_thr:.3f}")

# ---------------------------------------------------------------------
# 10. Re-evaluate on TEST with tuned threshold
# ---------------------------------------------------------------------
test_logits = trainer.predict(ds["test"]).predictions
test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()[:, 1]
test_pred = (test_probs >= best_thr).astype(int)
test_true = np.array(ds["test"]["labels"])  # ← and here


# ---------------------------------------------------------------------
# 11. Save model, tokenizer, and tuned threshold  ─────────────────────
# ---------------------------------------------------------------------
import json, os

SAVE_DIR = OUTPUT_DIR / "best_fragment_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1) Save tokenizer
tok.save_pretrained(SAVE_DIR)

# 2) Save model weights & an *augmented* config
cfg = model.encoder.config.to_dict()
cfg.update(
    {
        "architectures": ["FragmentClassifier"],
        "extra_numeric_features": 2,  # the two spaCy flags
        "cls_hidden_plus_flags": cfg["hidden_size"] + 2,
    }
)
with open(SAVE_DIR / "config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# We can’t rely on AutoModel.save_pretrained because HuggingFace
# doesn’t know our custom head; just save the whole torch state_dict.
torch.save(model.state_dict(), SAVE_DIR / "pytorch_model.bin")

# 3) Save tuned probability threshold
with open(SAVE_DIR / "threshold.json", "w") as f:
    json.dump({"best_threshold": float(best_thr)}, f)

print(f"\n📝  Everything saved to: {SAVE_DIR.resolve()}")
