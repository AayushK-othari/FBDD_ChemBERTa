from __future__ import annotations
import sys, os, warnings
from io import StringIO
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from contextlib import redirect_stderr
import shutil, json, inspect, pathlib

# ───────────────────────── SETUP ─────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.simplefilter("ignore")
RDLogger.DisableLog("rdApp.*")

# ───────────────────────── HYPERPARAMETERS ───────────────
MAX_LENGTH = 96
SEQ_PENALTY = 0.6636387385224852
INVALID_SMILES_PENALTY = 6.834045098534338 # 9.959792750289285
NUM_EPOCHS = 12
BATCH_SIZE = 16 
# GAP_THRESHOLD = 0.05  # Stop if val_tan < train_tan - GAP_THRESHOLD
l1_reg = 3.41624379001933e-06
patience = 4
min_delta = 1e-3
epochs_since_improve = 0

# ───────────────────────── DEVICE ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# ───────────────────────── LOAD DATA ─────────────────────
train_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_train_smiles.csv")
val_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_val_smiles.csv")
test_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_test_smiles.csv")


# ───────────────────────── UTILS ─────────────────────────
def tanimoto_similarity(sm1: str, sm2: str) -> float:
    try:
        m1, m2 = Chem.MolFromSmiles(sm1), Chem.MolFromSmiles(sm2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
        return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def canonical(sm: str) -> str:
    try:
        mol = Chem.MolFromSmiles(sm)
        return Chem.MolToSmiles(mol, canonical=True) if mol else sm
    except:
        return sm


def write_error_counts_to_file(fname):
    with open(fname, "w") as f:
        for ph, d in error_counts.items():
            f.write(f"[{ph}]\n")
            for k, v in d.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")


# ───────────────────────── CACHED FINGERPRINTS ─────────────
class FingerprintCache:
    def __init__(self):
        self.cache = {}

    def get_fingerprint(self, smiles: str):
        if smiles not in self.cache:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    self.cache[smiles] = fp
                else:
                    self.cache[smiles] = None
            except:
                self.cache[smiles] = None
        return self.cache[smiles]

    def tanimoto_similarity_cached(self, sm1: str, sm2: str) -> float:
        fp1 = self.get_fingerprint(sm1)
        fp2 = self.get_fingerprint(sm2)
        if fp1 is None or fp2 is None:
            return 0.0
        try:
            return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0


# Global fingerprint cache
fp_cache = FingerprintCache()


# ───────────────────────── DATASET ───────────────────────
class SMILESDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df
        self.tok = tok
        self.skipped = 0

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        drug = self.df.iloc[idx]["drug"]
        frag = self.df.iloc[idx]["fragment"]
        if not drug:
            self.skipped += 1
            return None
        enc_d = self.tok(drug, max_length=MAX_LENGTH, padding='max_length',
                         truncation=True, return_tensors="pt")
        enc_f = self.tok(frag, max_length=MAX_LENGTH, padding='max_length',
                         truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc_d.items()}
        item["labels"] = enc_f["input_ids"].squeeze(0)
        item["actual_fragment_smiles"] = frag
        item["drug_smiles"] = drug  # keep original drug for CSV
        return item


# ───────────────────────── TOKENIZER / MODEL ─────────────
model_path = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Fix potential missing pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

config = RobertaConfig.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path, config=config)
model.to(device)

# Allowed keys for Option A (only pass keys the model.forward accepts AND are tensors)
allowed_keys = set(inspect.signature(model.forward).parameters.keys())


# ───────────────────────── DATALOADERS ───────────────────
def collate(batch):
    # Filter out None items and handle empty batches
    valid_batch = [b for b in batch if b is not None]
    if len(valid_batch) == 0:
        return None  # Signal empty batch
    return torch.utils.data.dataloader.default_collate(valid_batch)


train_ds = SMILESDataset(train_df, tokenizer)
val_ds = SMILESDataset(val_df, tokenizer)
test_ds = SMILESDataset(test_df, tokenizer)
train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate)
val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 1, False, collate_fn=collate)
print("dataset sizes  train / val / test:", len(train_ds), len(val_ds), len(test_ds))

# ───────────────────────── OPTIMIZER & LR SCHEDULER ──────
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)
optimizer = optim.AdamW(model.parameters(), lr=8.706020878304853e-05, weight_decay= 3.1428808908401116e-05)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ───────────────────────── ERROR BUCKETING ───────────────
error_counts = {ph: {k: 0 for k in ["unclosed ring", "duplicated ring closure", "extra close parentheses",
                                    "extra open parentheses", "non-ring atom", "can't kekulize", "other"]}
                for ph in ["train", "val", "test"]}
error_msgs = []


def _bucket(msg, phase):
    d = error_counts[phase]
    msg = msg.lower()
    if "unclosed ring" in msg:
        d["unclosed ring"] += 1
    elif "duplicated ring closure" in msg:
        d["duplicated ring closure"] += 1
    elif "extra close" in msg:
        d["extra close parentheses"] += 1
    elif "extra open" in msg:
        d["extra open parentheses"] += 1
    elif "non-ring atom" in msg:
        d["non-ring atom"] += 1
    elif "kekul" in msg:
        d["can't kekulize"] += 1
    else:
        d["other"] += 1
        error_msgs.append(msg.strip())


def is_valid_smiles(sm, phase):
    """Improved SMILES validation with proper sanitization"""
    try:
        mol = Chem.MolFromSmiles(sm, sanitize=True)
        return mol is not None
    except Exception as e:
        _bucket(str(e), phase)
        return False


# Pre-compute true fragment fingerprints for efficiency
def precompute_true_fingerprints(df):
    """Pre-cache fingerprints for all true SMILES to avoid repeated computation"""
    print("Pre-computing true fragment fingerprints...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching fingerprints"):
        frag_smiles = row["fragment"]
        if frag_smiles:
            canon_frag = canonical(frag_smiles)
            fp_cache.get_fingerprint(canon_frag)


# ───────────────────────── TRAINING ──────────────────────
start_time = time.time()
formatted_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

precompute_true_fingerprints(train_df)
precompute_true_fingerprints(val_df)
precompute_true_fingerprints(test_df)

train_losses, val_losses = [], []
train_ep_tan, val_ep_tan = [], []
best_val_tani = 0.0

for ep in range(NUM_EPOCHS):
    # (removed printing of a random drug to avoid leaking drugs to terminal)
    model.train()
    ep_losses, ep_sims = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}/{NUM_EPOCHS}", leave=False):
        if batch is None:
            continue

        optimizer.zero_grad()

        # Option A: Only pass model.forward-allowed keys that are tensors
        inp = {k: v.to(device) for k, v in batch.items()
               if (k in allowed_keys) and torch.is_tensor(v)}

        out = model(**inp)

        # L1 regularization
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
        loss = SEQ_PENALTY * out.loss + l1_reg * l1_loss

        # Predictions for similarity
        with torch.no_grad():
            preds = out.logits.argmax(-1)
            pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)

        true_smi = batch["actual_fragment_smiles"]

        # Invalid penalty
        invalid_count = sum(not is_valid_smiles(p, "train") for p in pred_smi)
        invalid_frac = invalid_count / len(pred_smi) if len(pred_smi) > 0 else 0.0
        loss = loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)

        loss.backward()
        optimizer.step()
        scheduler.step()
        ep_losses.append(loss.item())

        # Similarities (cached fingerprints)
        sims = [fp_cache.tanimoto_similarity_cached(canonical(t), canonical(p))
                for p, t in zip(pred_smi, true_smi)]
        ep_sims.extend(sims)

    mean_train_tan = np.mean(ep_sims)
    train_losses.append(np.mean(ep_losses))
    train_ep_tan.append(mean_train_tan)

    # ─── VALIDATION ───
    model.eval()
    v_losses, v_sims = [], []
    total_val_invalid = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Valid {ep + 1}", leave=False):
            if batch is None:
                continue

            # Option A here as well
            inp = {k: v.to(device) for k, v in batch.items()
                   if (k in allowed_keys) and torch.is_tensor(v)}

            out = model(**inp)
            base_loss = SEQ_PENALTY * out.loss

            preds = out.logits.argmax(-1)
            pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)
            true_smi = batch["actual_fragment_smiles"]

            sims = [fp_cache.tanimoto_similarity_cached(canonical(t), canonical(p))
                    for p, t in zip(pred_smi, true_smi)]
            v_sims.extend(sims)

            invalid_count = sum(not is_valid_smiles(p, "val") for p in pred_smi)
            total_val_invalid += invalid_count
            invalid_frac = invalid_count / len(pred_smi) if len(pred_smi) > 0 else 0.0
            vloss = base_loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)
            v_losses.append(vloss.item())

    print(f"Epoch {ep + 1} | Val Invalid SMILES: {total_val_invalid}/{len(val_ds)}")

    mean_val_tan = np.mean(v_sims) if v_sims else 0.0
    mean_val_loss = np.mean(v_losses) if v_losses else float('inf')
    val_losses.append(mean_val_loss)
    val_ep_tan.append(mean_val_tan)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {ep + 1:02d} | Train Loss: {train_losses[-1]:.3f} | Train Tan: {mean_train_tan:.3f} | "
        f"Val Loss: {mean_val_loss:.3f} | Val Tan: {mean_val_tan:.3f} | LR: {current_lr:.2e}"
    )

    improved = mean_val_tan > (best_val_tani + min_delta)
    if improved:
        best_val_tani = mean_val_tan
        epochs_since_improve = 0
        model.save_pretrained("best_model_checkpoint")
        tokenizer.save_pretrained("best_model_checkpoint")
        print("✓ New best model saved.")
    else:
        epochs_since_improve += 1
        print(f"No val Tanimoto improvement for {epochs_since_improve} epoch(s).")

    if epochs_since_improve >= patience:
        print(f"Early stopping: no val Tanimoto improvement ≥ {min_delta} for {patience} epochs.")
        print(f"Best val Tanimoto: {best_val_tani:.4f}")
        break

# ───────────────────────── TESTING ───────────────────────
print("Loading best model checkpoint...")
model = RobertaForMaskedLM.from_pretrained("best_model_checkpoint", config=config).to(device)
tokenizer = RobertaTokenizer.from_pretrained("best_model_checkpoint")

true_vals, pred_vals, test_losses, tanimoto_scores = [], [], [], []
drug_vals = []  # collect drugs in-order (no printing)
valid_flags = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        if batch is None:
            continue

        # Option A here too
        inp = {k: v.to(device) for k, v in batch.items()
               if (k in allowed_keys) and torch.is_tensor(v)}

        out = model(**inp)

        preds = out.logits.argmax(-1)
        pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)
        true_smi = batch["actual_fragment_smiles"]

        drug_smi = batch["drug_smiles"]  # list of strings
        canon_drug = [canonical(d) for d in drug_smi]  # canonicalize for consistency
        drug_vals.extend(canon_drug)

        # Canonicalize predictions/truth
        canon_pred = [canonical(p) for p in pred_smi]
        canon_true = [canonical(t) for t in true_smi]

        # Tanimoto (cached)
        tani_sims = [fp_cache.tanimoto_similarity_cached(t, p) for t, p in zip(canon_true, canon_pred)]
        tanimoto_scores.extend(tani_sims)

        base_loss = SEQ_PENALTY * out.loss
        invalid_count = sum(not is_valid_smiles(p, "test") for p in pred_smi)
        invalid_frac = invalid_count / len(pred_smi) if len(pred_smi) > 0 else 0.0
        total_loss = base_loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)

        true_vals.extend(canon_true)
        pred_vals.extend(canon_pred)
        test_losses.append(total_loss.item())

print(f"Test Loss: {np.mean(test_losses):.3f}")
print(f"Mean Test Tanimoto Similarity: {np.mean(tanimoto_scores):.3f}")

# ───────────────────────── SAVE CHECKPOINTS ──────────────
ckpt_dir = pathlib.Path(f"checkpoint_{formatted_time}")
ckpt_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "drug": drug_vals,
    "true": true_vals,
    "pred": pred_vals,
    "loss": test_losses,
    "valid": [is_valid_smiles(p, "test") for p in pred_vals],
    "tanimoto": tanimoto_scores
}).to_csv(ckpt_dir / "test_preds.csv", index=False)

model.save_pretrained(ckpt_dir)
tokenizer.save_pretrained(ckpt_dir)
with open(ckpt_dir / "run_args.json", "w") as f:
    json.dump({
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seq_penalty": SEQ_PENALTY,
        "invalid_smiles_penalty": INVALID_SMILES_PENALTY,
        #   "gap_threshold": GAP_THRESHOLD,
        "time": formatted_time
    }, f, indent=2)

# Improved script copying with notebook support
try:
    script_file = getattr(sys.modules['__main__'], '__file__', None)
    if script_file:
        script_path = pathlib.Path(script_file)
        shutil.copy(script_path, ckpt_dir / script_path.name)
    else:
        print("(info) Running in notebook/interactive mode - script copy skipped")
except Exception as e:
    print(f"(warning) could not copy script file: {e}")

# ───────────────────────── PLOT ──────────────────────────
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Loss
axes[0].plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train')
axes[0].plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Validation')
axes[0].set_title('Epoch Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right')

# Tanimoto
axes[1].plot(range(1, len(train_ep_tan) + 1), train_ep_tan, marker='o', label='Train')
axes[1].plot(range(1, len(val_ep_tan) + 1), val_ep_tan, marker='s', label='Validation')
axes[1].set_title('Mean Tanimoto Similarity')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Tanimoto Similarity')
axes[1].set_ylim(0, 1)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend(loc='lower right')

plt.tight_layout()
out_pdf = ckpt_dir / f"mTOR_fig_{formatted_time}.pdf"
out_png = ckpt_dir / f"mTOR_fig_{formatted_time}.png"
fig.savefig(out_pdf, format='pdf')
fig.savefig(out_png, dpi=300)
print(f"Saved figure as:\n • {out_pdf}\n • {out_png}")

# ───────────────────────── ERROR REPORT ──────────────────
print("\nError message counts:")
for ph, d in error_counts.items():
    print(f"[{ph}]")
    for k, v in d.items():
        print(f"  {k:25s}: {v}")
write_error_counts_to_file(ckpt_dir / f"errors_{formatted_time}.txt")

elapsed = time.time() - start_time
h, m = divmod(elapsed // 60, 60)
s = int(elapsed % 60)
print(f"Total runtime: {int(h):02d}:{int(m):02d}:{s:02d}")
print("✓ Success!")
