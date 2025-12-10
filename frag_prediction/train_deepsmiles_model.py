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

# ───────────────────── DEEPSMILES SUPPORT ─────────────────────
# All chemistry happens on SMILES; the model sees DeepSMILES text.
import deepsmiles

DS_CONVERTER = deepsmiles.Converter(rings=True, branches=True)


def deep_to_smiles(ds: str) -> str | None:
    if not isinstance(ds, str):
        return None
    try:
        return DS_CONVERTER.decode(ds)
    except Exception:
        return None


def smiles_to_deep(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    try:
        return DS_CONVERTER.encode(smi)
    except Exception:
        return None


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
MAX_LENGTH = 64
SEQ_PENALTY = 0.6549843363313522
INVALID_SMILES_PENALTY = 14.87225684138008
NUM_EPOCHS = 10
BATCH_SIZE = 16
l1_reg = 3.8088910266740234e-07
patience = 4
min_delta = 1e-3
epochs_since_improve = 0

# ───────────────────────── DEVICE ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# ───────────────────────── LOAD DATA ─────────────────────
train_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_train_deepsmiles.csv")
val_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_val_deepsmiles.csv")
test_df = pd.read_csv("mtor_medchemfrag_final_canonicalized_dedup_test_deepsmiles.csv")


# ───────────────────────── UTILS (operate on DeepSMILES) ─
def tanimoto_similarity_deep(ds1: str, ds2: str) -> float:
    """Compute Tanimoto given DeepSMILES strings by decoding to SMILES first."""
    try:
        s1, s2 = deep_to_smiles(ds1), deep_to_smiles(ds2)
        if not s1 or not s2:
            return 0.0
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
        return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def canonical_deep(ds: str) -> str:
    """
    'Canonicalize' a DeepSMILES by decoding → RDKit canonical SMILES → re-encode to DeepSMILES.
    If decode fails, return the original DeepSMILES string.
    """
    try:
        smi = deep_to_smiles(ds)
        if not smi:
            return ds
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return ds
        can_smi = Chem.MolToSmiles(mol, canonical=True)
        re_ds = smiles_to_deep(can_smi)
        return re_ds if re_ds else ds
    except Exception:
        return ds


def write_error_counts_to_file(fname):
    with open(fname, "w") as f:
        for ph, d in error_counts.items():
            f.write(f"[{ph}]\n")
            for k, v in d.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")


# ───────────────────────── CACHED FINGERPRINTS ─────────────
class FingerprintCache:
    """
    Cache by canonical SMILES (not DeepSMILES).
    Callers must pass SMILES here.
    """

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
            except Exception:
                self.cache[smiles] = None
        return self.cache[smiles]

    def tanimoto_similarity_cached_smiles(self, sm1: str, sm2: str) -> float:
        """Inputs are SMILES strings."""
        fp1 = self.get_fingerprint(sm1)
        fp2 = self.get_fingerprint(sm2)
        if fp1 is None or fp2 is None:
            return 0.0
        try:
            return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.0


# Global fingerprint cache
fp_cache = FingerprintCache()


# ───────────────────────── DATASET ───────────────────────
class DeepSMILESDataset(Dataset):
    """
    NOTE: Columns remain the same names for minimal code churn:
      - "DRUG SMILES" and "FRAG_SMILES" now contain DeepSMILES text.
    The model sees DeepSMILES tokens. Chemistry is done by decode→SMILES.
    """

    def __init__(self, df, tok):
        self.df = df
        self.tok = tok
        self.skipped = 0

        # Fallback support if user renamed to DRUG_DEEPSMILES / FRAG_DEEPSMILES
        self.col_drug = "drug" if "drug" in df.columns else None
        self.col_frag = "fragment" if "fragment" in df.columns else None

        if self.col_drug is None or self.col_frag is None:
            raise KeyError("Expected DeepSMILES in columns 'DRUG SMILES'/'FRAG_SMILES' (or *_DEEPSMILES).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        drug_ds = self.df.iloc[idx][self.col_drug]
        frag_ds = self.df.iloc[idx][self.col_frag]
        if not isinstance(drug_ds, str) or len(drug_ds) == 0:
            self.skipped += 1
            return None

        # Tokenize DeepSMILES as text
        enc_d = self.tok(drug_ds, max_length=MAX_LENGTH, padding='max_length',
                         truncation=True, return_tensors="pt")
        enc_f = self.tok(frag_ds, max_length=MAX_LENGTH, padding='max_length',
                         truncation=True, return_tensors="pt")

        item = {k: v.squeeze(0) for k, v in enc_d.items()}
        item["labels"] = enc_f["input_ids"].squeeze(0)

        # Keep DeepSMILES strings for CSV/metrics; chemistry functions will decode.
        item["actual_fragment_smiles"] = frag_ds  # (DeepSMILES)
        item["drug_smiles"] = drug_ds  # (DeepSMILES)
        return item


# ───────────────────────── TOKENIZER / MODEL ─────────────
model_path = "aakothari/DeepBERTa_zinc_base_100k_v4"
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
    valid_batch = [b for b in batch if b is not None]
    if len(valid_batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(valid_batch)


train_ds = DeepSMILESDataset(train_df, tokenizer)
val_ds = DeepSMILESDataset(val_df, tokenizer)
test_ds = DeepSMILESDataset(test_df, tokenizer)
train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate)
val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 1, False, collate_fn=collate)
print("dataset sizes  train / val / test:", len(train_ds), len(val_ds), len(test_ds))

# ───────────────────────── OPTIMIZER & LR SCHEDULER ──────
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)
optimizer = optim.AdamW(model.parameters(), lr= 4.400379927768365e-05, weight_decay= 3.3655946722430514e-07)
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


def is_valid_deepsmiles(ds: str, phase: str) -> bool:
    """Validate a DeepSMILES by decoding → sanitizing SMILES."""
    try:
        smi = deep_to_smiles(ds)
        if not smi:
            return False
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return mol is not None
    except Exception as e:
        _bucket(str(e), phase)
        return False


# Pre-compute true fragment fingerprints for efficiency
def precompute_true_fingerprints(df):
    """
    Pre-cache fingerprints for all TRUE fragments.
    Decode DeepSMILES → canonical SMILES → cache FP.
    """
    print("Pre-computing true fragment fingerprints...")
    frag_col = "fragment"
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching fingerprints"):
        frag_ds = row[frag_col]
        if isinstance(frag_ds, str) and frag_ds:
            smi = deep_to_smiles(frag_ds)
            if smi:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        can_smi = Chem.MolToSmiles(mol, canonical=True)
                        fp_cache.get_fingerprint(can_smi)
                except Exception:
                    pass


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
    model.train()
    ep_losses, ep_sims = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}/{NUM_EPOCHS}", leave=False):
        if batch is None:
            continue

        optimizer.zero_grad()

        inp = {k: v.to(device) for k, v in batch.items()
               if (k in allowed_keys) and torch.is_tensor(v)}

        out = model(**inp)

        # L1 regularization
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
        loss = SEQ_PENALTY * out.loss + l1_reg * l1_loss

        # Predictions for similarity (DeepSMILES text)
        with torch.no_grad():
            preds = out.logits.argmax(-1)
            pred_deep = tokenizer.batch_decode(preds, skip_special_tokens=True)

        true_deep = batch["actual_fragment_smiles"]

        # Invalid penalty (validate via decode→SMILES)
        invalid_count = sum(not is_valid_deepsmiles(p, "train") for p in pred_deep)
        invalid_frac = invalid_count / len(pred_deep) if len(pred_deep) > 0 else 0.0
        loss = loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)

        loss.backward()
        optimizer.step()
        scheduler.step()
        ep_losses.append(loss.item())

        # Similarities using cached FPs on canonical SMILES
        sims = []
        for p_ds, t_ds in zip(pred_deep, true_deep):
            p_smi = deep_to_smiles(p_ds)
            t_smi = deep_to_smiles(t_ds)
            if p_smi and t_smi:
                # Canonicalize for cache key stability
                try:
                    mp = Chem.MolFromSmiles(p_smi)
                    mt = Chem.MolFromSmiles(t_smi)
                    if mp and mt:
                        cp = Chem.MolToSmiles(mp, canonical=True)
                        ct = Chem.MolToSmiles(mt, canonical=True)
                        sims.append(fp_cache.tanimoto_similarity_cached_smiles(ct, cp))
                        continue
                except Exception:
                    pass
            sims.append(0.0)
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

            inp = {k: v.to(device) for k, v in batch.items()
                   if (k in allowed_keys) and torch.is_tensor(v)}

            out = model(**inp)
            base_loss = SEQ_PENALTY * out.loss

            preds = out.logits.argmax(-1)
            pred_deep = tokenizer.batch_decode(preds, skip_special_tokens=True)
            true_deep = batch["actual_fragment_smiles"]

            sims = []
            for p_ds, t_ds in zip(pred_deep, true_deep):
                p_smi = deep_to_smiles(p_ds)
                t_smi = deep_to_smiles(t_ds)
                if p_smi and t_smi:
                    try:
                        mp = Chem.MolFromSmiles(p_smi)
                        mt = Chem.MolFromSmiles(t_smi)
                        if mp and mt:
                            cp = Chem.MolToSmiles(mp, canonical=True)
                            ct = Chem.MolToSmiles(mt, canonical=True)
                            sims.append(fp_cache.tanimoto_similarity_cached_smiles(ct, cp))
                            continue
                    except Exception:
                        pass
                sims.append(0.0)
            v_sims.extend(sims)

            invalid_count = sum(not is_valid_deepsmiles(p, "val") for p in pred_deep)
            total_val_invalid += invalid_count
            invalid_frac = invalid_count / len(pred_deep) if len(pred_deep) > 0 else 0.0
            vloss = base_loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)
            v_losses.append(vloss.item())

    print(f"Epoch {ep + 1} | Val Invalid DeepSMILES: {total_val_invalid}/{len(val_ds)}")

    mean_val_tan = np.mean(v_sims) if v_sims else 0.0
    mean_val_loss = np.mean(v_losses) if v_losses else float('inf')
    val_losses.append(mean_val_loss)
    val_ep_tan.append(mean_val_tan)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {ep + 1:02d} | Train Loss: {train_losses[-1]:.3f} | Train Tan: {mean_train_tan:.3f} | "
        f"Val Loss: {mean_val_loss:.3f} | Val Tan: {mean_val_tan:.3f} | LR: {current_lr:.2e}"
    )

    # --- GAP-BASED STOPPING ---
    gap_threshold = 0.20  # set the threshold you want, e.g. 0.10 (10%)
    gap = abs(mean_train_tan - mean_val_tan)

    if gap > gap_threshold:
        print(f"Early stopping: gap {gap:.3f} exceeded threshold {gap_threshold:.3f}.")
        break



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
drug_vals = []  # DeepSMILES
valid_flags = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        if batch is None:
            continue

        inp = {k: v.to(device) for k, v in batch.items()
               if (k in allowed_keys) and torch.is_tensor(v)}

        out = model(**inp)

        preds = out.logits.argmax(-1)
        pred_deep = tokenizer.batch_decode(preds, skip_special_tokens=True)
        true_deep = batch["actual_fragment_smiles"]  # DeepSMILES
        drug_deep = batch["drug_smiles"]  # DeepSMILES

        # Save DeepSMILES (no rename to keep structure)
        drug_vals.extend(drug_deep)

        # Decode to SMILES for metrics
        pred_smi = [deep_to_smiles(p) or "" for p in pred_deep]
        true_smi = [deep_to_smiles(t) or "" for t in true_deep]

        # Canonicalize SMILES before cache lookup
        canon_pred_smi, canon_true_smi = [], []
        for p, t in zip(pred_smi, true_smi):
            try:
                mp = Chem.MolFromSmiles(p) if p else None
                mt = Chem.MolFromSmiles(t) if t else None
                cp = Chem.MolToSmiles(mp, canonical=True) if mp else ""
                ct = Chem.MolToSmiles(mt, canonical=True) if mt else ""
            except Exception:
                cp, ct = "", ""
            canon_pred_smi.append(cp)
            canon_true_smi.append(ct)

        # Tanimoto (cached on SMILES)
        tani_sims = [fp_cache.tanimoto_similarity_cached_smiles(t, p) if t and p else 0.0
                     for t, p in zip(canon_true_smi, canon_pred_smi)]
        tanimoto_scores.extend(tani_sims)

        base_loss = SEQ_PENALTY * out.loss
        invalid_count = sum(not is_valid_deepsmiles(p, "test") for p in pred_deep)
        invalid_frac = invalid_count / len(pred_deep) if len(pred_deep) > 0 else 0.0
        total_loss = base_loss * (1.0 + INVALID_SMILES_PENALTY * invalid_frac)

        # Store DeepSMILES in final CSV (same keys: 'true'/'pred' now mean DeepSMILES)
        true_vals.extend([canonical_deep(t) for t in true_deep])
        pred_vals.extend([canonical_deep(p) for p in pred_deep])
        test_losses.append(total_loss.item())

print(f"Test Loss: {np.mean(test_losses):.3f}")
print(f"Mean Test Tanimoto Similarity: {np.mean(tanimoto_scores):.3f}")

# ───────────────────────── SAVE CHECKPOINTS ──────────────
ckpt_dir = pathlib.Path(f"checkpoint_{formatted_time}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Also include SMILES columns for convenience (derived from DeepSMILES)
df_out = pd.DataFrame({
    "drug": drug_vals,  # DeepSMILES
    "true": true_vals,  # DeepSMILES (canonicalized via round-trip)
    "pred": pred_vals,  # DeepSMILES (canonicalized via round-trip)
    "loss": test_losses,
    "valid": [is_valid_deepsmiles(p, "test") for p in pred_vals],
    "tanimoto": tanimoto_scores
})

# Optional: derived SMILES (not changing structure; just extra helpful cols)
try:
    df_out["true_SMILES"] = [deep_to_smiles(x) or "" for x in df_out["true"]]
    df_out["pred_SMILES"] = [deep_to_smiles(x) or "" for x in df_out["pred"]]
    df_out["drug_SMILES"] = [deep_to_smiles(x) or "" for x in df_out["drug"]]
except Exception:
    pass

df_out.to_csv(ckpt_dir / "test_preds.csv", index=False)

model.save_pretrained(ckpt_dir)
tokenizer.save_pretrained(ckpt_dir)
with open(ckpt_dir / "run_args.json", "w") as f:
    json.dump({
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seq_penalty": SEQ_PENALTY,
        "invalid_smiles_penalty": INVALID_SMILES_PENALTY,
        "time": formatted_time,
        "representation": "DeepSMILES"
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
print("✓ Success (DeepSMILES)!")
