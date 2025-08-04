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
import joblib
import random
from contextlib import redirect_stderr
import shutil, json, inspect, pathlib
from deepsmiles import Converter

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
SEQ_PENALTY = 1.0
INVALID_SMILES_PENALTY = 1.3
NUM_EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 4
l1_reg = 8e-7

# ───────────────────────── DEVICE ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# ───────────────────────── LOAD DATA ─────────────────────
train_df = pd.read_csv("mTORcanonical_deduplicated_mapped_deepSMILES_train.csv")
val_df = pd.read_csv("mTORcanonical_deduplicated_mapped_deepSMILES_val.csv")
test_df = pd.read_csv("mTORcanonical_deduplicated_mapped_deepSMILES_test.csv")

converter = Converter(rings=True, branches=True)


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


def decode_deepsmiles(smile):
    try:
        return converter.decode(smile)
    except:
        return ""


def canonical(sm: str) -> str:
    mol = Chem.MolFromSmiles(sm)
    return Chem.MolToSmiles(mol, canonical=True) if mol else sm


def write_error_counts_to_file(fname):
    with open(fname, "w") as f:
        for ph, d in error_counts.items():
            f.write(f"[{ph}]\n")
            for k, v in d.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")


# ───────────────────────── DATASET ───────────────────────
class SMILESDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df
        self.tok = tok
        self.skipped = 0

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        drug = self.df.iloc[idx]["DRUG_SMILES_DEEP"]
        frag = self.df.iloc[idx]["FRAG_SMILES_DEEP"]
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
        return item


# ───────────────────────── TOKENIZER / MODEL ─────────────
model_path = "aakothari/DeepBERTa_zinc_base_100k_v2"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
config = RobertaConfig.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path, config=config)
model.to(device)


# ───────────────────────── DATALOADERS ───────────────────
def collate(batch):
    return torch.utils.data.dataloader.default_collate([b for b in batch if b])


train_ds = SMILESDataset(train_df, tokenizer)
val_ds = SMILESDataset(val_df, tokenizer)
test_ds = SMILESDataset(test_df, tokenizer)
train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate)
val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 1, False, collate_fn=collate)
print("dataset sizes  train / val / test:", len(train_ds), len(val_ds), len(test_ds))

# ───────────────────────── OPTIMIZER & LEARNING-RATE SCHEDULER ─────────
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=12e-4)
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


def is_valid_smiles(sm: str, phase: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(sm, sanitize=False)
        if mol is None:
            _bucket("MolFromSmiles returned None", phase)
            return False

        # Sanitize safely with error catching
        Chem.SanitizeMol(mol, catchErrors=True)

        problems = Chem.DetectChemistryProblems(mol)
        if problems:
            _bucket(problems[0].Message(), phase)
            return False

        return True

    except Exception as e:
        _bucket(str(e), phase)
        return False


# ───────────────────────── TRAINING ──────────────────────
start_time = time.time()
formatted_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
train_losses, val_losses = [], []
train_ep_tan, val_ep_tan = [], []
best_val_tani = 0.0
epochs_no_improve = 0

for ep in range(NUM_EPOCHS):
    print("Sample SMILES:", train_df.sample(1)["DRUG_SMILES_DEEP"].values[0])
    model.train()
    ep_losses, ep_sims = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}/{NUM_EPOCHS}", leave=False):
        optimizer.zero_grad()
        inp = {k: v.to(device) for k, v in batch.items() if k != "actual_fragment_smiles"}
        out = model(**inp)
        # Compute normal loss

        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
        loss = SEQ_PENALTY * out.loss + l1_reg * l1_loss

        preds = out.logits.argmax(-1)
        pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)
        true_smi = batch["actual_fragment_smiles"]

        # Count invalids
        invalid_count = sum(not is_valid_smiles(decode_deepsmiles(p), "train") for p in pred_smi)
        invalid_frac = invalid_count / BATCH_SIZE

        # Scale the loss using penalty factor
        penalty_factor = 1.0 + INVALID_SMILES_PENALTY * invalid_frac
        loss = loss * penalty_factor

        loss.backward()
        optimizer.step()
        scheduler.step()
        ep_losses.append(loss.item())

        # CORRECT
        decoded_preds = [decode_deepsmiles(p) for p in pred_smi]
        decoded_trues = [decode_deepsmiles(t) for t in true_smi]

        sims = [tanimoto_similarity(canonical(t), canonical(p)) for p, t in zip(decoded_preds, decoded_trues)]
        ep_sims.extend(sims)

    mean_train_tan = np.mean(ep_sims)
    train_losses.append(ep_losses)
    train_ep_tan.append(mean_train_tan)

    # ─── VALIDATION ───
    model.eval()
    v_losses, v_sims = [], []
    total_val_invalid = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Valid {ep + 1}", leave=False):
            inp = {k: v.to(device) for k, v in batch.items() if k != "actual_fragment_smiles"}
            out = model(**inp)
            vloss = SEQ_PENALTY * out.loss

            preds = out.logits.argmax(-1)
            pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)
            true_smi = batch["actual_fragment_smiles"]

            decoded_preds = [decode_deepsmiles(p) for p in pred_smi]
            decoded_trues = [decode_deepsmiles(t) for t in true_smi]

            # Now, calculate similarity on the valid, standard SMILES strings
            sims = [tanimoto_similarity(canonical(t), canonical(p)) for p, t in zip(decoded_preds, decoded_trues)]
            v_sims.extend(sims)

            # Just count invalids for logging purposes
            invalid_count = sum(not is_valid_smiles(decode_deepsmiles(p), "val") for p in pred_smi)
            total_val_invalid += invalid_count

            v_losses.append(vloss.item())

    print(f"Epoch {ep + 1} | Val Invalid SMILES: {total_val_invalid}/{len(val_ds)}")

    mean_val_tan = np.mean(v_sims)
    mean_val_loss = np.mean(v_losses)
    val_losses.append(v_losses)
    val_ep_tan.append(mean_val_tan)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {ep + 1:02d} | Train Loss: {np.mean(ep_losses):.3f} | Train Tan: {mean_train_tan:.3f} | Val Loss: {np.mean(v_losses):.3f} | Val Tan: {mean_val_tan:.3f} | LR: {current_lr:.2e}")

    if mean_val_tan > best_val_tani + 1e-3:
        best_val_tani = mean_val_tan
        epochs_no_improve = 0
        model.save_pretrained("best_model_checkpoint")
        tokenizer.save_pretrained("best_model_checkpoint")
        print(f"✓ New best model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print(f"✗ Early stopping at epoch {ep + 1}. Best val tan: {best_val_tani:.4f}")
            break

# ───────────────────────── TESTING ───────────────────────
print("Loading best model checkpoint...")
model = RobertaForMaskedLM.from_pretrained("best_model_checkpoint", config=config).to(device)
tokenizer = RobertaTokenizer.from_pretrained("best_model_checkpoint")

true_vals, pred_vals, test_losses, tanimoto_scores = [], [], [], []
model.eval()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        inp = {k: v.to(device) for k, v in batch.items() if k != "actual_fragment_smiles"}
        out = model(**inp)

        preds = out.logits.argmax(-1)
        pred_smi = tokenizer.batch_decode(preds, skip_special_tokens=True)
        true_smi = batch["actual_fragment_smiles"]

        # Canonicalize both
        canon_pred = [canonical(decode_deepsmiles(p)) for p in pred_smi]
        canon_true = [canonical(decode_deepsmiles(t)) for t in true_smi]

        # Tanimoto similarity per prediction
        tani_sims = [tanimoto_similarity(t, p) for t, p in zip(canon_true, canon_pred)]
        tanimoto_scores.extend(tani_sims)

        # Validity penalty
        invalid_count = sum(not is_valid_smiles(decode_deepsmiles(p), "test") for p in pred_smi)
        total_loss = SEQ_PENALTY * out.loss + INVALID_SMILES_PENALTY * invalid_count

        # Accumulate results
        true_vals.extend(canon_true)
        pred_vals.extend(canon_pred)
        test_losses.append(total_loss.item())

print(f"Test Loss: {np.mean(test_losses):.3f}")
print(f"Mean Test Tanimoto Similarity: {np.mean(tanimoto_scores):.3f}")
# print(f"Invalid SMILES in test: {sum(not is_valid_smiles(p, 'test') for p in pred_vals)} / {len(pred_vals)}")


# ───────────────────────── SAVE CHECKPOINTS ──────────────
ckpt_dir = pathlib.Path(f"checkpoint_{formatted_time}")
ckpt_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "true": true_vals,
    "pred": pred_vals,
    "loss": test_losses,
    "valid": [is_valid_smiles(p , "test") for p in pred_vals],
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
        "time": formatted_time
    }, f, indent=2)

try:
    script_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    shutil.copy(script_path, ckpt_dir / script_path.name)
except Exception as e:
    print(f"(warning) could not copy script file: {e}")

# ───────────────────────── PLOT ──────────────────────────
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), [np.mean(l) for l in train_losses], label="Train")
plt.plot(range(1, len(val_losses) + 1), [np.mean(l) for l in val_losses], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_ep_tan) + 1), train_ep_tan, label="Train Tan")
plt.plot(range(1, len(val_ep_tan) + 1), val_ep_tan, label="Val Tan")
plt.title("Tanimoto Similarity")
plt.xlabel("Epoch")
plt.ylabel("Similarity")
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig(ckpt_dir / f"mTOR_fig_{formatted_time}.png")

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
