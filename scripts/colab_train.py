# ==============================================================================
# Google Colab / GPU Training Notebook Script for nomic-ai/CodeRankEmbed on APPS
# ==============================================================================
# Instructions:
# 1. In Google Colab, restart the runtime if you encountered OOM:
#    Runtime -> Restart session (or Restart runtime)
# 2. Paste and run this entire script in a single cell.
# 3. Once finished, it automatically zips and downloads `coderank-apps-tuned.zip`!
# ==============================================================================

# Step 1: Install required libraries
import subprocess
import sys
import os

os.environ["HF_TOKEN"] = "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"

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "datasets", "torch", "accelerate"])

import torch
import gc
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Free any residual PyTorch CUDA memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Step 2: Configure Optimized Parameters for T4 GPU (16GB VRAM)
MODEL_NAME = "nomic-ai/CodeRankEmbed"
EPOCHS = 3
BATCH_SIZE = 8        # Reduced to 8 to comfortably prevent CUDA OOM
MAX_SEQ_LENGTH = 512  # 512 tokens covers 95%+ of APPS queries (avg 300) & docs (avg 120)
LR = 2e-5
OUTPUT_DIR = "coderank-apps-tuned"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'})")

# Step 3: Load APPS Dataset & Assert Zero-Leakage
print("Loading CoIR-Retrieval/apps train split...")
qrels_train = load_dataset("CoIR-Retrieval/apps", split="train")
qrels_test = load_dataset("CoIR-Retrieval/apps", split="test")

train_qids = set(row["query-id"] for row in qrels_train)
test_qids = set(row["query-id"] for row in qrels_test)
overlap = train_qids.intersection(test_qids)

if overlap:
    raise ValueError(f"LEAKAGE DETECTED: {len(overlap)} query IDs overlap with test split! Aborting.")

print(f"Leakage check passed: 0 collisions out of {len(train_qids)} train queries.")

queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")

q_map = {row["_id"]: row["text"] for row in queries}
c_map = {row["_id"]: row["text"] for row in corpus}

train_examples = []
prefix = "Represent this query for searching relevant code: "

for row in qrels_train:
    qid = row["query-id"]
    cid = row["corpus-id"]
    q_text = q_map.get(qid, "")
    c_text = c_map.get(cid, "")
    if q_text and c_text:
        train_examples.append(InputExample(texts=[prefix + q_text, c_text]))

print(f"Constructed {len(train_examples)} clean training pairs.")

# Step 4: Fine-tune with Contrastive Loss & FP16 Mixed Precision
print(f"Loading base model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
model.max_seq_length = MAX_SEQ_LENGTH

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
print(f"Training for {EPOCHS} epochs ({len(train_dataloader)} steps/epoch) with FP16...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    use_amp=True,  # Enables Automatic Mixed Precision (FP16) to drastically cut VRAM & speed up training
    output_path=OUTPUT_DIR,
    show_progress_bar=True
)

print("Training finished! Compressing model weights...")
import shutil
shutil.make_archive("coderank-apps-tuned", "zip", OUTPUT_DIR)

print("Downloading coderank-apps-tuned.zip...")
try:
    from google.colab import files
    files.download("coderank-apps-tuned.zip")
except ImportError:
    print("Not running in Google Colab. Saved to coderank-apps-tuned.zip locally.")
