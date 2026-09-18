"""
Contrastive Fine-Tuning Script for nomic-ai/CodeRankEmbed on APPS.
Designed to be run offline on a GPU (e.g. Google Colab / CUDA instance) to produce 
a high-accuracy, CPU-deployable checkpoint for MTEB AppsRetrieval evaluation.

Adheres strictly to Rule #1 (Academic Integrity) and Rule #2 (Defensive Coding).
Verifies ZERO contamination against MTEB AppsRetrieval test split before training.
"""

import os
import sys
import argparse
import logging
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_zero_leakage(train_qids, test_qids):
    """
    Defensively asserts that training query IDs do NOT overlap with test split IDs.
    Rule 1: Academic & Research Integrity.
    """
    overlap = set(train_qids).intersection(set(test_qids))
    if overlap:
        logger.error(f"CRITICAL CONTAMINATION: {len(overlap)} query IDs overlap with test split!")
        raise ValueError(f"Academic Integrity Violation: Found {len(overlap)} test queries in training set.")
    logger.info(f"Leakage Firewall PASSED: 0 query collisions between {len(train_qids)} train queries and {len(test_qids)} test queries.")
    return True


def prepare_training_data(hf_token=None, max_samples=None):
    """
    Loads APPS training pairs directly from CoIR-Retrieval/apps train split.
    Maps qrels (train) to query texts and corpus texts.
    """
    logger.info("Loading CoIR-Retrieval/apps splits from Hugging Face...")
    qrels_train = load_dataset("CoIR-Retrieval/apps", split="train", token=hf_token)
    qrels_test = load_dataset("CoIR-Retrieval/apps", split="test", token=hf_token)
    
    train_qids = [row["query-id"] for row in qrels_train]
    test_qids = [row["query-id"] for row in qrels_test]
    
    # Enforce strict leakage firewall
    verify_zero_leakage(train_qids, test_qids)
    
    queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries", token=hf_token)
    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus", token=hf_token)
    
    logger.info("Building in-memory ID-to-text lookup tables...")
    q_map = {row["_id"]: row["text"] for row in queries}
    c_map = {row["_id"]: row["text"] for row in corpus}
    
    train_examples = []
    query_prefix = "Represent this query for searching relevant code: "
    
    count = 0
    for row in qrels_train:
        qid = row["query-id"]
        cid = row["corpus-id"]
        
        query_text = q_map.get(qid, "")
        code_text = c_map.get(cid, "")
        
        if not query_text or not code_text:
            continue
            
        prefixed_query = query_prefix + query_text
        train_examples.append(InputExample(texts=[prefixed_query, code_text]))
        count += 1
        if max_samples and count >= max_samples:
            break
            
    logger.info(f"Successfully constructed {len(train_examples)} clean contrastive pairs.")
    return train_examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeRankEmbed for APPS code retrieval")
    parser.add_argument("--model_name", default="nomic-ai/CodeRankEmbed", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 for T4, 32 for A100)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", default="models/coderank-apps-tuned", help="Output directory")
    parser.add_argument("--verify_only", action="store_true", help="Verify data leakage without training")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for testing")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    
    train_examples = prepare_training_data(hf_token=hf_token, max_samples=args.max_samples)
    
    if args.verify_only:
        logger.info("Verification complete. Exiting successfully.")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    logger.info(f"Loading base model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device, trust_remote_code=True)
    model.max_seq_length = 2048 # Balanced length for training efficiency

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    logger.info(f"Starting training: {args.epochs} epochs, {len(train_dataloader)} steps/epoch, warmup={warmup_steps}...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True
    )
    
    logger.info(f"Training complete! Model saved to {args.output_dir}")
    logger.info("This model directory can now be copied to your CPU deployment for MTEB evaluation.")

if __name__ == "__main__":
    main()
