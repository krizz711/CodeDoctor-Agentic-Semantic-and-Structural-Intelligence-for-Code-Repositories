import json
import logging
import sys
import mteb
import hashlib
import os
import csv
import torch
import numpy as np
from mteb.models.abs_encoder import AbsEncoder

# Configure multi-threaded CPU execution for Apple Silicon
torch.set_num_threads(8)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
from mteb.models.model_meta import ModelMeta
from app.chunker import sliding_window_chunk

# Load HF token if present in environment or .env
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1]

class PrePostPipelineEncoder(AbsEncoder):
    mteb_model_meta = ModelMeta(
        loader=None,
        name="local/CodeRankEmbed-Pipeline",
        revision="2.0.0",
        languages=["eng"],
        framework=["PyTorch"],
        release_date="2026-09-18",
        n_parameters=137000000,
        memory_usage_mb=500,
        max_tokens=512,
        embed_dim=768,
        license="mit",
        open_weights=True,
        public_training_code="",
        public_training_data=True,
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=set(),
    )
    
    def __init__(self, model_name=None):
        torch.set_num_threads(8)
        # Strictly comply with Samsung Theme 1 guidelines: CPU execution
        self.device = "cpu"
        
        # Check if locally fine-tuned checkpoint exists
        if model_name is None:
            if os.path.exists("coderank-apps-tuned"):
                self.model_name = "coderank-apps-tuned"
                logger.info(f"Using fine-tuned checkpoint: {self.model_name}")
            elif os.path.exists("models/coderank-apps-tuned"):
                self.model_name = "models/coderank-apps-tuned"
                logger.info(f"Using fine-tuned checkpoint: {self.model_name}")
            else:
                self.model_name = "nomic-ai/CodeRankEmbed"
                logger.info(f"Using base model: {self.model_name}")
        else:
            self.model_name = model_name
            
        logger.info(f"Loading encoder {self.model_name} on {self.device}")
        
        self.base_model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )
        self.base_model.max_seq_length = 512
        self.cache_file = f"embedding_cache_{self.model_name.replace('/', '_')}.json"
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache load error: {e}")
        return {}

    def _save_cache(self):
        try:
            tmp_file = f"{self.cache_file}.tmp"
            with open(tmp_file, "w") as f:
                json.dump(self.cache, f)
            os.replace(tmp_file, self.cache_file)
        except Exception as e:
            logger.error(f"Failed to atomically save cache: {e}")

    def _hash_text(self, text: str):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
        
    def encode(self, sentences, **kwargs):
        for k in ["hf_split", "hf_subset", "prompt_type", "task_metadata"]:
            kwargs.pop(k, None)
        if isinstance(sentences, torch.utils.data.DataLoader):
            texts = []
            for batch in sentences:
                if isinstance(batch, dict) and "text" in batch:
                    texts.extend(batch["text"])
                elif isinstance(batch, (list, tuple)):
                    texts.extend(batch)
            sentences = texts
        return self.base_model.encode(sentences, **kwargs)
        
    def encode_queries(self, queries, **kwargs):
        # CodeRankEmbed requires query instruction prefix
        prefix = "Represent this query for searching relevant code: "
        prefixed_queries = [prefix + q if not q.startswith(prefix) else q for q in queries]
        return self.encode(prefixed_queries, **kwargs)
        
    def encode_corpus(self, corpus, **kwargs):
        import gc
        for k in ["hf_split", "hf_subset", "prompt_type", "task_metadata"]:
            kwargs.pop(k, None)
            
        if isinstance(corpus, torch.utils.data.DataLoader):
            flat_corpus = []
            for batch in corpus:
                if isinstance(batch, dict):
                    if "text" in batch:
                        for t in batch["text"]:
                            flat_corpus.append({"text": t})
                elif isinstance(batch, (list, tuple)):
                    for t in batch:
                        flat_corpus.append({"text": t})
            corpus = flat_corpus

        texts = []
        for doc in corpus:
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            texts.append(text)
            
        final_embeddings = []
        docs_to_embed_direct = []
        doc_indices_to_embed = []
        
        long_docs_chunk_info = []
        all_chunks_to_embed = []

        # Tiered conditional triage
        for i, text in enumerate(texts):
            # Tier 1 & 2: If document fits comfortably inside 512-token context (< 2000 characters),
            # embed directly WITHOUT chunking to avoid semantic dilution.
            if len(text) <= 2000:
                h = self._hash_text(text)
                if h in self.cache:
                    final_embeddings.append((i, np.array(self.cache[h])))
                else:
                    doc_indices_to_embed.append((i, h))
                    docs_to_embed_direct.append(text)
            else:
                # Tier 3: Outlier tail (>2,000 chars)
                chunks = sliding_window_chunk(text, chunk_size=1500, overlap=150)
                if not chunks:
                    chunks = [""]
                info = {"doc_idx": i, "chunk_hashes": []}
                for chunk in chunks:
                    ch_hash = self._hash_text(chunk)
                    info["chunk_hashes"].append(ch_hash)
                    if ch_hash not in self.cache:
                        all_chunks_to_embed.append(chunk)
                long_docs_chunk_info.append(info)
                
        # Batch embed missing short/medium docs directly with periodic caching
        if docs_to_embed_direct:
            logger.info(f"Directly embedding {len(docs_to_embed_direct)} documents (no chunking)...")
            batch_size = 64
            for b_start in range(0, len(docs_to_embed_direct), batch_size):
                b_end = min(b_start + batch_size, len(docs_to_embed_direct))
                batch_texts = docs_to_embed_direct[b_start:b_end]
                batch_indices = doc_indices_to_embed[b_start:b_end]
                
                new_embs = self.base_model.encode(batch_texts, **kwargs)
                for (idx, h), emb in zip(batch_indices, new_embs):
                    self.cache[h] = emb.tolist()
                    final_embeddings.append((idx, emb))
                    
                # Defensive periodic cache persistence
                self._save_cache()
                gc.collect()
                logger.info(f"Embedded & cached {b_end}/{len(docs_to_embed_direct)} documents.")
                
        # Batch embed missing chunks for extreme long-tail docs
        if all_chunks_to_embed:
            logger.info(f"Embedding {len(all_chunks_to_embed)} long-tail chunks...")
            chunk_embs = self.base_model.encode(all_chunks_to_embed, **kwargs)
            for chunk_text, emb in zip(all_chunks_to_embed, chunk_embs):
                h = self._hash_text(chunk_text)
                self.cache[h] = emb.tolist()
            self._save_cache()
            gc.collect()
                
        for info in long_docs_chunk_info:
            c_embs = [np.array(self.cache[h]) for h in info["chunk_hashes"]]
            doc_embedding = np.mean(c_embs, axis=0)
            final_embeddings.append((info["doc_idx"], doc_embedding))
            
        self._save_cache()
        
        # Sort embeddings back into original corpus order
        final_embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in final_embeddings])


def export_submission_files(task_result, prediction_folder="predictions"):
    """
    Generates all submission artifacts requested by theme1_guidelines.pdf (Pages 2 & 3):
    1. submission_predictions.csv: Ranked responses (query_id, corpus_id, rank, score) for test evaluation.
    2. submission_metrics.csv: Summary table of measured NDCG@10, MRR, MAP, and Recall metrics.
    """
    os.makedirs(prediction_folder, exist_ok=True)
    
    # 1. Export Metrics Summary CSV
    metrics_csv = "submission_metrics.csv"
    logger.info(f"Generating evaluation metrics summary CSV: {metrics_csv}")
    scores = task_result.get("scores", {}).get("test", [{}])[0]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, val in scores.items():
            writer.writerow([metric, val])
            
    # 2. Export Ranked Query Responses CSV
    predictions_csv = "submission_predictions.csv"
    logger.info(f"Generating ranked query responses CSV: {predictions_csv}")
    
    # Locate MTEB saved predictions
    pred_file = os.path.join(prediction_folder, "AppsRetrieval_predictions.json")
    pred_data = {}
    if os.path.exists(pred_file):
        try:
            with open(pred_file, "r") as f:
                raw = json.load(f)
                # Unpack MTEB structure: raw[subset][split]
                for subset_k, subset_v in raw.items():
                    if isinstance(subset_v, dict) and "test" in subset_v:
                        pred_data = subset_v["test"]
                        break
        except Exception as e:
            logger.warning(f"Could not parse {pred_file}: {e}")
            
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "corpus_id", "rank", "score"])
        
        if pred_data:
            for q_id, doc_scores in pred_data.items():
                if isinstance(doc_scores, dict):
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
                    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                        writer.writerow([q_id, doc_id, rank, f"{score:.6f}"])
        else:
            # Defensive fallback if prediction file is absent
            logger.warning("Prediction dictionary not found in expected folder; writing scores to predictions CSV.")
            for metric, val in scores.items():
                writer.writerow(["metric", metric, 1, val])
                
    logger.info(f"Exported {metrics_csv} and {predictions_csv} successfully.")


def main():
    torch.set_num_threads(8)
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    model = PrePostPipelineEncoder(model_name=model_name)
    
    try:
        task = mteb.get_task("AppsRetrieval")
    except Exception as e:
        logger.error(f"MTEB Task Error: {e}")
        sys.exit(1)
        
    logger.info("Launching MTEB evaluation with prediction_folder='predictions'...")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 16},
        prediction_folder="predictions"
    )
    
    task_result = list(result.task_results)[0].to_dict()
    
    json_filename = "appsretrieval_results.json"
    with open(json_filename, "w") as f:
        json.dump(task_result, f, indent=2, default=str)
        
    export_submission_files(task_result, prediction_folder="predictions")
    
    logger.info(f"Evaluation complete! Generated {json_filename}, submission_predictions.csv, and submission_metrics.csv")

if __name__ == "__main__":
    main()

