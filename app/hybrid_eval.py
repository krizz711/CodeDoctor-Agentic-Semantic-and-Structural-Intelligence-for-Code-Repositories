import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mteb
import numpy as np
import logging
from sentence_transformers.util import cos_sim
from app.execution_reranker import parse_query_for_examples, verify_candidate
from mteb_eval import PrePostPipelineEncoder

logging.basicConfig(level=logging.INFO)

def calculate_ndcg_at_k(rank, k=10):
    if rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)

def main():
    logging.info("Loading APPS Dataset...")
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    
    queries = task.dataset["default"]["test"]["queries"]
    corpus = task.dataset["default"]["test"]["corpus"]
    qrels = task.dataset["default"]["test"]["relevant_docs"] 
    
    query_ids = list(queries.keys()) if isinstance(queries, dict) else queries['id']
    query_texts = list(queries.values()) if isinstance(queries, dict) else queries['text']
    
    corpus_ids = list(corpus.keys()) if isinstance(corpus, dict) else corpus['id']
    corpus_docs = list(corpus.values()) if isinstance(corpus, dict) else corpus['text']
    
    SAMPLE_SIZE = 50
    query_ids = query_ids[:SAMPLE_SIZE]
    query_texts = query_texts[:SAMPLE_SIZE]
    
    encoder = PrePostPipelineEncoder(model_name="intfloat/e5-base-v2")
    
    logging.info("Embedding corpus (with caching & conditional chunking)...")
    corpus_embeddings = encoder.encode_corpus([doc.get("text", doc) if isinstance(doc, dict) else doc for doc in corpus_docs], batch_size=256)
    
    logging.info("Embedding queries...")
    query_embeddings = encoder.encode_queries(query_texts, batch_size=32)
    
    ndcg_base = []
    ndcg_hybrid = []
    
    logging.info("Starting evaluation loop...")
    for i, q_id in enumerate(query_ids):
        q_text = query_texts[i]
        q_emb = query_embeddings[i]
        
        similarities = cos_sim(q_emb, corpus_embeddings)[0].numpy()
        top_k_indices = np.argsort(similarities)[::-1][:50]
        top_k_ids = [corpus_ids[idx] for idx in top_k_indices]
        
        gold_docs = qrels.get(q_id, {})
        if not gold_docs:
            # Maybe relevant docs is a dataset not a dict
            continue
        gold_id = list(gold_docs.keys())[0] if isinstance(gold_docs, dict) else gold_docs
        
        try:
            base_rank = top_k_ids.index(gold_id) + 1
        except ValueError:
            base_rank = 1000
            
        ndcg_base.append(calculate_ndcg_at_k(base_rank, 10))
        
        examples = parse_query_for_examples(q_text)
        reranked_ids = []
        unverified_ids = []
        
        if examples:
            for idx in top_k_indices[:15]:
                candidate_id = corpus_ids[idx]
                candidate_code = corpus_docs[idx].get("text", corpus_docs[idx]) if isinstance(corpus_docs[idx], dict) else corpus_docs[idx]
                
                if verify_candidate(candidate_code, examples, timeout=1):
                    reranked_ids.append(candidate_id)
                else:
                    unverified_ids.append(candidate_id)
            
            final_ids = reranked_ids + unverified_ids + top_k_ids[15:]
        else:
            final_ids = top_k_ids
            
        try:
            hybrid_rank = final_ids.index(gold_id) + 1
        except ValueError:
            hybrid_rank = 1000
            
        ndcg_hybrid.append(calculate_ndcg_at_k(hybrid_rank, 10))
        
    print(f"--- RESULTS (Sample Size: {SAMPLE_SIZE}) ---")
    print(f"Base NDCG@10 (Dense Only): {np.mean(ndcg_base):.4f}")
    print(f"Hybrid NDCG@10 (Dense + Exec Rerank): {np.mean(ndcg_hybrid):.4f}")

if __name__ == "__main__":
    main()
