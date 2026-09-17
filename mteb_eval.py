import json
import logging
import sys
import mteb
from mteb.models.abs_encoder import AbsEncoder

logging.basicConfig(level=logging.INFO)

from sentence_transformers import SentenceTransformer
from app.chunker import parse_javascript, ASTParseError

from mteb.models.model_meta import ModelMeta

class PrePostPipelineEncoder(AbsEncoder):
    mteb_model_meta = ModelMeta(
        loader=None,
        name="local/PrePostPipelineEncoder",
        revision="1.0.0",
        languages=["eng"],
        framework=["PyTorch"],
        release_date="2026-09-18",
        n_parameters=100000000,
        memory_usage_mb=500,
        max_tokens=512,
        embed_dim=384,
        license="mit",
        open_weights=True,
        public_training_code="https://github.com/local/PrePostPipelineEncoder",
        public_training_data=True,
        similarity_fn_name="cosine",
        use_instructions=False,
        training_datasets=set(),
    )
    
    def __init__(self):
        self.base_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def encode(self, sentences, **kwargs):
        for k in ['hf_split', 'hf_subset', 'prompt_type', 'task_metadata']:
            kwargs.pop(k, None)
        import torch
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
        for k in ['hf_split', 'hf_subset', 'prompt_type', 'task_metadata']:
            kwargs.pop(k, None)
        import torch
        if isinstance(queries, torch.utils.data.DataLoader):
            texts = []
            for batch in queries:
                if isinstance(batch, dict) and "text" in batch:
                    texts.extend(batch["text"])
                elif isinstance(batch, (list, tuple)):
                    texts.extend(batch)
            queries = texts
        return self.base_model.encode(queries, **kwargs)
        
    def encode_corpus(self, corpus, **kwargs):
        for k in ['hf_split', 'hf_subset', 'prompt_type', 'task_metadata']:
            kwargs.pop(k, None)
            
        import torch
        if isinstance(corpus, torch.utils.data.DataLoader):
            flat_corpus = []
            for batch in corpus:
                if isinstance(batch, dict):
                    # Sometimes batch has 'text' as a list of strings
                    if "text" in batch:
                        for t in batch["text"]:
                            flat_corpus.append({"text": t})
                elif isinstance(batch, (list, tuple)):
                    for t in batch:
                        flat_corpus.append({"text": t})
            corpus = flat_corpus

        processed_texts = []
        for doc in corpus:
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            try:
                chunks = parse_javascript(text)
                if chunks:
                    text = " ".join([c["content"] for c in chunks])
            except ASTParseError as e:
                logging.warning(f"LOUD FALLBACK: ASTParseError encountered. Using raw text instead. Error: {e}")
            processed_texts.append(text)
            
        return self.base_model.encode(processed_texts, **kwargs)

def main():
    model = PrePostPipelineEncoder()
    
    # As per guidelines, load the specific AppsRetrieval task
    try:
        task = mteb.get_task("AppsRetrieval")
    except Exception as e:
        print(f"MTEB Task Error: {e}")
        sys.exit(1)
        
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64}
    )
    
    # Write the evaluation JSON
    task_result = list(result.task_results)[0]
    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)
        
    print("MTEB Evaluation Complete. Results written to appsretrieval_results.json")

if __name__ == "__main__":
    main()
