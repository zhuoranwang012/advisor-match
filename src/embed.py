"""Encode text -> L2-normalized vectors. Encoder is switchable.

RUN LOCALLY: sentence-transformers downloads the model on first use; the dev
sandbox has no model access, so this file is written but not sandbox-tested.

Encoders
--------
  "minilm"  -> sentence-transformers/all-MiniLM-L6-v2   (primary; fast, tuned for similarity)
  "roberta" -> sentence-transformers/all-distilroberta-v1 (fair comparison: sentence-tuned RoBERTa)

Optional: encode_raw_roberta() reproduces the original raw roberta-base + mean-pooling
approach. Raw (non-sentence-tuned) embeddings are usually weaker for cosine similarity
(anisotropy) — useful to show the "tuned vs raw" gap, not for the main retriever.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

MODELS = {
    "minilm":  "sentence-transformers/all-MiniLM-L6-v2",
    "roberta": "sentence-transformers/all-distilroberta-v1",
}


def encode(texts, encoder="minilm", batch_size=64):
    """L2-normalized embeddings, shape (N, D), via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELS[encoder])
    embs = model.encode(list(texts), batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


def encode_raw_roberta(texts, batch_size=8, max_len=256):
    """OPTIONAL — reproduces original raw roberta-base mean-pooling (weaker baseline)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i+batch_size]), padding=True, truncation=True,
                     max_length=max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(v.cpu().numpy())
    v = np.vstack(out)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
