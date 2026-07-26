"""[REUSE ← run_matching.py + generate_embeddings_roberta.py] Encode text -> vectors.

RUN LOCALLY (needs to download the model; the sandbox can't).
Primary encoder: all-MiniLM-L6-v2. Also supports RoBERTa for the encoder comparison.
KEY CHANGE vs old code: encode PAPERS (paper-level index), not prof_text.
"""
# TODO: encode_papers(), encode_students(), save .npy to outputs/
