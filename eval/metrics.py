"""Ranking metrics with binary relevance (a professor is relevant or not).

hit@k        : did top-k contain at least one relevant professor
precision@k  : fraction of top-k that are relevant
ndcg@k       : normalized DCG
mrr          : reciprocal rank of the first relevant professor
Each takes a ranked list of professor names and a set of relevant names.
"""
import numpy as np


def hit_at_k(ranked, relevant, k=5):
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def precision_at_k(ranked, relevant, k=5):
    return sum(r in relevant for r in ranked[:k]) / k


def ndcg_at_k(ranked, relevant, k=5):
    dcg = sum((1.0 if r in relevant else 0.0) / np.log2(i + 2)
              for i, r in enumerate(ranked[:k]))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked, relevant):
    for i, r in enumerate(ranked):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0
