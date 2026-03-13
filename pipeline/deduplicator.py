"""
pipeline/deduplicator.py
------------------------
Anti-duplication module using TF-IDF + cosine similarity.

Design
------
- After all articles are generated, compute pairwise cosine similarity
  on the full article Markdown content.
- Any pair exceeding the configured threshold is flagged (or could trigger
  regeneration — the choice is explained in README).
- Updates each article's duplication score (0-15) inversely proportional
  to its maximum similarity with any peer article.

Why TF-IDF instead of embeddings?
- No extra API calls or GPU requirements
- Works offline and is fully deterministic
- Fast for batches up to a few hundred articles
- Extensible: swap to sentence-transformers with minimal changes
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .logger import logger
from .models import DuplicatePair


def compute_pairwise_similarity(
    slugs: List[str],
    texts: List[str],
) -> np.ndarray:
    """
    Compute a (N x N) cosine similarity matrix from a list of text documents.

    Parameters
    ----------
    slugs : article slug identifiers (same order as texts)
    texts : full article text strings

    Returns
    -------
    numpy ndarray of shape (N, N)
    """
    if len(texts) < 2:
        return np.zeros((len(texts), len(texts)))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        logger.error("[dedup] TF-IDF vectorization failed: %s", exc)
        return np.zeros((len(texts), len(texts)))

    return cosine_similarity(matrix)


def run_deduplication(
    slugs: List[str],
    texts: List[str],
    threshold: float = 0.85,
) -> Tuple[List[DuplicatePair], Dict[str, int]]:
    """
    Detect duplicate or near-duplicate articles and compute per-article
    duplication scores.

    Strategy for flagged pairs
    --------------------------
    We flag the pair and assign a low duplication score to both articles.
    Regeneration is not automatic — the summary.json reports the flagged
    pairs so the operator can decide to regenerate manually. This avoids
    infinite loops in edge cases (two near-identical topics).

    Parameters
    ----------
    slugs     : list of article slugs
    texts     : full Markdown content per article (same order)
    threshold : cosine similarity threshold above which a pair is flagged

    Returns
    -------
    pairs             : list of DuplicatePair objects for summary
    duplication_scores : dict mapping slug -> duplication_score (0-15)
    """
    logger.info("[dedup] Running pairwise deduplication on %d articles.", len(slugs))

    similarity_matrix = compute_pairwise_similarity(slugs, texts)
    n = len(slugs)

    pairs: List[DuplicatePair] = []
    flagged_slugs: set = set()

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                logger.warning(
                    "[dedup] Similarity %.3f >= %.2f between '%s' and '%s' — FLAGGED",
                    sim,
                    threshold,
                    slugs[i],
                    slugs[j],
                )
                pairs.append(
                    DuplicatePair(
                        slug_a=slugs[i],
                        slug_b=slugs[j],
                        similarity=round(sim, 4),
                        action="flagged",
                    )
                )
                flagged_slugs.add(slugs[i])
                flagged_slugs.add(slugs[j])

    # Compute per-article duplication score
    duplication_scores: Dict[str, int] = {}
    for i, slug in enumerate(slugs):
        # Maximum similarity with any other article (excluding self)
        row = similarity_matrix[i].copy()
        if n > 1:
            row[i] = 0.0  # exclude self-similarity
            max_sim = float(np.max(row))
        else:
            max_sim = 0.0

        if slug in flagged_slugs:
            # Penalised: near-duplicate detected
            dup_score = max(0, int((1.0 - max_sim) * 8))
        else:

            dup_score = int(15 * (1.0 - max_sim / threshold))
            dup_score = max(1, min(15, dup_score))

        duplication_scores[slug] = dup_score

    if pairs:
        logger.warning("[dedup] %d duplicate pair(s) detected.", len(pairs))
    else:
        logger.info("[dedup] No duplicates detected (threshold=%.2f).", threshold)

    return pairs, duplication_scores
