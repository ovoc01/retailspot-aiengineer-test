"""
pipeline/scorer.py
------------------
Quality scoring for generated articles.

Five criteria (total = 100 pts):
  1. Structure     [0-25] — presence & completeness of all required sections
  2. Readability   [0-20] — Flesch score (textstat) + paragraph length
  3. Sources       [0-20] — count, diversity of domains, HTTPS
  4. LLM-friendly  [0-20] — entity density, lists, FAQ quality, direct answers
  5. Duplication   [0-15] — updated by the deduplicator after batch generation
"""

import re
from typing import List, Optional
from urllib.parse import urlparse

import textstat

from .models import ArticleScore, ArticleStructure, ScoreDetails
from .logger import logger


def _score_structure(article: ArticleStructure, warnings: List[str]) -> int:
    score = 0

    # H1 title present and non-trivial
    if article.title and len(article.title) > 5:
        score += 3
    else:
        warnings.append("Title is too short or missing")

    # Meta description 150-160 chars
    meta_len = len(article.meta_description)
    if 150 <= meta_len <= 160:
        score += 3
    elif 100 <= meta_len < 150:
        score += 2
        warnings.append(
            f"Meta description too short ({meta_len} chars, target 150-160)"
        )
    elif meta_len > 160:
        score += 2
        warnings.append(f"Meta description too long ({meta_len} chars, target 150-160)")
    else:
        score += 0
        warnings.append(f"Meta description very short ({meta_len} chars)")

    # Introduction 3-5 lines
    intro_lines = [l for l in article.introduction.splitlines() if l.strip()]
    if 3 <= len(intro_lines) <= 5:
        score += 3
    elif len(intro_lines) < 3:
        score += 1
        warnings.append("Introduction too short (less than 3 lines)")
    else:
        score += 1
        warnings.append("Introduction too long (more than 5 lines)")

    # Table of contents present
    if article.table_of_contents and len(article.table_of_contents) >= 3:
        score += 2
    else:
        warnings.append("Table of contents missing or has fewer than 3 entries")

    # Minimum 4 H2 sections
    h2_count = len(article.sections)
    if h2_count >= 6:
        score += 6
    elif h2_count >= 4:
        score += 4
    elif h2_count >= 2:
        score += 2
        warnings.append(f"Only {h2_count} H2 sections (minimum 4 required)")
    else:
        score += 0
        warnings.append(f"Only {h2_count} H2 sections (minimum 4 required)")

    # FAQ minimum 5 questions
    faq_count = len(article.faq)
    if faq_count >= 7:
        score += 5
    elif faq_count >= 5:
        score += 4
    elif faq_count >= 3:
        score += 2
        warnings.append(f"FAQ has only {faq_count} questions (minimum 5 required)")
    else:
        score += 0
        warnings.append(f"FAQ too short ({faq_count} questions, minimum 5 required)")

    # Key takeaways 5-8 bullets
    kt_count = len(article.key_takeaways)
    if 5 <= kt_count <= 8:
        score += 3
    elif kt_count < 5:
        score += 1
        warnings.append(f"Only {kt_count} key takeaways (target 5-8)")
    else:
        score += 2
        warnings.append(f"{kt_count} key takeaways (target 5-8)")

    return min(score, 25)


def _score_readability(article: ArticleStructure, warnings: List[str]) -> int:
    """
    Use the Flesch reading ease score as the main readability signal.
    Score mapping is adapted for both English and French text.
    """
    # Assemble full body text for analysis
    body_parts = [article.introduction]
    for section in article.sections:
        body_parts.append(section.content)
        for sub in section.subsections:
            body_parts.append(sub.content)
    full_text = " ".join(body_parts)

    if len(full_text) < 100:
        warnings.append("Article body is too short for reliable readability scoring")
        return 5

    try:
        flesch = textstat.flesch_reading_ease(full_text)
    except Exception:
        return 10

    # Flesch thresholds (higher = easier to read)
    if flesch >= 60:
        score = 20
    elif flesch >= 50:
        score = 17
    elif flesch >= 40:
        score = 14
    elif flesch >= 30:
        score = 10
    else:
        score = 6
        warnings.append(
            f"Low readability score (Flesch: {flesch:.1f}). "
            "Consider shorter sentences and simpler vocabulary."
        )

    # Penalty for very long paragraphs
    long_paras = sum(
        1
        for section in article.sections
        for para in [section.content] + [s.content for s in section.subsections]
        if len(para.split()) > 120
    )
    if long_paras > 2:
        score = max(0, score - 3)
        warnings.append(f"{long_paras} paragraphs exceed 120 words. Break them up.")

    return min(score, 20)


def _score_sources(article: ArticleStructure, warnings: List[str]) -> int:
    sources = article.sources
    score = 0

    # Count
    n = len(sources)
    if n >= 5:
        score += 10
    elif n >= 3:
        score += 8
    elif n >= 1:
        score += 4
        warnings.append(f"Only {n} source(s). Minimum 3 recommended.")
    else:
        score += 0
        warnings.append("No sources found.")

    # HTTPS only
    https_count = sum(1 for s in sources if s.startswith("https://"))
    if https_count == n and n > 0:
        score += 4
    elif https_count > 0:
        score += 2
        warnings.append(f"{n - https_count} source(s) do not use HTTPS.")

    # Domain diversity
    domains = set()
    for src in sources:
        try:
            domain = urlparse(src).netloc
            if domain:
                domains.add(domain)
        except Exception:
            pass

    if len(domains) >= 3:
        score += 4
    elif len(domains) == 2:
        score += 2
    elif len(domains) == 1:
        score += 1
        warnings.append("All sources come from the same domain. Diversify sources.")

    # Plausible URL format (basic check)
    valid = sum(1 for s in sources if re.match(r"https?://[^\s/$.?#].[^\s]*", s))
    if valid == n and n > 0:
        score += 2
    elif valid < n:
        warnings.append(f"{n - valid} source URL(s) appear malformed.")

    return min(score, 20)


def _score_llm_friendly(article: ArticleStructure, warnings: List[str]) -> int:
    score = 0
    full_text = " ".join(
        [article.introduction]
        + [s.content for s in article.sections]
        + [s.content for sec in article.sections for s in sec.subsections]
        + [item.a for item in article.faq]
    )

    # Direct intro (does not start with "In this article" filler)
    filler_patterns = [
        r"(?i)^(in this article|dans cet article|welcome to|bienvenue|"
        r"today we|aujourd'hui nous|this guide will)"
    ]
    intro_start = article.introduction.strip()
    has_filler = any(re.match(p, intro_start) for p in filler_patterns)
    if not has_filler:
        score += 4
    else:
        score -= 2
        warnings.append("Introduction starts with filler text (not GEO-friendly)")

    # Lists / enumerations in body
    list_markers = len(re.findall(r"(?m)^[-*+]\s|^\d+\.\s", full_text))
    if list_markers >= 10:
        score += 4
    elif list_markers >= 5:
        score += 3
    elif list_markers >= 2:
        score += 1
    else:
        warnings.append("Very few lists or enumerations. Add bullet points for GEO.")

    # FAQ quality: answers that are long enough to be useful
    good_faq_answers = sum(1 for item in article.faq if len(item.a.split()) >= 20)
    if good_faq_answers >= len(article.faq) * 0.8 and article.faq:
        score += 4
    elif good_faq_answers >= len(article.faq) * 0.5:
        score += 2
    else:
        warnings.append("Many FAQ answers are too short. Aim for 20+ words per answer.")

    # Named entities: presence of brand/product names, standards, tools
    # Heuristic: capitalised multi-word phrases or ALL-CAPS acronyms
    entities = re.findall(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b|\b[A-Z]{2,6}\b", full_text
    )
    unique_entities = set(entities)
    if len(unique_entities) >= 10:
        score += 4
    elif len(unique_entities) >= 5:
        score += 3
    elif len(unique_entities) >= 2:
        score += 1
    else:
        warnings.append(
            "Few named entities detected. Add brand names, standards, tools."
        )

    # Key takeaways present and long enough
    good_kt = sum(1 for kt in article.key_takeaways if len(kt.split()) >= 8)
    if good_kt >= 4:
        score += 4
    elif good_kt >= 2:
        score += 2
    else:
        warnings.append("Key takeaways are too brief. Each should be at least 8 words.")

    return max(0, min(score, 20))


def _score_duplication_default() -> int:
    """
    Default full score: assume no duplication until deduplicator runs.
    The deduplicator calls update_duplication_score() to adjust.
    """
    return 15


def score_article(
    article: ArticleStructure,
    duplication_score: int = 15,
) -> ArticleScore:
    """
    Compute a full quality score for the given article.

    Parameters
    ----------
    article           : validated ArticleStructure
    duplication_score : pre-computed duplication score (0-15), default 15

    Returns
    -------
    ArticleScore with total, per-criterion breakdown, and warnings list
    """
    warnings: List[str] = []

    structure = _score_structure(article, warnings)
    readability = _score_readability(article, warnings)
    sources = _score_sources(article, warnings)
    llm_friendly = _score_llm_friendly(article, warnings)
    duplication = max(0, min(duplication_score, 15))

    total = structure + readability + sources + llm_friendly + duplication

    details = ScoreDetails(
        structure=structure,
        readability=readability,
        sources=sources,
        llm_friendly=llm_friendly,
        duplication=duplication,
    )

    logger.debug(
        "[scorer] %s | total=%d | struct=%d read=%d src=%d llm=%d dup=%d",
        article.title[:60],
        total,
        structure,
        readability,
        sources,
        llm_friendly,
        duplication,
    )

    return ArticleScore(total=total, details=details, warnings=warnings)
