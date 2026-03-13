"""
pipeline/exporter.py
--------------------
Export pipeline: writes articles to disk in three formats.

Outputs per article:
  - /articles/<slug>.md       — Markdown version
  - /articles/<slug>.html     — Full HTML with SEO meta tags (bonus)
  - /json/<slug>.json         — Publication-ready JSON payload

Global output:
  - summary.json              — Scores, warnings, duplicates, errors
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .logger import logger
from .models import Article, ArticleScore, ArticleStructure, Author, FAQItem, Summary
from .utils import assemble_html, assemble_markdown, make_slug


def _ensure_dirs(output_dir: str) -> None:
    for sub in ("articles", "json"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


def build_article(
    structure: ArticleStructure,
    score: ArticleScore,
    topic: str,
    language: str,
    tone: str,
) -> Article:
    """
    Convert a raw ArticleStructure + score into an exportable Article.
    """

    slug = make_slug(structure.title, language)

    markdown = assemble_markdown(structure)

    html = assemble_html(
        structure,
        slug,
        og_title=structure.title,
        og_description=structure.meta_description,
    )

    return Article(
        slug=slug,
        title=structure.title,
        meta_description=structure.meta_description,
        content_markdown=markdown,
        content_html=html,
        faq=structure.faq,
        key_takeaways=structure.key_takeaways,
        sources=structure.sources,
        author=structure.author,
        score=score,
        language=language,
        tone=tone,
        topic=topic,
        og_title=structure.title,
        og_description=structure.meta_description,
    )


def write_article(article: Article, output_dir: str) -> None:
    """Write all output files for a single article."""
    _ensure_dirs(output_dir)

    md_path = os.path.join(output_dir, "articles", f"{article.slug}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(article.content_markdown)
    logger.debug("[exporter] Markdown written: %s", md_path)

    if article.content_html:
        html_path = os.path.join(output_dir, "articles", f"{article.slug}.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(article.content_html)
        logger.debug("[exporter] HTML written: %s", html_path)

    json_path = os.path.join(output_dir, "json", f"{article.slug}.json")
    payload = _build_json_payload(article)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.debug("[exporter] JSON written: %s", json_path)


def _build_json_payload(article: Article) -> Dict[str, Any]:
    """Build the publication-ready JSON payload as specified."""
    score_dict: Dict[str, Any] = {}
    if article.score:
        score_dict = {
            "total": article.score.total,
            "details": {
                "structure": article.score.details.structure,
                "readability": article.score.details.readability,
                "sources": article.score.details.sources,
                "llm_friendly": article.score.details.llm_friendly,
                "duplication": article.score.details.duplication,
            },
            "warnings": article.score.warnings,
        }

    return {
        "slug": article.slug,
        "title": article.title,
        "meta_description": article.meta_description,
        "content_markdown": article.content_markdown,
        "faq": [{"q": item.q, "a": item.a} for item in article.faq],
        "key_takeaways": article.key_takeaways,
        "sources": article.sources,
        "author": {
            "name": article.author.name,
            "bio": article.author.bio,
            "methodology": article.author.methodology,
        },
        "score": score_dict,
        "language": article.language,
        "tone": article.tone,
        "topic": article.topic,
        "og_title": article.og_title,
        "og_description": article.og_description,
    }


def write_summary(summary: Summary, output_dir: str) -> None:
    """Write global summary.json."""
    _ensure_dirs(output_dir)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary.model_dump(), fh, ensure_ascii=False, indent=2)
    logger.info("[exporter] Summary written: %s", path)
