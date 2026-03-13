"""
pipeline/__init__.py
--------------------
Public API re-exports for the geo_pipeline package.
"""

from .config import Config
from .deduplicator import run_deduplication
from .exporter import build_article, write_article, write_summary
from .generator import generate_article
from .models import Article, ArticleScore, ArticleStructure, Summary, TopicInput
from .scorer import score_article

__all__ = [
    "Config",
    "generate_article",
    "score_article",
    "run_deduplication",
    "build_article",
    "write_article",
    "write_summary",
    "Article",
    "ArticleScore",
    "ArticleStructure",
    "Summary",
    "TopicInput",
]
