"""
pipeline/config.py
------------------
Centralised configuration loaded from environment variables.
All secrets stay in .env, never in code.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:

    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "120"))
    )

    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("RETRY_DELAY", "2.0"))
    )

    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    )

    use_multiprocessing: bool = field(
        default_factory=lambda: os.getenv("USE_MULTIPROCESSING", "false").lower()
        == "true"
    )
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "2")))

    enable_sources_retrieval: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SOURCES_RETRIEVAL", "false").lower()
        == "true"
    )

    enable_wordpress: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WORDPRESS", "false").lower() == "true"
    )
    wordpress_url: str = field(default_factory=lambda: os.getenv("WORDPRESS_URL", ""))
    wordpress_user: str = field(default_factory=lambda: os.getenv("WORDPRESS_USER", ""))
    wordpress_app_password: str = field(
        default_factory=lambda: os.getenv("WORDPRESS_APP_PASSWORD", "")
    )

    enable_rag: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RAG", "false").lower() == "true"
    )
    rag_docs_path: str = field(
        default_factory=lambda: os.getenv("RAG_DOCS_PATH", "./rag_docs")
    )
