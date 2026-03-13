"""
pipeline/generator.py
---------------------
Article generation via litellm (provider-agnostic LLM calls).

Key design decisions:
- Structured JSON output via explicit schema in prompt
- json_mode enabled for compatible models (openai, etc.)
- Tenacity-based retry with exponential backoff
- RAG context injection (optional bonus)
- Real source hints injection (optional bonus)
"""

import json
import logging
from typing import List, Optional

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .logger import logger
from .models import ArticleStructure
from .utils import extract_json_from_text

# litellm: suppress all verbose and info logs
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_JSON_SCHEMA = """
{
  "title": "string (H1, compelling, GEO-optimized)",
  "meta_description": "string (150-160 chars, includes main keyword)",
  "introduction": "string (3-5 lines max, direct answer, no filler)",
  "table_of_contents": ["string", "..."],
  "sections": [
    {
      "h2": "string",
      "content": "string (dense, factual, entity-rich)",
      "subsections": [
        {"h3": "string", "content": "string"}
      ]
    }
  ],
  "faq": [
    {"q": "string", "a": "string (direct, 2-4 sentences)"}
  ],
  "key_takeaways": ["string", "..."],
  "sources": ["https://...", "..."],
  "author": {
    "name": "string (fictional expert name)",
    "bio": "string (2-3 lines, domain expertise)",
    "methodology": ["string", "..."]
  }
}
"""

_SYSTEM_PROMPT = """You are an expert content strategist and GEO/GSO writer.
GEO (Generative Engine Optimization) means writing content optimised to be cited by AI models such as ChatGPT, Gemini, and Perplexity.

Rules for GEO-ready content:
- Start every answer directly (no 'In this article...' filler)
- Include explicit named entities (brands, standards, tools, organisations)
- Keep paragraphs short: 2-4 sentences max
- Use bullet points and numbered lists where logical
- FAQ answers must be self-contained and directly answer the question
- Dense information per word: no padding
- Include factual claims with plausible citations
- Sources must be real-looking, relevant HTTPS URLs from authoritative domains

You MUST respond with ONLY valid JSON matching the exact schema provided.
Do not include markdown fences, preamble, or any text outside the JSON.
"""


def _build_user_prompt(
    topic: str,
    language: str,
    tone: str,
    source_hints: Optional[List[str]] = None,
    rag_context: Optional[str] = None,
) -> str:
    lang_instruction = (
        "Write the entire article in FRENCH."
        if language == "fr"
        else "Write the entire article in ENGLISH."
    )

    tone_map = {
        "expert": "authoritative and precise, targeting professionals",
        "friendly": "approachable and conversational, targeting a general audience",
        "technique": "technical and detailed, targeting developers or engineers",
        "technique": "technical and detailed, targeting developers or engineers",
        "pedagogique": "educational and step-by-step, targeting beginners",
        "pratique": "practical and actionable, focusing on how-to guidance",
        "technical": "technical and detailed, targeting professionals",
        "informative": "balanced and informative, targeting a broad audience",
    }
    tone_description = tone_map.get(tone.lower(), "informative and clear")

    prompt_parts = [
        f"Generate a full GEO-ready article on the following topic: {topic}",
        "",
        f"Language: {language.upper()} — {lang_instruction}",
        f"Tone: {tone} — {tone_description}",
        "",
        "Requirements:",
        "- Title: compelling, keyword-rich H1",
        "- Meta description: 150-160 characters, includes the main keyword",
        "- Introduction: 3-5 lines maximum, direct answer to the topic",
        "- Table of contents: list all main H2 sections",
        "- Body: minimum 4 H2 sections, each with rich factual content",
        "  * Add H3 subsections where relevant",
        "- FAQ: minimum 5 question/answer pairs",
        "- Key takeaways: 5-8 bullet points summarising the article",
        "- Sources: minimum 3 real, authoritative HTTPS URLs",
        "- Author block: fictional expert name, 2-3 line bio, 3-bullet methodology",
        "",
    ]

    if rag_context:
        prompt_parts += [
            "Use the following reference material to enrich the article (cite as needed):",
            "---",
            rag_context,
            "---",
            "",
        ]

    if source_hints:
        prompt_parts += [
            "Prefer citing these real URLs when they are relevant:",
            *[f"- {url}" for url in source_hints],
            "",
        ]

    prompt_parts += [
        "Return ONLY valid JSON matching this exact schema:",
        _JSON_SCHEMA,
    ]

    return "\n".join(prompt_parts)


def _supports_json_mode(model: str) -> bool:
    """Return True for models known to support response_format=json_object."""
    lower = model.lower()
    return any(
        provider in lower
        for provider in ("openai/", "gpt-", "azure/", "groq/", "together/")
    )


def _call_llm_raw(
    config: Config,
    messages: list,
    attempt: int = 0,
) -> str:
    """Single LLM call via litellm. Returns raw text response."""
    kwargs: dict = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.request_timeout,
    }

    if _supports_json_mode(config.llm_model):
        kwargs["response_format"] = {"type": "json_object"}

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(
        (
            litellm.exceptions.RateLimitError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.APIConnectionError,
        )
    ),
    reraise=True,
)
def _call_with_retry(config: Config, messages: list) -> str:
    return _call_llm_raw(config, messages)


def generate_article(
    config: Config,
    topic: str,
    language: str = "en",
    tone: str = "informative",
    source_hints: Optional[List[str]] = None,
    rag_context: Optional[str] = None,
) -> ArticleStructure:
    """
    Generate a structured article for the given topic.

    Parameters
    ----------
    config        : pipeline configuration
    topic         : article topic string
    language      : 'en' or 'fr'
    tone          : tone descriptor (expert, friendly, technical, etc.)
    source_hints  : optional list of real URLs to prefer as sources (bonus)
    rag_context   : optional RAG context string injected into the prompt (bonus)

    Returns
    -------
    ArticleStructure validated Pydantic model

    Raises
    ------
    ValueError if the LLM response cannot be parsed after max_retries
    """
    logger.info("[generator] Starting: %s [%s/%s]", topic, language, tone)

    user_prompt = _build_user_prompt(topic, language, tone, source_hints, rag_context)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 1):
        try:
            raw = _call_with_retry(config, messages)
            logger.debug("[generator] Raw response length: %d chars", len(raw))

            data = extract_json_from_text(raw)
            if data is None:
                raise ValueError("Could not extract JSON from LLM response")

            # Validate and return
            article = ArticleStructure(**data)
            logger.info("[generator] Done: %s (attempt %d)", topic, attempt)
            return article

        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "[generator] Parsing failed (attempt %d/%d): %s",
                attempt,
                config.max_retries,
                exc,
            )
            if attempt < config.max_retries:
                # Add error feedback to the conversation and retry
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": raw if "raw" in dir() else "",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response had an issue: {exc}. "
                            "Please return ONLY valid JSON matching the required schema, "
                            "with no extra text, no markdown fences."
                        ),
                    },
                ]

    raise ValueError(
        f"Failed to generate article for '{topic}' after {config.max_retries} attempts. "
        f"Last error: {last_error}"
    )
