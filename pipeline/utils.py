"""
pipeline/utils.py
-----------------
Shared utility functions used across pipeline modules.
"""

import json
import re
import unicodedata
from typing import Any, Dict, Optional

from .models import ArticleStructure


def make_slug(text: str, language: str = "en") -> str:
    """
    Generate a URL-safe slug without external dependencies.
    Handles accents and special characters (Python 3 compatible).
    """
    # Normalisation NFKD pour séparer les caractères accentués de leurs accents
    text = unicodedata.normalize("NFKD", text)
    # Encodage en ASCII en ignorant les caractères non-ASCII (les accents)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Passage en minuscule
    text = text.lower()
    # Remplacement de tout ce qui n'est pas alphanumérique par un tiret
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Suppression des tirets multiples ou en début/fin de chaîne
    text = re.sub(r"-+", "-", text).strip("-")

    # Limitation de la longueur à 200 caractères sans couper de mot
    if len(text) <= 200:
        return text

    # On coupe à 200 et on recule jusqu'au dernier tiret pour ne pas couper un mot
    shortened = text[:200]
    last_dash = shortened.rfind("-")
    if last_dash > 0:
        return shortened[:last_dash]
    return shortened


def assemble_markdown(article: ArticleStructure) -> str:
    """
    Build a single Markdown string from the structured article data.
    This is the canonical Markdown representation stored in content_markdown.
    """
    lines: list[str] = []


    # H1 title
    lines.append(f"# {article.title}\n")

    # Meta description block (custom syntax, GEO-friendly)
    lines.append("> **Meta description:** " + article.meta_description + "\n")

    # Introduction
    lines.append("## Introduction\n")
    lines.append(article.introduction + "\n")

    # Table of contents
    lines.append("## Table of Contents\n")
    for i, entry in enumerate(article.table_of_contents, 1):
        lines.append(f"{i}. {entry}")
    lines.append("")

    # Main sections
    for section in article.sections:
        lines.append(f"\n## {section.h2}\n")
        if section.content:
            lines.append(section.content + "\n")
        for sub in section.subsections:
            lines.append(f"\n### {sub.h3}\n")
            if sub.content:
                lines.append(sub.content + "\n")

    # FAQ
    lines.append("\n## FAQ\n")
    for item in article.faq:
        lines.append(f"**Q: {item.q}**\n")
        lines.append(f"A: {item.a}\n")

    # Key takeaways
    lines.append("\n## Key Takeaways\n")
    for point in article.key_takeaways:
        lines.append(f"- {point}")
    lines.append("")

    # Sources
    lines.append("\n## Sources\n")
    for i, src in enumerate(article.sources, 1):
        lines.append(f"{i}. {src}")
    lines.append("")

    # Author block
    lines.append("\n## About the Author\n")
    lines.append(f"**{article.author.name}**\n")
    lines.append(article.author.bio + "\n")
    if article.author.methodology:
        lines.append("\n**Methodology:**\n")
        for m in article.author.methodology:
            lines.append(f"- {m}")
    lines.append("")

    return "\n".join(lines)


def assemble_html(
    article: ArticleStructure,
    slug: str,
    og_title: Optional[str] = None,
    og_description: Optional[str] = None,
) -> str:
    """
    Build a full HTML page from the structured article.
    Includes og:title and og:description meta tags (SEO bonus).
    """
    og_title = og_title or article.title
    og_description = og_description or article.meta_description

    sections_html = ""
    for section in article.sections:
        sections_html += f"<h2>{_esc(section.h2)}</h2>\n"
        sections_html += f"<p>{_esc(section.content)}</p>\n"
        for sub in section.subsections:
            sections_html += f"<h3>{_esc(sub.h3)}</h3>\n"
            sections_html += f"<p>{_esc(sub.content)}</p>\n"

    faq_html = ""
    for item in article.faq:
        faq_html += (
            f"<div class='faq-item'>"
            f"<strong>Q: {_esc(item.q)}</strong>"
            f"<p>A: {_esc(item.a)}</p>"
            f"</div>\n"
        )

    takeaways_html = "\n".join(f"<li>{_esc(p)}</li>" for p in article.key_takeaways)

    sources_html = "\n".join(
        f'<li><a href="{src}" rel="nofollow noopener">{src}</a></li>'
        for src in article.sources
    )

    toc_html = "\n".join(f"<li>{_esc(e)}</li>" for e in article.table_of_contents)

    methodology_html = "\n".join(
        f"<li>{_esc(m)}</li>" for m in article.author.methodology
    )

    return f"""<!DOCTYPE html>
<html lang="{_language_code(article)}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="{_esc(article.meta_description)}" />
  <meta property="og:title" content="{_esc(og_title)}" />
  <meta property="og:description" content="{_esc(og_description)}" />
  <meta property="og:type" content="article" />
  <meta property="og:url" content="https://example.com/{slug}" />
  <title>{_esc(article.title)}</title>
</head>
<body>
  <article>
    <h1>{_esc(article.title)}</h1>
    <p class="meta-description"><em>{_esc(article.meta_description)}</em></p>
    <section class="introduction">
      <h2>Introduction</h2>
      <p>{_esc(article.introduction)}</p>
    </section>
    <nav class="table-of-contents">
      <h2>Table of Contents</h2>
      <ol>{toc_html}</ol>
    </nav>
    <section class="body">{sections_html}</section>
    <section class="faq">
      <h2>FAQ</h2>
      {faq_html}
    </section>
    <section class="key-takeaways">
      <h2>Key Takeaways</h2>
      <ul>{takeaways_html}</ul>
    </section>
    <section class="sources">
      <h2>Sources</h2>
      <ol>{sources_html}</ol>
    </section>
    <section class="author">
      <h2>About the Author</h2>
      <p><strong>{_esc(article.author.name)}</strong></p>
      <p>{_esc(article.author.bio)}</p>
      <h3>Methodology</h3>
      <ul>{methodology_html}</ul>
    </section>
  </article>
</body>
</html>"""


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from an LLM text response.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries
    start = text.find("{")
    if start == -1:
        return None

    # Walk forward to find matching brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _language_code(article: ArticleStructure) -> str:
    """Return the language code attribute value."""
    return "fr" if getattr(article, "language", "en") == "fr" else "en"
