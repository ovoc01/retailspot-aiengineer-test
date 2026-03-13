"""
generate.py
-----------
CLI entry point for the GEO/GSO article generation pipeline.

Usage
-----
    python generate.py --input topics.json --output ./out

Optional flags
--------------
    --model          Override LLM model (e.g. "anthropic/claude-3-haiku-20240307")
    --threshold      Similarity threshold for deduplication (default 0.85)
    --workers        Number of parallel workers (default from .env)
    --parallel       Force multiprocessing mode on
    --no-rag         Disable RAG even if ENABLE_RAG=true
    --no-sources     Disable sources retrieval even if ENABLE_SOURCES_RETRIEVAL=true
    --no-wordpress   Disable WordPress export even if ENABLE_WORDPRESS=true
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pipeline.config import Config
from pipeline.deduplicator import run_deduplication
from pipeline.exporter import build_article, write_article, write_summary
from pipeline.generator import generate_article
from pipeline.logger import logger
from pipeline.models import (
    Article,
    ArticleStructure,
    DuplicatePair,
    Summary,
    SummaryEntry,
    TopicInput,
)
from pipeline.scorer import score_article

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEO/GSO Article Pipeline — generates, scores, and exports articles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to topics.json input file",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory (will be created if it does not exist)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="LLM model string (overrides LLM_MODEL env var)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Cosine similarity threshold for duplicate detection (default 0.85)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (overrides MAX_WORKERS env var)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable multiprocessing (overrides USE_MULTIPROCESSING env var)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG even if ENABLE_RAG=true in .env",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Disable sources retrieval",
    )
    parser.add_argument(
        "--no-wordpress",
        action="store_true",
        help="Disable WordPress export",
    )
    return parser.parse_args()


def load_topics(path: str) -> List[TopicInput]:
    if not os.path.isfile(path):
        logger.error("Input file not found: %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    topics = [TopicInput(**item) for item in raw]
    logger.info("Loaded %d topic(s) from %s", len(topics), path)
    return topics


def _init_rag(config: Config, disabled: bool):
    if disabled or not config.enable_rag:
        return None
    from pipeline.rag import RAGRetriever

    retriever = RAGRetriever(docs_path=config.rag_docs_path)
    retriever.build_index()
    return retriever if retriever.is_ready else None


def _init_wordpress(config: Config, disabled: bool):
    if disabled or not config.enable_wordpress:
        return None
    from pipeline.wordpress import WordPressClient

    return WordPressClient(config)


def process_single_topic(
    topic: TopicInput,
    config: Config,
    rag_retriever=None,
    source_hints: Optional[List[str]] = None,
) -> Tuple[Optional[ArticleStructure], Optional[str]]:
    """
    Generate one article. Returns (structure, error_message).
    """
    rag_context: Optional[str] = None
    if rag_retriever is not None:
        rag_context = rag_retriever.retrieve(topic.topic, top_k=3)
        if rag_context:
            logger.debug("[main] RAG context injected (%d chars)", len(rag_context))

    try:
        structure = generate_article(
            config=config,
            topic=topic.topic,
            language=topic.language,
            tone=topic.tone,
            source_hints=source_hints,
            rag_context=rag_context,
        )
        return structure, None
    except Exception as exc:
        logger.error("[main] Generation failed for '%s': %s", topic.topic, exc)
        return None, str(exc)


def run_pipeline(args: argparse.Namespace) -> None:
    config = Config()
    if args.model:
        config.llm_model = args.model
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.workers is not None:
        config.max_workers = args.workers
    if args.parallel:
        config.use_multiprocessing = True

    logger.info(
        "Model: %s | Threshold: %.2f", config.llm_model, config.similarity_threshold
    )

    topics = load_topics(args.input)
    os.makedirs(args.output, exist_ok=True)

    rag_retriever = _init_rag(config, disabled=args.no_rag)
    wp_client = _init_wordpress(config, disabled=args.no_wordpress)

    source_hints_map: Dict[str, List[str]] = {}
    if not args.no_sources and config.enable_sources_retrieval:
        from pipeline.sources import fetch_sources

        logger.info("[main] Sources retrieval enabled. Fetching URLs...")
        for t in topics:
            hints = fetch_sources(t.topic, language=t.language, max_results=5)
            if hints:
                source_hints_map[t.topic] = hints

    structures: Dict[str, ArticleStructure] = {}
    errors: Dict[str, str] = {}
    topic_map: Dict[str, TopicInput] = {t.topic: t for t in topics}

    if config.use_multiprocessing and len(topics) > 1:
        from pipeline.queue_worker import process_topics_parallel

        rag_context_map: Dict[str, str] = {}
        if rag_retriever:
            for t in topics:
                ctx = rag_retriever.retrieve(t.topic, top_k=3)
                if ctx:
                    rag_context_map[t.topic] = ctx

        parallel_results, parallel_errors = process_topics_parallel(
            topics=topics,
            config=config,
            source_hints_map=source_hints_map if source_hints_map else None,
            rag_context_map=rag_context_map if rag_context_map else None,
        )
        structures.update(parallel_results)
        errors.update(parallel_errors)

    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating articles", total=len(topics))
            for t in topics:
                progress.update(task, description=f"[cyan]{t.topic[:55]}")
                hints = source_hints_map.get(t.topic)
                structure, error = process_single_topic(
                    topic=t,
                    config=config,
                    rag_retriever=rag_retriever,
                    source_hints=hints,
                )
                if structure:
                    structures[t.topic] = structure
                    # SAUVEGARDE IMMÉDIATE : On sauve l'article dès qu'il est prêt
                    try:
                        temp_score = score_article(structure, duplication_score=15)
                        article = build_article(
                            structure=structure,
                            score=temp_score,
                            topic=t.topic,
                            language=t.language,
                            tone=t.tone,
                        )
                        write_article(article, args.output)
                        logger.info("[export] Saved: %s", article.slug)
                    except Exception as e:
                        logger.error(
                            "[export] Immediate save failed for '%s': %s", t.topic, e
                        )
                else:
                    errors[t.topic] = error or "Unknown error"
                progress.advance(task)

    logger.info(
        "Generation complete: %d success, %d failure(s).",
        len(structures),
        len(errors),
    )

    if not structures:
        logger.warning("[main] No articles were generated. Skipping further steps.")
        return

    # PHASE DE DÉDUPLICATION
    scored_structures: Dict[str, Tuple[ArticleStructure, Any]] = {}
    if len(structures) >= 2:
        logger.info("[main] Running final scoring and deduplication check...")
        slugs_list = []
        texts_list = []
        from pipeline.utils import assemble_markdown, make_slug

        for topic_str, structure in structures.items():
            slugs_list.append(make_slug(structure.title))
            texts_list.append(assemble_markdown(structure))

        try:
            duplicate_pairs, dup_scores_by_slug = run_deduplication(
                slugs=slugs_list,
                texts=texts_list,
                threshold=config.similarity_threshold,
            )

            for topic_str, structure in structures.items():
                slug = make_slug(structure.title)
                dup_score = dup_scores_by_slug.get(slug, 15)
                final_score = score_article(structure, duplication_score=dup_score)
                scored_structures[topic_str] = (structure, final_score)
        except Exception as e:
            logger.error(
                "[main] Deduplication/Refinement failed (using default scores): %s", e
            )
            duplicate_pairs = []
            for topic_str, structure in structures.items():
                scored_structures[topic_str] = (structure, score_article(structure, 15))
    else:
        duplicate_pairs = []
        for topic_str, structure in structures.items():
            scored_structures[topic_str] = (structure, score_article(structure, 15))

    # Mise à jour finale des articles (avec les scores définitifs)
    articles: List[Article] = []
    for topic_str, (structure, score) in scored_structures.items():
        t = topic_map[topic_str]
        try:
            article = build_article(structure, score, topic_str, t.language, t.tone)
            # print(article)
            write_article(article, args.output)
            articles.append(article)
        except Exception as e:
            logger.error("[main] Final export failed for %s: %s", topic_str, e)

    if wp_client:
        for article in articles:
            result = wp_client.publish(article)
            if result:
                logger.info("[wp] Published: %s", article.slug)

    entries: List[SummaryEntry] = []
    for article in articles:
        entries.append(
            SummaryEntry(
                slug=article.slug,
                topic=article.topic,
                language=article.language,
                score=article.score.total if article.score else None,
                warnings=article.score.warnings if article.score else [],
                status="success",
            )
        )
    for topic_str, err_msg in errors.items():
        entries.append(
            SummaryEntry(
                slug="",
                topic=topic_str,
                language=topic_map[topic_str].language,
                status="error",
                error=err_msg,
            )
        )

    scores = [e.score for e in entries if e.score is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    summary = Summary(
        total_articles=len(topics),
        successful=len(articles),
        failed=len(errors),
        average_score=avg_score,
        duplicates_detected=duplicate_pairs,
        articles=entries,
        errors=[{"topic": k, "error": v} for k, v in errors.items()],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    write_summary(summary, args.output)

    _print_report(articles, duplicate_pairs, errors, avg_score)


def _print_report(
    articles: List[Article],
    duplicate_pairs: List[DuplicatePair],
    errors: Dict[str, str],
    avg_score: Optional[float],
) -> None:
    console.print()
    console.rule("[bold green]Pipeline complete")

    table = Table(title="Article Scores", show_header=True, header_style="bold cyan")
    table.add_column("Slug", style="dim", no_wrap=True)
    table.add_column("Lang", width=5)
    table.add_column("Total", justify="right")
    table.add_column("Struct", justify="right")
    table.add_column("Read", justify="right")
    table.add_column("Src", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Dup", justify="right")
    table.add_column("Warnings", style="yellow")

    for a in articles:
        s = a.score
        if s is None:
            continue
        total_style = (
            "green" if s.total >= 75 else ("yellow" if s.total >= 55 else "red")
        )
        table.add_row(
            a.slug[:40],
            a.language,
            f"[{total_style}]{s.total}[/{total_style}]",
            str(s.details.structure),
            str(s.details.readability),
            str(s.details.sources),
            str(s.details.llm_friendly),
            str(s.details.duplication),
            str(len(s.warnings)),
        )

    console.print(table)

    if avg_score is not None:
        console.print(f"\n[bold]Average score:[/bold] {avg_score}/100")

    if duplicate_pairs:
        console.print(
            f"\n[bold red]{len(duplicate_pairs)} duplicate pair(s) detected:[/bold red]"
        )
        for pair in duplicate_pairs:
            console.print(
                f"  {pair.slug_a} <-> {pair.slug_b} (similarity={pair.similarity:.3f})"
            )

    if errors:
        console.print(f"\n[bold red]{len(errors)} topic(s) failed:[/bold red]")
        for topic, err in errors.items():
            console.print(f"  [red]{topic}[/red]: {err[:100]}")

    console.print()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
