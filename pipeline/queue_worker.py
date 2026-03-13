"""
pipeline/queue_worker.py
------------------------
Bonus: Queue / batch processing using Python's multiprocessing.

Architecture
------------
- A single producer enqueues TopicInput items into a JoinableQueue.
- N worker processes consume from the queue and return results via a result Queue.
- A collector process reads results and handles errors / retries.

Usage
-----
Run this module's process_topics_parallel() from generate.py when
USE_MULTIPROCESSING=true.

Caveats
-------
- Each worker imports litellm and opens its own API connections.
- Multiprocessing with LLM APIs risks hitting rate limits faster.
- Keep MAX_WORKERS <= 2 to stay within free-tier rate limits.
- Forking is disabled on Windows (spawn context is used instead).
"""

import multiprocessing as mp
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Config
from .logger import logger
from .models import ArticleStructure, TopicInput


def _worker_fn(
    topic_input: TopicInput,
    config_dict: Dict[str, Any],
    source_hints: Optional[List[str]],
    rag_context: Optional[str],
) -> Tuple[str, Optional[ArticleStructure], Optional[str]]:
    """
    Top-level function executed by each worker process.
    Must be importable at module level (no lambdas / closures).

    Returns (topic, ArticleStructure, error_message)
    """
    # Re-instantiate config inside the subprocess
    from .config import Config
    from .generator import generate_article

    config = Config(**config_dict)

    try:
        article = generate_article(
            config=config,
            topic=topic_input.topic,
            language=topic_input.language,
            tone=topic_input.tone,
            source_hints=source_hints,
            rag_context=rag_context,
        )
        return topic_input.topic, article, None
    except Exception as exc:
        tb = traceback.format_exc()
        return topic_input.topic, None, str(exc)


def process_topics_parallel(
    topics: List[TopicInput],
    config: Config,
    source_hints_map: Optional[Dict[str, List[str]]] = None,
    rag_context_map: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[str, bool], None]] = None,
) -> Tuple[Dict[str, ArticleStructure], Dict[str, str]]:
    """
    Process multiple topics in parallel using a multiprocessing pool.

    Parameters
    ----------
    topics             : list of TopicInput items to process
    config             : pipeline configuration
    source_hints_map   : optional {topic: [urls]} mapping
    rag_context_map    : optional {topic: context_str} mapping
    progress_callback  : optional fn(topic, success) called after each result

    Returns
    -------
    articles : dict {topic: ArticleStructure}
    errors   : dict {topic: error_message}
    """
    n_workers = min(config.max_workers, len(topics))
    logger.info("[queue] Processing %d topics with %d workers.", len(topics), n_workers)

    # Serialise config to dict so it can cross process boundary
    import dataclasses

    config_dict = dataclasses.asdict(config)

    # Build argument list
    args_list = [
        (
            t,
            config_dict,
            (source_hints_map or {}).get(t.topic),
            (rag_context_map or {}).get(t.topic),
        )
        for t in topics
    ]

    articles: Dict[str, ArticleStructure] = {}
    errors: Dict[str, str] = {}

    # Use spawn context for cross-platform compatibility (Windows + macOS)
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
        for result in pool.starmap(_worker_fn, args_list):
            topic, article, error = result
            if error:
                logger.error("[queue] Failed: %s — %s", topic, error[:120])
                errors[topic] = error
                if progress_callback:
                    progress_callback(topic, False)
            else:
                articles[topic] = article  # type: ignore[assignment]
                logger.info("[queue] Done: %s", topic)
                if progress_callback:
                    progress_callback(topic, True)

    logger.info(
        "[queue] Finished. %d success, %d failures.",
        len(articles),
        len(errors),
    )
    return articles, errors
