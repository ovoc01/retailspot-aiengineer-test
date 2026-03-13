"""
pipeline/wordpress.py
---------------------
Bonus: WordPress REST API client.

Publishes articles to a WordPress site via the /wp-json/wp/v2/posts endpoint
using Application Passwords for authentication (WP 5.6+).

Mock mode (ENABLE_WORDPRESS=false or missing credentials):
  Logs the full payload that *would* be sent, so the integration can be
  verified without a live WordPress instance.
"""

import json
from typing import Any, Dict, Optional

import requests
from requests.auth import HTTPBasicAuth

from .config import Config
from .logger import logger
from .models import Article


_WP_POSTS_ENDPOINT = "/wp-json/wp/v2/posts"
_TIMEOUT = 15


class WordPressClient:
    """Thin WordPress REST API client."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._is_mock = (
            not config.enable_wordpress
            or not config.wordpress_url
            or not config.wordpress_user
            or not config.wordpress_app_password
        )
        if self._is_mock:
            logger.info(
                "[wordpress] Running in MOCK mode — no real API calls will be made."
            )
        else:
            logger.info(
                "[wordpress] WordPress client initialised for %s", config.wordpress_url
            )

    def publish(self, article: Article) -> Optional[Dict[str, Any]]:
        """
        Publish an article to WordPress.

        Returns the API response dict on success (or mock payload in mock mode).
        Returns None on failure.
        """
        payload = self._build_payload(article)

        if self._is_mock:
            return self._mock_publish(article.slug, payload)

        return self._real_publish(payload)

    def _build_payload(self, article: Article) -> Dict[str, Any]:
        """Build the WordPress REST API post payload."""
        # Convert markdown to HTML for WordPress
        content = article.content_html or article.content_markdown

        return {
            "title": article.title,
            "content": content,
            "status": "draft",  # safer default: publish manually after review
            "excerpt": article.meta_description,
            "slug": article.slug,
            "meta": {
                "geo_score": article.score.total if article.score else None,
                "geo_language": article.language,
                "geo_tone": article.tone,
            },
            # Yoast SEO / RankMath compatible fields
            "yoast_head_json": {
                "og_title": article.og_title or article.title,
                "og_description": article.og_description or article.meta_description,
            },
        }

    def _real_publish(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = self.config.wordpress_url.rstrip("/") + _WP_POSTS_ENDPOINT
        auth = HTTPBasicAuth(
            self.config.wordpress_user,
            self.config.wordpress_app_password,
        )

        try:
            response = requests.post(
                url,
                json=payload,
                auth=auth,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                "[wordpress] Published post id=%s slug=%s",
                data.get("id"),
                data.get("slug"),
            )
            return data
        except requests.HTTPError as exc:
            logger.error(
                "[wordpress] HTTP error publishing %s: %s — %s",
                payload.get("slug"),
                exc,
                exc.response.text[:200] if exc.response else "",
            )
        except requests.RequestException as exc:
            logger.error("[wordpress] Request failed: %s", exc)

        return None

    def _mock_publish(self, slug: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "[wordpress][MOCK] Would POST to %s%s",
            self.config.wordpress_url or "https://your-site.com",
            _WP_POSTS_ENDPOINT,
        )
        logger.debug(
            "[wordpress][MOCK] Payload for '%s':\n%s",
            slug,
            json.dumps(payload, ensure_ascii=False, indent=2)[:500] + "...",
        )
        # Return a fake response that mirrors WordPress API shape
        return {
            "id": 0,
            "slug": slug,
            "status": "draft",
            "link": f"{self.config.wordpress_url or 'https://example.com'}/?p=0",
            "_mock": True,
        }
