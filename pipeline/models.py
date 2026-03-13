"""
pipeline/models.py
------------------
Pydantic v2 data models for the entire pipeline.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FAQItem(BaseModel):
    q: str = Field(..., description="Question")
    a: str = Field(..., description="Answer")


class Subsection(BaseModel):
    h3: str
    content: str


class Section(BaseModel):
    h2: str
    content: str
    subsections: List[Subsection] = Field(default_factory=list)


class Author(BaseModel):
    name: str
    bio: str
    methodology: List[str] = Field(default_factory=list)


class ArticleStructure(BaseModel):
    """Raw structured output from the LLM generation step."""

    title: str
    meta_description: str
    introduction: str
    table_of_contents: List[str]
    sections: List[Section]
    faq: List[FAQItem]
    key_takeaways: List[str]
    sources: List[str]
    author: Author

    @field_validator("meta_description")
    @classmethod
    def meta_length(cls, v: str) -> str:
        # Soft warning only – hard enforcement is in scorer
        return v.strip()

    @field_validator("sources")
    @classmethod
    def at_least_one_source(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one source is required")
        return v


class ScoreDetails(BaseModel):
    structure: int = Field(ge=0, le=25)
    readability: int = Field(ge=0, le=20)
    sources: int = Field(ge=0, le=20)
    llm_friendly: int = Field(ge=0, le=20)
    duplication: int = Field(ge=0, le=15)


class ArticleScore(BaseModel):
    total: int = Field(ge=0, le=100)
    details: ScoreDetails
    warnings: List[str] = Field(default_factory=list)


class Article(BaseModel):
    slug: str
    title: str
    meta_description: str
    content_markdown: str
    content_html: Optional[str] = None
    faq: List[FAQItem]
    key_takeaways: List[str]
    sources: List[str]
    author: Author
    score: Optional[ArticleScore] = None
    language: str
    tone: str
    topic: str
    og_title: Optional[str] = None
    og_description: Optional[str] = None


class TopicInput(BaseModel):
    topic: str
    language: str = "en"
    tone: str = "informative"


class DuplicatePair(BaseModel):
    slug_a: str
    slug_b: str
    similarity: float
    action: str = "flagged"


class SummaryEntry(BaseModel):
    slug: str
    topic: str
    language: str
    score: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None


class Summary(BaseModel):
    total_articles: int
    successful: int
    failed: int
    average_score: Optional[float] = None
    duplicates_detected: List[DuplicatePair] = Field(default_factory=list)
    articles: List[SummaryEntry] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str
