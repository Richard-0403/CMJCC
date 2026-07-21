"""TF-IDF lexical retrieval over job text using scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..domain.job import JobPosting
from ..utils.text import normalize_token
from .base import QuerySpec, RetrievalOutcome, RetrievedJob


def _job_document(job: JobPosting) -> str:
    parts = [
        job.title,
        job.normalized_title,
        job.role_family or "",
        " ".join(job.required_skills),
        " ".join(job.preferred_skills),
        " ".join(job.responsibilities),
        job.description,
        job.industry or "",
    ]
    return normalize_token(" ".join(p for p in parts if p))


class TfidfRetriever:
    """Lexical retriever built from a fitted TF-IDF matrix over the catalog."""

    name = "tfidf_retriever"

    def __init__(self, jobs: list[JobPosting]) -> None:
        self.jobs = jobs
        self.job_ids = [j.job_id for j in jobs]
        docs = [_job_document(j) for j in jobs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs) if docs else None

    def _similarities(self, query_text: str) -> np.ndarray:
        if self.matrix is None or not query_text.strip():
            return np.zeros(len(self.job_ids))
        q = self.vectorizer.transform([normalize_token(query_text)])
        sims = cosine_similarity(q, self.matrix)[0]
        return sims

    def retrieve(
        self, query: QuerySpec, jobs: list[JobPosting], pool_size: int
    ) -> RetrievalOutcome:
        sims = self._similarities(query.positive_text())
        scored = [
            RetrievedJob(job_id=self.job_ids[i], score=float(sims[i]),
                         components={"lexical": float(sims[i])})
            for i in range(len(self.job_ids))
        ]
        scored = [s for s in scored if s.score > 0.0]
        scored.sort(key=lambda s: (-s.score, s.job_id))
        return RetrievalOutcome(retrieved=scored[:pool_size], initial_pool_size=len(scored))
