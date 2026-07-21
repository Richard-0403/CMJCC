"""Hybrid retrieval: weighted fusion of lexical, structured and (optional) semantic.

    retrieval_score = w_lex * lexical + w_sem * semantic + w_struct * structured

All components are normalised to [0,1] and weights come from config. When no
semantic provider is configured, the semantic weight is redistributed over the
available components so the weights still sum to 1.
"""

from __future__ import annotations

from ..config import AppConfig
from ..domain.job import JobPosting
from .base import QuerySpec, RetrievalOutcome, RetrievedJob
from .structured import structured_score
from .tfidf import TfidfRetriever


class HybridRetriever:
    """Fuses lexical (TF-IDF) and structured role/skill similarity."""

    name = "hybrid_retriever"

    def __init__(self, jobs: list[JobPosting], config: AppConfig) -> None:
        self.jobs = jobs
        self.config = config
        self.tfidf = TfidfRetriever(jobs)
        self._by_id = {j.job_id: j for j in jobs}

        r = config.retrieval
        w_lex, w_sem, w_struct = r.lexical_weight, r.semantic_weight, r.structured_weight
        # No semantic provider -> redistribute its weight proportionally.
        base = w_lex + w_struct
        if base <= 0:
            w_lex, w_struct = 0.5, 0.5
        else:
            w_lex = w_lex + w_sem * (w_lex / base)
            w_struct = w_struct + w_sem * (w_struct / base)
        total = w_lex + w_struct
        self.w_lex = w_lex / total
        self.w_struct = w_struct / total

    def retrieve(
        self, query: QuerySpec, jobs: list[JobPosting], pool_size: int
    ) -> RetrievalOutcome:
        lex = self.tfidf._similarities(query.positive_text())
        lex_by_id = {self.tfidf.job_ids[i]: float(lex[i]) for i in range(len(self.tfidf.job_ids))}
        max_lex = max(lex_by_id.values(), default=0.0) or 1.0

        results: list[RetrievedJob] = []
        for job in jobs:
            lexical = lex_by_id.get(job.job_id, 0.0) / max_lex
            struct = structured_score(query, job)
            score = self.w_lex * lexical + self.w_struct * struct
            if score <= 0.0:
                continue
            results.append(RetrievedJob(
                job_id=job.job_id,
                score=round(score, 6),
                components={"lexical": round(lexical, 6), "structured": round(struct, 6)},
            ))
        results.sort(key=lambda s: (-s.score, s.job_id))
        return RetrievalOutcome(retrieved=results[:pool_size], initial_pool_size=len(results))


def make_retriever(jobs: list[JobPosting], config: AppConfig):
    """Factory selecting a retriever implementation from config."""
    provider = config.retrieval.provider
    if provider == "tfidf":
        return TfidfRetriever(jobs)
    if provider == "structured":
        from .structured import StructuredRetriever
        return StructuredRetriever()
    return HybridRetriever(jobs, config)
