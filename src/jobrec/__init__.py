"""Agent-Oriented Conversational Job Recommendation prototype (CMJCC).

This package implements a research prototype for constraint-aware conversational
job recommendation. Structured, typed state objects and evidence identifiers are
first-class citizens: candidate preferences, dialogue evidence, job context,
constraint decisions, ranking features and explanation claims are all persisted
as verifiable data objects rather than living only inside an LLM context.
"""

#: 0.2.0 -- the audit freeze. Everything that decides what a run produces or how it is scored
#: changed under 0.1.0 and is now settled: constraint-cue scoping and multi-value extraction
#: (P0-1), per-turn extraction snapshots replacing the re-parsing of dialogue history (P0-2),
#: salary thresholds read against a job's guaranteed minimum instead of range overlap (P0-3),
#: predicate-aware claim validation (P0-4), and no-match explanations scoped to the set that
#: was actually searched (P0-5).
#:
#: Declared rather than derived, and part of the experiment id, so bumping it separates every
#: post-freeze artifact from the pre-freeze ones by construction. Results produced under 0.1.0
#: are NOT comparable with 0.2.0 results: the extraction, the eligibility rule and the
#: grounding verdict all moved. The sealed thesis pair records 0.1.0 and stays valid as a
#: record of what was run, not as a baseline to diff against.
__version__ = "0.2.0"
CODE_VERSION = "0.2.0"
