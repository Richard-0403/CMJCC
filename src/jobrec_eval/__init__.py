"""CMJCC evaluation pipeline.

A standalone package (independent of the main application logic) that reads the
standard run bundles the prototype exports and produces reproducible metrics,
statistics, plots and an analysis report for RQ4.

Integrity notes (see the evaluation guide, sections 0 and 30):
- Human relevance/claim annotation is NOT fabricated. Relevance grades here come
  from a transparent, deterministic *automatic oracle* and are clearly labelled
  as such; explanation grounding comes from the system's claim validator. Slots
  for real human annotation are still emitted so raters can replace the proxy.
- Scenario-level results are always retained; nothing is reduced to a single
  mean without the paired structure.
"""

__version__ = "0.1.0"
EVAL_VERSION = "0.1.0"
