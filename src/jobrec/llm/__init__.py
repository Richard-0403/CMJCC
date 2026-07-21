"""LLM adapter layer.

Business code never calls a vendor SDK directly. All model access goes through
the ``LLMProvider`` protocol so that behaviour can be mocked offline, replayed,
or swapped without touching orchestration logic.
"""
