"""doc_indexer: index PDF/DOCX documents into PostgreSQL + pgvector and search them semantically.

Pipeline: extract -> clean -> chunk -> embed (Gemini) -> store (pgvector) -> search.
"""

__version__ = "1.0.0"
