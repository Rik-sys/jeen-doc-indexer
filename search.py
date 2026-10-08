#!/usr/bin/env python3
"""Semantic search over indexed chunks (cosine similarity with pgvector).

Examples:
    python search.py --query "login issue"
    python search.py --query "customer was charged twice" --top-k 3 --strategy paragraph
    python search.py --query "router keeps disconnecting" --filename example.pdf --min-score 0.6
    python search.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap

from doc_indexer import db
from doc_indexer.chunking import STRATEGIES
from doc_indexer.cli import run, setup_logging
from doc_indexer.config import load_settings
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.errors import InvalidArgumentError

log = logging.getLogger("doc_indexer")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search indexed documents by meaning.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", help="Natural-language search query (any language)")
    g.add_argument("--list", action="store_true", help="List indexed files and chunk counts")
    p.add_argument("--top-k", type=int, default=5, help="Number of results (default 5)")
    p.add_argument("--strategy", choices=STRATEGIES, help="Only search chunks made with this strategy")
    p.add_argument("--filename", help="Only search chunks from this file (base name, e.g. example.pdf)")
    p.add_argument("--min-score", type=float, help="Drop results below this similarity (0-1)")
    p.add_argument("--json", action="store_true", help="Print results as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def main(argv=None, embedder_factory=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    def _main() -> int:
        if args.top_k < 1 or args.top_k > 50:
            raise InvalidArgumentError("--top-k must be between 1 and 50")
        if args.min_score is not None and not -1.0 <= args.min_score <= 1.0:
            raise InvalidArgumentError("--min-score must be between -1 and 1")
        if args.query is not None and not args.query.strip():
            raise InvalidArgumentError("--query must not be empty")

        settings = load_settings(need_gemini=not args.list, need_db=True)
        conn = db.connect(settings, settings.embedding_dim)
        try:
            if args.list:
                rows = db.indexed_files(conn)
                if not rows:
                    log.info("The index is empty. Run index_documents.py first.")
                for name, strategy, n in rows:
                    log.info("%-40s %-10s %5d chunks", name, strategy, n)
                return 0

            if db.count_rows(conn) == 0:
                log.info('No results for "%s": the index is empty. Run index_documents.py first.', args.query)
                return 0

            embedder = (embedder_factory or (lambda s: GeminiEmbedder(
                s.gemini_api_key, s.embedding_model, s.embedding_dim)))(settings)
            qvec = embedder.embed_query(args.query)
            hits = db.search(conn, qvec, args.top_k, args.strategy, args.filename, args.min_score)
        finally:
            conn.close()

        if args.json:
            print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))
            return 0
        if not hits:
            filters = [f"{k}={v}" for k, v in (("strategy", args.strategy), ("filename", args.filename),
                                                 ("min-score", args.min_score)) if v is not None]
            extra = f" with {', '.join(filters)}" if filters else ""
            log.info('No results for "%s"%s. Try a broader query or remove filters.', args.query, extra)
            return 0

        log.info('Top %d results for "%s"\n', len(hits), args.query)
        for rank, h in enumerate(hits, 1):
            log.info("#%d  score=%.3f  %s  [%s]  chunk %d", rank, h.score, h.filename, h.split_strategy, h.chunk_index)
            body = textwrap.shorten(" ".join(h.chunk_text.split()), width=260, placeholder=" …")
            log.info(textwrap.indent(textwrap.fill(body, width=96), "    ") + "\n")
        return 0

    run(_main, args.verbose)
    return 0


if __name__ == "__main__":
    main()
