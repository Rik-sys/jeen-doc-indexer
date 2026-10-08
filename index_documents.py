#!/usr/bin/env python3
"""Index a PDF or DOCX file: extract -> clean -> chunk -> embed -> store in PostgreSQL/pgvector.

Examples:
    python index_documents.py --file ./docs/example.pdf --strategy paragraph
    python index_documents.py --file ./docs/example.pdf --strategy all
    python index_documents.py --file ./docs/example.pdf --strategy fixed --chunk-size 800 --overlap 150
    python index_documents.py --file ./docs/example.pdf --strategy sentence --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time

from doc_indexer import db
from doc_indexer.chunking import DEFAULTS, STRATEGIES, chunk_text
from doc_indexer.cli import run, setup_logging
from doc_indexer.config import load_settings
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.extract import extract_document

log = logging.getLogger("doc_indexer")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index a PDF/DOCX document for semantic search.")
    p.add_argument("--file", required=True, help="Path to a .pdf or .docx file")
    p.add_argument("--strategy", required=True, choices=[*STRATEGIES, "all"],
                   help="Chunking strategy, or 'all' to index with all three")
    p.add_argument("--chunk-size", type=int, help="Target chunk size in characters (strategy default if omitted)")
    p.add_argument("--overlap", type=int, help="Overlap in characters for the 'fixed' strategy (default 200)")
    p.add_argument("--replace", action="store_true",
                   help="Delete this file's existing chunks for the strategy before indexing")
    p.add_argument("--dry-run", action="store_true",
                   help="Extract and chunk only: no API calls, no database writes")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def main(argv=None, embedder_factory=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    def _main() -> int:
        t0 = time.perf_counter()
        doc = extract_document(args.file)
        pages = f" ({doc.pages} pages)" if doc.pages else ""
        log.info("✓ Extracted %s characters from %s%s", f"{doc.char_count:,}", doc.filename, pages)

        strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
        plans = []
        for strategy in strategies:
            chunks = chunk_text(doc.text, strategy, args.chunk_size, args.overlap)
            avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
            size = args.chunk_size or DEFAULTS[strategy]["chunk_size"]
            extra = f", overlap={args.overlap if args.overlap is not None else 200}" if strategy == "fixed" else ""
            log.info("✓ Split into %d chunks (strategy=%s, target<=%d chars%s, avg %d chars)",
                     len(chunks), strategy, size, extra, avg)
            plans.append((strategy, chunks))

        if args.dry_run:
            for strategy, chunks in plans:
                for c in chunks[:3]:
                    preview = c.text[:140].replace("\n", " ")
                    log.info("    [%s #%d] %s%s", strategy, c.index, preview, "…" if len(c.text) > 140 else "")
            log.info("✓ Dry run: nothing was embedded or stored")
            return 0

        settings = load_settings(need_gemini=True, need_db=True)
        conn = db.connect(settings, settings.embedding_dim)  # connect first: fail fast before spending API quota
        try:
            embedder = (embedder_factory or (lambda s: GeminiEmbedder(
                s.gemini_api_key, s.embedding_model, s.embedding_dim)))(settings)
            for strategy, chunks in plans:
                if args.replace:
                    removed = db.delete_document(conn, doc.filename, strategy)
                    if removed:
                        log.info("✓ Removed %d existing chunks for %s [%s]", removed, doc.filename, strategy)
                known = db.existing_hashes(conn, doc.filename, strategy)
                todo = [c for c in chunks if db.content_hash(c.text) not in known]
                skipped = len(chunks) - len(todo)
                if not todo:
                    log.info("✓ All %d chunks already indexed for %s [%s] - nothing to do",
                             len(chunks), doc.filename, strategy)
                    continue
                before = getattr(embedder, "batches_sent", 0)
                vectors = embedder.embed_documents([c.text for c in todo])
                batches = getattr(embedder, "batches_sent", 0) - before
                log.info("✓ Embedded %d chunks with %s (%d dims)%s", len(todo), settings.embedding_model,
                         settings.embedding_dim, f" in {batches} batch(es)" if batches else "")
                inserted = db.insert_chunks(conn, doc.filename, strategy,
                                            ((c.index, c.text, v) for c, v in zip(todo, vectors)))
                log.info("✓ Stored %d rows in %s (%d already indexed, skipped)", inserted, db.TABLE, skipped)
        finally:
            conn.close()
        log.info("✓ Done in %.1fs. Database: %s", time.perf_counter() - t0, settings.safe_db_target)
        return 0

    run(_main, args.verbose)
    return 0


if __name__ == "__main__":
    main()
