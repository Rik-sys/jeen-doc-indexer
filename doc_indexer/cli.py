"""Shared CLI plumbing: logging setup and clean error handling."""

from __future__ import annotations

import logging
import sys
from typing import Callable

from .errors import IndexerError

log = logging.getLogger("doc_indexer")


def setup_logging(verbose: bool) -> None:
    # Windows terminals default to a legacy code page; make sure ✓ and Hebrew print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s" if not verbose else "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(main: Callable[[], int], verbose: bool) -> None:
    """Run ``main`` and turn expected errors into a one-line message + exit code."""
    try:
        code = main()
    except IndexerError as exc:
        print(f"✗ Error: {exc.message}", file=sys.stderr)
        code = exc.exit_code
    except KeyboardInterrupt:
        print("✗ Interrupted", file=sys.stderr)
        code = 130
    except Exception as exc:  # unexpected: keep the message, show the traceback only with -v
        if verbose:
            log.exception("Unexpected error")
        print(f"✗ Unexpected error: {type(exc).__name__}: {exc}. Re-run with -v for details.", file=sys.stderr)
        code = 1
    sys.exit(code)
