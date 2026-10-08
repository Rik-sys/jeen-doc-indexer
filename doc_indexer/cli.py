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


def use_system_certificates() -> None:
    """Verify HTTPS against the operating system's certificate store.

    Python ships its own CA bundle (certifi). On corporate networks a proxy often
    re-signs HTTPS traffic with the company's root certificate, which the OS trusts
    but certifi does not, so API calls fail with CERTIFICATE_VERIFY_FAILED.
    ``truststore`` makes Python use the OS store instead. Verification stays on.
    """
    try:
        import truststore
    except ImportError:  # optional: without it Python's bundled CA list is used
        return
    truststore.inject_into_ssl()


def run(main: Callable[[], int], verbose: bool) -> None:
    """Run ``main`` and turn expected errors into a one-line message + exit code."""
    use_system_certificates()
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