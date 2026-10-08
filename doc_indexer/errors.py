"""Custom exceptions.

Every expected failure is an ``IndexerError`` subclass with a human-readable
message and a stable process exit code, so the CLIs can fail cleanly
(no tracebacks for expected problems) and scripts can react to the code.
"""


class IndexerError(Exception):
    """Base class for all expected errors."""

    exit_code = 10

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigError(IndexerError):
    """A required environment variable is missing or invalid."""

    exit_code = 1


class MissingFileError(IndexerError):
    """The input file does not exist or is not a regular file."""

    exit_code = 2


class UnsupportedFileError(IndexerError):
    """The file type is not supported, or its content does not match its extension."""

    exit_code = 3


class CorruptedFileError(IndexerError):
    """The file is encrypted, damaged or cannot be parsed."""

    exit_code = 3


class NoTextFoundError(IndexerError):
    """The document contains no extractable text (e.g. a scanned PDF)."""

    exit_code = 4


class EmbeddingError(IndexerError):
    """The embedding API failed after all retries, or returned an invalid response."""

    exit_code = 5


class DatabaseError(IndexerError):
    """PostgreSQL could not be reached or a query failed."""

    exit_code = 6


class InvalidArgumentError(IndexerError):
    """A CLI argument has an invalid value (e.g. overlap larger than chunk size)."""

    exit_code = 7
