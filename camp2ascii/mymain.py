"""Compatibility wrapper importing pipeline/ingest APIs."""

from camp2ascii.pipeline import main  # re-export for compatibility
from camp2ascii.ingest import parse_footer, ingest_tob3_data, ingest_tob2_data  # re-export

__all__ = ["main", "parse_footer", "ingest_tob3_data", "ingest_tob2_data"]