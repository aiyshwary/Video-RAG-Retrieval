"""Miscellaneous utilities for the video-RAG project."""

import os


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
