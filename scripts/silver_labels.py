"""Build LLM-annotated SILVER labels (not human gold) with two independent annotators + tie-break + Cohen's kappa.

Usage: python scripts/silver_labels.py emotion|compliance|extraction|intent
"""

from __future__ import annotations

import sys

from callgraph.config import get_config
from callgraph.logging_setup import setup_logging
from callgraph.silver import annotate

if __name__ == "__main__":
    setup_logging()
    annotate(get_config(), sys.argv[1])
