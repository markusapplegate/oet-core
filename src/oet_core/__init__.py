"""oet_core: Lightweight data processing toolkit.

Provides pure-Python implementations of algorithms and utilities for
building data pipelines and ETL workflows without heavy dependencies.
"""

from .algos import (
    binary_search,
    HashMap,
    set_verbose_logging as set_algos_verbose_logging,
)
from .utils import (
    Matrix,
    generate_matrix,
    validate_block_text,
    get_logger,
    log,
    GraphBuilder,
    generate_graph,
    SQLiteHelper,
    create_sqlite_helper,
    set_verbose_logging as set_utils_verbose_logging,
)
from .mintext import (
    Text,
    Corpus,
    set_mintext_verbose_logging,
)

__all__ = [
    "binary_search",
    "HashMap",
    "set_algos_verbose_logging",
    "Matrix",
    "generate_matrix",
    "validate_block_text",
    "get_logger",
    "log",
    "GraphBuilder",
    "generate_graph",
    "SQLiteHelper",
    "create_sqlite_helper",
    "set_utils_verbose_logging",
    "Text",
    "Corpus",
    "set_mintext_verbose_logging",
]

__version__ = "1.0.0"
