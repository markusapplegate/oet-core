"""Utility helpers for the oet_core package."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union


_VERBOSE_LOGGING = False
_LOG_IN_PROGRESS = False


def set_verbose_logging(enabled: bool) -> None:
    """Enable or disable verbose logging for this module."""
    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


class Matrix:
    """Matrix backed by nested Python lists."""

    def __init__(self, rows: int, cols: int, fill: Optional[Union[int, float, str]] = None) -> None:
        if _VERBOSE_LOGGING:
            log(
                f"Matrix.__init__ called with rows={rows}, cols={cols}, fill={fill}",
                level="info",
            )

        if not isinstance(rows, int) or not isinstance(cols, int):
            raise TypeError("rows and cols must be integers")

        if rows < 0 or cols < 0:
            raise ValueError("rows and cols must be non-negative")

        if fill is None:
            fill_value: Union[int, float, str] = 0
        elif isinstance(fill, (int, float, str)):
            fill_value = fill
        else:
            raise TypeError("fill must be an int, float, or str")

        self.rows = rows
        self.cols = cols
        self.data: List[List[Union[int, float, str]]] = [
            [fill_value for _ in range(cols)]
            for _ in range(rows)
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Matrix(rows={self.rows}, cols={self.cols}, data={self.data})"

    def get(self, row: int, col: int) -> Union[int, float, str]:
        """Return the value at ``(row, col)``."""
        if _VERBOSE_LOGGING:
            log(f"Matrix.get called with row={row}, col={col}", level="info")

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("row and column indices must be integers")

        if not (0 <= row < self.rows) or not (0 <= col < self.cols):
            raise IndexError("row or column out of range")

        return self.data[row][col]

    def set(self, row: int, col: int, value: Union[int, float, str]) -> None:
        """Set the value at ``(row, col)``."""
        if _VERBOSE_LOGGING:
            log(f"Matrix.set called with row={row}, col={col}, value={value}", level="info")

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("row and column indices must be integers")

        if not (0 <= row < self.rows) or not (0 <= col < self.cols):
            raise IndexError("row or column out of range")

        if not isinstance(value, (int, float, str)):
            raise TypeError("value must be an int, float, or str")

        self.data[row][col] = value

    def transpose(self) -> "Matrix":
        """Return a new matrix that is the transpose of this matrix."""
        if _VERBOSE_LOGGING:
            log("Matrix.transpose called", level="info")

        if self.rows == 0 or self.cols == 0:
            return Matrix(self.cols, self.rows)

        transposed = Matrix(self.cols, self.rows)
        transposed.data = [
            [self.data[row][col] for row in range(self.rows)]
            for col in range(self.cols)
        ]
        return transposed
    
    def to_symbolic(self) -> Any:
        """Convert this Matrix to a SymPy symbolic matrix.
        
        Requires SymPy to be installed. Elements that are strings will be
        parsed as symbolic expressions.
        
        Returns
        -------
        sympy.Matrix
            SymPy symbolic matrix representation.
        
        Examples
        --------
        >>> m = Matrix(2, 2)
        >>> m.set(0, 0, 'x')
        >>> m.set(0, 1, 'y')
        >>> m.set(1, 0, 2)
        >>> m.set(1, 1, 3)
        >>> sym_matrix = m.to_symbolic()
        >>> det = sym_matrix.det()  # x*3 - y*2
        """
        if _VERBOSE_LOGGING:
            log("Matrix.to_symbolic called", level="info")
        
        from .symbolics import matrix_to_symbolic
        return matrix_to_symbolic(self)
    
    def symbolic_determinant(self):
        """Compute symbolic determinant of this matrix.
        
        Matrix must be square. Requires SymPy to be installed.
        
        Returns
        -------
        SymbolicExpression
            Determinant as symbolic expression.
        
        Examples
        --------
        >>> m = Matrix(2, 2)
        >>> m.set(0, 0, 'a')
        >>> m.set(0, 1, 'b')
        >>> m.set(1, 0, 'c')
        >>> m.set(1, 1, 'd')
        >>> det = m.symbolic_determinant()  # a*d - b*c
        """
        if _VERBOSE_LOGGING:
            log("Matrix.symbolic_determinant called", level="info")
        
        from .symbolics import symbolic_determinant
        return symbolic_determinant(self)
    
    def symbolic_inverse(self) -> "Matrix":
        """Compute symbolic inverse of this matrix.
        
        Matrix must be square and invertible. Requires SymPy to be installed.
        
        Returns
        -------
        Matrix
            Inverse matrix with symbolic expressions.
        
        Examples
        --------
        >>> m = Matrix(2, 2)
        >>> m.set(0, 0, 'a')
        >>> m.set(0, 1, 'b')
        >>> m.set(1, 0, 'c')
        >>> m.set(1, 1, 'd')
        >>> inv = m.symbolic_inverse()
        """
        if _VERBOSE_LOGGING:
            log("Matrix.symbolic_inverse called", level="info")
        
        from .symbolics import symbolic_inverse
        return symbolic_inverse(self)


def generate_matrix(
    rows: int,
    cols: int,
    fill: Optional[Union[int, float, str]] = None,
) -> List[List[Union[int, float, str]]]:
    """Generate a list-of-lists matrix using :class:`Matrix`."""
    if _VERBOSE_LOGGING:
        log(f"generate_matrix called with rows={rows}, cols={cols}, fill={fill}", level="info")
    return Matrix(rows, cols, fill).data


def validate_block_text(block: str, file_ext: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate Markdown, JSON, or YAML text blocks."""
    if not isinstance(block, str):
        return False, "block must be a str"

    if "\x00" in block:
        return False, "input contains null byte"

    normalized_ext: Optional[str]
    if file_ext:
        normalized_ext = file_ext.lower().lstrip(".")
        if normalized_ext == "yml":
            normalized_ext = "yaml"
        if normalized_ext not in {"md", "json", "yaml"}:
            return False, f"unsupported extension hint: {file_ext}"
    else:
        normalized_ext = None

    import json as _json

    if normalized_ext is None:
        try:
            _json.loads(block)
            normalized_ext = "json"
        except Exception:
            try:
                import yaml as _yaml  # type: ignore

                try:
                    _yaml.safe_load(block)
                    normalized_ext = "yaml"
                except Exception:
                    normalized_ext = "md"
            except Exception:
                normalized_ext = "md"

    if normalized_ext == "json":
        try:
            _json.loads(block)
            return True, None
        except Exception as exc:  # pragma: no cover - depends on parser message
            return False, f"JSON error: {exc}"

    if normalized_ext == "yaml":
        try:
            import yaml as _yaml  # type: ignore
        except Exception:
            return False, "PyYAML is required for YAML validation. Install with: pip install pyyaml"

        try:
            _yaml.safe_load(block)
            return True, None
        except Exception as exc:  # pragma: no cover - depends on parser message
            return False, f"YAML error: {exc}"

    def _fence_balanced(fence: str) -> bool:
        return block.count(fence) % 2 == 0

    if not _fence_balanced("```"):
        return False, "Unclosed triple backtick code fence in markdown"

    if not _fence_balanced("~~~"):
        return False, "Unclosed tilde (~) code fence in markdown"

    if block.lstrip().startswith("---"):
        lines = block.splitlines()
        closing_found = any(line.strip() == "---" for line in lines[1:])
        if not closing_found:
            return False, "Unclosed YAML frontmatter '---' in markdown"

    return True, None


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    stream=None,
    timestamps: bool = True,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance."""
    logger_name = name or "oet_core"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    for handler in list(logger.handlers):
        if getattr(handler, "_oet_core_created", False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream) if stream is not None else logging.StreamHandler()
    if timestamps:
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=datefmt)
    else:
        formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    setattr(handler, "_oet_core_created", True)
    logger.addHandler(handler)

    return logger


def log(
    message: str,
    level: str = "info",
    name: Optional[str] = None,
    stream=None,
    timestamps: bool = True,
) -> None:
    """Convenience wrapper around :func:`get_logger`."""
    global _LOG_IN_PROGRESS

    if _LOG_IN_PROGRESS:
        return

    try:
        _LOG_IN_PROGRESS = True
        logger = get_logger(name=name, stream=stream, timestamps=timestamps)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)
    finally:
        _LOG_IN_PROGRESS = False


class GraphBuilder:
    """Fluent interface for constructing NetworkX graphs."""

    def __init__(self, directed: bool = False) -> None:
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:
            raise ImportError("networkx is required. Install with: pip install networkx") from exc

        self._nx = nx
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()

    def add_node(self, node: Any, **attrs: Any) -> "GraphBuilder":
        self.graph.add_node(node, **attrs)
        return self

    def add_edge(self, u: Any, v: Any, **attrs: Any) -> "GraphBuilder":
        self.graph.add_edge(u, v, **attrs)
        return self

    def build(self):
        return self.graph

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        graph_type = "directed" if self.directed else "undirected"
        return f"GraphBuilder({graph_type}, nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


def generate_graph(
    nodes: Optional[List[Any]] = None,
    edges: Optional[List[Tuple[Any, Any]]] = None,
    directed: bool = False,
    node_attrs: Optional[Dict[Any, Dict[str, Any]]] = None,
    edge_attrs: Optional[Dict[Tuple[Any, Any], Dict[str, Any]]] = None,
):
    """Generate a NetworkX graph from lists of nodes and edges."""
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:
        raise ImportError("networkx is required. Install with: pip install networkx") from exc

    graph = nx.DiGraph() if directed else nx.Graph()

    if nodes:
        for node in nodes:
            attributes = node_attrs.get(node, {}) if node_attrs else {}
            graph.add_node(node, **attributes)

    if edges:
        for edge in edges:
            if len(edge) != 2:
                raise ValueError(f"Edge must be a tuple of 2 elements, got: {edge}")
            u, v = edge
            attributes = edge_attrs.get(edge, {}) if edge_attrs else {}
            graph.add_edge(u, v, **attributes)

    return graph


class SQLiteHelper:
    """Lightweight wrapper for SQLite database operations.
    
    Provides simple helpers for common database tasks in data pipelines:
    connection management, query execution, bulk inserts, and schema creation.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize SQLite connection.
        
        Parameters
        ----------
        db_path:
            Path to database file, or ":memory:" for in-memory database.
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.__init__ called with db_path={db_path}", level="info")

        if not isinstance(db_path, str):
            raise TypeError("db_path must be a string")

        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close database connection."""
        if _VERBOSE_LOGGING:
            log("SQLiteHelper.close called", level="info")

        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "SQLiteHelper":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connection."""
        self.close()

    def execute(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict]] = None,
        commit: bool = True,
    ) -> sqlite3.Cursor:
        """Execute a SQL query.
        
        Parameters
        ----------
        query:
            SQL query to execute.
        params:
            Optional parameters for parameterized queries.
        commit:
            Whether to commit after execution (default: True).
        
        Returns
        -------
        sqlite3.Cursor
            Cursor object with query results.
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.execute called with query={query[:50]}...", level="info")

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if self.conn is None:
            raise RuntimeError("Database connection is closed")

        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if commit:
            self.conn.commit()

        return cursor

    def fetch_all(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query and fetch all results.
        
        Parameters
        ----------
        query:
            SQL SELECT query.
        params:
            Optional parameters for parameterized queries.
        
        Returns
        -------
        List[sqlite3.Row]
            List of result rows (dict-like objects).
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.fetch_all called with query={query[:50]}...", level="info")

        cursor = self.execute(query, params, commit=False)
        return cursor.fetchall()

    def fetch_one(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and fetch one result.
        
        Parameters
        ----------
        query:
            SQL SELECT query.
        params:
            Optional parameters for parameterized queries.
        
        Returns
        -------
        Optional[sqlite3.Row]
            Single result row or None if no results.
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.fetch_one called with query={query[:50]}...", level="info")

        cursor = self.execute(query, params, commit=False)
        return cursor.fetchone()

    def bulk_insert(
        self,
        table: str,
        columns: List[str],
        rows: List[Tuple],
    ) -> int:
        """Insert multiple rows efficiently.
        
        Parameters
        ----------
        table:
            Table name to insert into.
        columns:
            List of column names.
        rows:
            List of tuples containing row values.
        
        Returns
        -------
        int
            Number of rows inserted.
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.bulk_insert called with table={table}, rows={len(rows)}", level="info")

        if not isinstance(table, str):
            raise TypeError("table must be a string")
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")
        if not isinstance(rows, list):
            raise TypeError("rows must be a list")

        if not columns:
            raise ValueError("columns cannot be empty")
        if not rows:
            return 0

        placeholders = ", ".join(["?"] * len(columns))
        column_names = ", ".join(columns)
        query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

        if self.conn is None:
            raise RuntimeError("Database connection is closed")

        cursor = self.conn.cursor()
        cursor.executemany(query, rows)
        self.conn.commit()

        return cursor.rowcount

    def create_table(
        self,
        table: str,
        schema: Dict[str, str],
        if_not_exists: bool = True,
    ) -> None:
        """Create a table from a schema definition.
        
        Parameters
        ----------
        table:
            Table name to create.
        schema:
            Dictionary mapping column names to SQL types.
            Example: {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        if_not_exists:
            Add IF NOT EXISTS clause (default: True).
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.create_table called with table={table}", level="info")

        if not isinstance(table, str):
            raise TypeError("table must be a string")
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dict")
        if not schema:
            raise ValueError("schema cannot be empty")

        columns_def = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        query = f"CREATE TABLE {exists_clause}{table} ({columns_def})"

        self.execute(query)

    def table_exists(self, table: str) -> bool:
        """Check if a table exists in the database.
        
        Parameters
        ----------
        table:
            Table name to check.
        
        Returns
        -------
        bool
            True if table exists, False otherwise.
        """
        if _VERBOSE_LOGGING:
            log(f"SQLiteHelper.table_exists called with table={table}", level="info")

        if not isinstance(table, str):
            raise TypeError("table must be a string")

        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table,))
        return result is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        status = "open" if self.conn else "closed"
        return f"SQLiteHelper(db_path='{self.db_path}', status={status})"


def create_sqlite_helper(db_path: str = ":memory:") -> SQLiteHelper:
    """Create a SQLiteHelper instance.
    
    Parameters
    ----------
    db_path:
        Path to database file, or ":memory:" for in-memory database.
    
    Returns
    -------
    SQLiteHelper
        Configured SQLite helper instance.
    """
    if _VERBOSE_LOGGING:
        log(f"create_sqlite_helper called with db_path={db_path}", level="info")

    return SQLiteHelper(db_path)

