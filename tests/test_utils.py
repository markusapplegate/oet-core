"""Tests for oet_core.utils module."""
import sys
from pathlib import Path
from io import StringIO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oet_core.utils import (
    GraphBuilder,
    Matrix,
    SQLiteHelper,
    create_sqlite_helper,
    generate_graph,
    generate_matrix,
    get_logger,
    log,
    validate_block_text,
)


class TestMatrix:
    """Test cases for Matrix class."""

    def test_init(self):
        """Test Matrix initialization."""
        m = Matrix(3, 4)
        assert m.rows == 3
        assert m.cols == 4
        assert len(m.data) == 3
        assert len(m.data[0]) == 4

    def test_init_with_fill(self):
        """Test Matrix initialization with fill value."""
        m = Matrix(2, 3, fill=5)
        assert m.get(0, 0) == 5
        assert m.get(1, 2) == 5

        m2 = Matrix(2, 2, fill='x')
        assert m2.get(0, 0) == 'x'
        assert m2.get(1, 1) == 'x'

    def test_init_invalid(self):
        """Test Matrix initialization with invalid parameters."""
        try:
            Matrix(-1, 5)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            Matrix(3, 'not an int')
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_get(self):
        """Test Matrix get operation."""
        m = Matrix(3, 3, fill=0)
        m.data[1][2] = 42
        assert m.get(1, 2) == 42
        assert m.get(0, 0) == 0

    def test_get_out_of_bounds(self):
        """Test get with out of bounds indices."""
        m = Matrix(2, 2)
        try:
            m.get(5, 0)
            assert False, "Should raise IndexError"
        except IndexError:
            pass

        try:
            m.get(0, 5)
            assert False, "Should raise IndexError"
        except IndexError:
            pass

    def test_set(self):
        """Test Matrix set operation."""
        m = Matrix(3, 3)
        m.set(1, 1, 99)
        assert m.get(1, 1) == 99

        m.set(0, 2, 'hello')
        assert m.get(0, 2) == 'hello'

    def test_set_invalid_value(self):
        """Test set with invalid value type."""
        m = Matrix(2, 2)
        try:
            m.set(0, 0, [1, 2, 3])  # lists not allowed
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_transpose(self):
        """Test Matrix transpose operation."""
        m = Matrix(2, 3, fill=0)
        m.set(0, 0, 1)
        m.set(0, 1, 2)
        m.set(0, 2, 3)
        m.set(1, 0, 4)
        m.set(1, 1, 5)
        m.set(1, 2, 6)

        t = m.transpose()
        assert t.rows == 3
        assert t.cols == 2
        assert t.get(0, 0) == 1
        assert t.get(1, 0) == 2
        assert t.get(2, 0) == 3
        assert t.get(0, 1) == 4
        assert t.get(1, 1) == 5
        assert t.get(2, 1) == 6

    def test_transpose_empty(self):
        """Test transpose of empty matrix."""
        m = Matrix(0, 0)
        t = m.transpose()
        assert t.rows == 0
        assert t.cols == 0


class TestGenerateMatrix:
    """Test cases for generate_matrix function."""

    def test_basic(self):
        """Test basic matrix generation."""
        mat = generate_matrix(2, 3)
        assert len(mat) == 2
        assert len(mat[0]) == 3
        assert mat[0][0] == 0

    def test_with_fill(self):
        """Test matrix generation with fill value."""
        mat = generate_matrix(3, 2, fill=7)
        assert mat[0][0] == 7
        assert mat[2][1] == 7


class TestValidateBlockText:
    """Test cases for validate_block_text function."""

    def test_valid_json(self):
        """Test validation of valid JSON."""
        valid, error = validate_block_text('{"key": "value"}')
        assert valid is True
        assert error is None

        valid, error = validate_block_text('[1, 2, 3]', file_ext='json')
        assert valid is True

    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        valid, error = validate_block_text('{"key": invalid}', file_ext='json')
        assert valid is False
        assert error is not None

    def test_valid_markdown(self):
        """Test validation of valid Markdown."""
        md = "# Header\n\nSome text\n\n```python\ncode\n```"
        valid, error = validate_block_text(md, file_ext='md')
        assert valid is True
        assert error is None

    def test_invalid_markdown_unclosed_fence(self):
        """Test Markdown with unclosed code fence."""
        md = "# Header\n\n```python\ncode"
        valid, error = validate_block_text(md, file_ext='md')
        assert valid is False
        assert 'backtick' in error.lower()

    def test_invalid_markdown_unclosed_frontmatter(self):
        """Test Markdown with unclosed YAML frontmatter."""
        md = "---\ntitle: Test\n\nSome content"
        valid, error = validate_block_text(md, file_ext='md')
        assert valid is False
        assert 'frontmatter' in error.lower()

    def test_valid_yaml(self):
        """Test validation of valid YAML (requires PyYAML)."""
        yaml_text = "key: value\nlist:\n  - item1\n  - item2"
        valid, error = validate_block_text(yaml_text, file_ext='yaml')
        # Will pass if PyYAML is installed, otherwise will return error about PyYAML
        if valid:
            assert error is None
        else:
            assert 'PyYAML' in error

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        valid, error = validate_block_text(123)
        assert valid is False
        assert 'str' in error

    def test_null_byte(self):
        """Test with null byte in input."""
        valid, error = validate_block_text("text\x00with null")
        assert valid is False
        assert 'null' in error.lower()

    def test_auto_detection(self):
        """Test format auto-detection."""
        # Should detect as JSON
        valid, error = validate_block_text('{"key": "value"}')
        assert valid is True

        # Should detect as Markdown (invalid JSON/YAML)
        valid, error = validate_block_text('# Just a heading')
        assert valid is True


class TestLogging:
    """Test cases for logging functions."""

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger()
        assert logger is not None
        assert logger.name == "oet_core"

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger(name="test_logger")
        assert logger.name == "test_logger"

    def test_get_logger_with_stream(self):
        """Test logger with custom stream."""
        stream = StringIO()
        logger = get_logger(stream=stream, timestamps=False)
        logger.info("test message")
        output = stream.getvalue()
        assert "test message" in output

    def test_log_function(self):
        """Test log convenience function."""
        stream = StringIO()
        log("test message", level="info", stream=stream, timestamps=False)
        output = stream.getvalue()
        assert "test message" in output

    def test_log_different_levels(self):
        """Test log function with different levels."""
        stream = StringIO()
        log("info message", level="info", stream=stream, timestamps=False)
        log("warning message", level="warning", stream=stream, timestamps=False)
        output = stream.getvalue()
        assert "info message" in output
        assert "warning message" in output


class TestGraphBuilder:
    """Test cases for GraphBuilder class."""

    def test_init_undirected(self):
        """Test GraphBuilder initialization for undirected graph."""
        try:
            gb = GraphBuilder(directed=False)
            assert gb.directed is False
            assert gb.graph is not None
        except ImportError:
            # networkx not installed, skip test
            pass

    def test_init_directed(self):
        """Test GraphBuilder initialization for directed graph."""
        try:
            gb = GraphBuilder(directed=True)
            assert gb.directed is True
            assert gb.graph is not None
        except ImportError:
            pass

    def test_add_node(self):
        """Test adding nodes to graph."""
        try:
            gb = GraphBuilder()
            gb.add_node(1)
            gb.add_node(2, color='red')
            
            g = gb.build()
            assert 1 in g.nodes()
            assert 2 in g.nodes()
            assert g.nodes[2]['color'] == 'red'
        except ImportError:
            pass

    def test_add_edge(self):
        """Test adding edges to graph."""
        try:
            gb = GraphBuilder()
            gb.add_node(1)
            gb.add_node(2)
            gb.add_edge(1, 2, weight=5)
            
            g = gb.build()
            assert (1, 2) in g.edges() or (2, 1) in g.edges()  # undirected
        except ImportError:
            pass

    def test_method_chaining(self):
        """Test that methods support chaining."""
        try:
            gb = GraphBuilder()
            result = gb.add_node(1).add_node(2).add_edge(1, 2)
            assert result is gb  # Should return self
        except ImportError:
            pass


class TestGenerateGraph:
    """Test cases for generate_graph function."""

    def test_empty_graph(self):
        """Test generating an empty graph."""
        try:
            g = generate_graph()
            assert g.number_of_nodes() == 0
            assert g.number_of_edges() == 0
        except ImportError:
            pass

    def test_nodes_only(self):
        """Test generating graph with nodes only."""
        try:
            g = generate_graph(nodes=[1, 2, 3])
            assert g.number_of_nodes() == 3
            assert g.number_of_edges() == 0
        except ImportError:
            pass

    def test_edges_infer_nodes(self):
        """Test that nodes are inferred from edges."""
        try:
            g = generate_graph(edges=[(1, 2), (2, 3)])
            assert g.number_of_edges() == 2
            # Nodes should be inferred
            assert 1 in g.nodes()
            assert 2 in g.nodes()
            assert 3 in g.nodes()
        except ImportError:
            pass

    def test_directed_graph(self):
        """Test generating directed graph."""
        try:
            g = generate_graph(edges=[(1, 2)], directed=True)
            assert g.is_directed()
        except ImportError:
            pass

    def test_node_attributes(self):
        """Test graph with node attributes."""
        try:
            g = generate_graph(
                nodes=['A', 'B'],
                node_attrs={'A': {'color': 'red'}, 'B': {'color': 'blue'}}
            )
            assert g.nodes['A']['color'] == 'red'
            assert g.nodes['B']['color'] == 'blue'
        except ImportError:
            pass

    def test_edge_attributes(self):
        """Test graph with edge attributes."""
        try:
            g = generate_graph(
                edges=[(1, 2), (2, 3)],
                edge_attrs={(1, 2): {'weight': 10}}
            )
            assert g[1][2]['weight'] == 10
        except ImportError:
            pass

    def test_invalid_edge_format(self):
        """Test with invalid edge format."""
        try:
            generate_graph(edges=[(1, 2, 3)])  # Edges must be 2-tuples
            assert False, "Should raise ValueError"
        except (ValueError, ImportError):
            pass


class TestSQLiteHelper:
    """Test cases for SQLiteHelper class."""

    def test_init_memory(self):
        """Test SQLiteHelper initialization with in-memory database."""
        db = SQLiteHelper(":memory:")
        assert db.db_path == ":memory:"
        assert db.conn is not None
        db.close()

    def test_init_invalid(self):
        """Test SQLiteHelper initialization with invalid parameters."""
        try:
            SQLiteHelper(123)  # not a string
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_context_manager(self):
        """Test SQLiteHelper as context manager."""
        with SQLiteHelper(":memory:") as db:
            assert db.conn is not None
        assert db.conn is None  # should be closed

    def test_create_table(self):
        """Test table creation."""
        with SQLiteHelper(":memory:") as db:
            schema = {
                "id": "INTEGER PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "age": "INTEGER"
            }
            db.create_table("users", schema)
            assert db.table_exists("users")

    def test_create_table_invalid(self):
        """Test create_table with invalid parameters."""
        with SQLiteHelper(":memory:") as db:
            try:
                db.create_table(123, {})
                assert False, "Should raise TypeError"
            except TypeError:
                pass

            try:
                db.create_table("test", "not a dict")
                assert False, "Should raise TypeError"
            except TypeError:
                pass

            try:
                db.create_table("test", {})
                assert False, "Should raise ValueError"
            except ValueError:
                pass

    def test_table_exists(self):
        """Test table_exists method."""
        with SQLiteHelper(":memory:") as db:
            assert not db.table_exists("nonexistent")

            db.create_table("test_table", {"id": "INTEGER"})
            assert db.table_exists("test_table")

    def test_table_exists_invalid(self):
        """Test table_exists with invalid parameters."""
        with SQLiteHelper(":memory:") as db:
            try:
                db.table_exists(123)
                assert False, "Should raise TypeError"
            except TypeError:
                pass

    def test_execute(self):
        """Test execute method."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("users", {"id": "INTEGER", "name": "TEXT"})
            cursor = db.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
            assert cursor.rowcount == 1

    def test_execute_invalid(self):
        """Test execute with invalid parameters."""
        with SQLiteHelper(":memory:") as db:
            try:
                db.execute(123)
                assert False, "Should raise TypeError"
            except TypeError:
                pass

    def test_execute_closed_connection(self):
        """Test execute on closed connection."""
        db = SQLiteHelper(":memory:")
        db.close()
        try:
            db.execute("SELECT 1")
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass

    def test_fetch_all(self):
        """Test fetch_all method."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("users", {"id": "INTEGER", "name": "TEXT"})
            db.execute("INSERT INTO users VALUES (1, 'Alice')")
            db.execute("INSERT INTO users VALUES (2, 'Bob')")

            rows = db.fetch_all("SELECT * FROM users ORDER BY id")
            assert len(rows) == 2
            assert rows[0]["id"] == 1
            assert rows[0]["name"] == "Alice"
            assert rows[1]["id"] == 2
            assert rows[1]["name"] == "Bob"

    def test_fetch_one(self):
        """Test fetch_one method."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("users", {"id": "INTEGER", "name": "TEXT"})
            db.execute("INSERT INTO users VALUES (1, 'Alice')")

            row = db.fetch_one("SELECT * FROM users WHERE id = ?", (1,))
            assert row is not None
            assert row["id"] == 1
            assert row["name"] == "Alice"

            row = db.fetch_one("SELECT * FROM users WHERE id = ?", (999,))
            assert row is None

    def test_bulk_insert(self):
        """Test bulk_insert method."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("products", {
                "id": "INTEGER",
                "name": "TEXT",
                "price": "REAL"
            })

            rows = [
                (1, "Widget", 9.99),
                (2, "Gadget", 19.99),
                (3, "Doohickey", 29.99),
            ]

            count = db.bulk_insert("products", ["id", "name", "price"], rows)
            assert count == 3

            results = db.fetch_all("SELECT * FROM products ORDER BY id")
            assert len(results) == 3
            assert results[0]["name"] == "Widget"
            assert results[2]["price"] == 29.99

    def test_bulk_insert_empty(self):
        """Test bulk_insert with empty rows."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("test", {"id": "INTEGER"})
            count = db.bulk_insert("test", ["id"], [])
            assert count == 0

    def test_bulk_insert_invalid(self):
        """Test bulk_insert with invalid parameters."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("test", {"id": "INTEGER"})

            try:
                db.bulk_insert(123, ["id"], [(1,)])
                assert False, "Should raise TypeError"
            except TypeError:
                pass

            try:
                db.bulk_insert("test", "not a list", [(1,)])
                assert False, "Should raise TypeError"
            except TypeError:
                pass

            try:
                db.bulk_insert("test", "not a list", "not a list")
                assert False, "Should raise TypeError"
            except TypeError:
                pass

            try:
                db.bulk_insert("test", [], [(1,)])
                assert False, "Should raise ValueError"
            except ValueError:
                pass

    def test_bulk_insert_closed_connection(self):
        """Test bulk_insert on closed connection."""
        db = SQLiteHelper(":memory:")
        db.create_table("test", {"id": "INTEGER"})
        db.close()

        try:
            db.bulk_insert("test", ["id"], [(1,)])
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass

    def test_parameterized_queries(self):
        """Test parameterized queries with different parameter types."""
        with SQLiteHelper(":memory:") as db:
            db.create_table("users", {"id": "INTEGER", "name": "TEXT"})

            # Tuple parameters
            db.execute("INSERT INTO users VALUES (?, ?)", (1, "Alice"))

            # Dict parameters
            db.execute("INSERT INTO users VALUES (:id, :name)", {"id": 2, "name": "Bob"})

            rows = db.fetch_all("SELECT * FROM users ORDER BY id")
            assert len(rows) == 2
            assert rows[0]["name"] == "Alice"
            assert rows[1]["name"] == "Bob"

    def test_create_sqlite_helper_function(self):
        """Test create_sqlite_helper helper function."""
        db = create_sqlite_helper(":memory:")
        assert isinstance(db, SQLiteHelper)
        assert db.conn is not None
        db.close()

    def test_create_sqlite_helper_with_path(self):
        """Test create_sqlite_helper with file path."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = create_sqlite_helper(db_path)
            assert db.db_path == db_path
            db.create_table("test", {"id": "INTEGER"})
            db.close()

            # Verify file was created
            assert os.path.exists(db_path)


def run_tests():
    """Run all tests and report results."""
    test_classes = [
        TestMatrix,
        TestGenerateMatrix,
        TestValidateBlockText,
        TestLogging,
        TestGraphBuilder,
        TestGenerateGraph,
        TestSQLiteHelper,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    skipped_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"PASS {method_name}")
                passed_tests += 1
            except ImportError as e:
                print(f"SKIP {method_name}: Skipped (missing dependency: {e})")
                skipped_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"FAIL {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print('='*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Skipped: {len(skipped_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    
    if skipped_tests:
        print(f"\nSkipped tests (missing dependencies):")
        for class_name, method_name, error in skipped_tests:
            print(f"  - {class_name}.{method_name}")
    
    if not failed_tests:
        print(f"\nAll tests passed!")
        return True
    else:
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
