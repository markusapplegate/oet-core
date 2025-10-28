# API Documentation

Complete reference for the `oet-core` package.

## Installation

```bash
pip install oet-core
```

For local development with optional extras:

```bash
git clone https://github.com/markusapplegate/oet-core.git
cd oet-core
pip install -e .[dev,all]
```

## Module Overview

- `oet_core.algos`: Binary search helpers and a pure-Python hash map.
- `oet_core.mintext`: Text analysis primitives (Text, Corpus classes).
- `oet_core.symbolics`: Symbolic mathematics with SymPy (SymbolicExpression, SymbolicSolver, FormulaLibrary).
- `oet_core.utils`: Matrix helpers, SQLite wrappers, text validation, logging utilities, and graph builders.

Importing from the package root re-exports the primary entry points:

```python
from oet_core import binary_search, HashMap, Matrix, generate_matrix
from oet_core import Text, Corpus, SQLiteHelper, create_sqlite_helper
from oet_core import validate_block_text, get_logger, log, GraphBuilder, generate_graph
from oet_core import SymbolicExpression, SymbolicSolver, FormulaLibrary
from oet_core import matrix_to_symbolic, symbolic_to_matrix, symbolic_determinant, symbolic_inverse
```

You can also import directly from submodules:

```python
from oet_core.algos import binary_search, HashMap
from oet_core.mintext import Text, Corpus
from oet_core.utils import SQLiteHelper, Matrix
from oet_core.symbolics import SymbolicExpression, SymbolicSolver, FormulaLibrary
```

---

## Algorithms (`oet_core.algos`)

### `binary_search(pairs, x_target, y_target=None)`

Performs binary search on a sorted list of scalars or `(x, y)` pairs.

Parameters:
- `pairs` (`list`): Sorted data to inspect. Scalars or 2-item tuples/lists.
- `x_target`: Value to match against the scalar or first tuple element.
- `y_target` (optional): Second element to match when dealing with pairs.

Returns the index of the match or `None` if not found. When `y_target` is omitted the function returns the left-most index matching `x_target`.

```python
from oet_core import binary_search

# Scalars
numbers = [1, 3, 5, 7, 9]
assert binary_search(numbers, 7) == 3

# Pairs (x, y)
points = [(1, "alpha"), (3, "beta"), (3, "gamma"), (5, "delta")]
assert binary_search(points, 3) == 1          # left-most match
assert binary_search(points, 3, "gamma") == 2
assert binary_search(points, 2) is None
```

Enable diagnostic output with `set_algos_verbose_logging(True)`.

### `HashMap(initial_capacity=8)`

Separate-chaining hash map with automatic resizing.

Key methods:
- `put(key, value)` – Insert or update a key/value pair.
- `get(key, default=None)` – Retrieve a value or return `default`.
- `delete(key)` – Remove a key; returns `True` if removed.
- `contains(key)` – `True` when the key exists.
- `keys()`, `values()`, `items()` – Iterators over stored data.

```python
from oet_core import HashMap

cache = HashMap()
cache.put("language", "Python")
cache.put("year", 2025)

assert cache.get("language") == "Python"
assert cache.contains("year")

for key, value in cache.items():
    print(key, value)

cache.delete("language")
```

Verbose diagnostics can be toggled with `set_algos_verbose_logging(True)`.

---

## Utilities (`oet_core.utils`)

### `Matrix(rows, cols, fill=None)`

Lightweight 2D matrix stored as nested Python lists.

- `rows`, `cols`: Non-negative integers defining the shape.
- `fill`: Optional initial value (`int`, `float`, or `str`). Defaults to `0`.
- Methods: `get(row, col)`, `set(row, col, value)`, `transpose()`.

```python
from oet_core import Matrix

matrix = Matrix(2, 3, fill=1)
matrix.set(0, 2, 42)

transposed = matrix.transpose()
assert transposed.rows == 3 and transposed.cols == 2
```

### `generate_matrix(rows, cols, fill=None)`

Convenience wrapper returning a list-of-lists matrix by delegating to `Matrix`.

```python
from oet_core import generate_matrix

grid = generate_matrix(3, 3, fill=7)
assert grid[0][0] == 7
```

### `validate_block_text(block, file_ext=None)`

Validates JSON, YAML, or Markdown content.

- Autodetects the format when `file_ext` is omitted.
- Returns `(is_valid: bool, error_message: Optional[str])`.
- Requires `pyyaml` for YAML validation.

```python
from oet_core import validate_block_text

valid, error = validate_block_text('{"key": "value"}', file_ext="json")
assert valid and error is None

valid, error = validate_block_text('---\ntitle: Missing end')
assert not valid
print(error)
```

### Logging utilities

`oet_core` ships lightweight helpers around the standard library `logging` module:

- `get_logger(name=None, level=logging.INFO, stream=None, timestamps=True, datefmt=...)`
- `log(message, level="info", name=None, stream=None, timestamps=True)`
- `set_utils_verbose_logging(enabled)` toggles internal diagnostics for utilities.

Practical example capturing logs and enabling verbose output temporarily:

```python
from io import StringIO
from oet_core import (
    get_logger,
    log,
    generate_matrix,
    set_utils_verbose_logging,
)

buffer = StringIO()
logger = get_logger("demo", stream=buffer, timestamps=False)
logger.info("Pipeline started")

log("Inline status update", level="warning")

set_utils_verbose_logging(True)
generate_matrix(1, 1)  # emits verbose diagnostics while enabled
set_utils_verbose_logging(False)

print(buffer.getvalue())
```

### `GraphBuilder(directed=False)`

Fluent interface for constructing NetworkX graphs. Requires `networkx` to be installed.

```python
from oet_core import GraphBuilder

graph = (
    GraphBuilder(directed=True)
    .add_node("A", label="start")
    .add_node("B", label="end")
    .add_edge("A", "B", weight=3.5)
    .build()
)

assert graph.has_edge("A", "B")
```

### `generate_graph(nodes=None, edges=None, directed=False, node_attrs=None, edge_attrs=None)`

Creates a NetworkX graph from iterables of nodes and edges. Each entry in `node_attrs` or `edge_attrs` maps to dictionaries of attributes applied during creation.

```python
from oet_core import generate_graph

graph = generate_graph(
    nodes=["A", "B", "C"],
    edges=[("A", "B"), ("B", "C")],
    edge_attrs={("A", "B"): {"weight": 1.0}},
)

assert graph.number_of_nodes() == 3
assert graph["A"]["B"]["weight"] == 1.0
```

### `SQLiteHelper(db_path=":memory:")`

Lightweight wrapper for SQLite database operations. Manages connections, queries, bulk inserts, and schema creation for data pipelines. Part of Python's standard library - no external dependencies.

**Constructor**:
- `db_path`: Path to database file or `":memory:"` for in-memory database.

**Key Methods**:
- `create_table(table, schema, if_not_exists=True)` – Create table from schema dict.
- `table_exists(table)` – Check if table exists.
- `execute(query, params=None, commit=True)` – Execute SQL query.
- `fetch_all(query, params=None)` – Execute query and return all rows.
- `fetch_one(query, params=None)` – Execute query and return one row.
- `bulk_insert(table, columns, rows)` – Efficiently insert multiple rows.
- `close()` – Close database connection.

**Context Manager Support**: Use `with` statement for automatic cleanup.

```python
from oet_core import SQLiteHelper

# In-memory database
with SQLiteHelper(":memory:") as db:
    # Create table
    schema = {
        "id": "INTEGER PRIMARY KEY",
        "name": "TEXT NOT NULL",
        "price": "REAL"
    }
    db.create_table("products", schema)
    
    # Bulk insert
    rows = [
        (1, "Widget", 9.99),
        (2, "Gadget", 19.99),
        (3, "Doohickey", 29.99),
    ]
    db.bulk_insert("products", ["id", "name", "price"], rows)
    
    # Query data
    results = db.fetch_all("SELECT * FROM products WHERE price > ?", (10.0,))
    for row in results:
        print(f"{row['name']}: ${row['price']}")
    
    # Single row
    product = db.fetch_one("SELECT * FROM products WHERE id = ?", (1,))
    assert product["name"] == "Widget"

# Persistent database
db = SQLiteHelper("data/pipeline.db")
db.create_table("events", {"timestamp": "INTEGER", "message": "TEXT"})
db.execute("INSERT INTO events VALUES (?, ?)", (1234567890, "Started"))
db.close()
```

### `create_sqlite_helper(db_path=":memory:")`

Factory function that creates and returns a configured `SQLiteHelper` instance.

```python
from oet_core import create_sqlite_helper

db = create_sqlite_helper(":memory:")
db.create_table("cache", {"key": "TEXT PRIMARY KEY", "value": "TEXT"})
db.close()
```

---

## MinText: Text Analysis (`oet_core.utils`)

Lightweight text analysis primitives for research workflows. Zero required dependencies (stdlib only), with optional NumPy for efficient vectorization.

### `Text(content, metadata=None)`

Analyzes individual text documents with tokenization, frequency statistics, entropy, sentiment, and vectorization.

**Constructor Parameters**:
- `content` (`str`): Raw text content.
- `metadata` (`dict`, optional): Arbitrary metadata (id, source, timestamp, etc.).

**Key Methods**:

#### `tokenize(lowercase=True, remove_punctuation=True, min_length=1)`

Split text into tokens (words).

```python
from oet_core import Text

text = Text("Hello, World! How are you?")
tokens = text.tokenize()  # ['hello', 'world', 'how', 'are', 'you']

# Keep original case and punctuation
tokens = text.tokenize(lowercase=False, remove_punctuation=False)

# Filter short words
tokens = text.tokenize(min_length=3)  # ['hello', 'world', 'how', 'are', 'you']
```

#### `frequencies(**tokenize_kwargs)`

Compute token frequency distribution.

```python
text = Text("the cat and the dog")
freq = text.frequencies()
# {'the': 2, 'cat': 1, 'and': 1, 'dog': 1}

# Get top N most frequent words
top = text.top_words(n=3)
# [('the', 2), ('cat', 1), ('and', 1)]
```

#### `entropy(**tokenize_kwargs)`

Calculate Shannon entropy of token distribution (text diversity measure).

```python
text1 = Text("the the the the")
print(text1.entropy())  # 0.0 (no diversity)

text2 = Text("apple banana cherry date")
print(text2.entropy())  # ~2.0 (high diversity)
```

#### `sentiment(**tokenize_kwargs)`

Simple lexicon-based sentiment analysis.

```python
text = Text("This is great and wonderful! I love it.")
result = text.sentiment()
# {
#   'score': 3,           # positive - negative
#   'positive': 3,        # count of positive words
#   'negative': 0,        # count of negative words
#   'neutral': 5          # count of neutral words
# }
```

#### `vectorize(vocabulary=None, binary=False)`

Convert text to term frequency vector.

```python
text = Text("cat dog cat")

# Use text's own vocabulary
vector = text.vectorize()  # [2.0, 1.0] for ['cat', 'dog']

# Use custom vocabulary
vocab = ["cat", "dog", "bird"]
vector = text.vectorize(vocabulary=vocab)  # [2.0, 1.0, 0.0]

# Binary encoding (presence/absence)
binary_vec = text.vectorize(vocabulary=vocab, binary=True)  # [1.0, 1.0, 0.0]
```

#### `stats(**tokenize_kwargs)`

Comprehensive text statistics.

```python
text = Text("The quick brown fox jumps.")
stats = text.stats()
# {
#   'char_count': 25,
#   'token_count': 5,
#   'unique_tokens': 5,
#   'type_token_ratio': 1.0,
#   'entropy': 2.32,
#   'sentiment_score': 0,
#   'avg_token_length': 4.2
# }
```

#### Serialization

```python
# Export to dictionary
data = text.to_dict()
# {'content': '...', 'metadata': {...}}

# Import from dictionary
text = Text.from_dict(data)
```

### `Corpus(texts=None)`

Collection of `Text` objects with batch operations and persistence.

**Constructor Parameters**:
- `texts` (`list`, optional): Initial list of `Text` objects.

**Key Methods**:

#### Adding Texts

```python
from oet_core import Corpus, Text

corpus = Corpus()

# Add existing Text object
text = Text("Hello world")
corpus.add(text)

# Create and add from string
corpus.add_from_string("Another document", metadata={"id": 1})
```

#### `filter(predicate)`

Filter corpus by predicate function.

```python
corpus = Corpus([
    Text("Doc 1", metadata={"lang": "en"}),
    Text("Doc 2", metadata={"lang": "fr"}),
    Text("Doc 3", metadata={"lang": "en"}),
])

english_only = corpus.filter(lambda t: t.metadata.get("lang") == "en")
# Returns new Corpus with 2 texts
```

#### `vocabulary(min_freq=1, **tokenize_kwargs)`

Build vocabulary from all texts.

```python
corpus = Corpus([
    Text("the cat and the dog"),
    Text("the bird and the fish")
])

vocab = corpus.vocabulary()  # All unique words
common = corpus.vocabulary(min_freq=2)  # Words appearing 2+ times
# ['and', 'the']
```

#### `vectorize_all(vocabulary=None, binary=False, **tokenize_kwargs)`

Vectorize all texts using shared vocabulary.

```python
corpus = Corpus([
    Text("cat dog"),
    Text("dog bird"),
    Text("cat bird")
])

# Build vocabulary from corpus
vectors = corpus.vectorize_all()
# Returns 3×3 matrix (numpy array if available, else list of lists)

# Use custom vocabulary
vectors = corpus.vectorize_all(vocabulary=["cat", "dog", "bird", "fish"])
# Returns 3×4 matrix
```

#### `aggregate_stats(**tokenize_kwargs)`

Compute aggregate statistics across all texts.

```python
stats = corpus.aggregate_stats()
# {
#   'text_count': 5,
#   'total_tokens': 120,
#   'total_chars': 650,
#   'avg_tokens_per_text': 24.0,
#   'avg_chars_per_text': 130.0,
#   'vocabulary_size': 45
# }
```

#### Persistence with SQLite

```python
from oet_core import Corpus, SQLiteHelper

corpus = Corpus([
    Text("First document", metadata={"priority": "high"}),
    Text("Second document", metadata={"priority": "low"})
])

# Save to database
db = SQLiteHelper("corpus.db")
corpus.save_to_db(db, table="documents")

# Load from database
loaded = Corpus.load_from_db(db, table="documents")

# Load with WHERE clause
high_priority = Corpus.load_from_db(
    db, 
    table="documents", 
    where="id > ?", 
    params=(10,)
)

db.close()
```

#### Iteration and Indexing

```python
corpus = Corpus([Text("First"), Text("Second"), Text("Third")])

# Get by index
first = corpus[0]

# Iterate
for text in corpus:
    print(text.content)

# Length
print(len(corpus))  # 3
```

### Complete Example

```python
from oet_core import Text, Corpus, SQLiteHelper

# Analyze single text
text = Text("The quick brown fox jumps over the lazy dog.")
print(f"Tokens: {text.tokenize()}")
print(f"Entropy: {text.entropy():.2f}")
print(f"Sentiment: {text.sentiment()}")

# Build and analyze corpus
corpus = Corpus()
corpus.add_from_string("I love this product!", metadata={"rating": 5})
corpus.add_from_string("Terrible experience.", metadata={"rating": 1})
corpus.add_from_string("It's okay, works fine.", metadata={"rating": 3})

# Vocabulary and vectorization
vocab = corpus.vocabulary(min_freq=1)
vectors = corpus.vectorize_all(vocabulary=vocab)
print(f"Corpus matrix: {len(corpus)} texts × {len(vocab)} features")

# Aggregate analysis
stats = corpus.aggregate_stats()
print(f"Average tokens per text: {stats['avg_tokens_per_text']:.1f}")

# Persist to database
db = SQLiteHelper("reviews.db")
corpus.save_to_db(db, table="reviews")
db.close()
```

---

## Verbose Diagnostics

```

---

## Symbolic Mathematics (`oet_core.symbolics`)

The `symbolics` module provides symbolic mathematics capabilities using SymPy. Install with:

```bash
pip install oet-core[symbolic]
# or
pip install sympy>=1.12
```

### Core Classes

#### `SymbolicExpression(expression)`

Wrapper for SymPy expressions with convenient methods for manipulation and calculus.

**Parameters:**
- `expression` (`str` or SymPy expression): Mathematical expression to wrap.

**Methods:**

**Algebraic Manipulation:**
- `simplify()` – Simplify expression
- `expand()` – Expand products and powers
- `factor()` – Factor expression

**Calculus:**
- `differentiate(*variables)` – Compute derivatives
- `integrate(*variables, definite=False, limits=None)` – Compute integrals
- `limit(variable, point, direction="+-")` – Compute limits
- `taylor_series(variable, center=0, order=6)` – Taylor expansion

**Evaluation:**
- `substitute(substitutions)` – Substitute symbolic/numeric values
- `evaluate(values)` – Evaluate to numeric result
- `to_function(variables=None)` – Convert to callable Python function

**Conversion:**
- `to_latex()` – Convert to LaTeX string
- `to_string()` – Convert to string representation
- `get_variables()` – Extract all variable names

```python
from oet_core import SymbolicExpression

# Create and manipulate expressions
expr = SymbolicExpression("x**2 + 2*x + 1")
factored = expr.factor()  # (x + 1)**2
expanded = expr.expand()

# Calculus
derivative = expr.differentiate("x")  # 2*x + 2
integral = expr.integrate("x")  # x**3/3 + x**2 + x

# Definite integral
area = expr.integrate("x", definite=True, limits={"x": (0, 1)})

# Evaluation
result = expr.evaluate({"x": 5})  # 36.0

# Convert to function for fast numeric evaluation
f = expr.to_function()
values = [f(x=i) for i in range(10)]

# LaTeX for papers
latex = expr.to_latex()  # 'x^{2} + 2 x + 1'
```

#### `SymbolicSolver()`

Solve algebraic equations, systems of equations, and differential equations.

**Methods:**
- `solve(equation, variable=None)` – Solve single equation
- `solve_system(equations, variables=None)` – Solve system of equations
- `solve_ode(equation, function, variable)` – Solve ordinary differential equation

```python
from oet_core import SymbolicSolver

solver = SymbolicSolver()

# Solve algebraic equation
solutions = solver.solve("x**2 - 4 = 0", "x")  # [-2, 2]

# Solve system
result = solver.solve_system([
    "x + y = 5",
    "x - y = 1"
])  # {x: 3, y: 2}

# System with 3 variables
result = solver.solve_system([
    "x + y + z = 6",
    "2*x - y + z = 3",
    "x + 2*y - z = 1"
], ["x", "y", "z"])
```

#### `FormulaLibrary(db_path)`

SQLite-based storage for mathematical formulas with tagging and metadata.

**Parameters:**
- `db_path` (`str`): Path to SQLite database (use `:memory:` for in-memory).

**Methods:**
- `save_formula(name, formula, description="", tags=None, metadata=None, overwrite=False)`
- `load_formula(name)` – Returns dict with `name`, `formula`, `description`, `tags`, `metadata`, `expression`
- `search(name_pattern=None, tag=None, metadata_key=None, metadata_value=None)` – Search formulas
- `list_formulas()` – Get all formula names
- `delete_formula(name)` – Delete formula and related data
- `close()` – Close database connection

```python
from oet_core import FormulaLibrary

# Use context manager for automatic cleanup
with FormulaLibrary("physics.db") as lib:
    # Save formulas with metadata
    lib.save_formula(
        "kinetic_energy",
        "(1/2)*m*v**2",
        description="Kinetic energy formula",
        tags=["physics", "mechanics", "energy"],
        metadata={"units": "joules", "variables": "m,v"}
    )
    
    lib.save_formula(
        "force",
        "m*a",
        description="Newton's second law",
        tags=["physics", "mechanics"],
        metadata={"units": "newtons"}
    )
    
    # Search by tag
    energy_formulas = lib.search(tag="energy")
    
    # Search by metadata
    joule_formulas = lib.search(metadata_key="units", metadata_value="joules")
    
    # Load and use formula
    data = lib.load_formula("kinetic_energy")
    expr = data["expression"]  # SymbolicExpression object
    ke_func = expr.to_function()
    energy = ke_func(m=10, v=5)  # Calculate: 125.0 J
```

### Matrix Integration

The symbolics module integrates with oet-core's Matrix class for symbolic linear algebra.

**Functions:**
- `matrix_to_symbolic(matrix)` – Convert Matrix → SymPy Matrix
- `symbolic_to_matrix(sym_matrix)` – Convert SymPy Matrix → Matrix
- `symbolic_determinant(matrix)` – Compute symbolic determinant
- `symbolic_inverse(matrix)` – Compute symbolic inverse

**Matrix Methods:**
- `Matrix.to_symbolic()` – Convert to SymPy matrix
- `Matrix.symbolic_determinant()` – Compute determinant symbolically
- `Matrix.symbolic_inverse()` – Compute inverse symbolically

```python
from oet_core import Matrix, matrix_to_symbolic

# Create matrix with symbolic elements
m = Matrix(2, 2)
m.set(0, 0, 'a')
m.set(0, 1, 'b')
m.set(1, 0, 'c')
m.set(1, 1, 'd')

# Symbolic determinant
det = m.symbolic_determinant()  # a*d - b*c
print(det.simplify())

# Symbolic inverse
inv = m.symbolic_inverse()
print(inv.get(0, 0))  # 'd/(a*d - b*c)'

# Convert to SymPy for advanced operations
sym_matrix = m.to_symbolic()
eigenvals = sym_matrix.eigenvals()
eigenvects = sym_matrix.eigenvects()

# Numeric example
m2 = Matrix(2, 2)
m2.set(0, 0, 4)
m2.set(0, 1, 3)
m2.set(1, 0, 2)
m2.set(1, 1, 1)

det = m2.symbolic_determinant()
result = det.evaluate({})  # -2.0

inv = m2.symbolic_inverse()
# Inverse is [[-1/2, 3/2], [1, -2]]
```

### Utility Functions

**`parse_expression(expr_str)`** – Parse string to SymPy expression

```python
from oet_core import parse_expression

expr = parse_expression("sin(x)**2 + cos(x)**2")
```

**`validate_formula(formula)`** – Validate mathematical expression

```python
from oet_core import validate_formula

valid, error = validate_formula("x**2 + 2*x + 1")
if valid:
    print("Valid formula!")
```

**`set_symbolics_verbose_logging(enabled)`** – Enable diagnostic logging

```python
from oet_core import set_symbolics_verbose_logging, SymbolicExpression

set_symbolics_verbose_logging(True)
expr = SymbolicExpression("x**2")
# Logs: SymbolicExpression.__init__ called with expression=x**2
```

---

## Verbose Diagnostics

Both `oet_core.algos` and `oet_core.utils` expose `set_verbose_logging(enabled: bool)` functions. Enable them to surface internal diagnostic messages via the logging utilities:

```python
from oet_core import set_algos_verbose_logging, set_utils_verbose_logging, binary_search

set_algos_verbose_logging(True)
binary_search([1, 2, 3], 2)
set_algos_verbose_logging(False)
```

Pair verbose mode with `get_logger`/`log` to route messages to custom handlers or buffers during debugging sessions.
