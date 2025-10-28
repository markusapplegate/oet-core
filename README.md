# oet-core

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

> Lightweight data processing toolkit for Python

**oet-core** (Outer Element Taxonomy) is a minimal, pure-Python library for building data pipelines and ETL workflows without heavy dependencies. Search, transform, validate, and model your data using simple, readable implementations that prioritize portability over performance. 

Originally developed to support the Outer Element Taxonomy research framework, this toolkit is designed for **modular research workflows and rapid prototyping** where simplicity and reproducibility matter more than raw performance. Production-grade reliability for research computing - perfect for labs, experiments, and projects where you want to avoid pandas/numpy dependencies.

## Features

**Data Access & Storage**
- **Binary Search**: Fast lookups on sorted lists and coordinate pairs
- **HashMap**: Pure-Python hash table with automatic resizing and collision handling
- **SQLite Helpers**: Simple wrappers for database operations - queries, bulk inserts, schema management

**Data Transformation**
- **Matrix**: Operations for numerical data (transpose, get/set, generation)

**Text Analysis (MinText)**
- **Text**: Tokenization, frequency analysis, entropy, sentiment, and vectorization
- **Corpus**: Collection operations, vocabulary building, batch vectorization, and SQLite persistence

**I/O & Validation**
- **Text Validators**: Validate JSON, YAML, and Markdown content before processing

**Data Modeling**
- **Graph Generation**: Build NetworkX graphs programmatically for relationship modeling

**Symbolic Mathematics (Symbolics)**
- **SymbolicExpression**: Expression manipulation, simplification, expansion, factoring
- **Calculus**: Differentiation, integration, limits, Taylor series
- **SymbolicSolver**: Solve algebraic equations, systems of equations, and ODEs
- **FormulaLibrary**: SQLite-based formula storage with tagging and search
- **Formula Validation**: Parse and validate mathematical expressions
- **Matrix Integration**: Symbolic determinants, inverses, and conversions to/from SymPy matrices

**Observability**
- **Logging Helpers**: Lightweight logger factory and inline logging with opt-in verbosity

**Quality**
- **Well Tested**: 140+ tests covering algorithms, edge cases, and error handling

## Quick Start

### Installation

```bash
pip install oet-core
```

For local development with optional features:

```bash
git clone https://github.com/markusapplegate/oet-core.git
cd oet-core
pip install -e .[dev,all]
```

### Basic Usage

```python
from oet_core import binary_search, HashMap, Matrix, SQLiteHelper, Text, Corpus
from oet_core import SymbolicExpression, SymbolicSolver, FormulaLibrary

# Binary search
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
index = binary_search(numbers, 7)  # Returns: 6

# HashMap
hmap = HashMap()
hmap.put("name", "Alice")
print(hmap.get("name"))  # Returns: "Alice"

# Matrix operations
matrix = Matrix(3, 3, fill=0)
matrix.set(0, 0, 1)
transposed = matrix.transpose()

# SQLite database operations
with SQLiteHelper(":memory:") as db:
    db.create_table("users", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.execute("INSERT INTO users VALUES (?, ?)", (1, "Alice"))
    users = db.fetch_all("SELECT * FROM users")
    print(users[0]["name"])  # Returns: "Alice"

# Text analysis (MinText)
text = Text("The quick brown fox jumps over the lazy dog.")
tokens = text.tokenize()  # ['the', 'quick', 'brown', 'fox', ...]
sentiment = text.sentiment()  # {'score': 0, 'positive': 0, 'negative': 0, ...}
entropy = text.entropy()  # 3.17 bits (text diversity measure)

# Corpus operations with persistence
corpus = Corpus()
corpus.add_from_string("I love this product!", metadata={"rating": 5})
corpus.add_from_string("Terrible experience.", metadata={"rating": 1})

vocab = corpus.vocabulary()  # Build shared vocabulary
vectors = corpus.vectorize_all()  # Convert to term-frequency matrix

with SQLiteHelper(":memory:") as db:
    corpus.save_to_db(db, table="reviews")  # Persist to SQLite
    loaded = Corpus.load_from_db(db, table="reviews")

# Symbolic mathematics (requires SymPy)
expr = SymbolicExpression("x**2 + 2*x + 1")
expanded = expr.expand()  # Algebraic manipulation
derivative = expr.differentiate("x")  # 2*x + 2
integral = expr.integrate("x")  # x**3/3 + x**2 + x
result = expr.evaluate({"x": 5})  # 36.0

# Equation solving
solver = SymbolicSolver()
solutions = solver.solve("x**2 - 4 = 0", "x")  # [-2, 2]
system_solution = solver.solve_system(["x + y = 5", "x - y = 1"])  # {x: 3, y: 2}

# Formula library with SQLite
with FormulaLibrary("formulas.db") as library:
    library.save_formula(
        "quadratic",
        "a*x**2 + b*x + c",
        description="General quadratic equation",
        tags=["algebra", "polynomial"],
        metadata={"degree": "2"}
    )
    
    # Search and load formulas
    formulas = library.search(tag="algebra")
    formula = library.load_formula("quadratic")
    expr = formula["expression"]

# Convert symbolic to numeric function
f = expr.to_function()
value = f(x=3)  # Fast numeric evaluation

# Generate LaTeX for papers
latex = expr.to_latex()  # 'x^{2} + 2 x + 1'

# Matrix symbolic operations (requires SymPy)
m = Matrix(2, 2)
m.set(0, 0, 'a')
m.set(0, 1, 'b')
m.set(1, 0, 'c')
m.set(1, 1, 'd')

# Compute symbolic determinant
det = m.symbolic_determinant()  # a*d - b*c

# Compute symbolic inverse
inv = m.symbolic_inverse()

# Convert to SymPy matrix for advanced operations
sym_matrix = m.to_symbolic()
eigenvals = sym_matrix.eigenvals()

# Logging utilities
from io import StringIO
from oet_core import get_logger, log, set_utils_verbose_logging, generate_matrix

buffer = StringIO()
logger = get_logger("demo", stream=buffer, timestamps=False)
logger.info("Pipeline started")

log("Inline status update", level="warning")

set_utils_verbose_logging(True)
generate_matrix(1, 1)
set_utils_verbose_logging(False)
```

## Project Structure

```
oet-core/
├── README.md              # This file
├── docs/
│   ├── API_DOCS.md        # Complete API documentation
│   └── MINTEXT_GUIDE.md   # MinText quick reference
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── requirements.txt       # Development dependencies
├── pyproject.toml         # Package metadata
├── src/
│   ├── oet_core/
│   │   ├── __init__.py    # Package exports
│   │   ├── algos.py       # Algorithm implementations (binary_search, HashMap)
│   │   ├── mintext.py     # Text analysis (Text, Corpus)
│   │   ├── symbolics.py   # Symbolic mathematics (SymbolicExpression)
│   │   └── utils.py       # Utility helpers (Matrix, SQLite, logging, graphs)
│   ├── __init__.py        # Compatibility shim for legacy imports
│   └── utils.py           # Compatibility shim for legacy imports
└── tests/
    ├── __init__.py        # Test package
    ├── test_algos.py      # Algorithm tests
    ├── test_mintext.py    # MinText tests
    ├── test_utils.py      # Utility tests
    └── run_all_tests.py   # Test runner
```

## Running Tests

Comprehensive test suite with 140+ tests covering all modules.

### Run all tests:

```bash
python tests/run_all_tests.py
```

### Run specific test modules:

```bash
python tests/test_algos.py      # Test algorithms (binary_search, HashMap)
python tests/test_mintext.py    # Test text analysis (Text, Corpus)
python tests/test_symbolics.py  # Test symbolic mathematics (SymbolicExpression, SymbolicSolver, FormulaLibrary, Matrix integration)
python tests/test_utils.py      # Test utilities (Matrix, SQLite, validation, logging, graphs)
```

**Test Coverage:**
- **algos.py**: Binary search (scalars, pairs, duplicates, edge cases), HashMap (CRUD operations, resizing, collisions)
- **mintext.py**: Text tokenization, frequency analysis, entropy, sentiment, vectorization, Corpus operations, SQLite persistence
- **symbolics.py**: Expression parsing, validation, calculus (differentiation, integration, limits, series), symbolic-to-numeric conversion, LaTeX generation, equation solving (algebraic, systems, ODEs), formula library (SQLite storage, tagging, search), Matrix integration (symbolic determinants, inverses, conversions)
- **utils.py**: Matrix operations, text validation (JSON/YAML/Markdown), SQLite helpers, logging, graph building

*Note: Graph tests require `networkx` and symbolic tests require `sympy` to be installed (see optional dependencies).*

## Documentation

- **[API_DOCS.md](docs/API_DOCS.md)** - Complete API reference with examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## Design Philosophy

**Built for research workflows:**

Originally developed to support the Outer Element Taxonomy research framework, this library embodies principles essential for production research software:

- **Simplicity over speed**: Readable implementations researchers can understand, modify, and trust
- **Zero core dependencies**: Ensures reproducibility - works anywhere Python runs without dependency hell
- **Pure Python portability**: From laptops to HPC clusters to embedded systems
- **Modular design**: Mix and match components for rapid prototyping and experimentation
- **Production-ready**: Well-tested and documented - reliable enough for daily research use
- **Research-grade engineering**: Clear APIs, comprehensive tests, and proper versioning

**When to use oet-core:**
- **Production research computing** - reliable tools for daily research workflows
- Experimental prototyping and rapid iteration
- Reproducibility-critical environments where dependencies matter
- Teaching, learning, and understanding data structures
- Memory-constrained systems (embedded, serverless, HPC login nodes)
- Any project prioritizing simplicity and transparency over raw speed

**When NOT to use oet-core:**
- High-performance numerical computing at massive scale (use numpy/pandas)
- Enterprise data warehousing with strict SLAs
- When you need highly optimized algorithms for production data pipelines

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Optional Dependencies

- YAML validation support: `pip install oet-core[yaml]`
- Graph utilities: `pip install oet-core[graph]`
- Symbolic mathematics: `pip install oet-core[symbolic]`
- All extras and dev tooling: `pip install oet-core[all,dev]`

---

**Built with care following the principle of "minimum code, maximum value"**
