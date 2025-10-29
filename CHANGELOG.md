# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2025-10-28

### Added
- **Arithmetic Operator Overloading** for `SymbolicExpression`:
  - `+`, `-`, `*`, `/`, `**` operators for building expressions programmatically
  - Support for mixed operations with numbers (e.g., `expr + 5`, `3 * expr`)
  - Right-side operators (`__radd__`, `__rsub__`, etc.) for natural syntax
  - Unary operators: `-expr` (negation), `+expr` (positive), `abs(expr)` (absolute value)
  - Enables intuitive expression building: `(x + y) ** 2` instead of only string parsing
- 20 comprehensive tests for arithmetic operators with various combinations
- New documentation: `ARITHMETIC_OPERATORS.md` - Quick reference guide for symbolic operators

### Fixed
- Critical usability gap: `SymbolicExpression` now supports basic arithmetic operations
- Users can now build symbolic expressions programmatically instead of only using strings

## [1.1.0] - 2025-10-28

### Added
- **Symbolic Mathematics Module** (`oet_core.symbolics`):
  - `SymbolicExpression` class for expression manipulation, calculus, and conversion
  - `SymbolicSolver` class for solving algebraic equations, systems, and ODEs
  - `FormulaLibrary` class for SQLite-based formula storage with tagging and metadata
  - `parse_expression()` and `validate_formula()` utility functions
  - Comprehensive symbolic mathematics with optional SymPy dependency
- **Matrix Symbolic Integration**:
  - `Matrix.to_symbolic()` method to convert to SymPy matrices
  - `Matrix.symbolic_determinant()` for symbolic determinant computation
  - `Matrix.symbolic_inverse()` for symbolic matrix inversion
  - `matrix_to_symbolic()`, `symbolic_to_matrix()` conversion functions
  - `symbolic_determinant()` and `symbolic_inverse()` standalone functions
- 93 new tests for symbolic operations and Matrix integration (140+ tests total)
- Comprehensive documentation in API_DOCS.md for symbolics module
- Optional `symbolic` dependency group in pyproject.toml

## [1.0.0] - 2025-10-25

### Added
- Binary search helpers for lists and coordinate pairs
- Pure Python `HashMap` with automatic resizing and collision handling
- `Matrix` class with generation, mutation, and transpose utilities
- `SQLiteHelper` class for lightweight database operations (queries, bulk inserts, schema management)
- `create_sqlite_helper` factory function for SQLite connections
- Text block validators for JSON, YAML, and Markdown content
- Logging helpers with lazy initialization and opt-in verbose output
- Graph generation utilities built on top of `networkx`
- PEP 561 `py.typed` marker and documented public API surface

### Notes
- Initial public release of `oet-core`

[1.2.0]: https://github.com/markusapplegate/oet-core/releases/tag/v1.2.0
[1.1.0]: https://github.com/markusapplegate/oet-core/releases/tag/v1.1.0
[1.0.0]: https://github.com/markusapplegate/oet-core/releases/tag/v1.0.0
