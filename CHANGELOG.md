# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/markusapplegate/oet-core/releases/tag/v1.0.0
