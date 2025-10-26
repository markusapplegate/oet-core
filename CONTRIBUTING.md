# Contributing to oet-core

Thank you for your interest in contributing to oet-core! This document provides guidelines for contributing to the project.

## Philosophy

oet-core follows the principle of **"minimum code, maximum value"**:

- Keep implementations simple and readable
- Avoid unnecessary dependencies
- Prioritize clarity over cleverness
- Maintain comprehensive test coverage
- Document all public APIs

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/markusapplegate/oet-core.git
cd oet-core
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e .[dev,all]
```

4. Run tests to verify setup:
```bash
python tests/run_all_tests.py
```

## Development Workflow

### Making Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines below

3. **Add tests** for your changes in the appropriate test file:
   - `tests/test_algos.py` for algorithm changes
   - `tests/test_utils.py` for utility changes

4. **Run the test suite:**
```bash
python tests/run_all_tests.py
```


### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use type hints for function parameters and return values
- Keep functions focused and under 50 lines when possible
- Add docstrings to all public functions and classes
- Use clear, descriptive variable names

#### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.
    
    More detailed explanation if needed.
    
    Parameters
    ----------
    param1:
        Description of param1
    param2:
        Description of param2
    
    Returns
    -------
    bool
        Description of return value
    
    Examples
    --------
    >>> my_function(5, "hello")
    True
    """
    pass
```

### Testing

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Test edge cases and error conditions
- Use descriptive test names: `test_<what>_<condition>_<expected>`

Example:
```python
def test_binary_search_empty_list_returns_none(self):
    """Test that binary_search returns None for empty list."""
    result = binary_search([], 5)
    self.assertIsNone(result)
```

### Commit Messages

Write clear, concise commit messages:

```
Add feature: brief description

More detailed explanation of what changed and why.
Include any relevant context.
```

Good examples:
- `Add support for string keys in HashMap`
- `Fix binary_search handling of duplicate values`
- `Update Matrix.transpose() for empty matrices`

## Types of Contributions

### Bug Fixes

1. Check if an issue already exists
2. If not, create an issue describing the bug
3. Reference the issue in your PR

### New Features

1. Open an issue to discuss the feature first
2. Ensure it aligns with the project philosophy
3. Implement with tests and documentation
4. Update docs/API_DOCS.md if adding public APIs

### Documentation

- Fix typos, clarify explanations
- Add examples to docs/API_DOCS.md
- Improve README or other docs

### Tests

- Add missing test cases
- Improve test coverage
- Add edge case tests

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features or bug fixes
3. **Ensure all tests pass** before submitting
4. **Update docs/API_DOCS.md** if you added/changed public APIs
5. **Keep PRs focused** - one feature or fix per PR
6. **Write a clear PR description** explaining what and why

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] No unnecessary dependencies added

## Project Structure

```
oet-core/
├── src/                    # Source code
│   ├── oet_core/          # Package modules
│   │   ├── __init__.py    # Package exports
│   │   ├── algos.py       # Algorithm implementations
│   │   └── utils.py       # Utility helpers
│   ├── __init__.py        # Compatibility shim
│   └── utils.py           # Compatibility shim
├── tests/                 # Test suite
│   ├── test_algos.py      # Algorithm tests
│   ├── test_utils.py      # Utility tests
│   └── run_all_tests.py   # Test runner
├── requirements.txt       # Development dependencies
├── docs/
│   └── API_DOCS.md       # Library documentation
├── README.md             # Project overview
└── CONTRIBUTING.md       # This guide
```

## API Design Guidelines

When adding new features to the library:

1. **Keep it simple** - Prefer clarity over performance micro-optimizations
2. **Minimal API surface** - Expose only what's necessary
3. **Type hints** - Always use type hints for public APIs
4. **Error handling** - Validate inputs and provide clear error messages
5. **Documentation** - Document with examples in docstrings
6. **Testing** - Write comprehensive tests

## Questions?

If you have questions or need help:

1. Check existing documentation
2. Look at similar existing code
3. Open an issue for discussion

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to oet-core! 🚀
