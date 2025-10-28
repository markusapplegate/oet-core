# Release Instructions for v1.1.0

## ✅ Version Bumped to 1.1.0
- pyproject.toml
- VERSION
- src/oet_core/__init__.py
- CHANGELOG.md

---

## Step 1: Commit Changes

```bash
cd /Users/markus/Desktop/oet-core

# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "Release v1.1.0: Symbolic Mathematics & Matrix Integration

- Add SymbolicExpression class for expression manipulation and calculus
- Add SymbolicSolver for algebraic equations, systems, and ODEs
- Add FormulaLibrary for SQLite-based formula storage with tagging
- Integrate Matrix class with symbolic operations (determinant, inverse)
- Add matrix_to_symbolic(), symbolic_to_matrix() conversion functions
- Add 93 comprehensive tests (140+ tests total)
- Update documentation (README, API_DOCS, CHANGELOG, CONTRIBUTING)
- Add optional sympy>=1.12 dependency"

# Create version tag
git tag -a v1.1.0 -m "Release v1.1.0: Symbolic Mathematics & Matrix Integration

Major Features:
- Complete symbolic mathematics module with SymPy integration
- Matrix symbolic operations (determinant, inverse, conversions)
- Formula library with SQLite persistence and search
- Equation solver for algebraic, systems, and ODEs
- 93 new tests, comprehensive documentation

Breaking Changes: None
Dependencies: Optional sympy>=1.12 for symbolic features"

# Push to GitHub
git push origin main
git push origin v1.1.0
```

---

## Step 2: Create GitHub Release

1. Go to: https://github.com/markusapplegate/oet-core/releases/new
2. Choose tag: `v1.1.0`
3. Release title: `v1.1.0 - Symbolic Mathematics & Matrix Integration`
4. Description (copy from below):

```markdown
## 🎉 Major Feature Release: Symbolic Mathematics

This release adds comprehensive symbolic mathematics capabilities to oet-core, integrating SymPy with our existing Matrix class.

### ✨ New Features

**Symbolic Mathematics Module (`oet_core.symbolics`)**
- **SymbolicExpression**: Expression manipulation, simplification, expansion, factoring
- **Calculus Operations**: Differentiation, integration, limits, Taylor series
- **SymbolicSolver**: Solve algebraic equations, systems of equations, and ODEs
- **FormulaLibrary**: SQLite-based formula storage with tagging, metadata, and search
- **Conversion Tools**: Expression → numeric function, LaTeX export

**Matrix Symbolic Integration**
- Convert Matrix ↔ SymPy matrices (`matrix_to_symbolic`, `symbolic_to_matrix`)
- Symbolic determinant computation (`symbolic_determinant`)
- Symbolic matrix inversion (`symbolic_inverse`)
- Direct Matrix methods: `m.to_symbolic()`, `m.symbolic_determinant()`, `m.symbolic_inverse()`

### 📊 Testing & Quality
- **93 new tests** for symbolic operations and Matrix integration
- **140+ total tests** with 100% pass rate
- Comprehensive test coverage for all symbolic features

### 📚 Documentation
- Complete API reference in docs/API_DOCS.md
- Updated README with usage examples
- CHANGELOG with detailed feature list
- CONTRIBUTING guide updated

### 📦 Installation

```bash
# Install with symbolic mathematics support
pip install oet-core[symbolic]

# Or install SymPy separately
pip install oet-core
pip install sympy>=1.12
```

### 🚀 Quick Start

```python
from oet_core import SymbolicExpression, SymbolicSolver, Matrix

# Symbolic expressions
expr = SymbolicExpression("x**2 + 2*x + 1")
derivative = expr.differentiate("x")
result = expr.evaluate({"x": 5})

# Equation solving
solver = SymbolicSolver()
solutions = solver.solve("x**2 - 4 = 0", "x")  # [-2, 2]

# Matrix symbolic operations
m = Matrix(2, 2)
m.set(0, 0, 'a')
m.set(0, 1, 'b')
m.set(1, 0, 'c')
m.set(1, 1, 'd')
det = m.symbolic_determinant()  # a*d - b*c
```

### 🔄 Breaking Changes
None - fully backward compatible

### 🐛 Bug Fixes
None in this release

### 📝 Full Changelog
See [CHANGELOG.md](https://github.com/markusapplegate/oet-core/blob/main/CHANGELOG.md)

---

**Dependencies**: Python 3.8+, optional: sympy>=1.12
```

5. Attach assets (optional): You can build and attach wheel files
6. Click "Publish release"

---

## Step 3: Publish to PyPI (Optional)

If you want to publish to PyPI:

```bash
cd /Users/markus/Desktop/oet-core

# Install build tools
python3 -m pip install --upgrade build twine

# Build distribution packages
python3 -m build

# Check the packages
twine check dist/*

# Upload to Test PyPI first (recommended)
twine upload --repository testpypi dist/*
# Visit: https://test.pypi.org/project/oet-core/

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ oet-core[symbolic]

# If everything works, upload to real PyPI
twine upload dist/*
# Visit: https://pypi.org/project/oet-core/
```

**Note**: You'll need accounts on:
- TestPyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/

---

## Step 4: Verify Release

After publishing:

```bash
# Test fresh install
pip install oet-core[symbolic]==1.1.0

# Verify version
python3 -c "import oet_core; print(oet_core.__version__)"

# Test imports
python3 -c "from oet_core import SymbolicExpression, SymbolicSolver, FormulaLibrary, matrix_to_symbolic; print('✓ All imports successful')"

# Run tests
python3 -m unittest discover tests
```

---

## 🎉 Release Complete!

Once published:
- GitHub Release: https://github.com/markusapplegate/oet-core/releases/tag/v1.1.0
- PyPI (if published): https://pypi.org/project/oet-core/1.1.0/
- Documentation: https://github.com/markusapplegate/oet-core/blob/main/docs/API_DOCS.md
