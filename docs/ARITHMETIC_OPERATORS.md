# Arithmetic Operators Quick Reference

## Installation

To use symbolic mathematics with arithmetic operators:

```bash
pip install oet-core[symbolic]
```

## Overview

The `SymbolicExpression` class supports all standard Python arithmetic operators, allowing you to build symbolic expressions programmatically instead of only using strings.

## Basic Operators

### Addition (`+`)

```python
from oet_core import SymbolicExpression

x = SymbolicExpression("x")
y = SymbolicExpression("y")

# Expression + Expression
result = x + y          # x + y

# Expression + Number
result = x + 5          # x + 5

# Number + Expression (reverse)
result = 3 + x          # x + 3
```

### Subtraction (`-`)

```python
# Expression - Expression
result = x - y          # x - y

# Expression - Number
result = x - 2          # x - 2

# Number - Expression (reverse)
result = 10 - x         # 10 - x
```

### Multiplication (`*`)

```python
# Expression * Expression
result = x * y          # x*y

# Expression * Number
result = x * 3          # 3*x

# Number * Expression (reverse)
result = 5 * x          # 5*x
```

### Division (`/`)

```python
# Expression / Expression
result = x / y          # x/y

# Expression / Number
result = x / 2          # x/2

# Number / Expression (reverse)
result = 12 / x         # 12/x
```

### Exponentiation (`**`)

```python
# Expression ** Expression
result = x ** y         # x**y

# Expression ** Number
result = x ** 2         # x**2

# Number ** Expression (reverse)
result = 2 ** x         # 2**x
```

## Unary Operators

### Negation (`-`)

```python
x = SymbolicExpression("x")
result = -x             # -x
```

### Unary Plus (`+`)

```python
result = +x             # x
```

### Absolute Value (`abs()`)

```python
result = abs(x)         # Abs(x)
```

## Complex Examples

### Building Formulas

```python
from oet_core import SymbolicExpression

x = SymbolicExpression("x")

# Quadratic formula
quadratic = x**2 + 2*x + 1
print(quadratic)                    # x**2 + 2*x + 1

# Expand factored form
factored = (x + 1) ** 2
expanded = factored.expand()
print(expanded)                     # x**2 + 2*x + 1

# Build and factor
expr = (x + 3) * (x - 2)
expanded = expr.expand()
print(expanded)                     # x**2 + x - 6
```

### Dynamic Formula Generation

```python
def build_quadratic(a, b, c):
    """Build ax^2 + bx + c programmatically"""
    x = SymbolicExpression("x")
    return a*x**2 + b*x + c

# Create different quadratics
formula1 = build_quadratic(1, 2, 1)    # x**2 + 2*x + 1
formula2 = build_quadratic(2, -5, 3)   # 2*x**2 - 5*x + 3

# Factor them
print(formula1.factor())               # (x + 1)**2
print(formula2.factor())               # (2*x - 3)*(x - 1)
```

### Combining with Calculus

```python
x = SymbolicExpression("x")

# Build expression
f = x**3 + 2*x**2 - 5*x + 3

# Differentiate
f_prime = f.differentiate("x")
print(f_prime)                      # 3*x**2 + 4*x - 5

# Integrate
F = f.integrate("x")
print(F)                            # x**4/4 + 2*x**3/3 - 5*x**2/2 + 3*x

# Evaluate
result = f.evaluate({"x": 2})
print(result)                       # 5.0
```

### Multi-Variable Expressions

```python
x = SymbolicExpression("x")
y = SymbolicExpression("y")
z = SymbolicExpression("z")

# Build complex expression
expr = (x + y) * z + x**2 * y

# Expand
expanded = expr.expand()
print(expanded)                     # x**2*y + x*z + y*z

# Partial derivatives
dx = expr.differentiate("x")
dy = expr.differentiate("y")
dz = expr.differentiate("z")

print(dx)                           # 2*x*y + z
print(dy)                           # x**2 + z
print(dz)                           # x + y
```

### Converting to Functions

```python
x = SymbolicExpression("x")

# Build expression with operators
expr = 3*x**2 + 2*x + 1

# Convert to fast numeric function
f = expr.to_function()

# Evaluate many times efficiently
values = [f(x=i) for i in range(10)]
print(values)  # [1, 6, 17, 34, 57, 86, 121, 162, 209, 262]
```

## Operator Precedence

Operators follow standard Python precedence:

1. `**` (exponentiation) - highest
2. `-x`, `+x`, `abs(x)` (unary)
3. `*`, `/` (multiplication, division)
4. `+`, `-` (addition, subtraction) - lowest

Use parentheses to control evaluation order:

```python
x = SymbolicExpression("x")

# Without parentheses
expr1 = x + 2 * x       # x + 2*x = 3*x

# With parentheses
expr2 = (x + 2) * x     # (x + 2)*x = x**2 + 2*x
```

## Comparison: String vs Operator Building

### Before (String Only)

```python
# Old way - string concatenation
x_str = "x"
expr_str = f"({x_str} + 3) * ({x_str} - 2)"
expr = SymbolicExpression(expr_str)
```

**Problems:**
- Error-prone string concatenation
- Hard to build dynamically
- No IDE support or type checking
- Difficult to compose expressions

### After (With Operators)

```python
# New way - operators
x = SymbolicExpression("x")
expr = (x + 3) * (x - 2)
```

**Benefits:**
- ✅ Natural Python syntax
- ✅ Type-safe composition
- ✅ IDE autocomplete works
- ✅ Easy to build dynamically
- ✅ Readable and maintainable

## Common Patterns

### Pattern 1: Polynomial Builder

```python
def build_polynomial(coefficients):
    """Build a polynomial from coefficients [a0, a1, a2, ...]"""
    x = SymbolicExpression("x")
    result = coefficients[0]
    for i, coeff in enumerate(coefficients[1:], start=1):
        result = result + coeff * x**i
    return result

# Build 3x^2 + 2x + 1
poly = build_polynomial([1, 2, 3])
print(poly)  # 3*x**2 + 2*x + 1
```

### Pattern 2: Taylor Series Approximation

```python
import math

def manual_taylor(center, n_terms):
    """Build Taylor series for e^x manually"""
    x = SymbolicExpression("x")
    result = 1  # First term
    for n in range(1, n_terms):
        result = result + (x - center)**n / math.factorial(n)
    return result

# 4th order Taylor series for e^x around 0
taylor = manual_taylor(0, 4)
print(taylor)  # 1 + x + x**2/2 + x**3/6
```

### Pattern 3: Expression Templates

```python
def distance_formula(dim):
    """Build n-dimensional distance formula"""
    from oet_core import SymbolicExpression
    
    # Create variables dynamically
    vars = [SymbolicExpression(f"x{i}") for i in range(dim)]
    
    # Sum of squares
    sum_squares = vars[0]**2
    for v in vars[1:]:
        sum_squares = sum_squares + v**2
    
    return sum_squares

# 2D distance: x0^2 + x1^2
dist_2d = distance_formula(2)

# 3D distance: x0^2 + x1^2 + x2^2
dist_3d = distance_formula(3)
```

## Error Handling

```python
from oet_core import SymbolicExpression

x = SymbolicExpression("x")

# These work fine
result = x + 5          # ✓
result = 3 * x          # ✓
result = x ** 2         # ✓

# Invalid operations return NotImplemented
# (Python will try the other operand's method)
try:
    result = x + "string"   # Will raise TypeError
except TypeError as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Create variables once, reuse them:**
   ```python
   x = SymbolicExpression("x")
   y = SymbolicExpression("y")
   
   expr1 = x + y
   expr2 = x * y
   expr3 = x**2 + y**2
   ```

2. **Use descriptive variable names:**
   ```python
   # Good
   time = SymbolicExpression("t")
   velocity = SymbolicExpression("v")
   position = velocity * time
   
   # Less clear
   x = SymbolicExpression("x")
   y = SymbolicExpression("y")
   z = y * x
   ```

3. **Combine operators with methods:**
   ```python
   x = SymbolicExpression("x")
   
   # Build with operators
   expr = (x + 1)**2
   
   # Manipulate with methods
   expanded = expr.expand()
   derivative = expanded.differentiate("x")
   result = derivative.evaluate({"x": 5})
   ```

4. **Cache function conversions:**
   ```python
   x = SymbolicExpression("x")
   expr = x**3 + 2*x**2 + x
   
   # Convert once
   f = expr.to_function()
   
   # Evaluate many times (fast!)
   results = [f(x=i) for i in range(1000)]
   ```

## See Also

- [Full API Documentation](API_DOCS.md)
- [Main README](../README.md)
- [MinText Guide](MINTEXT_GUIDE.md)
