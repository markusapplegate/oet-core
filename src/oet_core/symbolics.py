"""Symbolics: Lightweight symbolic mathematics for oet_core.

Provides simple, dependency-free symbolic math operations using SymPy for
research workflows including expression manipulation, calculus, and equation solving.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from .utils import log


_VERBOSE_LOGGING = False


def set_symbolics_verbose_logging(enabled: bool) -> None:
    """Enable or disable verbose logging for symbolics module."""
    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


def _log_if_verbose(message: str) -> None:
    """Log message if verbose logging is enabled."""
    if _VERBOSE_LOGGING:
        log(message, level="info")


def _import_sympy():
    """Import sympy with helpful error message if not available."""
    try:
        import sympy
        return sympy
    except ImportError as exc:
        raise ImportError(
            "SymPy is required for symbolic operations. Install with: pip install oet-core[symbolic]"
        ) from exc


def parse_expression(expr_str: str, **kwargs) -> Any:
    """Parse a string into a SymPy expression.
    
    Parameters
    ----------
    expr_str:
        String representation of mathematical expression.
        Examples: "x**2 + 2*x + 1", "sin(x) + cos(y)", "exp(x*y)"
    **kwargs:
        Additional arguments passed to sympy.sympify.
    
    Returns
    -------
    sympy.Expr
        Parsed SymPy expression object.
    
    Examples
    --------
    >>> expr = parse_expression("x**2 + 2*x + 1")
    >>> expr = parse_expression("sin(x) + cos(y)")
    """
    _log_if_verbose(f"parse_expression called with expr_str='{expr_str}'")
    
    if not isinstance(expr_str, str):
        raise TypeError("expr_str must be a string")
    
    sp = _import_sympy()
    
    try:
        return sp.sympify(expr_str, **kwargs)
    except Exception as exc:
        raise ValueError(f"Failed to parse expression '{expr_str}': {exc}") from exc


def validate_formula(formula: str) -> tuple[bool, Optional[str]]:
    """Validate a mathematical formula string.
    
    Checks if a string can be successfully parsed as a SymPy expression
    and returns validation status with error message if invalid.
    
    Parameters
    ----------
    formula:
        Mathematical formula string to validate.
    
    Returns
    -------
    tuple[bool, Optional[str]]
        Tuple of (is_valid, error_message). error_message is None if valid.
    
    Examples
    --------
    >>> valid, error = validate_formula("x**2 + 2*x + 1")
    >>> assert valid and error is None
    
    >>> valid, error = validate_formula("x**2 + + y")
    >>> assert not valid
    """
    _log_if_verbose(f"validate_formula called with formula='{formula}'")
    
    if not isinstance(formula, str):
        return False, "formula must be a string"
    
    if not formula.strip():
        return False, "formula cannot be empty"
    
    try:
        parse_expression(formula)
        return True, None
    except Exception as exc:
        return False, str(exc)


class SymbolicExpression:
    """Wrapper for SymPy expressions with research-friendly interface.
    
    Provides simplified access to common symbolic operations including
    simplification, calculus, substitution, and conversion to numeric functions.
    """
    
    def __init__(self, expression: Union[str, Any]) -> None:
        """Initialize a symbolic expression.
        
        Parameters
        ----------
        expression:
            Mathematical expression as string or existing SymPy expression.
            String examples: "x**2 + 2*x + 1", "sin(x)*cos(y)", "exp(-x**2)"
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + 2*x + 1")
        >>> expr = SymbolicExpression("sin(x) + cos(x)")
        """
        _log_if_verbose(f"SymbolicExpression.__init__ called with expression={expression}")
        
        sp = _import_sympy()
        
        if isinstance(expression, str):
            self.expr = parse_expression(expression)
        elif isinstance(expression, sp.Basic):
            self.expr = expression
        else:
            raise TypeError("expression must be a string or SymPy expression")
        
        self._cached_function: Optional[Callable] = None
        self._cached_vars: Optional[tuple] = None
    
    def simplify(self) -> "SymbolicExpression":
        """Simplify the expression algebraically.
        
        Returns
        -------
        SymbolicExpression
            New expression with simplified form.
        
        Examples
        --------
        >>> expr = SymbolicExpression("(x + 1)**2")
        >>> simplified = expr.simplify()  # x**2 + 2*x + 1
        """
        _log_if_verbose("SymbolicExpression.simplify called")
        
        sp = _import_sympy()
        simplified = sp.simplify(self.expr)
        return SymbolicExpression(simplified)
    
    def expand(self) -> "SymbolicExpression":
        """Expand the expression.
        
        Returns
        -------
        SymbolicExpression
            New expression with expanded form.
        
        Examples
        --------
        >>> expr = SymbolicExpression("(x + 1)**2")
        >>> expanded = expr.expand()  # x**2 + 2*x + 1
        """
        _log_if_verbose("SymbolicExpression.expand called")
        
        sp = _import_sympy()
        expanded = sp.expand(self.expr)
        return SymbolicExpression(expanded)
    
    def factor(self) -> "SymbolicExpression":
        """Factor the expression.
        
        Returns
        -------
        SymbolicExpression
            New expression with factored form.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 - 1")
        >>> factored = expr.factor()  # (x - 1)*(x + 1)
        """
        _log_if_verbose("SymbolicExpression.factor called")
        
        sp = _import_sympy()
        factored = sp.factor(self.expr)
        return SymbolicExpression(factored)
    
    def differentiate(self, *variables: str) -> "SymbolicExpression":
        """Compute derivative with respect to one or more variables.
        
        Parameters
        ----------
        *variables:
            Variable names to differentiate with respect to.
            Multiple variables compute higher-order or partial derivatives.
        
        Returns
        -------
        SymbolicExpression
            Derivative expression.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**3 + 2*x**2")
        >>> derivative = expr.differentiate("x")  # 3*x**2 + 4*x
        
        >>> expr = SymbolicExpression("x**2 * y**2")
        >>> partial = expr.differentiate("x", "y")  # 4*x*y
        """
        _log_if_verbose(f"SymbolicExpression.differentiate called with variables={variables}")
        
        if not variables:
            raise ValueError("At least one variable required for differentiation")
        
        sp = _import_sympy()
        
        # Convert variable names to SymPy symbols
        symbols = [sp.Symbol(var) for var in variables]
        
        result = self.expr
        for sym in symbols:
            result = sp.diff(result, sym)
        
        return SymbolicExpression(result)
    
    def integrate(self, *variables: str, definite: bool = False, 
                  limits: Optional[Dict[str, tuple[float, float]]] = None) -> "SymbolicExpression":
        """Compute integral with respect to one or more variables.
        
        Parameters
        ----------
        *variables:
            Variable names to integrate with respect to.
        definite:
            Whether to compute definite integral (requires limits).
        limits:
            Dictionary mapping variable names to (lower, upper) bound tuples.
            Only used when definite=True.
        
        Returns
        -------
        SymbolicExpression
            Integrated expression (or numeric result for definite integrals).
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2")
        >>> integral = expr.integrate("x")  # x**3/3
        
        >>> expr = SymbolicExpression("x*y")
        >>> integral = expr.integrate("x", "y")
        
        >>> expr = SymbolicExpression("x**2")
        >>> result = expr.integrate("x", definite=True, limits={"x": (0, 1)})
        """
        _log_if_verbose(f"SymbolicExpression.integrate called with variables={variables}")
        
        if not variables:
            raise ValueError("At least one variable required for integration")
        
        sp = _import_sympy()
        
        result = self.expr
        
        for var in variables:
            sym = sp.Symbol(var)
            
            if definite and limits and var in limits:
                lower, upper = limits[var]
                result = sp.integrate(result, (sym, lower, upper))
            else:
                result = sp.integrate(result, sym)
        
        return SymbolicExpression(result)
    
    def substitute(self, substitutions: Dict[str, Union[float, int, str]]) -> "SymbolicExpression":
        """Substitute values for variables in the expression.
        
        Parameters
        ----------
        substitutions:
            Dictionary mapping variable names to values.
            Values can be numbers or other symbolic expressions.
        
        Returns
        -------
        SymbolicExpression
            Expression with substitutions applied.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + y")
        >>> result = expr.substitute({"x": 2, "y": 3})  # 7
        
        >>> expr = SymbolicExpression("x + y")
        >>> result = expr.substitute({"x": "2*z"})  # 2*z + y
        """
        _log_if_verbose(f"SymbolicExpression.substitute called with substitutions={substitutions}")
        
        if not isinstance(substitutions, dict):
            raise TypeError("substitutions must be a dictionary")
        
        sp = _import_sympy()
        
        # Convert keys to SymPy symbols and values to SymPy expressions
        subs_dict = {}
        for var, value in substitutions.items():
            sym = sp.Symbol(var)
            if isinstance(value, str):
                subs_dict[sym] = parse_expression(value)
            else:
                subs_dict[sym] = value
        
        result = self.expr.subs(subs_dict)
        return SymbolicExpression(result)
    
    def evaluate(self, values: Dict[str, Union[float, int]]) -> Union[float, complex]:
        """Evaluate expression numerically with given variable values.
        
        Parameters
        ----------
        values:
            Dictionary mapping variable names to numeric values.
        
        Returns
        -------
        float or complex
            Numeric result of evaluation.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + 2*x + 1")
        >>> result = expr.evaluate({"x": 5})  # 36.0
        """
        _log_if_verbose(f"SymbolicExpression.evaluate called with values={values}")
        
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary")
        
        sp = _import_sympy()
        
        # Convert keys to symbols
        subs_dict = {sp.Symbol(var): val for var, val in values.items()}
        
        result = self.expr.subs(subs_dict)
        
        # Convert to float/complex
        try:
            return complex(result) if result.has(sp.I) else float(result)
        except Exception as exc:
            raise ValueError(f"Cannot evaluate to numeric value: {exc}") from exc
    
    def limit(self, variable: str, point: Union[float, int, str], 
              direction: str = "+-") -> "SymbolicExpression":
        """Compute limit of expression as variable approaches a point.
        
        Parameters
        ----------
        variable:
            Variable name to take limit with respect to.
        point:
            Point to approach. Can be numeric or string like "oo" for infinity.
        direction:
            Direction of approach: "+-" (both sides), "+" (from right), "-" (from left).
        
        Returns
        -------
        SymbolicExpression
            Limit result.
        
        Examples
        --------
        >>> expr = SymbolicExpression("sin(x)/x")
        >>> lim = expr.limit("x", 0)  # 1
        
        >>> expr = SymbolicExpression("1/x")
        >>> lim = expr.limit("x", "oo")  # 0
        """
        _log_if_verbose(f"SymbolicExpression.limit called with variable={variable}, point={point}")
        
        sp = _import_sympy()
        
        sym = sp.Symbol(variable)
        
        # Handle infinity
        if isinstance(point, str) and point.lower() in ("oo", "inf", "infinity"):
            point = sp.oo
        
        result = sp.limit(self.expr, sym, point, dir=direction)
        return SymbolicExpression(result)
    
    def taylor_series(self, variable: str, center: Union[float, int] = 0, 
                      order: int = 6) -> "SymbolicExpression":
        """Compute Taylor series expansion.
        
        Parameters
        ----------
        variable:
            Variable to expand around.
        center:
            Point to expand around (default: 0 for Maclaurin series).
        order:
            Order of expansion (number of terms).
        
        Returns
        -------
        SymbolicExpression
            Taylor series expansion.
        
        Examples
        --------
        >>> expr = SymbolicExpression("exp(x)")
        >>> series = expr.taylor_series("x", center=0, order=4)
        """
        _log_if_verbose(f"SymbolicExpression.taylor_series called with variable={variable}, order={order}")
        
        sp = _import_sympy()
        
        sym = sp.Symbol(variable)
        series = self.expr.series(sym, center, order).removeO()
        return SymbolicExpression(series)
    
    def get_variables(self) -> List[str]:
        """Extract all variable symbols from the expression.
        
        Returns
        -------
        List[str]
            Sorted list of variable names.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + y*z")
        >>> vars = expr.get_variables()  # ['x', 'y', 'z']
        """
        _log_if_verbose("SymbolicExpression.get_variables called")
        
        symbols = self.expr.free_symbols
        return sorted([str(sym) for sym in symbols])
    
    def to_function(self, variables: Optional[List[str]] = None) -> Callable:
        """Convert symbolic expression to callable numeric function.
        
        Creates an optimized numeric function using SymPy's lambdify.
        Result is cached for performance.
        
        Parameters
        ----------
        variables:
            List of variable names defining function signature.
            If None, automatically detected from expression.
        
        Returns
        -------
        Callable
            Numeric function accepting keyword arguments.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + 2*x + 1")
        >>> f = expr.to_function()
        >>> result = f(x=5)  # 36
        
        >>> expr = SymbolicExpression("x*y + z")
        >>> f = expr.to_function(variables=["x", "y", "z"])
        >>> result = f(x=1, y=2, z=3)  # 5
        """
        _log_if_verbose("SymbolicExpression.to_function called")
        
        sp = _import_sympy()
        
        if variables is None:
            variables = self.get_variables()
        
        var_tuple = tuple(variables)
        
        # Check cache
        if self._cached_function is not None and self._cached_vars == var_tuple:
            return self._cached_function
        
        # Create symbols
        symbols = [sp.Symbol(var) for var in variables]
        
        # Lambdify with numpy support if available
        try:
            import numpy as np
            modules = [np, "math"]
        except ImportError:
            modules = ["math"]
        
        func = sp.lambdify(symbols, self.expr, modules=modules)
        
        # Wrap to accept keyword arguments
        def wrapped(**kwargs):
            args = [kwargs.get(var, 0) for var in variables]
            return func(*args)
        
        # Cache result
        self._cached_function = wrapped
        self._cached_vars = var_tuple
        
        return wrapped
    
    def to_latex(self) -> str:
        """Convert expression to LaTeX representation.
        
        Returns
        -------
        str
            LaTeX string representation.
        
        Examples
        --------
        >>> expr = SymbolicExpression("x**2 + 2*x + 1")
        >>> latex = expr.to_latex()  # 'x^{2} + 2 x + 1'
        """
        _log_if_verbose("SymbolicExpression.to_latex called")
        
        sp = _import_sympy()
        return sp.latex(self.expr)
    
    def to_string(self) -> str:
        """Convert expression to string representation.
        
        Returns
        -------
        str
            String representation of expression.
        """
        return str(self.expr)
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.expr)
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"SymbolicExpression('{self.expr}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if isinstance(other, SymbolicExpression):
            sp = _import_sympy()
            return sp.simplify(self.expr - other.expr) == 0
        return False


class SymbolicSolver:
    """Equation solver for symbolic expressions.
    
    Provides methods for solving algebraic equations, systems of equations,
    and differential equations.
    """
    
    def __init__(self) -> None:
        """Initialize the symbolic solver."""
        _log_if_verbose("SymbolicSolver.__init__ called")
        self.sp = _import_sympy()
    
    def solve(
        self,
        equation: Union[str, SymbolicExpression],
        variable: Optional[str] = None,
        **kwargs,
    ) -> List[Any]:
        """Solve an algebraic equation for a variable.
        
        Parameters
        ----------
        equation:
            Equation to solve. Can be a string like "x**2 - 4 = 0" or "x**2 - 4"
            (assumes = 0), or a SymbolicExpression.
        variable:
            Variable to solve for. If None and only one variable exists,
            that variable is used.
        **kwargs:
            Additional arguments passed to sympy.solve.
        
        Returns
        -------
        List[Any]
            List of solutions. May contain numbers, expressions, or SymPy objects.
        
        Examples
        --------
        >>> solver = SymbolicSolver()
        >>> solutions = solver.solve("x**2 - 4 = 0", "x")  # [-2, 2]
        >>> solutions = solver.solve("x**2 + 2*x + 1 = 0", "x")  # [-1]
        """
        _log_if_verbose(f"SymbolicSolver.solve called with equation={equation}, variable={variable}")
        
        # Parse equation
        if isinstance(equation, str):
            # Handle equations with "=" sign
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                expr = parse_expression(lhs) - parse_expression(rhs)
            else:
                expr = parse_expression(equation)
        elif isinstance(equation, SymbolicExpression):
            expr = equation.expr
        else:
            raise TypeError("equation must be a string or SymbolicExpression")
        
        # Determine variable to solve for
        if variable is None:
            symbols = list(expr.free_symbols)
            if len(symbols) == 0:
                raise ValueError("No variables found in equation")
            elif len(symbols) > 1:
                raise ValueError(f"Multiple variables found: {symbols}. Specify which to solve for.")
            variable = str(symbols[0])
        
        symbol = self.sp.Symbol(variable)
        
        # Solve the equation
        try:
            solutions = self.sp.solve(expr, symbol, **kwargs)
            return solutions
        except Exception as exc:
            raise ValueError(f"Failed to solve equation: {exc}") from exc
    
    def solve_system(
        self,
        equations: List[Union[str, SymbolicExpression]],
        variables: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[Any, Any]:
        """Solve a system of equations.
        
        Parameters
        ----------
        equations:
            List of equations to solve simultaneously.
        variables:
            List of variables to solve for. If None, automatically detected.
        **kwargs:
            Additional arguments passed to sympy.solve.
        
        Returns
        -------
        Dict[Any, Any]
            Dictionary mapping variables to their solutions.
            For multiple solutions, returns a list of solution dictionaries.
        
        Examples
        --------
        >>> solver = SymbolicSolver()
        >>> solutions = solver.solve_system(["x + y = 5", "x - y = 1"])
        >>> # {x: 3, y: 2}
        """
        _log_if_verbose(f"SymbolicSolver.solve_system called with {len(equations)} equations")
        
        # Parse equations
        exprs = []
        for eq in equations:
            if isinstance(eq, str):
                if "=" in eq:
                    lhs, rhs = eq.split("=", 1)
                    expr = parse_expression(lhs) - parse_expression(rhs)
                else:
                    expr = parse_expression(eq)
            elif isinstance(eq, SymbolicExpression):
                expr = eq.expr
            else:
                raise TypeError("equations must be strings or SymbolicExpressions")
            exprs.append(expr)
        
        # Determine variables
        if variables is None:
            all_symbols = set()
            for expr in exprs:
                all_symbols.update(expr.free_symbols)
            variables = [str(sym) for sym in all_symbols]
        
        symbols = [self.sp.Symbol(var) for var in variables]
        
        # Solve the system
        try:
            solutions = self.sp.solve(exprs, symbols, **kwargs)
            return solutions
        except Exception as exc:
            raise ValueError(f"Failed to solve system: {exc}") from exc
    
    def solve_ode(
        self,
        equation: Union[str, SymbolicExpression],
        function: str,
        **kwargs,
    ) -> Any:
        """Solve an ordinary differential equation (ODE).
        
        Parameters
        ----------
        equation:
            Differential equation to solve.
        function:
            Function to solve for (e.g., "y(x)").
        **kwargs:
            Additional arguments passed to sympy.dsolve.
        
        Returns
        -------
        Any
            Solution to the ODE as a SymPy expression.
        
        Examples
        --------
        >>> solver = SymbolicSolver()
        >>> # Solve dy/dx = y
        >>> solution = solver.solve_ode("Derivative(y(x), x) - y(x)", "y(x)")
        """
        _log_if_verbose(f"SymbolicSolver.solve_ode called with function={function}")
        
        # Parse equation
        if isinstance(equation, str):
            expr = parse_expression(equation)
        elif isinstance(equation, SymbolicExpression):
            expr = equation.expr
        else:
            raise TypeError("equation must be a string or SymbolicExpression")
        
        # Parse function
        func = self.sp.sympify(function)
        
        # Solve the ODE
        try:
            solution = self.sp.dsolve(expr, func, **kwargs)
            return solution
        except Exception as exc:
            raise ValueError(f"Failed to solve ODE: {exc}") from exc
    
    def __repr__(self) -> str:  # pragma: no cover
        """Developer representation."""
        return "SymbolicSolver()"


class FormulaLibrary:
    """SQLite-based storage for symbolic formulas.
    
    Provides a persistent library for storing, retrieving, and organizing
    mathematical formulas with metadata and tagging support.
    """
    
    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize formula library with SQLite database.
        
        Parameters
        ----------
        db_path:
            Path to SQLite database file, or ":memory:" for in-memory database.
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> library = FormulaLibrary("formulas.db")
        """
        _log_if_verbose(f"FormulaLibrary.__init__ called with db_path={db_path}")
        
        # Import SQLiteHelper
        from .utils import SQLiteHelper
        
        self.db = SQLiteHelper(db_path)
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        # Formulas table
        self.db.create_table(
            "formulas",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT UNIQUE NOT NULL",
                "formula": "TEXT NOT NULL",
                "description": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
        )
        
        # Tags table
        self.db.create_table(
            "tags",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "formula_id": "INTEGER NOT NULL",
                "tag": "TEXT NOT NULL",
                "FOREIGN KEY": "(formula_id) REFERENCES formulas(id)",
            },
        )
        
        # Metadata table
        self.db.create_table(
            "metadata",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "formula_id": "INTEGER NOT NULL",
                "key": "TEXT NOT NULL",
                "value": "TEXT",
                "FOREIGN KEY": "(formula_id) REFERENCES formulas(id)",
            },
        )
    
    def save_formula(
        self,
        name: str,
        formula: Union[str, SymbolicExpression],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ) -> int:
        """Save a formula to the library.
        
        Parameters
        ----------
        name:
            Unique name for the formula.
        formula:
            Formula expression as string or SymbolicExpression.
        description:
            Optional description of the formula.
        tags:
            Optional list of tags for categorization.
        metadata:
            Optional metadata dictionary (all values converted to strings).
        overwrite:
            If True, overwrite existing formula with same name.
        
        Returns
        -------
        int
            Formula ID in the database.
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> formula_id = library.save_formula(
        ...     "quadratic",
        ...     "a*x**2 + b*x + c",
        ...     description="General quadratic equation",
        ...     tags=["algebra", "polynomial"],
        ...     metadata={"degree": "2", "domain": "real"}
        ... )
        """
        _log_if_verbose(f"FormulaLibrary.save_formula called with name={name}")
        
        # Convert formula to string
        if isinstance(formula, SymbolicExpression):
            formula_str = formula.to_string()
        elif isinstance(formula, str):
            # Validate the formula
            valid, error = validate_formula(formula)
            if not valid:
                raise ValueError(f"Invalid formula: {error}")
            formula_str = formula
        else:
            raise TypeError("formula must be a string or SymbolicExpression")
        
        # Check if formula exists
        existing = self.db.fetch_one("SELECT id FROM formulas WHERE name = ?", (name,))
        
        if existing:
            if not overwrite:
                raise ValueError(f"Formula '{name}' already exists. Use overwrite=True to replace.")
            
            # Delete existing formula and related data
            formula_id = existing["id"]
            self.db.execute("DELETE FROM tags WHERE formula_id = ?", (formula_id,))
            self.db.execute("DELETE FROM metadata WHERE formula_id = ?", (formula_id,))
            self.db.execute("DELETE FROM formulas WHERE id = ?", (formula_id,))
        
        # Insert formula
        self.db.execute(
            "INSERT INTO formulas (name, formula, description) VALUES (?, ?, ?)",
            (name, formula_str, description),
        )
        
        formula_id = self.db.fetch_one("SELECT last_insert_rowid() as id")["id"]
        
        # Insert tags
        if tags:
            tag_rows = [(formula_id, tag) for tag in tags]
            self.db.bulk_insert("tags", ["formula_id", "tag"], tag_rows)
        
        # Insert metadata
        if metadata:
            meta_rows = [(formula_id, key, str(value)) for key, value in metadata.items()]
            self.db.bulk_insert("metadata", ["formula_id", "key", "value"], meta_rows)
        
        return formula_id
    
    def load_formula(self, name: str) -> Dict[str, Any]:
        """Load a formula by name.
        
        Parameters
        ----------
        name:
            Name of the formula to load.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing formula data including:
            - name: Formula name
            - formula: Formula string
            - expression: SymbolicExpression object
            - description: Description text
            - tags: List of tags
            - metadata: Metadata dictionary
            - created_at: Creation timestamp
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> library.save_formula("test", "x**2 + 1")
        >>> data = library.load_formula("test")
        >>> expr = data["expression"]
        """
        _log_if_verbose(f"FormulaLibrary.load_formula called with name={name}")
        
        # Fetch formula
        formula_row = self.db.fetch_one(
            "SELECT * FROM formulas WHERE name = ?",
            (name,)
        )
        
        if not formula_row:
            raise ValueError(f"Formula '{name}' not found")
        
        formula_id = formula_row["id"]
        
        # Fetch tags
        tag_rows = self.db.fetch_all(
            "SELECT tag FROM tags WHERE formula_id = ?",
            (formula_id,)
        )
        tags = [row["tag"] for row in tag_rows]
        
        # Fetch metadata
        meta_rows = self.db.fetch_all(
            "SELECT key, value FROM metadata WHERE formula_id = ?",
            (formula_id,)
        )
        metadata = {row["key"]: row["value"] for row in meta_rows}
        
        return {
            "name": formula_row["name"],
            "formula": formula_row["formula"],
            "expression": SymbolicExpression(formula_row["formula"]),
            "description": formula_row["description"],
            "tags": tags,
            "metadata": metadata,
            "created_at": formula_row["created_at"],
        }
    
    def search(
        self,
        name_pattern: Optional[str] = None,
        tag: Optional[str] = None,
        metadata_key: Optional[str] = None,
        metadata_value: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search formulas by various criteria.
        
        Parameters
        ----------
        name_pattern:
            SQL LIKE pattern to match formula names (e.g., "quad%").
        tag:
            Tag to filter by.
        metadata_key:
            Metadata key to filter by.
        metadata_value:
            Metadata value to filter by (requires metadata_key).
        
        Returns
        -------
        List[Dict[str, Any]]
            List of formula data dictionaries matching the criteria.
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> library.save_formula("quadratic", "a*x**2 + b*x + c", tags=["algebra"])
        >>> results = library.search(tag="algebra")
        """
        _log_if_verbose("FormulaLibrary.search called")
        
        # Build query
        query = "SELECT DISTINCT f.name FROM formulas f"
        conditions = []
        params = []
        
        if tag:
            query += " JOIN tags t ON f.id = t.formula_id"
            conditions.append("t.tag = ?")
            params.append(tag)
        
        if metadata_key:
            query += " JOIN metadata m ON f.id = m.formula_id"
            conditions.append("m.key = ?")
            params.append(metadata_key)
            
            if metadata_value:
                conditions.append("m.value = ?")
                params.append(metadata_value)
        
        if name_pattern:
            conditions.append("f.name LIKE ?")
            params.append(name_pattern)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute search
        rows = self.db.fetch_all(query, tuple(params))
        
        # Load full formula data for each result
        results = []
        for row in rows:
            results.append(self.load_formula(row["name"]))
        
        return results
    
    def list_formulas(self) -> List[str]:
        """List all formula names in the library.
        
        Returns
        -------
        List[str]
            List of all formula names.
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> library.save_formula("f1", "x**2")
        >>> library.save_formula("f2", "x**3")
        >>> names = library.list_formulas()  # ["f1", "f2"]
        """
        _log_if_verbose("FormulaLibrary.list_formulas called")
        
        rows = self.db.fetch_all("SELECT name FROM formulas ORDER BY name")
        return [row["name"] for row in rows]
    
    def delete_formula(self, name: str) -> bool:
        """Delete a formula from the library.
        
        Parameters
        ----------
        name:
            Name of the formula to delete.
        
        Returns
        -------
        bool
            True if formula was deleted, False if not found.
        
        Examples
        --------
        >>> library = FormulaLibrary(":memory:")
        >>> library.save_formula("test", "x**2")
        >>> library.delete_formula("test")  # True
        """
        _log_if_verbose(f"FormulaLibrary.delete_formula called with name={name}")
        
        # Check if exists
        formula_row = self.db.fetch_one(
            "SELECT id FROM formulas WHERE name = ?",
            (name,)
        )
        
        if not formula_row:
            return False
        
        formula_id = formula_row["id"]
        
        # Delete related data
        self.db.execute("DELETE FROM tags WHERE formula_id = ?", (formula_id,))
        self.db.execute("DELETE FROM metadata WHERE formula_id = ?", (formula_id,))
        self.db.execute("DELETE FROM formulas WHERE id = ?", (formula_id,))
        
        return True
    
    def close(self) -> None:
        """Close the database connection."""
        _log_if_verbose("FormulaLibrary.close called")
        self.db.close()
    
    def __enter__(self) -> "FormulaLibrary":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:  # pragma: no cover
        """Developer representation."""
        count = len(self.list_formulas())
        return f"FormulaLibrary(formulas={count}, db='{self.db.db_path}')"


# ============================================================================
# Matrix Integration
# ============================================================================

def matrix_to_symbolic(matrix: "Matrix") -> Any:
    """Convert oet_core Matrix to SymPy symbolic matrix.
    
    Parameters
    ----------
    matrix:
        Matrix object from oet_core.utils.
    
    Returns
    -------
    sympy.Matrix
        SymPy symbolic matrix representation.
    
    Examples
    --------
    >>> from oet_core import Matrix
    >>> m = Matrix(2, 2)
    >>> m.set(0, 0, 'x')
    >>> m.set(0, 1, 'y')
    >>> m.set(1, 0, 2)
    >>> m.set(1, 1, 3)
    >>> sym_matrix = matrix_to_symbolic(m)
    """
    from .utils import Matrix
    
    if not isinstance(matrix, Matrix):
        raise TypeError("matrix must be an oet_core Matrix object")
    
    _log_if_verbose(f"matrix_to_symbolic called with matrix shape ({matrix.rows}, {matrix.cols})")
    
    sp = _import_sympy()
    
    # Convert each element to symbolic expression
    sym_data = []
    for row in range(matrix.rows):
        sym_row = []
        for col in range(matrix.cols):
            value = matrix.get(row, col)
            if isinstance(value, str):
                # Parse as symbolic expression
                sym_row.append(parse_expression(value))
            else:
                # Numeric value
                sym_row.append(value)
        sym_data.append(sym_row)
    
    return sp.Matrix(sym_data)


def symbolic_to_matrix(sym_matrix: Any) -> "Matrix":
    """Convert SymPy symbolic matrix to oet_core Matrix.
    
    Parameters
    ----------
    sym_matrix:
        SymPy Matrix object.
    
    Returns
    -------
    Matrix
        oet_core Matrix object with string representations of symbolic elements.
    
    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols('x y')
    >>> sym_m = sp.Matrix([[x, y], [1, 2]])
    >>> matrix = symbolic_to_matrix(sym_m)
    """
    _log_if_verbose("symbolic_to_matrix called")
    
    from .utils import Matrix
    
    sp = _import_sympy()
    
    if not isinstance(sym_matrix, sp.Matrix):
        raise TypeError("sym_matrix must be a SymPy Matrix object")
    
    rows, cols = sym_matrix.shape
    result = Matrix(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            elem = sym_matrix[i, j]
            # Store as string if symbolic, otherwise as numeric value
            if elem.is_number:
                try:
                    result.set(i, j, float(elem))
                except Exception:
                    result.set(i, j, str(elem))
            else:
                result.set(i, j, str(elem))
    
    return result


def symbolic_determinant(matrix: "Matrix") -> SymbolicExpression:
    """Compute symbolic determinant of a matrix.
    
    Parameters
    ----------
    matrix:
        Square Matrix object from oet_core.utils.
    
    Returns
    -------
    SymbolicExpression
        Determinant as symbolic expression.
    
    Examples
    --------
    >>> from oet_core import Matrix
    >>> m = Matrix(2, 2)
    >>> m.set(0, 0, 'a')
    >>> m.set(0, 1, 'b')
    >>> m.set(1, 0, 'c')
    >>> m.set(1, 1, 'd')
    >>> det = symbolic_determinant(m)  # a*d - b*c
    """
    _log_if_verbose(f"symbolic_determinant called with matrix shape ({matrix.rows}, {matrix.cols})")
    
    from .utils import Matrix
    
    if not isinstance(matrix, Matrix):
        raise TypeError("matrix must be an oet_core Matrix object")
    
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square to compute determinant")
    
    sym_matrix = matrix_to_symbolic(matrix)
    det = sym_matrix.det()
    
    return SymbolicExpression(det)


def symbolic_inverse(matrix: "Matrix") -> "Matrix":
    """Compute symbolic inverse of a matrix.
    
    Parameters
    ----------
    matrix:
        Square Matrix object from oet_core.utils.
    
    Returns
    -------
    Matrix
        Inverse matrix with symbolic expressions.
    
    Examples
    --------
    >>> from oet_core import Matrix
    >>> m = Matrix(2, 2)
    >>> m.set(0, 0, 'a')
    >>> m.set(0, 1, 'b')
    >>> m.set(1, 0, 'c')
    >>> m.set(1, 1, 'd')
    >>> inv = symbolic_inverse(m)
    """
    _log_if_verbose(f"symbolic_inverse called with matrix shape ({matrix.rows}, {matrix.cols})")
    
    from .utils import Matrix
    
    if not isinstance(matrix, Matrix):
        raise TypeError("matrix must be an oet_core Matrix object")
    
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square to compute inverse")
    
    sym_matrix = matrix_to_symbolic(matrix)
    
    try:
        inv_matrix = sym_matrix.inv()
    except Exception as exc:
        raise ValueError(f"Matrix is not invertible: {exc}") from exc
    
    return symbolic_to_matrix(inv_matrix)
