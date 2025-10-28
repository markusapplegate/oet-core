"""Tests for symbolics module."""

import unittest
from io import StringIO

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# Skip all tests if SymPy not available
skipIfNoSympy = unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not available")


@skipIfNoSympy
class TestParseExpression(unittest.TestCase):
    """Test parse_expression function."""

    def test_parse_simple_expression(self):
        from oet_core import parse_expression
        expr = parse_expression("x**2 + 2*x + 1")
        self.assertIsNotNone(expr)
        self.assertEqual(str(expr), "x**2 + 2*x + 1")

    def test_parse_trig_functions(self):
        from oet_core import parse_expression
        expr = parse_expression("sin(x) + cos(y)")
        self.assertIsNotNone(expr)

    def test_parse_invalid_expression(self):
        from oet_core import parse_expression
        with self.assertRaises(ValueError):
            parse_expression("x**2 + +")

    def test_parse_non_string(self):
        from oet_core import parse_expression
        with self.assertRaises(TypeError):
            parse_expression(123)


@skipIfNoSympy
class TestValidateFormula(unittest.TestCase):
    """Test validate_formula function."""

    def test_validate_valid_formula(self):
        from oet_core import validate_formula
        valid, error = validate_formula("x**2 + 2*x + 1")
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validate_invalid_formula(self):
        from oet_core import validate_formula
        # SymPy is actually quite lenient - "x**2 + + y" gets parsed as "x**2 + y"
        # Use a truly invalid expression
        valid, error = validate_formula("x**2 + @#$")
        self.assertFalse(valid)
        self.assertIsNotNone(error)

    def test_validate_empty_formula(self):
        from oet_core import validate_formula
        valid, error = validate_formula("")
        self.assertFalse(valid)
        self.assertEqual(error, "formula cannot be empty")

    def test_validate_non_string(self):
        from oet_core import validate_formula
        valid, error = validate_formula(123)
        self.assertFalse(valid)
        self.assertEqual(error, "formula must be a string")


@skipIfNoSympy
class TestSymbolicExpression(unittest.TestCase):
    """Test SymbolicExpression class."""

    def test_init_from_string(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x + 1")
        self.assertIsNotNone(expr.expr)

    def test_init_from_sympy_expr(self):
        from oet_core import SymbolicExpression
        import sympy as sp
        sympy_expr = sp.sympify("x**2")
        expr = SymbolicExpression(sympy_expr)
        self.assertEqual(str(expr), "x**2")

    def test_init_invalid_type(self):
        from oet_core import SymbolicExpression
        with self.assertRaises(TypeError):
            SymbolicExpression(123)

    def test_simplify(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x*y + y**2")
        # This should remain as is or simplify depending on SymPy version
        simplified = expr.simplify()
        self.assertIsInstance(simplified, SymbolicExpression)

    def test_expand(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("(x + 1)**2")
        expanded = expr.expand()
        self.assertEqual(str(expanded), "x**2 + 2*x + 1")

    def test_factor(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 - 1")
        factored = expr.factor()
        self.assertEqual(str(factored), "(x - 1)*(x + 1)")

    def test_differentiate_single_variable(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**3")
        derivative = expr.differentiate("x")
        self.assertEqual(str(derivative), "3*x**2")

    def test_differentiate_multiple_variables(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 * y**2")
        derivative = expr.differentiate("x", "y")
        self.assertEqual(str(derivative), "4*x*y")

    def test_differentiate_no_variables(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        with self.assertRaises(ValueError):
            expr.differentiate()

    def test_integrate_indefinite(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        integral = expr.integrate("x")
        self.assertEqual(str(integral), "x**3/3")

    def test_integrate_definite(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x")
        integral = expr.integrate("x", definite=True, limits={"x": (0, 1)})
        # Result should be 1/2
        self.assertEqual(float(integral.expr), 0.5)

    def test_integrate_no_variables(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        with self.assertRaises(ValueError):
            expr.integrate()

    def test_substitute_numeric(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + y")
        result = expr.substitute({"x": 2, "y": 3})
        self.assertEqual(str(result), "7")

    def test_substitute_symbolic(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x + y")
        result = expr.substitute({"x": "2*z"})
        # SymPy may reorder terms, so check both are present
        result_str = str(result)
        self.assertIn("2*z", result_str)
        self.assertIn("y", result_str)

    def test_substitute_non_dict(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        with self.assertRaises(TypeError):
            expr.substitute("invalid")

    def test_evaluate(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x + 1")
        result = expr.evaluate({"x": 5})
        self.assertEqual(result, 36.0)

    def test_evaluate_multiple_vars(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x*y + z")
        result = expr.evaluate({"x": 2, "y": 3, "z": 4})
        self.assertEqual(result, 10.0)

    def test_evaluate_non_dict(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        with self.assertRaises(TypeError):
            expr.evaluate("invalid")

    def test_limit_finite(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("(x**2 - 1)/(x - 1)")
        lim = expr.limit("x", 1)
        self.assertEqual(str(lim), "2")

    def test_limit_infinity(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("1/x")
        lim = expr.limit("x", "oo")
        self.assertEqual(str(lim), "0")

    def test_taylor_series(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("exp(x)")
        series = expr.taylor_series("x", center=0, order=4)
        # Should be approximately 1 + x + x**2/2 + x**3/6
        self.assertIn("x**3", str(series))

    def test_get_variables(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + y*z")
        variables = expr.get_variables()
        self.assertEqual(variables, ["x", "y", "z"])

    def test_get_variables_single(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x + 1")
        variables = expr.get_variables()
        self.assertEqual(variables, ["x"])

    def test_to_function(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x + 1")
        f = expr.to_function()
        result = f(x=5)
        self.assertEqual(result, 36)

    def test_to_function_multiple_vars(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x*y + z")
        f = expr.to_function(variables=["x", "y", "z"])
        result = f(x=2, y=3, z=4)
        self.assertEqual(result, 10)

    def test_to_function_caching(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        f1 = expr.to_function()
        f2 = expr.to_function()
        # Should return cached function
        self.assertIs(f1, f2)

    def test_to_latex(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 2*x + 1")
        latex = expr.to_latex()
        self.assertIn("x", latex)
        self.assertIsInstance(latex, str)

    def test_to_string(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2 + 1")
        string = expr.to_string()
        self.assertEqual(string, "x**2 + 1")

    def test_str(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        self.assertEqual(str(expr), "x**2")

    def test_repr(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("x**2")
        self.assertIn("SymbolicExpression", repr(expr))

    def test_equality_same(self):
        from oet_core import SymbolicExpression
        expr1 = SymbolicExpression("x**2 + 2*x + 1")
        expr2 = SymbolicExpression("(x + 1)**2")
        # These should be equal after simplification
        self.assertEqual(expr1, expr2)

    def test_equality_different(self):
        from oet_core import SymbolicExpression
        expr1 = SymbolicExpression("x**2")
        expr2 = SymbolicExpression("x**3")
        self.assertNotEqual(expr1, expr2)


@skipIfNoSympy
class TestVerboseLogging(unittest.TestCase):
    """Test verbose logging functionality."""

    def test_set_verbose_logging(self):
        from oet_core import set_symbolics_verbose_logging, SymbolicExpression
        
        # Enable verbose logging
        set_symbolics_verbose_logging(True)
        
        # Create expression with logging
        buffer = StringIO()
        from oet_core import get_logger
        logger = get_logger("oet_core", stream=buffer)
        
        expr = SymbolicExpression("x**2")
        
        # Disable verbose logging
        set_symbolics_verbose_logging(False)


@skipIfNoSympy
class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_complex_expression(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("I*x")  # I is imaginary unit
        result = expr.evaluate({"x": 1})
        self.assertEqual(result, 1j)

    def test_expression_with_constants(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("pi*x")
        import math
        result = expr.evaluate({"x": 1})
        self.assertAlmostEqual(result, math.pi, places=5)

    def test_trig_functions(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("sin(x)")
        derivative = expr.differentiate("x")
        self.assertEqual(str(derivative), "cos(x)")

    def test_exponential_functions(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("exp(x)")
        derivative = expr.differentiate("x")
        self.assertEqual(str(derivative), "exp(x)")

    def test_log_functions(self):
        from oet_core import SymbolicExpression
        expr = SymbolicExpression("log(x)")
        derivative = expr.differentiate("x")
        self.assertEqual(str(derivative), "1/x")


class TestImportError(unittest.TestCase):
    """Test behavior when SymPy is not available."""

    def test_import_error_message(self):
        # This test assumes SymPy might not be available
        # We can't really test this if SymPy is installed
        # But we can at least verify the error message is correct
        from oet_core.symbolics import _import_sympy
        try:
            _import_sympy()
        except ImportError as e:
            self.assertIn("SymPy is required", str(e))
            self.assertIn("pip install oet-core[symbolic]", str(e))


@skipIfNoSympy
class TestSymbolicSolver(unittest.TestCase):
    """Test SymbolicSolver class."""

    def test_init(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        self.assertIsNotNone(solver.sp)

    def test_solve_simple_equation(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        solutions = solver.solve("x**2 - 4 = 0", "x")
        self.assertEqual(len(solutions), 2)
        self.assertIn(-2, solutions)
        self.assertIn(2, solutions)

    def test_solve_without_equals(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        solutions = solver.solve("x**2 - 4", "x")
        self.assertEqual(len(solutions), 2)

    def test_solve_auto_variable(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        solutions = solver.solve("x**2 - 1 = 0")
        self.assertEqual(len(solutions), 2)

    def test_solve_multiple_variables_error(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        with self.assertRaises(ValueError):
            solver.solve("x**2 + y = 0")

    def test_solve_no_variables_error(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        with self.assertRaises(ValueError):
            solver.solve("4 = 0")

    def test_solve_with_expression(self):
        from oet_core.symbolics import SymbolicSolver, SymbolicExpression
        solver = SymbolicSolver()
        expr = SymbolicExpression("x**2 - 9")
        solutions = solver.solve(expr, "x")
        self.assertEqual(len(solutions), 2)

    def test_solve_system_two_equations(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        solutions = solver.solve_system(["x + y = 5", "x - y = 1"], ["x", "y"])
        # Solutions should be x=3, y=2
        self.assertEqual(solutions[solver.sp.Symbol("x")], 3)
        self.assertEqual(solutions[solver.sp.Symbol("y")], 2)

    def test_solve_system_auto_variables(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        solutions = solver.solve_system(["x + y = 10", "2*x + y = 15"])
        self.assertIsNotNone(solutions)

    def test_solve_system_with_expressions(self):
        from oet_core.symbolics import SymbolicSolver, SymbolicExpression
        solver = SymbolicSolver()
        expr1 = SymbolicExpression("x + y - 5")
        expr2 = SymbolicExpression("x - y - 1")
        solutions = solver.solve_system([expr1, expr2])
        self.assertIsNotNone(solutions)

    def test_solve_ode_simple(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        # Solve dy/dx = y
        solution = solver.solve_ode("Derivative(y(x), x) - y(x)", "y(x)")
        self.assertIsNotNone(solution)

    def test_repr(self):
        from oet_core.symbolics import SymbolicSolver
        solver = SymbolicSolver()
        self.assertIn("SymbolicSolver", repr(solver))


@skipIfNoSympy
class TestFormulaLibrary(unittest.TestCase):
    """Test FormulaLibrary class."""

    def test_init_memory(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        self.assertIsNotNone(library.db)
        library.close()

    def test_save_formula_string(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        formula_id = library.save_formula("test", "x**2 + 1")
        self.assertIsInstance(formula_id, int)
        self.assertGreater(formula_id, 0)
        library.close()

    def test_save_formula_expression(self):
        from oet_core.symbolics import FormulaLibrary, SymbolicExpression
        library = FormulaLibrary(":memory:")
        expr = SymbolicExpression("x**2 + 1")
        formula_id = library.save_formula("test", expr)
        self.assertGreater(formula_id, 0)
        library.close()

    def test_save_formula_with_metadata(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        formula_id = library.save_formula(
            "quadratic",
            "a*x**2 + b*x + c",
            description="General quadratic",
            tags=["algebra", "polynomial"],
            metadata={"degree": "2", "domain": "real"}
        )
        self.assertGreater(formula_id, 0)
        library.close()

    def test_save_duplicate_error(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2")
        with self.assertRaises(ValueError):
            library.save_formula("test", "x**3")
        library.close()

    def test_save_duplicate_overwrite(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2")
        formula_id = library.save_formula("test", "x**3", overwrite=True)
        self.assertGreater(formula_id, 0)
        
        # Verify it was overwritten
        data = library.load_formula("test")
        self.assertEqual(data["formula"], "x**3")
        library.close()

    def test_save_invalid_formula(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        with self.assertRaises(ValueError):
            library.save_formula("test", "x**2 + @#$")
        library.close()

    def test_load_formula(self):
        from oet_core.symbolics import FormulaLibrary, SymbolicExpression
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2 + 1", description="Test formula")
        
        data = library.load_formula("test")
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["formula"], "x**2 + 1")
        self.assertEqual(data["description"], "Test formula")
        self.assertIsInstance(data["expression"], SymbolicExpression)
        library.close()

    def test_load_formula_not_found(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        with self.assertRaises(ValueError):
            library.load_formula("nonexistent")
        library.close()

    def test_load_formula_with_tags(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2", tags=["algebra", "quadratic"])
        
        data = library.load_formula("test")
        self.assertEqual(len(data["tags"]), 2)
        self.assertIn("algebra", data["tags"])
        self.assertIn("quadratic", data["tags"])
        library.close()

    def test_load_formula_with_metadata(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2", metadata={"key1": "value1", "key2": "value2"})
        
        data = library.load_formula("test")
        self.assertEqual(len(data["metadata"]), 2)
        self.assertEqual(data["metadata"]["key1"], "value1")
        self.assertEqual(data["metadata"]["key2"], "value2")
        library.close()

    def test_search_by_name(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("quadratic", "x**2")
        library.save_formula("cubic", "x**3")
        
        results = library.search(name_pattern="quad%")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "quadratic")
        library.close()

    def test_search_by_tag(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("f1", "x**2", tags=["algebra"])
        library.save_formula("f2", "sin(x)", tags=["trigonometry"])
        library.save_formula("f3", "x**3", tags=["algebra"])
        
        results = library.search(tag="algebra")
        self.assertEqual(len(results), 2)
        names = [r["name"] for r in results]
        self.assertIn("f1", names)
        self.assertIn("f3", names)
        library.close()

    def test_search_by_metadata(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("f1", "x**2", metadata={"type": "polynomial"})
        library.save_formula("f2", "sin(x)", metadata={"type": "trigonometric"})
        
        results = library.search(metadata_key="type", metadata_value="polynomial")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "f1")
        library.close()

    def test_search_no_results(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2")
        
        results = library.search(tag="nonexistent")
        self.assertEqual(len(results), 0)
        library.close()

    def test_list_formulas(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("f1", "x**2")
        library.save_formula("f2", "x**3")
        library.save_formula("f3", "x**4")
        
        names = library.list_formulas()
        self.assertEqual(len(names), 3)
        self.assertIn("f1", names)
        self.assertIn("f2", names)
        self.assertIn("f3", names)
        library.close()

    def test_list_formulas_empty(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        names = library.list_formulas()
        self.assertEqual(len(names), 0)
        library.close()

    def test_delete_formula(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2")
        
        result = library.delete_formula("test")
        self.assertTrue(result)
        
        names = library.list_formulas()
        self.assertEqual(len(names), 0)
        library.close()

    def test_delete_formula_not_found(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        result = library.delete_formula("nonexistent")
        self.assertFalse(result)
        library.close()

    def test_context_manager(self):
        from oet_core.symbolics import FormulaLibrary
        with FormulaLibrary(":memory:") as library:
            library.save_formula("test", "x**2")
            names = library.list_formulas()
            self.assertEqual(len(names), 1)

    def test_repr(self):
        from oet_core.symbolics import FormulaLibrary
        library = FormulaLibrary(":memory:")
        library.save_formula("test", "x**2")
        repr_str = repr(library)
        self.assertIn("FormulaLibrary", repr_str)
        self.assertIn("formulas=1", repr_str)
        library.close()


@skipIfNoSympy
class TestMatrixIntegration(unittest.TestCase):
    """Test Matrix integration with symbolic operations."""

    def test_matrix_to_symbolic(self):
        from oet_core import Matrix, matrix_to_symbolic
        m = Matrix(2, 2)
        m.set(0, 0, 'x')
        m.set(0, 1, 'y')
        m.set(1, 0, 2)
        m.set(1, 1, 3)
        
        sym_matrix = matrix_to_symbolic(m)
        self.assertEqual(sym_matrix.shape, (2, 2))
        # Check that symbolic elements are present
        self.assertTrue(str(sym_matrix[0, 0]) == 'x')
        self.assertTrue(str(sym_matrix[0, 1]) == 'y')

    def test_symbolic_to_matrix(self):
        from oet_core import Matrix, symbolic_to_matrix
        import sympy as sp
        x, y = sp.symbols('x y')
        sym_m = sp.Matrix([[x, y], [1, 2]])
        
        matrix = symbolic_to_matrix(sym_m)
        self.assertEqual(matrix.rows, 2)
        self.assertEqual(matrix.cols, 2)
        self.assertEqual(matrix.get(0, 0), 'x')
        self.assertEqual(matrix.get(0, 1), 'y')
        self.assertEqual(matrix.get(1, 0), 1.0)
        self.assertEqual(matrix.get(1, 1), 2.0)

    def test_symbolic_determinant(self):
        from oet_core import Matrix, symbolic_determinant
        m = Matrix(2, 2)
        m.set(0, 0, 'a')
        m.set(0, 1, 'b')
        m.set(1, 0, 'c')
        m.set(1, 1, 'd')
        
        det = symbolic_determinant(m)
        # Determinant should be a*d - b*c
        det_str = str(det.simplify())
        # Check that both terms appear
        self.assertTrue('a' in det_str and 'd' in det_str)
        self.assertTrue('b' in det_str and 'c' in det_str)

    def test_symbolic_determinant_numeric(self):
        from oet_core import Matrix, symbolic_determinant
        m = Matrix(2, 2)
        m.set(0, 0, 4)
        m.set(0, 1, 3)
        m.set(1, 0, 2)
        m.set(1, 1, 1)
        
        det = symbolic_determinant(m)
        result = det.evaluate({})
        self.assertEqual(result, -2.0)

    def test_symbolic_inverse(self):
        from oet_core import Matrix, symbolic_inverse
        m = Matrix(2, 2)
        m.set(0, 0, 'a')
        m.set(0, 1, 'b')
        m.set(1, 0, 'c')
        m.set(1, 1, 'd')
        
        inv = symbolic_inverse(m)
        self.assertEqual(inv.rows, 2)
        self.assertEqual(inv.cols, 2)
        # Inverse exists and contains symbolic expressions
        self.assertTrue(isinstance(inv.get(0, 0), str))

    def test_symbolic_inverse_numeric(self):
        from oet_core import Matrix, symbolic_inverse
        m = Matrix(2, 2)
        m.set(0, 0, 4)
        m.set(0, 1, 3)
        m.set(1, 0, 2)
        m.set(1, 1, 1)
        
        inv = symbolic_inverse(m)
        # Check one element: for [[4,3],[2,1]], inverse is [[-1/2, 3/2],[1, -2]]
        elem = float(inv.get(0, 0))
        self.assertAlmostEqual(elem, -0.5, places=5)

    def test_matrix_to_symbolic_method(self):
        from oet_core import Matrix
        m = Matrix(2, 2)
        m.set(0, 0, 'x')
        m.set(0, 1, 1)
        m.set(1, 0, 2)
        m.set(1, 1, 'y')
        
        sym_matrix = m.to_symbolic()
        self.assertEqual(sym_matrix.shape, (2, 2))

    def test_matrix_symbolic_determinant_method(self):
        from oet_core import Matrix
        m = Matrix(2, 2)
        m.set(0, 0, 'a')
        m.set(0, 1, 'b')
        m.set(1, 0, 'c')
        m.set(1, 1, 'd')
        
        det = m.symbolic_determinant()
        det_str = str(det.simplify())
        self.assertTrue('a' in det_str and 'd' in det_str)

    def test_matrix_symbolic_inverse_method(self):
        from oet_core import Matrix
        m = Matrix(2, 2)
        m.set(0, 0, 4)
        m.set(0, 1, 3)
        m.set(1, 0, 2)
        m.set(1, 1, 1)
        
        inv = m.symbolic_inverse()
        self.assertEqual(inv.rows, 2)
        self.assertEqual(inv.cols, 2)

    def test_symbolic_determinant_non_square(self):
        from oet_core import Matrix, symbolic_determinant
        m = Matrix(2, 3)  # Non-square
        with self.assertRaises(ValueError):
            symbolic_determinant(m)

    def test_symbolic_inverse_non_square(self):
        from oet_core import Matrix, symbolic_inverse
        m = Matrix(2, 3)  # Non-square
        with self.assertRaises(ValueError):
            symbolic_inverse(m)

    def test_matrix_to_symbolic_invalid_type(self):
        from oet_core import matrix_to_symbolic
        with self.assertRaises(TypeError):
            matrix_to_symbolic("not a matrix")

    def test_symbolic_to_matrix_invalid_type(self):
        from oet_core import symbolic_to_matrix
        with self.assertRaises(TypeError):
            symbolic_to_matrix("not a sympy matrix")


if __name__ == "__main__":
    unittest.main()
