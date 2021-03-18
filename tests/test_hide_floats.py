import sympy
import sympytorch


def test_hide_floats():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    e = sympy.UnevaluatedExpr

    z = 3.0 * x + (4.0 * y + 1.0) ** 2 - 1
    z_ = sympytorch.hide_floats(z)
    assert z_ == x * e(3.0) - 1 + e(16.0) * (y + e(0.25)) ** 2

