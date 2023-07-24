from typing import Dict, TypeVar

import sympy


ExprType = TypeVar("ExprType", bound=sympy.Expr)


def hide_floats(expression: ExprType) -> ExprType:
    _memodict: Dict[ExprType, ExprType] = {}
    return _hide_floats(expression, _memodict)


def _hide_floats(expression: ExprType, _memodict: Dict[ExprType, ExprType]) -> ExprType:
    try:
        return _memodict[expression]
    except KeyError:
        pass

    if issubclass(expression.func, (sympy.Min, sympy.Max)):
        evaluate = False
    else:
        evaluate = True

    new_expression: ExprType
    if issubclass(expression.func, sympy.Float):
        new_expression = sympy.UnevaluatedExpr(expression)  # type: ignore
    elif issubclass(expression.func, sympy.Integer):
        new_expression = expression
    elif issubclass(expression.func, sympy.Symbol):
        new_expression = expression
    else:
        new_expression = expression.func(
            *[_hide_floats(arg, _memodict) for arg in expression.args],
            evaluate=evaluate
        )  # type: ignore
    _memodict[expression] = new_expression
    return new_expression
