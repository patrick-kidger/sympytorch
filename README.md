<h1 align="center">sympytorch</h1>

Turn SymPy expressions into PyTorch Modules.

SymPy floats (optionally) become trainable parameters. SymPy symbols are inputs to the Module.

Optimise your symbolic expressions via gradient descent!

## Installation

```bash
pip install sympytorch
```
Requires Python 3.7+ and PyTorch 1.6.0+ and SymPy 1.7.1+.

## Example

```python
import sympy, torch, sympytorch

x = sympy.symbols('x_name')
cosx = 1.0 * sympy.cos(x)
sinx = 2.0 * sympy.sin(x)
mod = sympytorch.SymPyModule(expressions=[cosx, sinx])

x_ = torch.rand(3)
out = mod(x_name=x_)  # out has shape (3, 2)

assert torch.equal(out[:, 0], x_.cos())
assert torch.equal(out[:, 1], 2 * x_.sin())
assert out.requires_grad  # from the two Parameters initialised as 1.0 and 2.0
assert {x.item() for x in mod.parameters()} == {1.0, 2.0}
```

## API

```python
sympytorch.SymPyModule(*, expressions, extra_funcs=None)
```
Where:
- `expressions` is a list of SymPy expressions.
- `extra_funcs` is a dictionary mapping from custom `sympy.Function`s to their PyTorch implementation. Defaults to no extra functions.

Instances of `SymPyModule` can be called, passing the values of the symbols as in the above example.

`SymPyModule` has a method `.sympy()`, which returns the corresponding list of SymPy expressions. (Which may not be the same as the expressions it was initialised with, if the values of its Parameters have been changed, i.e. have been learnt.)

Wrapping floats in `sympy.UnevaluatedExpr` will cause them not to be trained, by registering them as buffers rather than parameters.

```python
sympytorch.hide_floats(expression)
```
As a convenience, `hide_floats` will take an expression and return a new expression with every float wrapped in a `sympy.UnevaluatedExpr`, so that it is interpreted as a buffer rather than a parameter.

## Extensions

Not every PyTorch or SymPy operation is supported -- just the ones that I found I've needed! There's a dictionary [here](./sympytorch/sympy_module.py#L12) that lists the supported operations. Feel free to submit PRs for any extra operations you think should be in by default. You can also use the `extra_funcs` argument to specify extra functions, including custom functions.
