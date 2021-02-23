# sympytorch

A micro-library as a convenience for turning SymPy expressions into PyTorch Modules.

All SymPy floats become trainable parameters. All SymPy symbols are inputs to the Module.

## Installation

```bash
pip install git+https://github.com/patrick-kidger/sympytorch.git
```

## Example

```python
import sympy, torch, sympytorch

x = sympy.symbols('x_name')
cosx = 1.0 * sympy.cos(x)
sinx = 1.0 * sympy.sin(x)

mod = sympytorch.SympyModule(expressions=[cosx, sinx])
x_ = torch.rand(3)
out = mod(x_name=x_)  # returns a list of SymPy expressions
assert out[0] == x_.cos()
assert out[1] == x_.sin()
assert out.requires_grad  # from the two Parameters initialised as 1.0
```

## API

The API consists of a single object, `SymPyModule`.

It is initialised as `SymPyModule(*, expressions)`, where `expressions` is a list of SymPy expressions.  
It can be called, passing the values of the symbols as in the above example.  
It has a method `.sympy()`, which returns the corresponding list of SymPy expressions. (Which may not be the same as the expressions it was initialised with, if the values of its Parameters have been changed, e.g. have been learnt.)

## Extensions

Not every PyTorch or SymPy operation is supported -- just the ones that I found I've needed! There's a dictionary [here](./sympytorch/sympy_module.py#L5) that lists the supported operations. Feel free to submit PRs for any extra operations you need.
