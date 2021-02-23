# sympytorch

A micro-library as a convenience for turning SymPy expressions into PyTorch Modules.

All SymPy floats become trainable parameters; all SymPy symbols are inputs to the Module.

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
out = mod(x_name=x_)
assert out[0] == x_.cos()
assert out[1] == x_.sin()
assert out.requires_grad  # from the two Parameters initialised as 1.0
```
