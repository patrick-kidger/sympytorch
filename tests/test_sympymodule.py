import sympy, torch, sympytorch
import numpy as np

def test_example():
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


def test_grad():
    x = sympy.symbols('x_name')
    y = 1.0 * x
    mod = sympytorch.SymPyModule(expressions=[y])
    out = mod(x_name=torch.ones(()))
    out.backward()
    with torch.no_grad():
        for param in mod.parameters():
            param += param.grad
    expr, = mod.sympy()
    assert expr == 2.0 * x


def test_reduce():
    x, y = sympy.symbols('x y')
    z = 2 * x * y
    mod = sympytorch.SymPyModule(expressions=[z])
    mod(x=torch.rand(2), y=torch.rand(2))

    z = 2 + x + y
    mod = sympytorch.SymPyModule(expressions=[z])
    mod(x=torch.rand(2), y=torch.rand(2))


def test_special_subclasses():
    x, y = sympy.symbols('x y')
    z = x - 1
    w = y * 0

    mod = sympytorch.SymPyModule(expressions=[z, w])
    assert mod.sympy() == [x - 1, sympy.Integer(0)]


def test_constants():
    x = sympy.symbols('x')
    y = 2.0 * x + sympy.UnevaluatedExpr(1.0)
    mod = sympytorch.SymPyModule(expressions=[y])
    assert mod.sympy() == [y]
    assert set(p.item() for p in mod.parameters()) == {2.0}
    assert set(b.item() for b in mod.buffers()) == {1.0}

def test_custom_function():
    x, y = sympy.symbols('x y')
    f = sympy.Function('f')
    z = x + f(y)
    extra_funcs = {f: lambda y_: y_ ** 2}
    mod = sympytorch.SymPyModule(expressions=[z], extra_funcs=extra_funcs)
    assert mod.sympy() == [z]
    assert mod(x=1, y=2) == 1 + 2 ** 2

def test_rationals():
    xvals = np.random.randn(100)
    x = sympy.symbols('x')
    y =  x*sympy.Rational(2,7)
    mod = sympytorch.SymPyModule(expressions=[y])
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    assert len([p.item() for p in mod.parameters()])==0
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:,0]
    error = y_tilde.detach().numpy()-xvals*2/7
    assert (error**2).mean()<1e-10, "error:{}".format((error**2).mean())
    
def test_half1():
    xvals = np.random.randn(100)
    x = sympy.symbols('x')
    y =  abs(x)**sympy.S.Half
    mod = sympytorch.SymPyModule(expressions=[y])
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:,0]
    error = y_tilde.detach().numpy()-np.abs(xvals)**0.5
    assert (error**2).mean()<1e-10, "error:{}".format((error**2).mean())
    
def test_half2():
    xvals = np.random.randn(100)
    y = sympy.parse_expr("sqrt(Abs(x))")
    mod = sympytorch.SymPyModule(expressions=[y])
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    assert len([p.item() for p in mod.parameters()])==0
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:,0]
    error = y_tilde.detach().numpy()-np.abs(xvals)**0.5
    assert (error**2).mean()<1e-10, "error:{}".format((error**2).mean())