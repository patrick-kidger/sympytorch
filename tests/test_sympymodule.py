import numpy as np
import sympy
import torch

import sympytorch


def test_example():
    x = sympy.symbols("x_name")
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
    x = sympy.symbols("x_name")
    y = 1.0 * x
    mod = sympytorch.SymPyModule(expressions=[y])
    out = mod(x_name=torch.ones(()))
    out.backward()
    with torch.no_grad():
        for param in mod.parameters():
            param += param.grad
    (expr,) = mod.sympy()
    assert expr == 2.0 * x


def test_reduce():
    x, y = sympy.symbols("x y")
    z = 2 * x * y
    mod = sympytorch.SymPyModule(expressions=[z])
    mod(x=torch.rand(2), y=torch.rand(2))

    z = 2 + x + y
    mod = sympytorch.SymPyModule(expressions=[z])
    mod(x=torch.rand(2), y=torch.rand(2))


def test_special_subclasses():
    x, y = sympy.symbols("x y")
    z = x - 1
    w = y * 0
    u = sympy.Integer(1)

    mod = sympytorch.SymPyModule(expressions=[z, w, u])
    assert mod.sympy() == [x - 1, sympy.Integer(0), sympy.Integer(1)]


def test_constants():
    x = sympy.symbols("x")
    y = 2.0 * x + sympy.UnevaluatedExpr(1.0)
    mod = sympytorch.SymPyModule(expressions=[y])
    assert mod.sympy() == [y]
    assert set(p.item() for p in mod.parameters()) == {2.0}
    assert set(b.item() for b in mod.buffers()) == {1.0}


def test_custom_function():
    x, y = sympy.symbols("x y")
    f = sympy.Function("f")
    z = x + f(y)
    extra_funcs = {f: lambda y_: y_**2}
    mod = sympytorch.SymPyModule(expressions=[z], extra_funcs=extra_funcs)
    assert mod.sympy() == [z]
    assert mod(x=1, y=2) == 1 + 2**2


def test_rationals():
    xvals = np.random.randn(100)
    x = sympy.symbols("x")
    y = x * sympy.Rational(2, 7)
    mod = sympytorch.SymPyModule(expressions=[y])
    mod.to(torch.float64)
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    assert len([p.item() for p in mod.parameters()]) == 0
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:, 0]
    error = y_tilde.detach().numpy() - xvals * 2 / 7
    assert (error**2).mean() < 1e-10, "error:{}".format((error**2).mean())


def test_half1():
    xvals = np.random.randn(100)
    x = sympy.symbols("x")
    y = abs(x) ** sympy.S.Half
    mod = sympytorch.SymPyModule(expressions=[y])
    mod.to(torch.float64)
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:, 0]
    error = y_tilde.detach().numpy() - np.abs(xvals) ** 0.5
    assert (error**2).mean() < 1e-10, "error:{}".format((error**2).mean())


def test_half2():
    xvals = np.random.randn(100)
    y = sympy.parse_expr("sqrt(Abs(x))")
    mod = sympytorch.SymPyModule(expressions=[y])
    mod.to(torch.float64)
    assert mod.sympy() == [y], "mod: {}, y:{}".format(mod.sympy(), y)
    assert len([p.item() for p in mod.parameters()]) == 0
    y_tilde = mod(x=torch.tensor(xvals, dtype=torch.float64))[:, 0]
    error = y_tilde.detach().numpy() - np.abs(xvals) ** 0.5
    assert (error**2).mean() < 1e-10, "error:{}".format((error**2).mean())


def test_constants():
    constants = [sympy.pi, sympy.E, sympy.GoldenRatio, sympy.TribonacciConstant, sympy.EulerGamma, sympy.Catalan]
    mod = sympytorch.SymPyModule(expressions=constants)
    mod.to(torch.float64)
    assert mod.sympy() == constants, "mod: {}, y:{}".format(mod.sympy(), constants)
    assert len([p.item() for p in mod.parameters()]) == 0
    torch.testing.assert_allclose(mod(),torch.tensor([float(c) for c in constants]))


def test_complex():
    # Simple complex number handing test
    x = sympy.symbols('x')

    complex_func_torch = sympytorch.SymPyModule(expressions=[x * sympy.I, 
                                                            sympy.conjugate(x), 
                                                            sympy.sqrt(sympy.conjugate(x*sympy.I) * x*sympy.I)])
    
    out = complex_func_torch(x=torch.tensor(2.0,dtype=torch.double)).detach().numpy()
    assert out[0].item() == 2.0j, "Expected 2j, eval:{}".format(out[0].item())
    assert out[1].item() == 2.0, "Expected 2, eval:{}".format(out[1].item())
    assert out[2].item() == 2.0, "Expected 2, eval:{}".format(out[2].item())

    # Complex number handling test with complex parameters
    out = complex_func_torch(x=torch.tensor(2.0j,dtype=torch.complex128)).detach().numpy()
    assert out[0].item() == -2.0, "Expected -2, eval:{}".format(out[0].item())
    assert out[1].item() == -2.0j, "Expected -2j, eval:{}".format(out[1].item())
    assert out[2].item() == 2.0, "Expected 2, eval:{}".format(out[2].item())

    # Comparison with numpy using spherical harmonics
    theta, phi = sympy.symbols("theta phi")
    max_l = 2
    m_list = range(-max_l, max_l + 1)

    # spherical harmonics from l=-2 to l=2
    func_list = [sympy.simplify(sympy.functions.special.spherical_harmonics\
            .Znm(max_l, m, theta, phi).expand(func=True)).evalf() for m in m_list]

    # Numpy and Torch based functions
    func_list_np = [sympy.lambdify([theta, phi], func, "numpy") for func in func_list ]
    func_list_torch = sympytorch.SymPyModule(expressions=func_list)

    np_eval = np.array(list(map(lambda i: func_list_np[i](np.pi/3,np.pi/3), np.arange(5))))
    torch_eval= func_list_torch(theta=torch.tensor(np.pi/3,dtype=torch.double),
                                phi=torch.tensor(np.pi/3,dtype=torch.double))

    # Correctness within single precision
    error = np.sum(np.abs(torch_eval.detach().numpy() - np_eval))
    assert error < 1e-7, "np v torch complex error:{}".format(error)

def test_integers():
    m = sympytorch.SymPyModule(expressions=[sympy.core.numbers.Zero()])
    assert m() == torch.tensor([0.0])

    m = sympytorch.SymPyModule(expressions=[sympy.core.numbers.One()])
    assert m() == torch.tensor([1.0])

    m = sympytorch.SymPyModule(expressions=[sympy.core.numbers.NegativeOne()])
    assert m() == torch.tensor([-1.0])

    for i in range(-10, 10):
        m = sympytorch.SymPyModule(expressions=[sympy.core.numbers.Integer(i)])
        assert m() == torch.tensor([float(i)])

