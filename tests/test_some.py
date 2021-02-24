import sympy, torch, sympytorch


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

