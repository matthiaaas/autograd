function forward_autodiff(f::Function, x::Real)
    return f(Dual(x, 1.0)).dual
end
