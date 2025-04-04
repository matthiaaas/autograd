# As seen in https://www.youtube.com/watch?v=QwFLA5TrviI

struct Dual{T<:Real}
    real::T
    dual::T
end

Dual(real::T, dual::T) where {T<:Real} = Dual{T}(real, dual)

Base.show(io::IO, d::Dual) = print(io, "Dual($(d.real), $(d.dual)ε)")

Base.:+(a::Dual, b::Dual) = Dual(a.real + b.real, a.dual + b.dual)
Base.:+(a::Real, b::Dual) = Dual(a + b.real, b.dual)
Base.:+(a::Dual, b::Real) = Dual(a.real + b, a.dual)

Base.:-(a::Dual, b::Dual) = Dual(a.real - b.real, a.dual - b.dual)
Base.:-(a::Real, b::Dual) = Dual(a - b.real, -b.dual)
Base.:-(a::Dual, b::Real) = Dual(a.real - b, a.dual)

Base.:*(a::Dual, b::Dual) = Dual(a.real * b.real, a.real * b.dual + a.dual * b.real)
Base.:*(a::Real, b::Dual) = Dual(a * b.real, a * b.dual)
Base.:*(a::Dual, b::Real) = Dual(a.real * b, a.dual * b)

Base.:/(a::Dual, b::Dual) = Dual(a.real / b.real, (a.dual * b.real - a.real * b.dual) / (b.real^2))
Base.:/(a::Real, b::Dual) = Dual(a / b.real, (-a * b.dual) / (b.real^2))
Base.:/(a::Dual, b::Real) = Dual(a.real / b, a.dual / b)

Base.:^(a::Dual, n::Real) = Dual(a.real^n, n * a.real^(n - 1) * a.dual)

Base.:sin(a::Dual) = Dual(sin(a.real), cos(a.real) * a.dual)
Base.:cos(a::Dual) = Dual(cos(a.real), -sin(a.real) * a.dual)
Base.:exp(a::Dual) = Dual(exp(a.real), exp(a.real) * a.dual)
