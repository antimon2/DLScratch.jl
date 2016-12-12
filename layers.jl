### AbstractLayer
abstract AbstractLayer

### MulLayer
type MulLayer{T} <: AbstractLayer
    x::T
    y::T
end

(::Type{MulLayer{T}}){T}() = MulLayer(zero(T), zero(T))

function forward{T}(lyr::MulLayer{T}, x::T, y::T)
    lyr.x = x
    lyr.y = y
    x * y
end
@inline forward{T}(lyr::MulLayer{T}, x, y) = forward(lyr, T(x), T(y))

function backward{T}(lyr::MulLayer{T}, dout::T)
    dx = dout * lyr.y
    dy = dout * lyr.x
    (dx, dy)
end

### AddLayer
type AddLayer <: AbstractLayer end

function forward(lyr::AddLayer, x, y)
    x + y
end

function backward(lyr::AddLayer, dout)
    (dout, dout)
end

### ActivationLayer
abstract ActivationLayer{T} <: AbstractLayer

### ReluLayer
type ReluLayer{T} <: ActivationLayer{T}
    mask::AbstractArray{Bool}
    (::Type{ReluLayer{T}}){T}() = new{T}()
end

function forward{T}(lyr::ReluLayer{T}, x::AbstractArray{T})
    mask = lyr.mask = (x .<= 0)
    out = copy(x)
    out[mask] = zero(T)
    out
end

function backward{T}(lyr::ReluLayer{T}, dout::AbstractArray{T})
    dout[lyr.mask] = zero(T)
    dout
end

### SigmoidLayer
type SigmoidLayer{T} <: ActivationLayer{T}
    out::T
    (::Type{SigmoidLayer{T}}){T}() = new{T}()
end

function forward{T}(lyr::SigmoidLayer{T}, x::T)
    lyr.out = 1 ./ (1 .+ exp(-x))
end

function backward{T}(lyr::SigmoidLayer{T}, dout::T)
    dout .* (1 .- lyr.out) .* lyr.out
end

### AffineLayer
type AffineLayer{T} <: AbstractLayer
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    x::AbstractArray{T}
    dW::AbstractMatrix{T}
    db::AbstractVector{T}
    function (::Type{AffineLayer}){T}(W::AbstractMatrix{T}, b::AbstractVector{T})
        lyr = new{T}()
        lyr.W = W
        lyr.b = b
        lyr
    end
end

function forward{T}(lyr::AffineLayer{T}, x::AbstractArray{T})
    lyr.x = x
    lyr.W * x .+ lyr.b
end

function backward{T}(lyr::AffineLayer{T}, dout::AbstractArray{T})
    dx = lyr.W' * dout
    lyr.dW = dout * lyr.x'
    lyr.db = _sumvec(dout)
    dx
end
@inline _sumvec{T}(dout::AbstractVector{T}) = dout
@inline _sumvec{T}(dout::AbstractMatrix{T}) = vec(mapslices(sum, dout, 2))
@inline _sumvec{T,N}(dout::AbstractArray{T,N}) = vec(mapslices(sum, dout, 2:N))

### SoftmaxWithLossLayer
function softmax{T<:AbstractFloat}(a::AbstractVector{T})
    c = maximum(a)  # オーバーフロー対策
    exp_a = exp(a .- c)
    exp_a ./ sum(exp_a)
end

function softmax{T<:AbstractFloat}(a::AbstractMatrix{T})
    mapslices(softmax, a, 1)
end

function crossentropyerror(y::Vector, t::Vector)
    δ = 1e-7  # アンダーフロー対策
    # -sum(t .* log(y .+ δ))
    -(t ⋅ log(y .+ δ))
end
function crossentropyerror(y::Matrix, t::Matrix)
    batch_size = size(y, 2)
    δ = 1e-7  # アンダーフロー対策
    # -sum(t .* log(y .+ δ)) / batch_size
    -vecdot(t, log(y .+ δ)) / batch_size
end
function crossentropyerror(y::Matrix, t::Vector)
    batch_size = size(y, 2)
    δ = 1e-7  # アンダーフロー対策
    -sum([log(y[t[i]+1, i]) for i=1:batch_size] .+ δ) / batch_size
end

type SoftmaxWithLossLayer{T} <: AbstractLayer
    loss::T
    y::AbstractArray{T}
    t::AbstractArray{T}
    (::Type{SoftmaxWithLossLayer{T}}){T}() = new{T}()
end

function forward{T}(lyr::SoftmaxWithLossLayer{T}, x::AbstractArray{T}, t::AbstractArray{T})
    lyr.t = t
    y = lyr.y = softmax(x)
    lyr.loss = crossentropyerror(y, t)
end

function backward{T}(lyr::SoftmaxWithLossLayer{T}, dout::T=1)
    dout .* _swlvec(lyr.y, lyr.t)
end
@inline _swlvec{T}(y::AbstractArray{T}, t::AbstractVector{T}) = y .- t
@inline _swlvec{T}(y::AbstractArray{T}, t::AbstractMatrix{T}) = (y .- t) / size(t)[2]