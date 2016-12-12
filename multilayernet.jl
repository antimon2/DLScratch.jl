type MultiLayerNet{T}
    fclyrs::AbstractVector{AffineLayer{T}}
    aclyrs::AbstractVector{ActivationLayer{T}}
    lastlyr::SoftmaxWithLossLayer{T}
end

function (::Type{MultiLayerNet{T}}){T}(
        input_size::Int, 
        hidden_sizes::AbstractVector{Int}, 
        output_size::Int)
    input_sizes = [input_size;hidden_sizes]
    output_sizes = [hidden_sizes;output_size]
    weights = [_he_initializer(T, isz) .* randn(T, osz, isz) for 
                        (isz, osz)=zip(input_sizes, output_sizes)]
    biases = [zeros(T, osz) for osz=output_sizes]
    fclyrs = AffineLayer{T}[AffineLayer(W, b) for (W,b)=zip(weights, biases)]
    aclyrs = ActivationLayer{T}[ReluLayer{T}() for _=hidden_sizes]
    MultiLayerNet(fclyrs, aclyrs, SoftmaxWithLossLayer{T}())
end
@inline _he_initializer{T}(::Type{T}, input_size::Int) = sqrt(T(2)/input_size)

function predict{T}(net::MultiLayerNet{T}, x::AbstractArray{T})
    z = x
    N = length(net.fclyrs)
    for i = 1:N-1
        a = forward(net.fclyrs[i], z)
        z = forward(net.aclyrs[i], a)
    end
    a = forward(net.fclyrs[N], z)
    # softmax(a)
    a
end

function loss{T}(net::MultiLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    y = predict(net, x)
    forward(net.lastlyr, y, t)
end

function accuracy{T}(net::MultiLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    y = vec(mapslices(indmax, predict(net, x), 1))
    if ndims(t) > 1 t = vec(mapslices(indmax, t, 1)) end
    mean(y .== t)
end

immutable MultiLayerNetGrads{T}
    Ws::Vector{AbstractMatrix{T}}
    bs::Vector{AbstractVector{T}}
end

@inline _get_W{T}(lyr::AffineLayer{T}) = lyr.W
@inline _get_b{T}(lyr::AffineLayer{T}) = lyr.b
@inline _get_dW{T}(lyr::AffineLayer{T}) = lyr.dW
@inline _get_db{T}(lyr::AffineLayer{T}) = lyr.db

function Base.gradient{T}(net::MultiLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    N = length(net.fclyrs)
    # forward
    loss(net, x, t)
    # backward
    dout = one(T)
    dz = backward(net.lastlyr, dout)
    da = backward(net.fclyrs[N], dz)
    for i=N-1:-1:1
        dz = backward(net.aclyrs[i], da)
        da = backward(net.fclyrs[i], dz)
    end
    # Ws = map(_get_dW, net.fclyrs)
    Ws = AbstractMatrix{T}[lyr.dW for lyr=net.fclyrs]
    # bs = map(_get_db, net.fclyrs)
    bs = AbstractVector{T}[lyr.db for lyr=net.fclyrs]
    MultiLayerNetGrads(Ws, bs)
end

function applygradient!{T}(net::MultiLayerNet{T}, grads::MultiLayerNetGrads{T}, learning_rate::T)
    N = length(net.fclyrs)
    for i=1:N
        net.fclyrs[i].W .-= learning_rate .* grads.Ws[i]
        net.fclyrs[i].b .-= learning_rate .* grads.bs[i]
    end
end
