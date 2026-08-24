#------------------------------------------------------------------------------
# microgpt.jl
# This is a carbon copy of Andrej Karpathy's "microgpt"
# https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
# launch with juila -t auto for multithreading speed
#------------------------------------------------------------------------------

using Random
using LinearAlgebra
import Base: +, -, *, /, ^, log, exp

if !ispath("input.txt")
    URL = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    download(URL, "input.txt")
end

docs = readlines("input.txt")
shuffle!(docs)
println("num docs: $(length(docs))")

uchars = sort(collect(Set(join(docs))))
BOS = length(uchars) + 1 # Julia is 1-indexed
vocab_size = BOS
println("vocab size: $BOS")

relu(x::Real) = x > 0 ? x : 0.0
softmax(logits) = begin
    m = maximum(val.(logits))           # Float64
    exps = exp.(logits .- m)
    exps ./ sum(exps)                   # uses zero(V) now, so it's fine
end
draw(p) = min(length(p), searchsortedfirst(cumsum(p), rand()))
rmsnorm(x) = begin
    ms = sum(xi*xi for xi in x) / length(x)
    x .* ((ms + 1e-5) ^ (-0.5))
end

#------------------------------------------------------------------------------
# Autodiff (tape-based)
#------------------------------------------------------------------------------

struct V
    i::Int
end

# tape storage (SoA)
const X  = Float64[]   # primal values
const G  = Float64[]   # adjoints
const A  = Int[]       # parent a (0 = none)
const B  = Int[]       # parent b (0 = none)
const dA = Float64[]   # ∂out/∂a
const dB = Float64[]   # ∂out/∂b

@inline val(v::V)  = X[v.i]
@inline grad(v::V) = G[v.i]

@inline function pushnode(x::Float64, a::Int, b::Int, da::Float64, db::Float64)
    push!(X, x); push!(G, 0.0)
    push!(A, a); push!(B, b)
    push!(dA, da); push!(dB, db)
    return V(length(X))
end

@inline leaf(x::Real) = pushnode(Float64(x), 0, 0, 0.0, 0.0)

zero_grad!(start::Int) = (resize!(X,start); resize!(G,start); resize!(A,start); resize!(B,start);
                      resize!(dA,start); resize!(dB,start))

function backward!(y::V, start::Int)
    G[y.i] = 1.0
    @inbounds for i in length(X):-1:start+1
        gi = G[i]; a = A[i]; b = B[i]
        a != 0 && (G[a] += dA[i] * gi)
        b != 0 && (G[b] += dB[i] * gi)
    end
    return nothing
end

val(x::Real) = Float64(x)
@inline bin(u::V, v::V, y::Float64, du::Float64, dv::Float64) = pushnode(y, u.i, v.i, du, dv)
@inline un(u::V, y::Float64, du::Float64)                      = pushnode(y, u.i, 0,  du, 0.0)

Base.zero(v::V) = leaf(0.0)
Base.one(::Type{V})  = leaf(1.0)
Base.broadcastable(v::V) = Ref(v)
LinearAlgebra.adjoint(v::V) = v
+(u::V, v::V)    = bin(u, v, val(u)+val(v), 1.0,  1.0)
-(u::V, v::V)    = bin(u, v, val(u)-val(v), 1.0, -1.0)
*(u::V, v::V)    = (xu=val(u); xv=val(v); bin(u, v, xu*xv, xv, xu))
/(u::V, v::V)    = (xu=val(u); xv=val(v); bin(u, v, xu/xv, 1/xv, -xu/(xv*xv)))
^(u::V, p::Real) = (xu=val(u); pf=Float64(p); un(u, xu^pf, pf*xu^(pf-1)))
log(u::V)        = (xu=val(u); un(u, log(xu), 1/xu))
exp(u::V)        = (xu=val(u); e=exp(xu); un(u, e, e))
relu(u::V)       = (xu=val(u); un(u, xu>0 ? xu : 0.0, xu>0 ? 1.0 : 0.0))
+(u::V, c::Real) = un(u, val(u)+Float64(c), 1.0)
-(u::V, c::Real) = un(u, val(u)-Float64(c), 1.0)
*(u::V, c::Real) = un(u, val(u)*Float64(c), Float64(c))
/(u::V, c::Real) = un(u, val(u)/Float64(c), 1/Float64(c))
+(c::Real, u::V) = u + c
-(c::Real, u::V) = leaf(c) - u
*(c::Real, u::V) = u * c
/(c::Real, u::V) = leaf(c) / u
-(u::V)          = (-1.0) * u

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
n_embd = 16                 # embedding dimension
n_head = 4                  # number of attention heads
n_layer = 4                 # number of layers
block_size = 16             # maximum sequence length

mutable struct Layer{M<:AbstractMatrix}
    ℚ::M; 𝕂::M; 𝕍::M; 𝕆::M; 𝔽𝕔₁::M; 𝔽𝕔₂::M
end

mutable struct GPT{M<:AbstractMatrix}
    𝗘::M; 𝗣::M; lm_head::M
    layers::Vector{Layer{M}};
    n_embd::Int; n_head::Int; n_layer::Int; block_size::Int
end

mutable struct KVCache{T}
    K::Array{T,4}; V::Array{T,4}; t::Int
end

KVCache(::Type{T}, n_layer, n_embd, n_head, block_size) where {T} = 
    KVCache(Array{T}(undef, n_embd ÷ n_head, block_size, n_head, n_layer),
            Array{T}(undef, n_embd ÷ n_head, block_size, n_head, n_layer),
            0)

"""Single parameter vector; index into it with views for the model"""
function gpt_views(buf::AbstractVector, vocab_size, block_size, n_embd, n_head, n_layer)
    i = 1; take(m, n) = (r = i:(i+m*n-1); i += m*n; reshape(@view(buf[r]), m, n))
    wte = take(vocab_size, n_embd); wpe = take(block_size,  n_embd);  lm  = take(vocab_size, n_embd)
    layers = [Layer(take(n_embd,n_embd), take(n_embd,n_embd), take(n_embd,n_embd), take(n_embd,n_embd),
                    take(n_embd,4n_embd), take(4n_embd,n_embd))
              for _ in 1:n_layer]
    return GPT(wte, wpe, lm, layers, n_embd, n_head, n_layer, block_size)
end

#------------------------------------------------------------------------------
# Architecture
#------------------------------------------------------------------------------
function (ω::GPT)(token_id, pos_id, 𝚱)
    (; 𝗘, 𝗣, lm_head, layers, n_embd, n_head, n_layer, block_size) = ω
    t = (𝚱.t += 1)
    (; K, V) = 𝚱
    head_dim = n_embd ÷ n_head
    tok_emb = @view  𝗘[token_id, :]   # token embedding
    pos_emb = @view  𝗣[pos_id, :]     # position embedding
    x = tok_emb .+ pos_emb            # joint token and position embedding
    x = rmsnorm(x)
    for (li, layer) in enumerate(layers)
        (; ℚ, 𝕂, 𝕍, 𝕆, 𝔽𝕔₁, 𝔽𝕔₂) = layer
        xᵣ = x                  # Residual
        x = rmsnorm(x)
        q = reshape(ℚ' * x, head_dim, n_head)
        @views K[:, t, :, li] .= reshape(𝕂' * x, head_dim, n_head)
        @views V[:, t, :, li] .= reshape(𝕍' * x, head_dim, n_head)
        heads = similar(q)
        @views for head in 1:n_head
            qₕ = q[:, head] 
            Kₕ = K[:, 1:t, head, li]
            Vₕ = V[:, 1:t, head, li] 
            logits = (Kₕ' * qₕ) ./ √head_dim
            w = softmax(logits)
            heads[:, head] .= Vₕ * w
        end
        x = 𝕆' * vec(heads)
        x .+= xᵣ                # Residual highway
        # MLP Block
        xᵣ = x                  # Residual for MLP             
        x = (𝔽𝕔₂' * relu.(𝔽𝕔₁' * rmsnorm(x))) .+ xᵣ
    end
    return lm_head * x
end

#------------------------------------------------------------------------------
# Optimization
#------------------------------------------------------------------------------
nparams(vocab_size, block_size, n_embd, n_layer) = (2*vocab_size + block_size)*n_embd + 12*n_layer*n_embd^2
θ = [leaf(0.08 * randn()) for _ in 1:nparams(vocab_size, block_size, n_embd, n_layer)]
🤖 = gpt_views(θ, vocab_size, block_size, n_embd, n_head, n_layer)

# Adam
lr, β₁, β₂, eps = 0.01, 0.85, 0.99, 1e-8
m = zeros(Float64, length(θ))
v = zeros(Float64, length(θ))

tokenize = doc -> vcat(BOS, indexin(doc, uchars), BOS)
start = length(X)
num_steps = length(docs)
for step in 1:num_steps
    zero_grad!(start) 
    doc = docs[mod1(step, length(docs))]             # Go through docs in order, repeating if step > length(docs)
    tokens = tokenize(doc)
    n = min(block_size, length(tokens)-1)
    𝚱 = KVCache(V, n_layer, n_embd, n_head, block_size)
    start = length(X)
    losses = V[]
    for pos_id in 1:n
        token_id, target_id = tokens[pos_id:pos_id+1]
        logits = 🤖(token_id, pos_id, 𝚱)
        probs = softmax(logits)
        push!(losses, -log(probs[target_id]))
    end

    loss = sum(losses) / n
    backward!(loss, start)
    lrₜ = lr * (1 - (step - 1) / num_steps)

    Threads.@threads :static for j in eachindex(θ)
        i = θ[j].i
        g = G[i]
        m[j] = β₁*m[j] + (1-β₁)*g
        v[j] = β₂*v[j] + (1-β₂)*g*g
        m̂ = m[j] / (1 - β₁^step)
        v̂ = v[j] / (1 - β₂^step)
        X[i] -= lrₜ * m̂ / (sqrt(v̂) + eps)
        G[i] = 0.0
    end
    println("step $(lpad(step,4)) / $(num_steps) | loss $(round(val(loss), digits=4)) | tape $(length(X))")
end

#------------------------------------------------------------------------------
# Inference
#------------------------------------------------------------------------------

θf = val.(θ);
🧠 = gpt_views(θf, vocab_size, block_size, n_embd, n_head, n_layer);
T = .5;

Threads.@threads :static for Sᵢ in 1:100
    𝚱 = KVCache(Float64, n_layer, n_embd, n_head, block_size)
    token_id = BOS    
    chars = Char[]
    for pos_id in 1:block_size
        logits = 🧠(token_id, pos_id, 𝚱)
        probs = softmax(logits ./ T)
        token_id = draw(probs)
        if token_id == BOS; break; end
        push!(chars, uchars[token_id])
    end
    println("sample $Sᵢ: $(String(chars)) \tin docs?: $(String(chars) in docs)")
end