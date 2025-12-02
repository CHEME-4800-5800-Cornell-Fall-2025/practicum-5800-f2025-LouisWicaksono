# throw(ErrorException("Oppps! No methods defined in src/Factory.jl. What should you do here?"))
include("Types.jl")

"""
    build(::Type{MyClassicalHopfieldNetworkModel}; memories)

Constructs a Hopfield network from a set of memories using the Hebbian learning rule.
- `memories`: N×K matrix of ±1 integers (columns are images)
"""
function build(::Type{MyClassicalHopfieldNetworkModel}; memories::Matrix{Int32})
    N, K = size(memories)

    # Hebbian learning rule
    W = zeros(Float64, N, N)
    for k in 1:K
        s = memories[:,k]
        W .+= s * s'
    end
    W ./= K

    # remove self-connections
    for i in 1:N
        W[i,i] = 0.0
    end

    # bias vector set to zero
    b = zeros(Float64, N)

    # compute energies of each stored memory
    energies = [ -0.5 * sum((W * memories[:,k]) .* memories[:,k]) for k in 1:K ]

    return MyClassicalHopfieldNetworkModel(N, W, b, memories, energies)
end

# Wrapper to accept NamedTuple (needed because main file calls build with a NamedTuple)
function build(::Type{MyClassicalHopfieldNetworkModel}, args::NamedTuple)
    return build(MyClassicalHopfieldNetworkModel; memories=args.memories)
end
