"""
    MyClassicalHopfieldNetworkModel

Represents a classical Hopfield network with Hebbian weights.
- `N`: number of neurons (pixels)
- `W`: weight matrix
- `b`: bias vector
- `memories`: stored memories (columns = images)
- `energy`: energies of each stored memory
"""
struct MyClassicalHopfieldNetworkModel
    N::Int
    W::Matrix{Float64}
    b::Vector{Float64}
    memories::Matrix{Int32}
    energy::Vector{Float64}
end