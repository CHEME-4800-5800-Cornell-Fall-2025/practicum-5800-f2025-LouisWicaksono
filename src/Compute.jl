# throw(ErrorException("Oppps! No methods defined in src/Compute.jl. What should you do here?"))
include("Types.jl")

"""
    energy(model, s)

Compute the Hopfield energy of state vector `s`.
"""
function energy(model::MyClassicalHopfieldNetworkModel, s::AbstractVector{<:Integer})
    return -0.5 * sum((model.W * s) .* s) - sum(model.b .* s)
end

"""
    update_neuron!(model, s, i)

Asynchronously update neuron `i` in state vector `s`.
"""
function update_neuron!(model::MyClassicalHopfieldNetworkModel, s::Vector{Int32}, i::Int)
    h = sum(model.W[i,:] .* s) - model.b[i]
    s[i] = sign(h) == 0 ? s[i] : sign(h)
end

"""
    recover(model, sₒ, true_image_energy; maxiterations, patience, miniterations_before_convergence)

Run the recovery algorithm starting from corrupted state `sₒ`.
Returns:
- `frames`: Dict mapping iteration → state vector
- `energydictionary`: Dict mapping iteration → energy
"""
function recover(model::MyClassicalHopfieldNetworkModel,
                 sₒ::Vector{Int32},
                 true_image_energy::Float64;
                 maxiterations::Int=1000,
                 patience::Int=5,
                 miniterations_before_convergence::Union{Int,Nothing}=nothing)

    N = model.N
    s = copy(sₒ)
    frames = Dict{Int,Vector{Int32}}()
    energydictionary = Dict{Int,Float64}()

    if isnothing(miniterations_before_convergence)
        miniterations_before_convergence = patience
    end

    converged = false
    t = 1
    stable_count = 0

    while !converged && t ≤ maxiterations
        # asynchronous update
        i = rand(1:N)
        update_neuron!(model, s, i)

        # record state and energy
        frames[t] = copy(s)
        energydictionary[t] = energy(model, s)

        # check convergence
        if t ≥ miniterations_before_convergence
            if t > 1 && frames[t] == frames[t-1]
                stable_count += 1
            else
                stable_count = 0
            end
            if stable_count ≥ patience
                converged = true
            end
        end

        t += 1
    end

    return frames, energydictionary
end

"""
    decode(s; rows, cols)

Convert a flattened ±1 state vector back into a 2D image array.
Maps -1 → 0.0 (black), +1 → 1.0 (white).
"""
function decode(s::AbstractVector{<:Integer}; rows::Int=28, cols::Int=28)
    img = reshape(s, rows, cols)
    return Float32.(img .== 1)
end

"""
    hamming(a, b)

Compute the Hamming distance between two binary vectors.
"""
function hamming(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    @assert length(a) == length(b)
    return sum(a .!= b)
end