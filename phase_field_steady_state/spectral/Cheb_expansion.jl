using FastGaussQuadrature
using ClassicalOrthogonalPolynomials

# chebyshev_expansion(x) = Σ aᵢ * Tᵢ(x)
function chebyshev_expansion(a_i, x)
    res = 0.
    for i=0:( length(a_i) - 1 )
        res += a_i[i+1] * chebyshevt(i, x)
    end
    return res
end

#cheb_nodes = [cos(pi/num_of_nodes * (1/2 + k)) for k=0:num_of_nodes-1]
function calculate_cheb_colloc_expansion_coeffs(test_function::Function, N; dist_from_boundary=0.)
    nodes, _ = gausschebyshev(N-2)

    f_rhs::Vector{Float64} = [[test_function(nodes[i]) for i in eachindex(nodes)]; test_function(-1+dist_from_boundary); test_function(1-dist_from_boundary)]

    A::Matrix{Float64} = Matrix{Float64}(undef, N::Int64, N::Int64)
    for i in eachindex(nodes)
        A[i, :] = [chebyshevt(k, nodes[i]) for k=0:N-1]
    end
    A[N-1, :] = [chebyshevt(k, -1) for k=0:N-1]
    A[N, :] = [chebyshevt(k, 1) for k=0:N-1]

    a_i::Vector{Float64} = A \ f_rhs

    return a_i
end