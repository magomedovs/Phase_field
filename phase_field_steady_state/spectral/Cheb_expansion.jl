using FastGaussQuadrature
using ClassicalOrthogonalPolynomials

# chebyshev_expansion(x) = Σ aᵢ * Tᵢ(x)
function chebyshev_expansion(a_i::Vector{Float64}, x::Float64)::Float64
    chebT = ChebyshevT()
    return chebT[x, 1:length(a_i)]' * a_i
    #=
    res = 0.
    for i=0:( length(a_i) - 1 )
        res += a_i[i+1] * chebyshevt(i, x)
    end
    return res
    =#
    #=
    res = a_i' * [chebyshevt(i, x) for i=0:( length(a_i) - 1 )]
    return res
    =#
end

#cheb_nodes = [cos(pi/num_of_nodes * (1/2 + k)) for k=0:num_of_nodes-1]
function calculate_cheb_expansion_coeffs(test_function::Function, N; dist_from_boundary=0.)
    cheb_nodes, _ = gausschebyshev(N-2)
    nodes = [-1 + dist_from_boundary; cheb_nodes; 1 - dist_from_boundary]
    #nodes = [-cos(j * pi / (N-1)) for j=0:(N-1)]

    chebT = ChebyshevT()
    A::Matrix{Float64} = chebT[nodes, 1:N]  #println("A size = ", size(A))
    f_rhs::Vector{Float64} = [test_function(node) for node in nodes]    #println("f_rhs size = ", size(f_rhs))
    
    a_i::Vector{Float64} = A \ f_rhs

    return a_i
end