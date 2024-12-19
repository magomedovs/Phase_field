using ClassicalOrthogonalPolynomials

# chebyshev_expansion(x) = Σ aᵢ * Tᵢ(x)
function chebyshev_expansion(a_i::Vector{Float64}, x::Float64)::Float64
    chebT = chebyshevt()
    return chebT[x, 1:length(a_i)]' * a_i
end

function calculate_cheb_expansion_coeffs(
    test_function::Function, N; 
    dist_from_boundary=0.)
    
    nodes = [cos(j * pi / (N-1)) for j=(N-1):-1:0]
    nodes[1] += dist_from_boundary
    nodes[end] -= dist_from_boundary

    chebT = chebyshevt()
    A::Matrix{Float64} = chebT[nodes, 1:N]  #println("A size = ", size(A))
    f_rhs::Vector{Float64} = [test_function(node) for node in nodes]    #println("f_rhs size = ", size(f_rhs))
    
    a_i::Vector{Float64} = A \ f_rhs

    return a_i
end