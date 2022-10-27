# subscript l stands for left, m -- middle, r -- right
f(phi_l, phi_m, phi_r, T_m, c) = α * (phi_r - 2phi_m + phi_l) + γ * c * (phi_r - phi_l) + phi_m * (1 - phi_m) * (phi_m - 1/2 - m(T_m))
g(phi_l, phi_r, T_l, T_m, T_r, c) = (T_r - 2T_m + T_l) / Δh + c * (T_r - T_l) /2 + c * (phi_r - phi_l) / (2 * S)

# derivatives for Jacobian matrix
p_phi_l(c) = α - γ * c                                                              # ∂f_{i}/∂φ_{i-1}
p_phi_m(phi_m, T_m) = -2α + (2phi_m - 3phi_m^2) - (1 - 2phi_m) * (1/2 + m(T_m))     # ∂f_{i}/∂φ_{i}
p_phi_r(c) = α + γ * c                                                              # ∂f_{i}/∂φ_{i+1}
p_T_m(phi_m, T_m) = -phi_m * (1 - phi_m) * m_prime(T_m)                             # ∂f_{i}/∂T_{i}
p_c(phi_l, phi_r) = γ * (phi_r - phi_l)                                             # ∂f_{i}/∂c

q_phi_l(c) = -c/(2 * S)                                                             # ∂g_{i}/∂φ_{i-1}
q_phi_r(c) = c/(2 * S)                                                              # ∂g_{i}/∂φ_{i+1}
q_T_l(c) = (2 - Δh * c) / (2 * Δh)                                                  # ∂g_{i}/∂T_{i-1}
q_T_m() = -2 / Δh                                                                   # ∂g_{i}/∂T_{i}
q_T_r(c) = (2 + Δh * c) / (2 * Δh)                                                  # ∂g_{i}/∂T_{i+1}
q_c(phi_l, phi_r, T_l, T_r) = 1/2 * (T_r - T_l) + 1/(2 * S) * (phi_r - phi_l)       # ∂g_{i}/∂c


# Newton Raphson method to solve a system of nonlinear algebraic equations
# F'(x_k) ⋅ Y = -F(x_k), where Y = X_{k+1} - X_k
function solve_NR(init_phi, init_T, init_c)
    phi = copy(init_phi)
    T = copy(init_T)
    c = copy(init_c)

    # X = [phi_2, ..., phi_(I-1), phi_(I+1), ..., phi_(M-1),    T_2, ...,    T_(M-1),      c   ]
    # X = [X[1] , ...,    X[I-2],    X[I-1], ...,  X[dim-1], X[dim], ..., X[2*dim-1],  X[2*dim]]
    X = Vector{Float64}(undef, 2*dim)
    Y = Vector{Float64}(undef, 2*dim)
    X_new = Vector{Float64}(undef, 2*dim)

    X = [phi[2:(Interface_index-1)]; phi[(Interface_index+1):(end-1)]; T[2:(end-1)]; c]
    rhs_vec = create_rhs_vec(phi, T, c)    # right hand side
    Jac_mat = create_Jac_mat(phi, T, c)
    Y = Jac_mat \ rhs_vec
    X_new = Y + X

    #prob = LinearProblem(Jac_mat, rhs_vec)
    #sol = solve(prob, UMFPACKFactorization())
    #X_new = sol.u + X

    phi[2:(Interface_index-1)] = X_new[1:(Interface_index-1-1)]
    phi[(Interface_index+1):(end-1)] = X_new[(Interface_index-1):(dim-1)]
    T[2:(end-1)] = X_new[(dim):(end-1)]
    c = X_new[end]

    number_of_iterations = 1

    println()
    println("Iteration number ", number_of_iterations)
    println("norm(X_new - X, Inf) = ", norm(X_new - X, Inf))
    println("norm(F(X), inf) = ", norm(-rhs_vec, Inf))

    while norm(X_new - X, Inf) > 1.0e-10
        X = [phi[2:(Interface_index-1)]; phi[(Interface_index+1):(end-1)]; T[2:(end-1)]; c]
        rhs_vec = create_rhs_vec(phi, T, c)    # right hand side
        Jac_mat = create_Jac_mat(phi, T, c)
        Y = Jac_mat \ rhs_vec
        X_new = Y + X

        #prob = LinearProblem(Jac_mat, rhs_vec)
        #sol = solve(prob, UMFPACKFactorization())
        #X_new = sol.u + X

        phi[2:(Interface_index-1)] = X_new[1:(Interface_index-1-1)]
        phi[(Interface_index+1):(end-1)] = X_new[(Interface_index-1):(dim-1)]
        T[2:(end-1)] = X_new[(dim):(end-1)]
        c = X_new[end]

        number_of_iterations += 1

        println()
        println("Iteration number ", number_of_iterations)
        println("norm(X_new - X, Inf) = ", norm(X_new - X, Inf))
        println("norm(F(X), inf) = ", norm(-rhs_vec, Inf))
    end
    println()
    println("Number of iterations ", number_of_iterations)
    return phi, T, c

end


function create_Jac_mat(phi::Vector{Float64}, T::Vector{Float64}, c)#::SparseMatrixCSC{Float64, Int64}
    #Jac_mat::Matrix{Float64} = spzeros(2*dim, 2*dim)     # creating SparseArrays matrix

    A_main_diag = spzeros(dim)
    B_main_diag = spzeros(dim)
    D_main_diag = spzeros(dim)
    dfdc = spzeros(dim)
    dgdc = spzeros(dim)

    # creating main diagonal of matrix A
    for i in eachindex(A_main_diag)
        A_main_diag[i] = p_phi_m(phi[i+1], T[i+1])
    end

    A = spdiagm(0 => A_main_diag, 1 => fill(p_phi_r(c), dim-1), -1 => fill(p_phi_l(c), dim-1)) # filling main, upper and lower diagonals of the matrix A

    # creating main diagonal of matrix B
    for i in eachindex(B_main_diag)
        B_main_diag[i] = p_T_m(phi[i+1], T[i+1])
    end

    B = spdiagm(0 => B_main_diag) # filling main diagonal of the matrix B

    # creating matrix C
    C = spdiagm(1 => fill(q_phi_r(c), dim-1), -1 => fill(q_phi_l(c), dim-1)) # filling upper and lower diagonals of the matrix C

    # creating main diagonal of matrix D
    for i in eachindex(D_main_diag)
        D_main_diag[i] = q_T_m()
    end

    D = spdiagm(0 => D_main_diag, 1 => fill(q_T_r(c), dim-1), -1 => fill(q_T_l(c), dim-1)) # filling main, upper and lower diagonals of the matrix D

    for i in eachindex(dfdc)
        dfdc[i] = p_c(phi[i+1-1], phi[i+1+1])
    end

    for i in eachindex(dgdc)
        dgdc[i] = q_c(phi[i+1-1], phi[i+1+1], T[i+1-1], T[i+1+1])
    end

    Jac_mat::SparseMatrixCSC{Float64, Int64} = [A[:,1:(Interface_index-1-1)] A[:,(Interface_index-1+1):dim] B dfdc; C[:,1:(Interface_index-1-1)] C[:,(Interface_index-1+1):dim] D dgdc]

    println()
    println("Jac_mat max = ", findmax(nonzeros(Jac_mat))[1])
    println("Jac_mat min = ", findmin(nonzeros(Jac_mat))[1])
    println("Jac_mat absmin = ", findmin(map(x->abs(x), nonzeros(Jac_mat)))[1])

    return Jac_mat
    #return A, B, C, D
end


function create_rhs_vec(phi::Vector{Float64}, T::Vector{Float64}, c)
    f_vec = Vector{Float64}(undef, (M - 2))     # f_vec = [f_2, ..., f_(M-1)]
    g_vec = Vector{Float64}(undef, (M - 2))     # g_vec = [g_2, ..., g_(M-1)]
    for i=2:(M-1)
        f_vec[i-1] = f(phi[i-1], phi[i], phi[i+1], T[i], c)
        g_vec[i-1] = g(phi[i-1], phi[i+1], T[i-1], T[i], T[i+1], c)
    end
    return -[f_vec; g_vec]
end
