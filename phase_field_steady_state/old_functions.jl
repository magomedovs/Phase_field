
function main_NR(init_phi, init_T, init_c)
    phi = copy(init_phi)
    T = copy(init_T)
    c = copy(init_c)

    # X = [phi_2, ..., phi_(I-1), c, phi_(I+1), ..., phi_(M-1), T_2, ..., T_(M-1)]^T
    X = Vector{Float64}(undef, 2*dim)
    Y = Vector{Float64}(undef, 2*dim)
    X_new = Vector{Float64}(undef, 2*dim)

    X = [phi[2:(Interface_index-1)]; c; phi[(Interface_index+1):(end-1)]; T[2:(end-1)]]
    rhs_vec = create_rhs_vec(phi, T, c)    # right hand side
    Jac_mat = create_Jac_mat(phi, T, c)
    Y = Jac_mat \ rhs_vec
    X_new = Y + X

    phi[2:(Interface_index-1)] = X_new[1:(Interface_index-2)]
    phi[(Interface_index+1):(end-1)] = X_new[(Interface_index):dim]
    T[2:(end-1)] = X_new[(dim+1):end]
    c = X_new[Interface_index-1]

    number_of_iterations = 1
    while norm(X_new - X, Inf) > 2*10^-10
        X = [phi[2:(Interface_index-1)]; c; phi[(Interface_index+1):(end-1)]; T[2:(end-1)]]
        rhs_vec = create_rhs_vec(phi, T, c)    # right hand side
        Jac_mat = create_Jac_mat(phi, T, c)
        Y = Jac_mat \ rhs_vec
        X_new = Y + X

        phi[2:(Interface_index-1)] = X_new[1:(Interface_index-2)]
        phi[(Interface_index+1):(end-1)] = X_new[(Interface_index):dim]
        T[2:(end-1)] = X_new[(dim+1):end]
        c = X_new[Interface_index-1]

        number_of_iterations += 1
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

    A[:, (Interface_index-1)] .= dfdc
    C[:, (Interface_index-1)] .= dgdc

    Jac_mat::SparseMatrixCSC{Float64, Int64} = [A B; C D]

    #v = [dfdc; dgdc]
    #Jac_mat[:, (Interface_index-1)] .= v

    return Jac_mat
end


using OffsetArrays

function create_Jac_mat_OffsetArr(phi::Vector{Float64}, T::Vector{Float64}, c)#::SparseMatrixCSC{Float64, Int64}
    #Jac_mat::Matrix{Float64} = spzeros(2*dim, 2*dim)     # creating SparseArrays matrix

    #A = spzeros(dim)
    #B = spzeros(dim)
    #C = spzeros(dim)
    #D = spzeros(dim)

    # creating main diagonal of matrix A
    A_main_diag = OffsetArray(zeros(dim), 2:(dim+1))    # now indexing in the array begins from 2
    for i in eachindex(A_main_diag)
        A_main_diag[i] = p_phi_m(phi[i], T[i])
    end

    A = spdiagm(0 => A_main_diag, 1 => fill(p_phi_r(c), dim-1), -1 => fill(p_phi_l(c), dim-1)) # filling main, upper and lower diagonals of the matrix A

    # creating main diagonal of matrix B
    B_main_diag = OffsetArray(zeros(dim), 2:(dim+1))    # now indexing in the array begins from 2
    for i in eachindex(B_main_diag)
        B_main_diag[i] = p_T_m(phi[i], T[i])
    end

    B = spdiagm(0 => B_main_diag) # filling main diagonal of the matrix B

    # creating matrix C
    C = spdiagm(1 => fill(q_phi_r(c), dim-1), -1 => fill(q_phi_l(c), dim-1)) # filling upper and lower diagonals of the matrix C

    # creating main diagonal of matrix D
    D_main_diag = OffsetArray(zeros(dim), 2:(dim+1))    # now indexing in the array begins from 2
    for i in eachindex(D_main_diag)
        D_main_diag[i] = q_T_m()
    end

    D = spdiagm(0 => D_main_diag, 1 => fill(q_T_r(c), dim-1), -1 => fill(q_T_l(c), dim-1)) # filling main, upper and lower diagonals of the matrix D

    dfdc = OffsetArray(zeros(dim), 2:(dim+1))    # now indexing in the array begins from 2
    for i in eachindex(dfdc)
        dfdc[i] = p_c(phi[i-1], phi[i+1])
    end

    dgdc = OffsetArray(zeros(dim), 2:(dim+1))    # now indexing in the array begins from 2
    for i in eachindex(dgdc)
        dgdc[i] = q_c(phi[i-1], phi[i+1], T[i-1], T[i+1])
    end

    v = [OffsetArrays.no_offset_view(dfdc); OffsetArrays.no_offset_view(dgdc)]

#    for mat in [A,B,C,D]
#        mat = OffsetArrays.no_offset_view(mat)
#    end

    Jac_mat::SparseMatrixCSC{Float64, Int64} = [A B; C D]
    Jac_mat[:, (Interface_index-1)] .= v

    return Jac_mat
end


function create_Jac_mat_long(phi::Vector{Float64}, T::Vector{Float64}, c)

    Jac_mat = spzeros(2*dim, 2*dim)     # creating SparseArrays matrix

    # fill first row
    Jac_mat[1, 1] = p_phi_m(phi[2], T[2])                                               # ∂f_{2}/∂φ_{2}
    Jac_mat[1, 2] = p_phi_r(c)                                                          # ∂f_{2}/∂φ_{3}
    Jac_mat[1, Interface_index-1] = p_c(phi[2-1], phi[2+1])                             # ∂f_{2}/∂c
    Jac_mat[1, dim + 1] = p_T_m(phi[2], T[2])                                           # ∂f_{2}/∂T_{2}

    Jac_mat[dim, dim - 1] = p_phi_l(c)                                                  # ∂f_{M-1}/∂φ_{M-2}
    Jac_mat[dim, dim] = p_phi_m(phi[dim-1], T[dim-1])                                   # ∂f_{M-1}/∂φ_{M-1}
    Jac_mat[dim, Interface_index-1] = p_c(phi[(M-1)-1], phi[(M-1)+1])                   # ∂f_{M-1}/∂c
    Jac_mat[dim, dim + dim] = p_T_m(phi[M-1], T[M-1])                                   # ∂f_{M-1}/∂T_{M-1}


    Jac_mat[dim + 1, 2] = q_phi_r(c)                                                    # ∂g_{2}/∂φ_{3}
    Jac_mat[dim + 1, Interface_index-1] = q_c(phi[2-1], phi[2+1], T[2-1], T[2+1])       # ∂g_{2}/∂c
    Jac_mat[dim + 1, dim + 1] = q_T_m()                                                   # ∂g_{2}/∂T_{2}
    Jac_mat[dim + 1, dim + 2] = q_T_r(c)                                                # ∂g_{2}/∂T_{3}

    Jac_mat[2*dim, dim-1] = q_phi_l(c)                                                  # ∂g_{M-1}/∂φ_{M-2}
    Jac_mat[2*dim, Interface_index-1] = q_c(phi[(dim+1)-1], phi[(dim+1)+1], T[(dim+1)-1], T[(dim+1)+1]) # ∂g_{M-1}/∂c
    Jac_mat[2*dim, 2*dim-1] = q_T_l(c)                                                 # ∂g_{M-1}/∂T_{M-2}
    Jac_mat[2*dim, 2*dim] = q_T_m()                                                       # ∂g_{M-1}/∂T_{M-1}


    # fill 2:(dim-1) and (dim+2):(2*dim-1) rows
    for i=2:(dim-1)
        Jac_mat[i, i-1] = p_phi_l(c)
        Jac_mat[i, i] = p_phi_m(phi[i+1], T[i+1])
        Jac_mat[i, i+1] = p_phi_r(c)
        Jac_mat[i, Interface_index-1] = p_c(phi[(i+1)-1], phi[(i+1)+1])
        Jac_mat[i, dim + i] = p_T_m(phi[i+1], T[i+1])

        Jac_mat[dim + i, i-1] = q_phi_l(c)
        Jac_mat[dim + i, i+1] = q_phi_r(c)
        Jac_mat[dim + i, Interface_index-1] = q_c(phi[(i+1)-1], phi[(i+1)+1], T[(i+1)-1], T[(i+1)+1])
        Jac_mat[dim + i, dim + (i-1)] = q_T_l(c)
        Jac_mat[dim + i, dim + i] = q_T_m()
        Jac_mat[dim + i, dim + (i+1)] = q_T_r(c)
    end

    return Jac_mat

end
