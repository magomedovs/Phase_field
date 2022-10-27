using Arpack
using ArnoldiMethod

function solve_eigen(phi, T, c::Float64, k::Float64, NUM::Int64)

    #NUM = 501
    h = H / (NUM - 1)
    DIM = (NUM - 2)

    # Ax = λx

    β_1 = ϵ_0^2 / (τ * h^2) - c / (2 * h)

    g_1(i::Int64) = -ϵ_0^2 * k^2 + (1 - 2 * phi(-H/2 + h * (i - 1))) * (phi(-H/2 + h * (i - 1)) - 1/2 - m(T(-H/2 + h * (i - 1)))) + phi(-H/2 + h * (i - 1)) * (1 - phi(-H/2 + h * (i - 1)))
    β_2(i::Int64) = -2 * ϵ_0^2 / (τ * h^2) + g_1(i) / τ

    β_3 = ϵ_0^2 / (τ * h^2) + c / (2 * h)

    g_2(i::Int64) = m_prime(T(-H/2 + h * (i - 1))) * phi(-H/2 + h * (i - 1)) * (1 - phi(-H/2 + h * (i - 1)))
    β_4(i::Int64) = -g_2(i) / τ

    γ_1 = c / (2 * h * S)
    γ_2 = 1 / h^2 - c / (2 * h)
    γ_3 = -2 / h^2 - k^2
    γ_4 = 1 / h^2 + c / (2 * h)

#=
    # both sides of Ax = λx multiplied by h

    β_1 = ϵ_0^2 / (τ) - c * h / (2)

    g_1(i::Int64) = -ϵ_0^2 * k^2 + (1 - 2 * phi(-H/2 + h * (i - 1))) * (phi(-H/2 + h * (i - 1)) - 1/2 - m(T(-H/2 + h * (i - 1)))) + phi(-H/2 + h * (i - 1)) * (1 - phi(-H/2 + h * (i - 1)))
    β_2(i::Int64) = -2 * ϵ_0^2 / (τ) + g_1(i) * h^2 / τ

    β_3 = ϵ_0^2 / (τ) + c * h / (2)

    g_2(i::Int64) = m_prime(T(-H/2 + h * (i - 1))) * phi(-H/2 + h * (i - 1)) * (1 - phi(-H/2 + h * (i - 1)))
    β_4(i::Int64) = -g_2(i) * h^2 / τ

    γ_1 = c * h / (2 * S)
    γ_2 = 1 - c * h / (2)
    γ_3 = -2 - k^2 * h^2
    γ_4 = 1 + c * h / (2)
=#

    A_main_diag = spzeros(DIM)
    B_main_diag = spzeros(DIM)

    # creating main diagonal of matrix A
    for i in eachindex(A_main_diag)
        A_main_diag[i] = β_2(i + 1) # (i + 1) as we go from 2 to M-1 indices
    end

    A = spdiagm(0 => A_main_diag, 1 => fill(β_3, DIM-1), -1 => fill(β_1, DIM-1)) # filling main, upper and lower diagonals of the matrix A

    # creating main diagonal of matrix B
    for i in eachindex(B_main_diag)
        B_main_diag[i] = β_4(i + 1)
    end

    B = spdiagm(0 => B_main_diag) # filling main diagonal of the matrix B

    C = spdiagm(1 => fill(γ_1, DIM-1), -1 => fill(-γ_1, DIM-1))

    D = spdiagm(0 => fill(γ_3, DIM), 1 => fill(γ_4, DIM-1), -1 => fill(γ_2, DIM-1))

    Mat::SparseMatrixCSC{Float64, Int64} = [A B; C D]

    #return A, B, C, D
    #return Mat

    println()
    println("NUM = ", NUM)
    println("k = ", k)

    number_of_eigenvals = 300
    println("number_of_eigenvals = ", number_of_eigenvals)

    # Computing eigenvalues of the matrix Mat

    # ArnoldiMethod
    decomp, history = @time partialschur(Mat, nev=number_of_eigenvals, tol=1e-10, which=LR());
    λs, X = partialeigen(decomp)
    println("max Re(λs) = ", findmax(map(x->x.re, λs))[1])

    return λs, X  #, A, B, C, D

    # Arpack
    #λ, ϕ = @time eigs(Mat, nev = number_of_eigenvals, which=:LR)
    #return λ, ϕ

end
