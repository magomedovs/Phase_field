using Printf
using Plots
#using CairoMakie
#using ApproxFun
using LinearAlgebra
using LsqFit
using DelimitedFiles
#using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using ArnoldiMethod
using SparseArrays

include("../structs_and_functions.jl")
include("../spectral/Cheb_expansion.jl")

function solve_eigen(phi, T, p::Params, k::Float64, NUM::Int64;
    c::Float64 = p.c_sharp_lim, 
    interval_start::Float64 = -3.2, interval_end::Float64 = 3.2,
    save_solution=false)

    #NUM = 501
    h::Float64 = (interval_end - interval_start) / (NUM - 1)
    DIM::Int64 = (NUM - 2)

    # Ax = λx

    β_1 = p.ϵ_0^2 / (p.τ * h^2) - c / (2 * h)

    g_1(i::Int64) = -p.ϵ_0^2 * k^2 + (1 - 2 * phi(interval_start + h * (i - 1))) * (phi(interval_start + h * (i - 1)) - 1/2 - m(T(interval_start + h * (i - 1)), p)) + phi(interval_start + h * (i - 1)) * (1 - phi(interval_start + h * (i - 1)))
    β_2(i::Int64) = -2 * p.ϵ_0^2 / (p.τ * h^2) + g_1(i) / p.τ

    β_3 = p.ϵ_0^2 / (p.τ * h^2) + c / (2 * h)

    g_2(i::Int64) = m_prime(T(interval_start + h * (i - 1)), p) * phi(interval_start + h * (i - 1)) * (1 - phi(interval_start + h * (i - 1)))
    β_4(i::Int64) = -g_2(i) / p.τ

    γ_1 = c / (2 * h * p.S)
    γ_2 = 1 / h^2 - c / (2 * h)
    γ_3 = -2 / h^2 - k^2
    γ_4 = 1 / h^2 + c / (2 * h)

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

    number_of_eigenvals = 650
    println("number_of_eigenvals = ", number_of_eigenvals)

    # Computing eigenvalues of the matrix Mat

    # ArnoldiMethod
    decomp, history = @time partialschur(Mat, nev=number_of_eigenvals, tol=1e-10, which=LR());
    λs, X = partialeigen(decomp)
    λs_max_re, λs_max_re_ind = findmax(map(x->x.re, λs)) #findmax(map(x->x.re, λs))[1], findmax(map(x->x.re, λs))[2]
    println("max Re(λs) = ", λs_max_re)

    if save_solution
        λs_max_re = λs_max_re
        u = Real.(X[1:NUM-2, λs_max_re_ind])
        v = Real.(X[NUM-1:2*(NUM-2), λs_max_re_ind])

        open("lambdas_k_$(k)_NUM_$(NUM).txt", "w") do io
            writedlm(io, λs)
        end
        open("uvlambdamax_k_$(k)_NUM_$(NUM).txt", "w") do io
            writedlm(io, [u v fill(λs_max_re, length(u))])
        end
    end

    return λs, X  #, A, B, C, D

end

file_name = "phase_field_steady_state/finite-difference/spectral_tests_data_alpha_coef_5.0/spectral_solution_coeffs_NUM_600.txt"
phi, T, c_computed = read_solution_from_file(file_name)

phi_computed(x::Float64)::Float64 = chebyshev_expansion(phi, x)
T_computed(x::Float64)::Float64 = chebyshev_expansion(T, x)

params = Params(1.2, 0.005)
const alpha_coef = 5.0
const α = alpha_coef / params.c_sharp_lim   # find appropriate / optimal value !

phi_computed_on_line(eta::Float64)::Float64 = phi_computed(tanh(eta/α))
T_computed_on_line(eta::Float64)::Float64 = T_computed(tanh(eta/α))


interval_start = -12.0
interval_end = 2.0

NUM = 28001
#k_test = 1.3 #1.0e1
for k_test in [range(0.5, 0.9, step=0.1); range(1., 19., step=1); range(20., 100., step=10)]
    λs_test, X_test = solve_eigen(
        phi_computed_on_line, T_computed_on_line, params, k_test, NUM;
        c = c_computed, interval_start = interval_start, interval_end = interval_end,
        save_solution=true
    )
end

#@profview 
#=
u = Real.(X_test[1:NUM-2, findmax(map(x->x.re, λs_test))[2]])
v = Real.(X_test[NUM-1:2*(NUM-2), findmax(map(x->x.re, λs_test))[2]])
#plot(range(-H/2, H/2, length = NUM-2), [u, v], label=["u" "v"], xlabel="η", title="k=$k_test")

λs_test_max_re = findmax(map(x->x.re, λs_test))[1]

C0 = u[findmax(abs.(u))[2]] * 4 * sqrt(2)
u_composite(x, p::Params) = C0 / (4 * sqrt(2) * cosh(x / (p.ϵ_0 * 2 * sqrt(2)))^2)

function v_composite(x, p::Params; front_vel=p.c_sharp_lim)
    D2 = v[findmax(abs.(u))[2]] / p.ϵ_0 # - sign(v[div(NUM-2, 2) + 1]) * ϵ_0 * front_vel  # ??? guessed parameter value  # front_vel^2 / (2 * S * sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2)) * C0

    B1 = D2 - C0 * front_vel / (2 * p.S)
    A1 = D2 + C0 * front_vel / (2 * p.S)

    mu_left = (-front_vel + sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2))/2
    mu_right = (-front_vel - sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2))/2


    if x>=0
        return p.ϵ_0 * (B1 * exp(mu_right * x) - C0 * front_vel / (2 * p.S) * tanh(x / (p.ϵ_0 * 2 * sqrt(2))) + C0 * front_vel / (2 * p.S))
    else
        return p.ϵ_0 * (A1 * exp(mu_left * x) - C0 * front_vel / (2 * p.S) * tanh(x / (p.ϵ_0 * 2 * sqrt(2))) - C0 * front_vel / (2 * p.S))
    end
end


span = range(interval_start, interval_end, length=Int(NUM-2))

plot(
    span, u, 
    #xlims=(-0.1, 0.1),
    label="u computed"
)
plot!(
    span, x -> u_composite(x, params),
    label="u approx"
)

plot(
    span, 
    v,
    #yaxis=:log,
    #xlims=(-0.1, 0.1),
    label="v computed"
)
plot!(
    span,
    x -> v_composite(x, params),
    label="v approx"
)

plot(
    span, 
    abs.(v),
    yaxis=:log,
    #xlims=(0., 2.),
    label="v computed"
)
=#

#=
tempspan = range(-0.3, 0.3, length=Int(1e3))
plot(
    tempspan,
    x -> u_composite(x, params),
    label="u approx",
    xlabel="η"
)
plot!(
    tempspan,
    x -> v_composite(x, params),
    label="v approx",
    #yaxis=:log,
    legend=:topleft,
    xlabel="η"
)
#savefig("~/Desktop/uv_approx.pdf")

tempspan1 = range(-1., 1., length=Int(1e3))
temp_alpha = α
plot(
    tempspan1,
    x -> phi_approx(temp_alpha * atanh(x), params),
    label="phi",
    title="α=$(round(temp_alpha, digits=3))",
    xlabel="x"
)
plot!(
    tempspan1,
    x -> T_approx(temp_alpha * atanh(x), params),
    label="T",
    #yaxis=:log,
    legend=:topleft,
    xlabel="x"
)
#savefig("~/Desktop/phiT_approx_alpha_$(round(temp_alpha, digits=3)).pdf")
=#

#=
x = Fun(interval_start..interval_end)
#d = Interval(interval_start, interval_end)
S = space(x)
D = ApproxFun.Derivative(S)

Bu = Dirichlet(S)
Bv = Dirichlet(S)
=#
#E = Operator(I,d)

# https://juliaapproximation.github.io/ApproxFun.jl/latest/generated/Eigenvalue/
# https://discourse.julialang.org/t/solving-for-eigenvalues-of-an-equation-system-in-approxfun-jl/33682

#=
x = Fun(-8..8)
V = x^2/2
L = -𝒟^2/2 + V
S = space(x)
B = Dirichlet(S)
λ, v = ApproxFun.eigs(B, L, 500,tolerance=1E-10);

p = Plots.plot(V; legend=false, ylim=(-Inf, λ[22]))
for k=1:20
    Plots.plot!(real(v[k]/norm(v[k]) + λ[k]), )
end
p
=#

#=
H = (interval_end - interval_start)
h_list = [H / 7001, H / 10001, H / 20001]
c_list = [22.01990175633657, 20.9258446450055, 20.13655602069039]
res_estim = extrapolate_func(h_list, c_list)
=#