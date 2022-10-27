using Printf
using Plots
using LinearAlgebra
using NLsolve
using ForwardDiff
using SparseArrays

include("../structs_and_functions.jl")

function calculate_finite_diff(NUM::Int64, p::Params; save_solution=false, output_filename="finite_diff_solution_NUM_$(NUM).txt")
  
    println("Number of terms = $(NUM)")

    function f!(F, U)

        interval_start = -1.
        interval_end = 1.

        h = (interval_end - interval_start) / (NUM - 1)
        x(i) = interval_start + (i-1) * h

        #Threads.@threads 
        # equations for ϕ
        for i=2 : (NUM-1)
            F[i - 1] = (p.ϵ_0 / α)^2 * (1 - x(i)^2)^2 * (U[i+1] - 2*U[i] + U[i-1]) / h^2 + 
                            (1 / α)^2 * (1 - x(i)^2) * (α * p.τ * U[2*NUM + 1] - 2 * p.ϵ_0^2 * x(i)) * (U[i+1] - U[i-1]) / (2 * h) + 
                            U[i] * (1 - U[i]) * (U[i] - 1/2 - m(U[NUM + i], p))
        end

        # equations for T
        # First order
        for i=2 : (NUM-1)
            F[(NUM-2) + i - 1] = 1/α * (1 - x(i)^2) * (U[NUM + i + 1] - U[NUM + i - 1]) / (2 * h) + 
                                        U[2*NUM + 1] * (U[NUM + i] + 1/p.S * U[i] - 1/p.S)
        end

        F[(2*(NUM-2) + 1) : (2*(NUM-2) + 5)] = [
            # B.C. for ϕ
            U[1],
            U[NUM] - 1,
            U[div(NUM, 2) + 1] - 1/2,
            
            # B.C. for T
            U[NUM+1] - 1/p.S,
            U[2*NUM],
        ]
    
    end


    function j!(J::SparseMatrixCSC{Float64, Int64}, U::Vector{Float64})

        interval_start::Float64 = -1.
        interval_end::Float64 = 1.
    
        h::Float64 = (interval_end - interval_start) / (NUM - 1)
        z(i::Int64)::Float64 = interval_start + (i-1) * h
    
        df_dphi_m(i::Int64)::Float64 = (p.ϵ_0 / α)^2 * (1 - z(i)^2)^2 * -2 / h^2 + 
                        (1 - U[i]) * (U[i] - 1/2 - m(U[NUM + i], p)) + -U[i] * (U[i] - 1/2 - m(U[NUM + i], p)) + U[i] * (1 - U[i])
        df_dphi_r(i::Int64)::Float64 = (p.ϵ_0 / α)^2 * (1 - z(i)^2)^2 / h^2 + (1 / α)^2 * (1 - z(i)^2) * (α * p.τ * U[2*NUM + 1] - 2 * p.ϵ_0^2 * z(i)) / (2 * h)
        df_dphi_l(i::Int64)::Float64 = (p.ϵ_0 / α)^2 * (1 - z(i)^2)^2 / h^2 - (1 / α)^2 * (1 - z(i)^2) * (α * p.τ * U[2*NUM + 1] - 2 * p.ϵ_0^2 * z(i)) / (2 * h)
        df_dT(i::Int64)::Float64 = -U[i] * (1 - U[i]) * m_prime(U[NUM + i], p)
        df_dc(i::Int64)::Float64 = (1 - z(i)^2) * p.τ / α * (U[i+1] - U[i-1]) / (2 * h)
    
        dg_dphi::Float64 = U[2*NUM + 1] / p.S
        dg_dT_m::Float64 = U[2*NUM + 1]
        dg_dT_r(i::Int64)::Float64 = 1 / α * (1 - z(i)^2) / (2 * h)
        dg_dT_l(i::Int64)::Float64 = 1 / α * (1 - z(i)^2) / -(2 * h)
        dg_dc(i::Int64)::Float64 = U[NUM + i] + (U[i] - 1) / p.S
    
        A::SparseMatrixCSC = sparse(
            [[i for i=1:NUM-2] ; [i for i=1:NUM-2] ; [i for i=1:NUM-2]],
            [[i for i=1:NUM-2] ; [i for i=2:NUM-1] ; [i for i=3:NUM]],
            [[df_dphi_l(i) for i=2:NUM-1] ; [df_dphi_m(i) for i=2:NUM-1] ; [df_dphi_r(i) for i=2:NUM-1]]
        )
    
        B::SparseMatrixCSC = sparse(
            [i for i=1:NUM-2],
            [i for i=2:NUM-1],
            [df_dT(i) for i=2:NUM-1],
            NUM-2, NUM
        )
    
        df_dc_vec::SparseVector = sparse([df_dc(i) for i=2:NUM-1])
    
        C::SparseMatrixCSC = sparse(
            [i for i=1:NUM-2],
            [i for i=2:NUM-1],
            [dg_dphi for i=2:NUM-1],
            NUM-2, NUM
        )
    
        D::SparseMatrixCSC = sparse(
            [[i for i=1:NUM-2] ; [i for i=1:NUM-2] ; [i for i=1:NUM-2]],
            [[i for i=1:NUM-2] ; [i for i=2:NUM-1] ; [i for i=3:NUM]],
            [[dg_dT_l(i) for i=2:NUM-1] ; [dg_dT_m for i=2:NUM-1] ; [dg_dT_r(i) for i=2:NUM-1]]
        )
         
        dg_dc_vec::SparseVector = sparse([dg_dc(i) for i=2:NUM-1])
    #=
        E = spzeros(5, 2 * NUM + 1)
        E[1, 1] = 1
        E[2, NUM] = 1
        E[3, div(NUM, 2) + 1] = 1
        E[4, NUM + 1] = 1
        E[5, 2 * NUM] = 1
    =#
        E::SparseMatrixCSC = sparse(
            [1, 2, 3, 4, 5], 
            [1, NUM, div(NUM, 2) + 1, NUM + 1, 2 * NUM],
            [1., 1., 1., 1., 1.],
            5, 2 * NUM + 1
        )
    
        J[:, :]::SparseMatrixCSC{Float64, Int64} = sparse([A B df_dc_vec; C D dg_dc_vec; E])
    end

    x = range(-1, 1, length=NUM)

    phi_init_sol = [phi_approx((α * atanh(i)), p) for i in x] 
    T_init_sol = [T_approx((α * atanh(i)), p) for i in x] 
    T_init_sol[1] = T_approx((α * atanh(-1 + 1e-13)), p)
    T_init_sol[end] = T_approx((α * atanh(1 - 1e-13)), p)

    if isnan(T_init_sol[1]) || isnan(T_init_sol[end]) || isnan(phi_init_sol[1]) || isnan(phi_init_sol[end])
        println("Initial guess values contain NaN!")
        return 0
    end

    df = OnceDifferentiable(f!, j!, rand(2*NUM + 1), rand(2*NUM + 1), spzeros(2*NUM + 1, 2*NUM + 1))

    sol = @time nlsolve(
        df,
        [phi_init_sol; T_init_sol; 15.], 
        method = :newton, #autodiff = :forward,
        ftol=1e-8, xtol=1e-16, show_trace=true, iterations=5
    )

    phi_sol = sol.zero[1:NUM]
    T_sol = sol.zero[NUM+1:2*NUM]
    c_computed = sol.zero[end]
    println("Computed velocity c = $(c_computed)")
    
    if save_solution
        write_solution_to_file(output_filename, phi_sol, T_sol, c_computed)
    end

    return phi_sol, T_sol, c_computed, sol

end

params = Params(1.2, 0.005)

const alpha_coef = 2.3
const α = alpha_coef / params.c_sharp_lim   # find appropriate / optimal value !
println("α = $(α)")

NUM = 20001
phi_sol, T_sol, c_computed = calculate_finite_diff(NUM, params; save_solution=false)


function plot_solution()
    x = range(-1, 1, length=NUM)

    plot(
        x, phi_sol,# xlims=(1-1e-5, 1), ylims=(0, 1e-6),
        #ylabel="f(x)",
        xlabel="x",
        label="phi(x)",
        #legend=:bottomleft
    )
    plot!(
        x, T_sol, 
        #ylabel="f(x)",
        xlabel="x",
        label="T(x)",
        #legend=:bottomleft
    )
end

#plot_solution()


#=
f(x) = sin(x[1]) * cos(x[2])

a = SparseVector(ForwardDiff.gradient(f, [1.3, 2.1, 0.33]))'
b = SparseVector(ForwardDiff.gradient(f, [0.23, 4.3, 0.78]))'

mat = [a; b]
=#

#=
jac = spzeros(2*NUM + 1, 2*NUM + 1)
a = [phi_init_coefs; T_init_coefs; 15.]
@code_warntype j_auto!(jac, a)
=#
#=
farr = zeros(2*NUM + 1)
a = [phi_init_coefs; T_init_coefs; 15.]
@code_warntype f!(farr, a)
=#


#=

function j_auto!(J::SparseMatrixCSC{Float64, Int64}, U::Vector{Float64})

    interval_start::Float64 = -1.
    interval_end::Float64 = 1.

    h::Float64 = (interval_end - interval_start) / (NUM - 1)
    x(i::Int64)::Float64 = interval_start + (i-1) * h

    J_1::SparseMatrixCSC{Float64, Int64} = spzeros(NUM-2, (2*NUM+1))
    J_2::SparseMatrixCSC{Float64, Int64} = spzeros(NUM-2, (2*NUM+1))
    J_3::SparseMatrixCSC{Float64, Int64} = spzeros(5, (2*NUM+1))
    # equations for ϕ
    for i=2 : (NUM-1)
        J_1[i - 1, :]::SparseMatrixCSC{Float64, Int64} = sparse(ForwardDiff.gradient(
            U -> (ϵ_0 / α)^2 * (1 - x(i)^2)^2 * (U[i+1] - 2*U[i] + U[i-1]) / h^2 + 
                        (1 / α)^2 * (1 - x(i)^2) * (α * τ * U[2*NUM + 1] - 2 * ϵ_0^2 * x(i)) * (U[i+1] - U[i-1]) / (2 * h) + 
                        U[i] * (1 - U[i]) * (U[i] - 1/2 - m(U[NUM + i])), U
                        ))
        
        
    end

    # equations for T
    # First order
    for i=2 : (NUM-1)
        J_2[i - 1, :]::SparseMatrixCSC{Float64, Int64} = sparse(ForwardDiff.gradient(
            U -> 1/α * (1 - x(i)^2) * (U[NUM + i + 1] - U[NUM + i - 1]) / (2 * h) + 
                                    U[2*NUM + 1] * (U[NUM + i] + 1/S * U[i] - 1/S), U
                                    ))
    end
    
    J_3[1:5, :]::SparseMatrixCSC{Float64, Int64} = sparse([
    # B.C. for ϕ
    SparseVector(ForwardDiff.gradient(U -> U[1], U))';
    SparseVector(ForwardDiff.gradient(U -> U[NUM] - 1, U))';
    SparseVector(ForwardDiff.gradient(U -> U[div(NUM, 2) + 1] - 1/2, U))';
    
    # B.C. for T
    SparseVector(ForwardDiff.gradient(U -> U[NUM+1] - 1/S, U))';
    SparseVector(ForwardDiff.gradient(U -> U[2*NUM], U))';
    ])

    J_temp::SparseMatrixCSC{Float64, Int64} = SparseMatrixCSC{Float64, Int64}([J_1; J_2: J_3])

    J[:, :]::SparseMatrixCSC{Float64, Int64} = J_temp[:, :]
    #J[:, :]::SparseMatrixCSC{Float64, Int64} = sparse([J_1; J_2: J_3])

    #dropzeros!(J)

end

=#