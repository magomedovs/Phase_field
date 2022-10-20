using Printf
using Plots
using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using NLsolve
using ForwardDiff
using SparseArrays


const S = 1.2         #length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.2  # Stefan number, vary from 0.5 to 2
const ϵ_0 = 0.005     #length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.005
const a_1 = 0.9 * (ϵ_0*100)
const β = 10.0
const τ = 0.0003 * (ϵ_0*100)^2

m(T) = (a_1 / pi) * atan(β * (1 - T))
m_prime(T) = -(a_1 * β / pi) * 1/(1 + (β * (1 - T))^2)

const c_sharp_lim = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))

const alpha_coef = 2.3
const α = alpha_coef / c_sharp_lim   # find appropriate / optimal value !

const NUM = length(ARGS) >= 1 ? parse(Int64, ARGS[1]) : 20001

println("Number of terms = $(NUM)")
println("α = $(α)")

phi_init_band(x; shift=0)::Float64 = (tanh((x - shift) / (ϵ_0 * 2 * sqrt(2))) + 1 ) / 2

function T_composite_solution(x; shift=0, c=c_sharp_lim)::Float64    
    function T_outer_solution(x, shift)
        if (x <= shift)
            return 1/S
        else
            return 1/S * exp((-x + shift) * c)
        end
    end

    function T_inner_solution(x, shift)::Float64
        return 1/S + ϵ_0 * (-(c * sqrt(2) / S) * log(2) - (c * sqrt(2) / S) * log(cosh((x - shift)/(ϵ_0 * 2^(3/2)))) + ((x - shift) / ϵ_0) * ( - c / (2 * S)) )
    end

    if (x <= shift)
        return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S
    else
        return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S + (x - shift) * (c / S)
    end
end


function f!(F, U)

    interval_start = -1.
    interval_end = 1.

    h = (interval_end - interval_start) / (NUM - 1)
    x(i) = interval_start + (i-1) * h

    #Threads.@threads 
    # equations for ϕ
    for i=2 : (NUM-1)
        F[i - 1] = (ϵ_0 / α)^2 * (1 - x(i)^2)^2 * (U[i+1] - 2*U[i] + U[i-1]) / h^2 + 
                        (1 / α)^2 * (1 - x(i)^2) * (α * τ * U[2*NUM + 1] - 2 * ϵ_0^2 * x(i)) * (U[i+1] - U[i-1]) / (2 * h) + 
                        U[i] * (1 - U[i]) * (U[i] - 1/2 - m(U[NUM + i]))
    end

    # equations for T
    # First order
    for i=2 : (NUM-1)
        F[(NUM-2) + i - 1] = 1/α * (1 - x(i)^2) * (U[NUM + i + 1] - U[NUM + i - 1]) / (2 * h) + 
                                    U[2*NUM + 1] * (U[NUM + i] + 1/S * U[i] - 1/S)
    end

    F[(2*(NUM-2) + 1) : (2*(NUM-2) + 5)] = [
        # B.C. for ϕ
        U[1],
        U[NUM] - 1,
        U[div(NUM, 2) + 1] - 1/2,
        
        # B.C. for T
        U[NUM+1] - 1/S,
        U[2*NUM],
    ]
  
end

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


function j!(J::SparseMatrixCSC{Float64, Int64}, U::Vector{Float64})
#U = rand(2*NUM + 1, 2*NUM + 1)
    interval_start::Float64 = -1.
    interval_end::Float64 = 1.

    h::Float64 = (interval_end - interval_start) / (NUM - 1)
    z(i::Int64)::Float64 = interval_start + (i-1) * h

    df_dphi_m(i::Int64)::Float64 = (ϵ_0 / α)^2 * (1 - z(i)^2)^2 * -2 / h^2 + 
                    (1 - U[i]) * (U[i] - 1/2 - m(U[NUM + i])) + -U[i] * (U[i] - 1/2 - m(U[NUM + i])) + U[i] * (1 - U[i])
    df_dphi_r(i::Int64)::Float64 = (ϵ_0 / α)^2 * (1 - z(i)^2)^2 / h^2 + (1 / α)^2 * (1 - z(i)^2) * (α * τ * U[2*NUM + 1] - 2 * ϵ_0^2 * z(i)) / (2 * h)
    df_dphi_l(i::Int64)::Float64 = (ϵ_0 / α)^2 * (1 - z(i)^2)^2 / h^2 - (1 / α)^2 * (1 - z(i)^2) * (α * τ * U[2*NUM + 1] - 2 * ϵ_0^2 * z(i)) / (2 * h)
    df_dT(i::Int64)::Float64 = -U[i] * (1 - U[i]) * m_prime(U[NUM + i])
    df_dc(i::Int64)::Float64 = (1 - z(i)^2) * τ / α * (U[i+1] - U[i-1]) / (2 * h)

    dg_dphi::Float64 = U[2*NUM + 1] / S
    dg_dT_m::Float64 = U[2*NUM + 1]
    dg_dT_r(i::Int64)::Float64 = 1 / α * (1 - z(i)^2) / (2 * h)
    dg_dT_l(i::Int64)::Float64 = 1 / α * (1 - z(i)^2) / -(2 * h)
    dg_dc(i::Int64)::Float64 = U[NUM + i] + (U[i] - 1) / S

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

phi_init_coefs = phi_init_band.((α * atanh.(x)))
T_init_coefs = T_composite_solution.((α * atanh.(x)))
T_init_coefs[1] = T_composite_solution((α * atanh(-1 + 1e-13)))
T_init_coefs[end] = T_composite_solution((α * atanh(1 - 1e-13)))

if isnan(T_init_coefs[1]) || isnan(phi_init_coefs[1])
    println("Initial guess values contain NaN!")
end

df = OnceDifferentiable(f!, j!, rand(2*NUM + 1), rand(2*NUM + 1), spzeros(2*NUM + 1, 2*NUM + 1))

#@profview 

sol = @time nlsolve(df, [phi_init_coefs; T_init_coefs; 15.], method = :newton,
        ftol=1e-8, xtol=1e-16, show_trace=true, iterations=5)

c_computed = sol.zero[end]
println("Computed velocity c = $(c_computed)")

#=
# Writing computed data into file
io = open("/Users/shamilmagomedov/Desktop/finite_diff_calculated_c.txt", "a")

write(io, "$(2. / (NUM - 1)) $c_computed $c_sharp_lim $S $ϵ_0\n")

close(io)
=#

#=
plot(
    x, sol.zero[1:NUM],# xlims=(1-1e-5, 1), ylims=(0, 1e-6),
    #ylabel="f(x)",
    xlabel="x",
    label="phi(x)",
    #legend=:bottomleft
)
plot!(
    x, sol.zero[NUM+1:2*NUM], 
    #ylabel="f(x)",
    xlabel="x",
    label="T(x)",
    #legend=:bottomleft
)
=#

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