using Printf
using LinearAlgebra
using Plots
using Distributions
#using TimerOutputs
using Profile
using ProgressMeter
using DelimitedFiles
#using NLsolve

include("types.jl")
include("set_initial_conditions_Kobayashi.jl")
include("save_and_get_matrices.jl")
include("FVM_explicit_step.jl")
include("ADI_implicit_step.jl")
include("stop_function.jl")
include("functions.jl")
include("calculate.jl")
include("calculate_with_shifting.jl")

#===========================================================================#
# Stochastic term (matrix from which the values will be taken during the calculation)
hi() = rand(Uniform(-1, 1), 1)

θ(p::Float64, q::Float64)::Float64 = atan(q , p)
ϵ(p::Float64, q::Float64)::Float64 = ϵ_0 * (1 + δ * cos(j*(θ(p, q) - θ_0)))
ϵ_prime(p::Float64, q::Float64)::Float64 = - ϵ_0 * δ * j * sin(j*(θ(p, q) - θ_0))

# ∂U/∂t = ∇ ⋅ [ ∇J(U) ] + R
# R - reaction term
#R(phi::Matrix{Float64}, T::Matrix{Float64}, i::Int64, j::Int64)::Float64 = phi[i, j] * (1 - phi[i, j]) * (phi[i, j] - 1/2 - ( (a_1/π) * atan(β * (1 - T[i, j])) ) ) # + a_2/2 * hi()[1] ) )
phi_Source_Term(phi::Matrix{Float64}, T::Matrix{Float64})::Matrix{Float64} = @. (1/τ) * phi * (1 - phi) * (phi - 1/2 - ( (a_1/π) * atan(β * (1 - T)) ) ) # + a_2/2 * hi()[1] ) )
T_Source_Term(old_phi::Matrix{Float64}, new_phi::Matrix{Float64}, Δt::Float64)::Matrix{Float64} = @.  - (1 / S) * (new_phi - old_phi) / Δt

# здесь аргументы ddx и ddy это функции, которые будут выдавать производную в точке (конечная разность)
# ∂U/∂t = ∇ ⋅ [ ∇J(U) ] + R
# ∇J = (J_x, J_y)  -  векторная функция
phi_x(phi::Matrix{Float64}, i::Int64, j::Int64, ddx::Function, ddy::Function)::Float64 = ((ddx(phi, i, j))^2 + (ddy(phi, i, j))^2)^2 < eps(Float64) ? zero(Float64) : (1/τ) * ϵ(ddx(phi, i, j), ddy(phi, i, j)) * (ddx(phi, i, j) * ϵ(ddx(phi, i, j), ddy(phi, i, j)) - ddy(phi, i, j) * ϵ_prime(ddx(phi, i, j), ddy(phi, i, j)))
phi_y(phi::Matrix{Float64}, i::Int64, j::Int64, ddx::Function, ddy::Function)::Float64 = ((ddx(phi, i, j))^2 + (ddy(phi, i, j))^2)^2 < eps(Float64) ? zero(Float64) : (1/τ) * ϵ(ddx(phi, i, j), ddy(phi, i, j)) * (ddy(phi, i, j) * ϵ(ddx(phi, i, j), ddy(phi, i, j)) + ddx(phi, i, j) * ϵ_prime(ddx(phi, i, j), ddy(phi, i, j)))

T_x(T::Matrix{Float64}, i::Int64, j::Int64, ddx::Function, ddy::Function)::Float64 = D * ddx(T, i, j)
T_y(T::Matrix{Float64}, i::Int64, j::Int64, ddx::Function, ddy::Function)::Float64 = D * ddy(T, i, j)

#===========================================================================#
const tol = 2 * 10^(-12)   # tolerance for my_stop_func and another functions

#===========================================================================#
const D = 1  # thermal diffusivity in Heat/Diffusion equation (also included in ADI_implicit_step.jl)

const S = 1.2 # Stefan number, vary from 0.5 to 2
const ϵ_0 = 0.005
const δ = 0.0 # 0.04

const θ_0 = π/2
const j = 6  # 6
const a_1 = 0.9 * (ϵ_0*100)
const a_2 = 0.01
const β = 10.0
const τ = 0.0003 * (ϵ_0*100)^2

H1 = 5.0   # Domain_side_length
M1 = 1001    # количество отрезков разбиения области вдоль оси x и y (можем выбирать)
Time_Period1 = 0.02      # период наблюдения (можем выбирать)

# Создаем структуру типа Discretization
scheme = create_Discretization(scheme_type=Explicit(), domain_side_length=H1, mesh_number=M1, time=Time_Period1)

#===========================================================================#
x = range(-H1/2, H1/2, length = M1)    # значит длина шага будет H / (M-1)
y = copy(x)

# creating initial conditions matrices
init_phi = ones(M1, M1)
init_T = zeros(M1, M1)
A = 0.0 #0.02     # amplitude
k = 20.0      # wave number

set_initial_phi!(init_phi, H1, M1; smoothness="continious", shape="band", band_height=0.5, amplitude=A, wave_number=k)
set_initial_T!(init_T, H1, M1; smoothness="continious", band_height=0.5, amplitude=A, wave_number=k)

#init_phi = get_matrix("/Users/admin/Desktop/init_1/phi_501_t0.6_S1.2_d0_e0.01.txt")
#init_T = get_matrix("/Users/admin/Desktop/init_1/T_501_t0.6_S1.2_d0_e0.01.txt")

# array of matrices of initial conditions
init_cond_matrices = [init_phi, init_T]

# Устанавливаем граничные условия на верхней и нижней границах области.
# Граничные условия "Dirichlet" будут зафиксированы такими, какие установлены в начальном условии.
# Граничные условия "Neumann" -- адиабатические стенки (нет потока через границу).
Top_bc = Neumann()  # "Dirichlet" or "Neumann"
Bottom_bc = Neumann()

# массив моментов времени, в которые хотим сохранить решение в файл
time_moments = []#[0.01; [i for i=0.05:0.05:scheme.time]] # [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, Time_Period]

# @time
# Juno.@profiler

#phi, T, time_end = calculate(scheme, init_cond_matrices; time_moments_to_make_shots=time_moments, number_of_shots_per_time_moment=3, top_bc=Top_bc, bottom_bc=Bottom_bc, stop_function=my_stop_func)

phi, T, time_end = calculate_with_shifting(scheme, init_cond_matrices; time_moments_to_make_shots=time_moments, number_of_shots_per_time_moment=1, top_bc=Top_bc, bottom_bc=Bottom_bc, stop_function=my_stop_func)


#println(time_end)

#save_two_matrices(phi, T; my_str="")

#=
open(string("phi_", M, "_t", Time_Period, "_S", S, "_d", δ, "_e", ϵ_0 ,".txt"), "w") do io
    writedlm(io, phi)
end

open(string("T_", M, "_t", Time_Period, "_S", S, "_d", δ, "_e", ϵ_0 ,".txt"), "w") do io
    writedlm(io, T)
end
=#

#=
open("phi_0.txt", "w") do io
    writedlm(io, phi)
end

open("T_0.txt", "w") do io
    writedlm(io, T)
end
=#

# Writing data into the file.

# https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/
#open("delim_file.txt", "w") do io
#    writedlm(io, [x y])
#end

#matrix = readdlm("delim_file.txt", '\t', Float64, '\n')
