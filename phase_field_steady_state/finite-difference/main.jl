using Printf
using Plots
using LinearAlgebra
using LinearSolve
using SparseArrays
using DataFrames
using CSV
using DelimitedFiles
using Interpolations
#using ImplicitEquations
#using Statistics
#using ProfileView
#using TimerOutputs     # https://github.com/KristofferC/TimerOutputs.jl

include("init_cond_functions.jl")
include("functions_Newton_Raphson.jl")
include("save_data.jl")
include("linear_sys_eigenvals.jl")

H = 10     # domain size
S = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.2 #1.1   # Stefan number, vary from 0.5 to 2
ϵ_0 = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.005 #0.0025
a_1 = 0.9 * (ϵ_0*100)
β = 10.0
τ = 0.0003 * (ϵ_0*100)^2

m(T) = (a_1 / pi) * atan(β * (1 - T))
m_prime(T) = -(a_1 * β / pi) * 1/(1 + (β * (1 - T))^2)

M = length(ARGS) >= 3 ? parse(Int64, ARGS[3]) : 100001    # number of mesh points for Newton--Raphson. Must be odd number
Δh = H / (M - 1)

α = (ϵ_0 / Δh)^2
γ = τ / (2 * Δh)
dim = (M - 2)

x = range(-H/2, H/2, length = M)

Interface_index = div(M, 2) + 1   # interface location index

# boundary conditions:
phi_1 = 0
phi_I = 1/2
phi_M = 1
T_1 = 1/S
T_M = T_composite_solution(x[end], 0)  # 0

# initially set arrays
init_phi = [phi_init_band(i, 0) for i=x]
init_phi[Interface_index] = phi_I

init_T = [T_composite_solution(i, 0) for i=x]
init_c = 10

println("S = ", S)
println("ϵ_0 = ", ϵ_0)
println("M = ", M)
#println(length(ARGS))

# @time
phi, T, c = @time solve_NR(init_phi, init_T, init_c)
#@profview main_NR(init_phi, init_T, init_c)

#save_two_matrices(phi, T)
#save_data(c)

println("Computed c = ", c)
println()

c_sharp_lim = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))    # velocity of the wave derived analytically
println("Velocity from sharp interface limit : ", c_sharp_lim)

plot([x, x], [init_phi, init_T], label=["approximate ϕ" "approximate T"], legend=:right)
plot!([x, x], [phi, T], label=["computed ϕ" "computed T"], legend=:right)

#phi_new = [phi_init_band(i, 0) for i=x]
#T_new = [T_composite_solution(i, 0, c=c_extrap) for i=x]
#plot!([x, x], [phi_new, T_new], label=["ϕ ext" "T ext"], legend=:right)

#=
# Writing computed data into file
io = open("/Users/shamilmagomedov/Desktop/calculated_c.txt", "a")

write(io, "$Δh $c $c_sharp_lim $S $ϵ_0\n")

close(io)
=#

#==

# Calculation of eigenvalues of the linear system

# Creating interpolants
phi_itp = interpolate(phi, BSpline(Cubic(Natural(OnGrid()))));
phi_sitp = scale(phi_itp, x);

T_itp = interpolate(T, BSpline(Cubic(Natural(OnGrid()))));
T_sitp = scale(T_itp, x);

NUM = 3001

# ArnoldiMethod
k_test = 1.0e1
λs_test, X_test = solve_eigen(phi_sitp, T_sitp, c, k_test, NUM)
u = Real.(X_test[1:NUM-2, findmax(map(x->x.re, λs_test))[2]])
v = Real.(X_test[NUM-1:2*(NUM-2), findmax(map(x->x.re, λs_test))[2]])
plot(range(-H/2, H/2, length = NUM-2), [u, v], label=["u" "v"], xlabel="η", title="k=$k_test")

λs_test_max_re = findmax(map(x->x.re, λs_test))[1]

front_vel = c_sharp_lim

C0 = u[div(NUM-2, 2) + 1] * 4 * sqrt(2)
D2 = v[div(NUM-2, 2) + 1] / ϵ_0 # - sign(v[div(NUM-2, 2) + 1]) * ϵ_0 * front_vel  # ??? guessed parameter value  # front_vel^2 / (2 * S * sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2)) * C0

B1 = D2 - C0 * front_vel / (2 * S)
A1 = D2 + C0 * front_vel / (2 * S)

mu_left = (-front_vel + sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2))/2
mu_right = (-front_vel - sqrt(4*(λs_test_max_re + k_test^2) + front_vel^2))/2

u_composite(x) = C0 / (4 * sqrt(2) * cosh(x / (ϵ_0 * 2 * sqrt(2)))^2)
plot(range(-H/2, H/2, length = NUM-2), u, label="u", legend=:topleft) #, xlims=(-H/2, -0.2), ylims=(-0.005, 0.005)
plot!(x, x->u_composite(x), style=:solid, label="u asymptotic solution,\n ϵ_0=$ϵ_0")

function v_composite(x)
    if x>=0
        return ϵ_0 * (B1 * exp(mu_right * x) - C0 * front_vel / (2 * S) * tanh(x / (ϵ_0 * 2 * sqrt(2))) + C0 * front_vel / (2 * S))
    else
        return ϵ_0 * (A1 * exp(mu_left * x) - C0 * front_vel / (2 * S) * tanh(x / (ϵ_0 * 2 * sqrt(2))) - C0 * front_vel / (2 * S))
    end
end

plot(range(-H/2, H/2, length = NUM-2), v, label="v computed", legend=:topleft)#, xlims=(0.499, 0.5), ylims=(-1e-8, 1e-10)) #, xlims=(-H/2, -1.), ylims=(-1e-5, 1e-5)) #, xlims=(-H/2, -0.2), ylims=(-0.005, 0.005)
plot!(x, x->v_composite(x), style=:solid, label="v asymptotic solution,\n ϵ_0=$ϵ_0")#, xlims=(-0.01, 0.01))#, ylims=(-1e-5, 1e-5))
#savefig("/Users/shamilmagomedov/Desktop/plots_uv/v_computed_zoomed.pdf")
#savefig("/Users/shamilmagomedov/Desktop/plots_uv/v_asympt.pdf")

v_asympt(mu, some_const, x) = some_const * exp(mu * x)
plot!(range(-H/2, 0, length = NUM-2), x->v_asympt(mu_left, ϵ_0 * A1, x), style = :dashdot, label="const * exp($(round(mu_left; digits=3))x)")
plot!(range(0.05, H/2, length = NUM-2), x->v_asympt(mu_right, ϵ_0 * B1, x), style = :dashdot, label="const * exp($(round(mu_right; digits=3))x)")

# yaxis=:log plot of v and asympt on the left side
v_itp = interpolate(v, BSpline(Cubic(Natural(OnGrid()))));
v_sitp = scale(v_itp, range(-H/2, H/2, length = NUM-2));
plot(range(-H/2, H/2, length = NUM-2), abs.(v_sitp), legend=:bottomright, yaxis=:log, xlim=(-H/2, 0), ylim=(1e-5, 1e-1))#, shape=:circle)
plot!(range(-H/2, H/2, length = NUM-2), x->abs(v_asympt(mu_left, ϵ_0 * A1, x)), style = :dashdot, label="const * exp($(round(mu_left; digits=3))x)")#, yaxis=:log)
#plot!(range(-H/2, H/2, length = NUM-2), x->abs(v_composite(x)), style=:solid, label="v asymptotic solution,\n ϵ_0=$ϵ_0", yaxis=:log)

#savefig("/home/shamil/Desktop/plot_v_asympt_k_" * "$k_test" * "_eps_" * "$ϵ_0" * ".pdf")
#savefig("/home/shamil/Desktop/eigenfunc_NUM_2001_eps_0.0025.pdf")

==#

# Arpack
#λ, ϕ = solve_eigen(phi_sitp, T_sitp, c, 0.5, NUM)
#plot(Real.(ϕ[1:NUM, 1]))
#plot!(Real.(ϕ[NUM:end, 1]))

#=
# plot (k, lambda)

#span_start = -5
#span_end = 1.5
#span_num = 20 # span_end - span_start + 1
#span = 10 .^ range(span_start, span_end, length=span_num)
span = range(0.001, 20, length=40)

lambda_arr = zeros(length(span))
for i in eachindex(span)
    λs, X = solve_eigen(phi_sitp, T_sitp, c, span[i], NUM)
    lambda_arr[i] = findmax(map(x->x.re, λs))[1]   # λs[end].re
end
plot(span, lambda_arr, shape=:circle, xaxis=:log, legend=:bottomleft, label="λ(k), c=$c", xlabel="k", ylabel="λ")

#savefig("/home/shamil/Desktop/eigenval_NUM_2001_eps_0.0025.pdf")

=#

#=

# plot (NUM, lambda)
span = 1000 .* [1, 2, 4] #[1000; 3000; 5000; 7000; 1000 .* range(10, 50, step=10)] #range(1, 50, length=50)
k1 = 1.0e-1

lambda_arr = zeros(length(span))
for i in eachindex(span)
    λs, X = solve_eigen(phi_sitp, T_sitp, c, k1, Int(span[i]))
    lambda_arr[i] = findmax(map(x->x.re, λs))[1]   # λs[end].re
end
plot(1 ./ span[1:end], lambda_arr[1:end], xaxis=:log, yaxis=:log, shape=:circle, legend=:topleft, xlabel="h", ylabel="λ", label="k = $k1")

inds = [1, 2, 3]

h_list = [1/span[i] for i in inds]
c_list = [lambda_arr[i] for i in inds]
res_estim = extrapolate_c(h_list, c_list)
lambda_ext = res_estim[1]
h2_coef_lam = res_estim[2]

# Plot of lambda(h, k1) with extrapolated value of lambda
errors_arr = abs.(lambda_arr .- lambda_ext) / lambda_ext

plot(1 ./ span, errors_arr,
    label="|lambda(h) - lambda_ext| / lambda_ext", xlabel="h", ylabel="rel error", legend=:topleft, xaxis=:log, yaxis=:log, 
    yformatter=:scientific, shape=:circle)

plot!(range(findmin(1 ./ span)[1], findmax(1 ./ span)[1], length=Int(1e5)),
    x -> (h2_coef_lam * x^2 / lambda_ext), xaxis=:log, yaxis=:log,
     label="$(round(h2_coef_lam, digits=10)) / lambda_ext * h^2")

=#

#=
new_span = span[1:end-1]
new_lambda_arr = lambda_arr[1:end-1]
errors_arr = [abs(new_lambda_arr[i] - lambda_arr[end]) / lambda_arr[end] for i=1:length(new_lambda_arr)]
plot(1 ./ new_span, errors_arr, xaxis=:log, yaxis=:log, shape=:circle, legend=:topleft, xlabel="h", ylabel="error", label="k = $k1")
=#

#=
x0 = span[end]
y0 = lambda_arr[end]
a = x0 - span[end-1]
b = y0 - lambda_arr[end-1]
line_eq(x) = y0 + b/a * (x - x0)
plot!(span[1:end], x -> line_eq(x), xaxis=:log, yaxis=:log)
=#

#best_val, X = solve_eigen(phi_sitp, T_sitp, c, k1, 5000)
#plot(Real.(X[1:5000, findmax(map(x->x.re, best_val))[2]]))
#plot!(Real.(X[5000:end, findmax(map(x->x.re, best_val))[2]]))


#plot(range(1e-4, 3 * 1e1, length=Int(1e4)), x -> x^(2), xaxis=:log, yaxis=:log)
#plot!(range(1e-4, 3 * 1e1, length=Int(1e4)), x -> exp(3x), yaxis=:log)

