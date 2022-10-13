using Printf
using Plots
using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using NLsolve

S = 1.2         #length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.2  # Stefan number, vary from 0.5 to 2
ϵ_0 = 0.005     #length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.005
a_1 = 0.9 * (ϵ_0*100)
β = 10.0
τ = 0.0003 * (ϵ_0*100)^2

m(T) = (a_1 / pi) * atan(β * (1 - T))
m_prime(T) = -(a_1 * β / pi) * 1/(1 + (β * (1 - T))^2)

c_sharp_lim = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))

alpha_coef = 2.1
α = alpha_coef / c_sharp_lim   # find appropriate / optimal value !


NUM = 1000

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


function f!(F, x)


    interval_start = -1.
    interval_end = 1.

    h = (interval_end - interval_start) / (NUM - 1)

    # equations for ϕ
    Threads.@threads for i=2 : (NUM-1)
        F[i - 1] = (ϵ_0 / α)^2 * (1 - (interval_start + (i-1) * h)^2)^2 * (x[i+1] - 2*x[i] + x[i-1]) / h^2 + 
                        (1 / α)^2 * (1 - (interval_start + (i-1) * h)^2) * (α * τ * x[2*NUM + 1] - 2 * ϵ_0^2 * (interval_start + (i-1) * h)) * (x[i+1] - x[i-1]) / (2 * h) + 
                        x[i] * (1 - x[i]) * (x[i] - 1/2 - m(x[NUM + i]))
    end

    # equations for T
    # First order
    Threads.@threads for i=2 : (NUM-1)
        F[(NUM-2) + i - 1] = 1/α * (1 - (interval_start + (i-1) * h)^2) * (x[NUM + i + 1] - x[NUM + i - 1]) / (2 * h) + 
                                    x[2*NUM + 1] * (x[NUM + i] + 1/S * x[i] - 1/S)
    end

    F[(2*(NUM-2) + 1) : (2*(NUM-2) + 5)] = [
        # B.C. for ϕ
        x[1],
        x[NUM] - 1,
        x[div(NUM, 2) + 1] - 1/2,
        
        # B.C. for T
        x[NUM+1] - 1/S,
        x[2*NUM],
    ]
  
end

x = range(-1, 1, length=NUM)

phi_init_coefs = phi_init_band.((α * atanh.(x)))
T_init_coefs = T_composite_solution.((α * atanh.(x)))
T_init_coefs[1] = T_composite_solution((α * atanh(-1 + 1e-13)))
T_init_coefs[end] = T_composite_solution((α * atanh(1 - 1e-13)))

if isnan(T_init_coefs[1]) || isnan(phi_init_coefs[1])
    println("Initial guess values contain NaN!")
end

sol = @time nlsolve(f!, [phi_init_coefs; T_init_coefs; 15.], autodiff = :forward, method = :newton,
        ftol=1e-10, xtol=1e-16, show_trace=true)#, iterations=50)    

println("Number of terms = $(NUM)")
println("α = $(α)")
println("Computed velocity c = $(sol.zero[end])")

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