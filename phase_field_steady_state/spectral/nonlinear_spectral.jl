using Printf
using Plots
using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using NLsolve

const S = 1.2         #length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.2  # Stefan number, vary from 0.5 to 2
const ϵ_0 = 0.005     #length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.005
const a_1 = 0.9 * (ϵ_0*100)
const β = 10.0
const τ = 0.0003 * (ϵ_0*100)^2

m(T) = (a_1 / pi) * atan(β * (1 - T))
m_prime(T) = -(a_1 * β / pi) * 1/(1 + (β * (1 - T))^2)

const c_sharp_lim = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))

const alpha_coef = 2.1
const α = alpha_coef / c_sharp_lim   # find appropriate / optimal value !

const NUM = 200

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

# chebyshev_expansion(x) = Σ aᵢ * Tᵢ(x)
function chebyshev_expansion(a_i, x)
    res = 0.
    for i=0:( length(a_i) - 1 )
        res += a_i[i+1] * chebyshevt(i, x)
    end
    return res
end

#cheb_nodes = [cos(pi/num_of_nodes * (1/2 + k)) for k=0:num_of_nodes-1]
function calculate_cheb_colloc_expansion_coeffs(test_function::Function, N; dist_from_boundary=1e-13)
    nodes, _ = gausschebyshev(N-2)

    f_rhs::Vector{Float64} = [[test_function(nodes[i]) for i in eachindex(nodes)]; test_function(-1+dist_from_boundary); test_function(1-dist_from_boundary)]

    A::Matrix{Float64} = Matrix{Float64}(undef, N::Int64, N::Int64)
    for i in eachindex(nodes)
        A[i, :] = [chebyshevt(k, nodes[i]) for k=0:N-1]
    end
    A[N-1, :] = [chebyshevt(k, -1) for k=0:N-1]
    A[N, :] = [chebyshevt(k, 1) for k=0:N-1]

    a_i::Vector{Float64} = A \ f_rhs

    return a_i
end


function f!(F, x)

    nodes, _ = gausschebyshev(NUM - 2)

    F[1:5] = [
        # B.C. for ϕ
        x[1 : NUM]' * [chebyshevt(k, -1) for k=0:NUM-1],
        x[1 : NUM]' * [chebyshevt(k, 1) for k=0:NUM-1] - 1,
        x[1 : NUM]' * [chebyshevt(k, 0) for k=0:NUM-1] - 1/2,
        
        # B.C. for T
        x[NUM+1 : 2*NUM]' * [chebyshevt(k, -1) for k=0:NUM-1] - 1/S,
        x[NUM+1 : 2*NUM]' * [chebyshevt(k, 1) for k=0:NUM-1],
    ]

    # equations for ϕ
    #=
    for i=1 : NUM-2 #6:(6 - 1 + NUM-2)
        F[5 + i] = -(ϵ_0 / α)^2 * (1 - nodes[i]^2) * (x[1 : NUM]' * [k * ((k+1) * chebyshevt(k, nodes[i]) - chebyshevu(k, nodes[i])) for k=0:NUM-1]) +
                    #1/α^2 * (1 - nodes[i]^2) * (α * τ * x[2*NUM + 1] - 2 * ϵ_0^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                    (1 - nodes[i]^2) * (τ/α * x[2*NUM + 1] - 2 * (ϵ_0 / α)^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                    (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) * (1 - x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) *
                    (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1] - 1/2 - m(x[NUM+1 : 2*NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]))
    end
    =#
    Threads.@threads for i=1 : NUM-2 #6:(6 - 1 + NUM-2)
        F[5 + i] = -ϵ_0^2 * (1 - nodes[i]^2) * (x[1 : NUM]' * [k * ((k+1) * chebyshevt(k, nodes[i]) - chebyshevu(k, nodes[i])) for k=0:NUM-1]) +
                    #1/α^2 * (1 - nodes[i]^2) * (α * τ * x[2*NUM + 1] - 2 * ϵ_0^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                    (1 - nodes[i]^2) * (α * τ * x[2*NUM + 1] - 2 * ϵ_0^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                    α^2 * (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) * (1 - x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) *
                    (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1] - 1/2 - m(x[NUM+1 : 2*NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]))
    end

    # equations for T
    # First order
    Threads.@threads for i=1 : NUM-2
        F[5 + NUM-2 + i] = 1/α * (1 - nodes[i]^2) * (x[NUM+1 : 2*NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                            x[2*NUM + 1] * (x[NUM+1 : 2*NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) +
                            x[2*NUM + 1] / S * (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) - x[2*NUM + 1] / S 
    end
    #=
    # Second order
    for i=1 : NUM-2
        F[5 + NUM-2 + i] = -(1 - nodes[i]^2) * (x[NUM+1 : 2*NUM]' * [k * ((k+1) * chebyshevt(k, nodes[i]) - chebyshevu(k, nodes[i])) for k=0:NUM-1]) + 
                            (1 - nodes[i]^2) * (α * x[2*NUM + 1] - 2 * nodes[i]) * (x[NUM+1 : 2*NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                            x[2*NUM + 1] * α / S * (1 - nodes[i]^2) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]])
    end
    =#
end

phi_init_coefs = calculate_cheb_colloc_expansion_coeffs(x -> phi_init_band((α * atanh(x))), NUM)
T_init_coefs = calculate_cheb_colloc_expansion_coeffs(x -> T_composite_solution((α * atanh(x))), NUM)

if isnan(T_init_coefs[1]) || isnan(phi_init_coefs[1])
    println("Initial guess values contain NaN!")
end

sol = @time nlsolve(f!, [phi_init_coefs; T_init_coefs; 15.], autodiff = :forward, method = :newton,
        ftol=1e-13, xtol=1e-16, show_trace=true)#, iterations=50)    
    
x = range(-1, 1, length=Int(1e4))

phi_computed(x) = chebyshev_expansion(sol.zero[1:NUM], x)
T_computed(x) = chebyshev_expansion(sol.zero[NUM+1:2*NUM], x)

println("Number of terms = $(NUM)")
println("α = $(α)")
println("Computed velocity c = $(sol.zero[end])")

plot(
    x, x -> phi_computed(x),# xlims=(1-1e-5, 1), ylims=(0, 1e-6),
    #ylabel="f(x)",
    xlabel="x",
    label="phi(x)",
    #legend=:bottomleft
)
plot!(
    x, x -> T_computed(x), 
    #ylabel="f(x)",
    xlabel="x",
    label="T(x)",
    #legend=:bottomleft
)

#=
plot!(
    x, x -> phi_init_band(α * atanh(x)),
    #ylabel="f(x)",
    xlabel="x",
    label="T(x)",
    #legend=:bottomleft
)
=#

#=

eta_span = range(-0.5, 0.5, length=Int(1e4))
plot(
    eta_span, eta -> T_computed(tanh(eta/α)), 
    #ylabel="f(x)",
    xlabel="x",
    label="T(x)",
    #yaxis=:log,
    #legend=:bottomleft
)
plot!(
    eta_span, eta -> T_composite_solution(eta), #, c=sol.zero[end]
    #ylabel="f(x)",
    xlabel="x",
    label="T(x) approximate",
    #legend=:bottomleft
)

plot!(
    eta_span, eta -> phi_computed(tanh(eta/α)), 
    #ylabel="f(x)",
    xlabel="x",
    label="phi(x)",
    #legend=:bottomleft
)
plot!(
    eta_span, eta -> phi_init_band(eta), #, c=sol.zero[end]
    #ylabel="f(x)",
    xlabel="x",
    label="phi(x) approximate",
    #legend=:bottomleft
)

#savefig("/Users/shamilmagomedov/Desktop/spectral_plot.pdf")

=#