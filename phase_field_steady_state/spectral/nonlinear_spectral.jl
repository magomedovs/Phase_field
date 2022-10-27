using Printf
using Plots
using FastGaussQuadrature
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using NLsolve
using DelimitedFiles

include("Cheb_expansion.jl")
include("../structs_and_functions.jl")

function calculate_spectral(NUM::Int64, p::Params; save_solution=false, output_filename="spectral_solution_coeffs_NUM_$(NUM).txt")

    println("Number of terms = $(NUM)")

    function f!(F, x)#, p::Params)

        nodes, _ = gausschebyshev(NUM - 2)

        F[1:5] = [
            # B.C. for ϕ
            x[1 : NUM]' * [chebyshevt(k, -1) for k=0:NUM-1],
            x[1 : NUM]' * [chebyshevt(k, 1) for k=0:NUM-1] - 1,
            x[1 : NUM]' * [chebyshevt(k, 0) for k=0:NUM-1] - 1/2,
            
            # B.C. for T
            x[NUM+1 : 2*NUM]' * [chebyshevt(k, -1) for k=0:NUM-1] - 1/p.S,
            x[NUM+1 : 2*NUM]' * [chebyshevt(k, 1) for k=0:NUM-1],
        ]

        # equations for ϕ
        Threads.@threads for i=1 : NUM-2
            F[5 + i] = -p.ϵ_0^2 * (1 - nodes[i]^2) * (x[1 : NUM]' * [k * ((k+1) * chebyshevt(k, nodes[i]) - chebyshevu(k, nodes[i])) for k=0:NUM-1]) +
                        #1/α^2 * (1 - nodes[i]^2) * (α * p.τ * x[2*NUM + 1] - 2 * p.ϵ_0^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                        (1 - nodes[i]^2) * (α * p.τ * x[2*NUM + 1] - 2 * p.ϵ_0^2 * nodes[i]) * (x[1 : NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                        α^2 * (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) * (1 - x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) *
                        (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1] - 1/2 - m(x[NUM+1 : 2*NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1], p))
        end

        # equations for T
        Threads.@threads for i=1 : NUM-2
            F[5 + NUM-2 + i] = (1 - nodes[i]^2) * (x[NUM+1 : 2*NUM]' * [0; [k * chebyshevu(k-1, nodes[i]) for k=1:NUM-1]]) +
                                α * x[2*NUM + 1] * (x[NUM+1 : 2*NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) +
                                α * x[2*NUM + 1] / p.S * (x[1 : NUM]' * [chebyshevt(k, nodes[i]) for k=0:NUM-1]) - 
                                α * x[2*NUM + 1] / p.S 
        end
    end

    #f!(F, x) = f!(F, x, p)

    phi_init_coefs = calculate_cheb_colloc_expansion_coeffs(x -> phi_approx((α * atanh(x)), p), NUM; dist_from_boundary=1e-13)
    T_init_coefs = calculate_cheb_colloc_expansion_coeffs(x -> T_approx((α * atanh(x)), p), NUM; dist_from_boundary=1e-13)

    if isnan(T_init_coefs[1]) || isnan(T_init_coefs[end]) || isnan(phi_init_coefs[1]) || isnan(phi_init_coefs[end])
        println("Initial guess values contain NaN!")
        return 0
    end

    sol = @time nlsolve(
        f!, [phi_init_coefs; T_init_coefs; 15.], 
        autodiff = :forward, method = :newton, 
        ftol=1e-13, xtol=1e-16, show_trace=true, iterations=7
    )    

    phi_expansion_coeffs = sol.zero[1:NUM]
    T_expansion_coeffs = sol.zero[NUM+1:2*NUM]
    c_computed = sol.zero[end]
    println("Computed velocity c = $(c_computed)")
    
    if save_solution
        write_solution_to_file(output_filename, phi_expansion_coeffs, T_expansion_coeffs, c_computed)
    end

    return phi_expansion_coeffs, T_expansion_coeffs, c_computed, sol

end

params = Params(1.2, 0.005)

const alpha_coef = 2.3
const α = alpha_coef / params.c_sharp_lim   # find appropriate / optimal value !
println("α = $(α)")

NUM = 200      # must be an even number!
#for NUM in [100] #[range(10, 50, step=4); range(60, 100, step=10); range(150, 1100, step=50)]
    phi_expansion_coeffs, T_expansion_coeffs, c_computed = calculate_spectral(NUM, params; save_solution=false)
#end

#phi_expansion_coeffs, T_expansion_coeffs, c_computed = read_solution_from_file("spectral_solution_coeffs_NUM_$(NUM).txt")

phi_computed(x) = chebyshev_expansion(phi_expansion_coeffs, x)
T_computed(x) = chebyshev_expansion(T_expansion_coeffs, x)

function plot_solution()
    x = range(-1, 1, length=Int(1e4))

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
end

#plot_solution()

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