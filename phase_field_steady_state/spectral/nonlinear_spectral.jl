using Printf
using Plots
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using NLsolve
using DelimitedFiles
using LaTeXStrings

include("Cheb_expansion.jl")
include("../structs_and_functions.jl")

function calculate_spectral(
    NUM::Int64, p::Params; 
    save_solution=false, 
    output_filename="spectral_solution_coeffs_NUM_$(NUM).txt")
    println("Number of terms = $(NUM)")

    nodes = [cos(j * pi / (NUM-1)) for j=(NUM-1):-1:0]

    function f!(F, a)#, p::Params)

        chebT = chebyshevt()
        phi_expansion(x) = a[1 : NUM]' * chebT[x, 1:NUM]
        T_expansion(x)   = a[NUM+1 : 2*NUM]' * chebT[x, 1:NUM]
        
        # chebU = chebyshevu()
        dphi_expansion(x)               = a[1 : NUM]' * [0; [k * chebyshevu(k-1, x) for k=1:NUM-1]]
        ddphi_expansion_numerator(x)    = a[1 : NUM]' * [k * ((k+1) * chebyshevt(k, x) - chebyshevu(k, x)) for k=0:NUM-1]
        dT_expansion(x)                 = a[NUM+1 : 2*NUM]' * [0; [k * chebyshevu(k-1, x) for k=1:NUM-1]]

        wave_speed = a[2*NUM + 1]

        Threads.@threads for i=1 : NUM
            x = nodes[i]
            # equations for ϕ
            F[i] = -p.ϵ_0^2 * (1 - x^2) * ddphi_expansion_numerator(x) +
                    (1 - x^2) * (α * p.τ * wave_speed - 2 * p.ϵ_0^2 * x) * dphi_expansion(x) +
                    α^2 * phi_expansion(x) * (1 - phi_expansion(x)) *
                    (phi_expansion(x) - 1/2 - m(T_expansion(x), p))

            # equations for T
            F[NUM + i] = (1 - x^2) * dT_expansion(x) +
                    α * wave_speed * T_expansion(x) +
                    α * wave_speed / p.S * phi_expansion(x) - 
                    α * wave_speed / p.S 
        end

        # Modify some equations to account for the boundary conditions
        # B.C. for ϕ
        F[1]    = phi_expansion(nodes[1]) - 0           # at x = -1
        F[NUM]  = phi_expansion(nodes[NUM]) - 1         # at x = 1
        # B.C. for T
        # F[NUM + 1] = T_expansion(nodes[1]) - 1/p.S      # at x = -1. This third condition seems excessive.
        F[2*NUM] = T_expansion(nodes[NUM]) - 0.         # at x = 1

        # Additional equation needed to close the system (because of the unknown wave_speed).
        F[2*NUM + 1] = phi_expansion(0) - 1/2           # at x = 0
        
    end

    #f!(F, x) = f!(F, x, p)

    phi_init_coefs = calculate_cheb_expansion_coeffs(
        x -> phi_approx((α * atanh(x)), p), NUM; 
        dist_from_boundary=1e-13)
    T_init_coefs = calculate_cheb_expansion_coeffs(
        x -> T_approx((α * atanh(x)), p), NUM; 
        dist_from_boundary=1e-13)

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

const alpha_coef = 5.0
const α = alpha_coef / params.c_sharp_lim   # find appropriate / optimal value !
println("α = $(α)")

NUM = 300      # must be an even number!
#for NUM in [100] #[range(10, 50, step=4); range(60, 100, step=10); range(150, 1100, step=50)]
    phi_expansion_coeffs, T_expansion_coeffs, c_computed = calculate_spectral(
        NUM, params; 
        save_solution=false)
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
        label=L"\phi(x)",
        #legend=:bottomleft
    )
    plot!(
        x, x -> T_computed(x), 
        #ylabel="f(x)",
        xlabel=L"x",
        label=L"T(x)",
        #legend=:bottomleft
    )
end
function plot_coeffs()
    scatter(
        abs.(phi_expansion_coeffs),
        label="phi_expansion_coeffs",
        yaxis=:log,
        shape=:circle
    )
    scatter!(
        abs.(T_expansion_coeffs),
        label="T_expansion_coeffs",
        yaxis=:log,
        shape=:circle
    )
end

plot_solution()
plot_coeffs()

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