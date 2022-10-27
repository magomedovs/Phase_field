using DelimitedFiles

struct Params
    S::Float64              # 1.2
    ϵ_0::Float64            # 0.005
    a_1::Float64            # 0.9 * (ϵ_0*100)
    β::Float64              # 10.0
    τ::Float64              # 0.0003 * (ϵ_0*100)^2
    c_sharp_lim::Float64    # ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))

    Params(iS, iϵ_0, ia_1, iβ, iτ) = new(
        iS, iϵ_0, ia_1, iβ, iτ,
        iϵ_0 * ia_1 * sqrt(2) / (pi * iτ) * atan(iβ * (1.0 - 1/iS))
    )
end

Params(iS, iϵ_0) = Params(
    iS, 
    iϵ_0, 
    0.9 * (iϵ_0*100),
    10.0,
    0.0003 * (iϵ_0*100)^2,
)

m(T, p::Params) = (p.a_1 / pi) * atan(p.β * (1 - T))
m_prime(T, p::Params) = -(p.a_1 * p.β / pi) * 1/(1 + (p.β * (1 - T))^2)

phi_approx(x, p::Params)::Float64 = (tanh(x / (p.ϵ_0 * 2 * sqrt(2))) + 1 ) / 2

function T_approx(x, p::Params; c = p.c_sharp_lim)::Float64    
    function T_outer_solution(x)
        if (x <= 0.)
            return 1/p.S
        else
            return 1/p.S * exp(-x * c)
        end
    end

    function T_inner_solution(x)::Float64
        return 1/p.S + p.ϵ_0 * (-(c * sqrt(2) / p.S) * log(2) - (c * sqrt(2) / p.S) * log(cosh(x/(p.ϵ_0 * 2^(3/2)))) + (x / p.ϵ_0) * ( - c / (2 * p.S)) )
    end

    if (x <= 0.)
        return T_inner_solution(x) + T_outer_solution(x) - 1/p.S
    else
        return T_inner_solution(x) + T_outer_solution(x) - 1/p.S + x * (c / p.S)
    end
end

function write_solution_to_file(filename::String, phi, T, c_computed::Float64)
    if isempty(phi) || isempty(T)
        println("Output array is empty!")
        return 0
    end
    # Writing computed data into file
    open(filename, "w") do io
        writedlm(io, [phi T fill(c_computed, length(phi))])
    end
end

function read_solution_from_file(filename::String)
    Mat = readdlm(filename, Float64)
    phi, T, c_computed = Mat[:, 1], Mat[:, 2], Mat[1, 3]
    return phi, T, c_computed
end
