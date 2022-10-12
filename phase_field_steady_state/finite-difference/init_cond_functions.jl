
phi_init_band(x::Float64, shift)::Float64 = (tanh((x - shift) / (ϵ_0 * 2 * sqrt(2))) + 1 ) / 2


function T_composite_solution(x, shift; c = 0)
    if (c == 0)
        c = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))    # velocity of the wave
    end
    
    function T_outer_solution(x, shift)
        if (x <= shift)
            return 1/S
        else
            return 1/S * exp((-x + shift) * c)  # calculate_velocity_analytical(phi_data_middle, T_data_middle)
        end
    end

    function T_inner_solution(x, shift)
        return 1/S + ϵ_0 * (-(c * sqrt(2) / S) * log(2) - (c * sqrt(2) / S) * log(cosh((x - shift)/(ϵ_0 * 2^(3/2)))) + ((x - shift) / ϵ_0) * ( - c / (2 * S)) )
    end

    if (x <= shift)
        return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S
    else
        return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S + (x - shift) * (c / S)
    end
end
