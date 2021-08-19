# smooth initial condition for phi function
phi_init_circle(x::Float64, y::Float64, shift::Float64)::Float64 = (tanh((sqrt(x^2 + y^2) - shift) / (ϵ_0 * 2 * sqrt(2) * (1 + δ))) + 1 ) / 2
phi_init_band(x::Float64, shift::Float64)::Float64 = (tanh((x - shift) / (ϵ_0 * 2 * sqrt(2) * (1 + δ))) + 1 ) / 2

function set_initial_phi!(U::Matrix{Float64}, H::Float64, M::Int64; shape="circle", smoothness="discontinious", band_height=0.5, amplitude=0.0, wave_number=5.0)

    x = range(-H/2, H/2, length = M)    # значит длина шага будет H / (M-1)
    y = copy(x)

    seed = 0.1 #M / 60
    if (shape=="circle")
        if (smoothness=="discontinious")
            for i=1:M, j=1:M
                if (((i - div(M,2) - 1)*H/M)^2 + ((j - div(M,2) - 1)*H/M)^2 <= (seed^2))        # ядро в центре области
#                if((i - div(M,2) - 1)^2+(j - 1)^2 < (seed^2))                   # ядро в центре нижней границы области
                   U[i,j] = zero(Float64) #zero(phi[1, 1])
                end
            end
        elseif (smoothness=="continious")
            U .= [phi_init_circle(i, j, seed) for i=x, j=y]
        end
    end

    #band_height = 0.5 #M / 5
    if (shape=="band")
        if (smoothness=="discontinious")
            for i=1:M, j=1:M
                if (j * H/M  - amplitude * cos(H/M * i * wave_number) <= band_height)						 # плоский фронт вдоль нижней границы
                   U[i,j] = zero(Float64) #zero(phi[1, 1])
                end
            end
        elseif (smoothness=="continious")
            for i=1:M
                U[i, :] = [phi_init_band(j, -H/2 + band_height + amplitude * cos(H/M * i * wave_number)) for j=y]  # amplitude and wave_number are parameters of perturbation of initial conditions
            end
        end
    end

end

function set_initial_T!(U::Matrix{Float64}, H::Float64, M::Int64; smoothness="discontinious", band_height=0.5, amplitude=0.0, wave_number=5.0)
    T_0::Float64 = 1/S

    c = ϵ_0 * a_1 * sqrt(2) / (pi * τ) * atan(β * (1.0 - 1/S))    # velocity of the wave

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

    function T_composite_solution(x, shift)
        if (x <= shift)
            return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S
        else
            return T_inner_solution(x, shift) + T_outer_solution(x, shift) - 1/S + (x - shift) * (c / S)
        end
    end

    #band_height = 0.5
    if (smoothness=="continious")
        for i=1:M, j=1:M
            U[i, j] = T_composite_solution(j * H/M, band_height + amplitude * cos(H/M * i * wave_number))
            #if (j * H/M - amplitude * cos(H/M * i * wave_number)) <= band_height
            #    U[i, j] = T_0
            #else
            #    U[i, j] = T_0 * exp(( -(j*H/M - band_height - amplitude * cos(H/M * i * wave_number)) ) * c )
            #end
        end
    elseif (smoothness=="discontinious")
        for i=1:M, j=1:M
            if (j * H/M - amplitude * cos(H/M * i * wave_number)) <= band_height
                U[i, j] = 1.0
            else
                U[i, j] = 0.0
            end
        end
    end

end
