# Функция, делающая N шагов по времени для вычисления функций температуры и фазового поля
function calculate_with_shifting(scheme::Discretization, init_cond_matrices; time_moments_to_make_shots=[], number_of_shots_per_time_moment::Int64=1, top_bc::Boundary_condition_Type=Neumann(), bottom_bc::Boundary_condition_Type=Neumann(), stop_function::Function=const_false_function)

    #x = range(-scheme.domain_side_length/2, scheme.domain_side_length/2, length = scheme.mesh_number)    # значит длина шага будет scheme.domain_side_length / (scheme.mesh_number-1)
    #y = copy(x)

    # setting initial conditions
    phi::Matrix{Float64} = copy(init_cond_matrices[1])
    T::Matrix{Float64} = copy(init_cond_matrices[2])

    new_phi::Matrix{Float64} = copy(phi)
    new_T::Matrix{Float64} = copy(T)

    N::Int64 = Int64(floor(scheme.time / scheme.Δt))        # number of time steps

    # Устанавливаем шаги по времени для явной и неявной схем
    if isa(scheme.scheme_type, Implicit)
        println()
        println("Implicit ADI scheme for T")
    elseif isa(scheme.scheme_type, Explicit)
        println()
        println("Explicit scheme for T")
    end
    println("N = ", N)
    println("Δt = ", scheme.Δt)
    println("M = ", scheme.mesh_number)
    println("Δh = ", scheme.Δh)

    # формируем дискретные значения моментов времени, в которые будет сделан снимок решения для сохранения
    numbers_N_to_make_shots_in = []
    number_of_time_moments = length(time_moments_to_make_shots)
    if (number_of_time_moments > 0)
        for i=1:number_of_time_moments
            append!(numbers_N_to_make_shots_in, Int64(floor(time_moments_to_make_shots[i] / scheme.Δt)))
        end
    end

    # Делаем N шагов по времени вычисления искомых функций. Отдельная реализация для явной и неявной схемы для уравнения температуры.
    if isa(scheme.scheme_type, Implicit)

        #@showprogress 1 "Computing..."
        @showprogress 1 "Computing..." for i = 1:N
            FVM_explicit_step!(phi, new_phi, phi_x, phi_y, scheme.mesh_number, scheme.Δh, scheme.Δt, phi_Source_Term(phi, T); top_bc=Neumann(), bottom_bc=Neumann())
            ADI_implicit_step!(T, new_T, scheme.mesh_number, scheme.Δh, scheme.Δt, T_Source_Term(phi, new_phi, scheme.Δt))
            phi = copy(new_phi)

            # сохраняем решения в файл
            if (number_of_time_moments > 0)
                for j=1:number_of_time_moments
                    if ((i > (numbers_N_to_make_shots_in[j] - number_of_shots_per_time_moment)) && (i <= numbers_N_to_make_shots_in[j]))
                        save_two_matrices(phi, T; time_of_calculation=time_moments_to_make_shots[j], my_str=string(number_of_shots_per_time_moment - (numbers_N_to_make_shots_in[j] - i)))
                    end
                end
            end

            # Проверяем, выполняется ли условие для остановки расчета и сдвигаем решение
            if stop_function(phi, T, scheme.mesh_number)
                #return phi, T, i*scheme.Δt
                interface_location = find_interface_index(phi)
                start_cut_index::Int64 = interface_location - div(scheme.mesh_number, 5)
                phi = create_shifted_solution_matrix(phi, start_cut_index)
                T = create_shifted_solution_matrix(T, start_cut_index)
                println()
                println("Time of shifting ", i*scheme.Δt)
            end
        end

        # Проверяем, добрались ли мы до конца интервала по времени
        if (scheme.time - scheme.Δt * N) < 2 * eps(Float64)
            return phi, T, N*scheme.Δt
        else
            Δt_end = scheme.time - scheme.Δt * N
            FVM_explicit_step!(phi, new_phi, phi_x, phi_y, scheme.mesh_number, scheme.Δh, Δt_end, phi_Source_Term(phi, T); top_bc=Neumann(), bottom_bc=Neumann())
            ADI_implicit_step!(T, new_T, scheme.mesh_number, scheme.Δh, Δt_end, T_Source_Term(phi, new_phi, Δt_end))
            phi = copy(new_phi)
            return phi, T, N*scheme.Δt+Δt_end
        end

    elseif isa(scheme.scheme_type, Explicit)

        @showprogress 1 "Computing..." for i = 1:N
            FVM_explicit_step!(phi, new_phi, phi_x, phi_y, scheme.mesh_number, scheme.Δh, scheme.Δt, phi_Source_Term(phi, T); top_bc=top_bc, bottom_bc=bottom_bc)
            FVM_explicit_step!(T, new_T, T_x, T_y, scheme.mesh_number, scheme.Δh, scheme.Δt, T_Source_Term(phi, new_phi, scheme.Δt); top_bc=top_bc, bottom_bc=bottom_bc)
            phi = copy(new_phi)
            T = copy(new_T)

            # сохраняем решения в файл
            if (number_of_time_moments > 0)
                for j=1:number_of_time_moments
                    if ((i > (numbers_N_to_make_shots_in[j] - number_of_shots_per_time_moment)) && (i <= numbers_N_to_make_shots_in[j]))
                        save_two_matrices(phi, T; time_of_calculation=time_moments_to_make_shots[j], my_str=string(number_of_shots_per_time_moment - (numbers_N_to_make_shots_in[j] - i)))
                    end
                end
            end

            # Проверяем, выполняется ли условие для остановки расчета и сдвигаем решение
            if stop_function(phi, T, scheme.mesh_number)
                #return phi, T, i*scheme.Δt
                interface_location = find_interface_index(phi)
                start_cut_index::Int64 = interface_location - div(scheme.mesh_number, 5)
                phi = create_shifted_solution_matrix(phi, start_cut_index)
                T = create_shifted_solution_matrix(T, start_cut_index)
                println()
                println("Time of shifting ", i*scheme.Δt)
            end
        end

        # Проверяем, добрались ли мы до конца интервала по времени
        if (scheme.time - scheme.Δt * N) < 2 * eps(Float64)
            return phi, T, N*scheme.Δt
        else
            Δt_end = scheme.time - scheme.Δt * N
            FVM_explicit_step!(phi, new_phi, phi_x, phi_y, scheme.mesh_number, scheme.Δh, Δt_end, phi_Source_Term(phi, T); top_bc=top_bc, bottom_bc=bottom_bc)
            FVM_explicit_step!(T, new_T, T_x, T_y, scheme.mesh_number, scheme.Δh, Δt_end, T_Source_Term(phi, new_phi, Δt_end); top_bc=top_bc, bottom_bc=bottom_bc)
            phi = copy(new_phi)
            T = copy(new_T)
            return phi, T, N*scheme.Δt+Δt_end
        end

    end

end

const_false_function(phi, T, M)::Bool = false
