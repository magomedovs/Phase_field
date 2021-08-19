# Функция для скачивания матрицы из файла
function get_matrix(my_path::String)::Matrix{Float64}
    temp::Matrix{Float64} = readdlm(my_path, '\t', Float64, '\n')
    #return readdlm(my_path, '\t', Float64, '\n') #convert(Matrix{Float64}, readdlm(my_path, '\t', Float64, '\n'))
    return temp
end


# Функция, сохраняющая в файл матрицы.
# Аргумент-строка my_str будет добавлен в конец названия файла.
function  save_two_matrices(phi, T; my_str="", time_of_calculation="")
    modified_str = ""
    if (my_str != "")
        modified_str = string("_", my_str)
    end

    M = size(phi, 1)
    open(string("phi_", M, "_t", time_of_calculation, "_S", S, "_d", δ, "_e", ϵ_0, modified_str, ".txt"), "w") do io
        writedlm(io, phi)
    end

    open(string("T_", M, "_t", time_of_calculation, "_S", S, "_d", δ, "_e", ϵ_0, modified_str, ".txt"), "w") do io
        writedlm(io, T)
    end
end
