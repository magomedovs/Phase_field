function find_interface_index(U::Matrix{Float64})::Int64

    val = 1/2  # 2*10^-32

    dim = size(U, 1)
    my_vec = U[div(dim, 2) + 1, :]
    for i=1:(dim-1)
        if ((my_vec[i] <= val)&&(my_vec[i+1] > val))
            return i
            #break
        end
    end
    return -1  # вывод в случае неудачи
end

function create_shifted_solution_matrix(U::Matrix{Float64}, start_cut_index::Int64)
    dim  = size(U, 1)

    new_U = zeros(dim, dim)

    temp_num = (dim - (start_cut_index-1))
    new_U[:, 1:temp_num] .= U[:, start_cut_index:dim]   # отсекаем ту часть матрицы решения, которую хотим сохранить для продолжения
    new_U[:, (temp_num+1):dim] .= U[:, dim]     # продолжаем усеченную матрицу решения до конца области

    return new_U
end
