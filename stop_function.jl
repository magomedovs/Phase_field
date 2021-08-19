# Функция, которая определяет условие для остановки расчета.
function my_stop_func(phi, T, M)::Bool
    #tol::Float64 = 2 * 10^(-8)
    if (find_interface_index(phi) >= (M - div(M, 3)))&&((abs(1 - phi[div(M, 2)+1, M]) > tol) || (abs(T[div(M, 2)+1, M]) > tol))
        return true
    else
        return false
    end
end
