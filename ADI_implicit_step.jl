# Функция Формирующая трехдиагональную матрицу A
function create_mat(γ::Float64, M::Int64)::Tridiagonal

    # нижняя диагональ
    diag_low = fill(-γ, M - 1)
    diag_low[end] = -2γ    # меняем значение последнего элемента вектора

    # главная диагональ
    diag_main = fill(1 + 2γ, M)

    # верхняя диагональ
    diag_up = fill(-γ, M - 1)
    diag_up[1] = -2γ

    return Tridiagonal(diag_low, diag_main, diag_up)
end


# Функция, делающая один шаг ADI method
function ADI_implicit_step!(u::Matrix{Float64}, new_u::Matrix{Float64}, M::Int64, Δh::Float64, Δt::Float64, Source_Term::Matrix{Float64})

    #M = size(u, 1)
    α = (Δt/2 * D) / Δh^2
    A::Tridiagonal = create_mat(α, M)
    B::Matrix{Float64} = Matrix{Float64}(undef, M::Int64, M::Int64)

    # матрица правых частей B
    B[:, 1] = @. (1 - 2α) * u[:, 1] + 2α * u[:, 2] + (Δt/2) * Source_Term[:, 1] #[ - (1 / S) * ( new_phi[i,1] - phi[i,1] ) for i = 1:M]
    B[:, M] = @. 2α * u[:, M - 1] + (1 - 2α) * u[:, M] + (Δt/2) * Source_Term[:, M] #[ - (1 / S) * ( new_phi[i,M] - phi[i,M] ) for i = 1:M]
    #Threads.@threads
    Threads.@threads for j = 2:(M - 1)
        B[:, j] = @. α * u[:, j - 1] + (1 - 2α) * u[:, j] + α * u[:, j + 1] + (Δt/2) * Source_Term[:, j] #[ - (1 / S) * ( new_phi[i,j] - phi[i,j] ) for i = 1:M]
    end

    Threads.@threads for j = 1:M
        new_u[:, j] = A \ B[:, j]
    end

    B[1, :] = @. (1 - 2α) * new_u[1, :] + 2α * new_u[2, :] + (Δt/2) * Source_Term[1, :] #[ - (1 / S) * ( new_phi[1,j] - phi[1,j] ) for j = 1:M]
    B[M, :] = @. 2α * new_u[M - 1, :] + (1 - 2α) * new_u[M, :] + (Δt/2) * Source_Term[M, :] #[ - (1 / S) * ( new_phi[M,j] - phi[M,j] ) for j = 1:M]

    Threads.@threads for i = 2:(M - 1)
        B[i, :] = @. α * new_u[i - 1, :] + (1 - 2α) * new_u[i, :] + α * new_u[i + 1, :] + (Δt/2) * Source_Term[i, :] #[ - (1 / S) * ( new_phi[i,j] - phi[i,j] ) for j = 1:M]
    end

    Threads.@threads for i = 1:M
        u[i, :] = A \ B[i, :]
    end

end
