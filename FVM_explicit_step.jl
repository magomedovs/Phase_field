# Сумма потоков через четыре границы индивидуального объема, умноженная на коэффициент flows_coef,
# который зависит от расположения клетки относительно границы области.
#function flows_sum(U::Matrix{Float64}, U_x::Function, U_y::Function, i::Int64, j::Int64, flows_coef::Int64)::Float64
#    return flows_coef * ( (J1(U, U_x, i, j) - J2(U, U_x, i, j)) + (J3(U, U_y, i, j) - J4(U, U_y, i, j)) )
#end

function FVM_explicit_step!(U::Matrix{Float64}, new_U::Matrix{Float64}, U_x::Function, U_y::Function, M::Int64, Δh::Float64, Δt::Float64, Source_Term::Matrix{Float64}; top_bc::Boundary_condition_Type=Neumann(), bottom_bc::Boundary_condition_Type=Neumann())

        # Производные (разностные выражения производных)

        # Rx - derivative with respect to x in the center of the Right side of the volume
        Rx(U, i::Int64, j::Int64)::Float64 = ( U[i+1, j] - U[i, j] ) / ( Δh )

        # Ry - derivative with respect to y in the center of the Right side of the volume
        Ry(U, i::Int64, j::Int64)::Float64 = ( U[i, j+1] + U[i+1, j+1] - U[i, j-1] - U[i+1, j-1] ) / ( 4 * Δh )

        # Lx - derivative with respect to x in the center of the Left side of the volume
        Lx(U, i::Int64, j::Int64)::Float64 = ( U[i, j] - U[i-1, j] ) / ( Δh ) #Rx(U, i-1, j)

        # Ly - derivative with respect to y in the center of the Left side of the volume
        Ly(U, i::Int64, j::Int64)::Float64 = ( U[i, j+1] + U[i-1, j+1] - U[i, j-1] - U[i-1, j-1] ) / ( 4 * Δh ) #Ry(U, i-1, j)

        # Tx - derivative with respect to x in the center of the Top side of the volume
        Tx(U, i::Int64, j::Int64)::Float64 = ( U[i+1, j] + U[i+1, j+1] - U[i-1, j] - U[i-1, j+1] ) / ( 4 * Δh )

        # Ty - derivative with respect to y in the center of the Top side of the volume
        Ty(U, i::Int64, j::Int64)::Float64 = ( U[i, j+1] - U[i, j] ) / ( Δh )

        # Bx - derivative with respect to x in the center of the Bottom side of the volume
        Bx(U, i::Int64, j::Int64)::Float64 = ( U[i+1, j] + U[i+1, j-1] - U[i-1, j] - U[i-1, j-1] ) / ( 4 * Δh ) #Tx(U, i, j-1)

        # By - derivative with respect to y in the center of the Bottom side of the volume
        By(U, i::Int64, j::Int64)::Float64 = ( U[i, j] - U[i, j-1] ) / ( Δh ) #Ty(U, i, j-1)


        # Производные (разностные выражения производных) около границ области

        # Далее i несущественная переменная
        # Производные на правой границе области
        right_bound_Ty(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[size(U, 1), j+1] - U[size(U, 1), j]) + U[size(U, 1)-1, j+1] - U[size(U, 1)-1, j] ) / ( 4 * Δh )
        right_bound_By(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[size(U, 1), j] - U[size(U, 1), j-1]) + U[size(U, 1)-1, j] - U[size(U, 1)-1, j-1] ) / ( 4 * Δh ) #right_bound_Ty(U, i, j-1)

        right_bound_Tx(U, i::Int64, j::Int64)::Float64 = ( U[size(U, 1), j+1] + U[size(U, 1), j] - U[size(U, 1)-1, j+1] - U[size(U, 1)-1, j] ) / ( 2 * Δh )
        right_bound_Bx(U, i::Int64, j::Int64)::Float64 = ( U[size(U, 1), j] + U[size(U, 1), j-1] - U[size(U, 1)-1, j] - U[size(U, 1)-1, j-1] ) / ( 2 * Δh ) #right_bound_Tx(U, i, j-1)

        #  Производные на левой границе области
        left_bound_Ty(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[1, j+1] - U[1, j]) + U[2, j+1] - U[2, j] ) / ( 4 * Δh )
        left_bound_By(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[1, j] - U[1, j-1]) + U[2, j] - U[2, j-1] ) / ( 4 * Δh ) #left_bound_Ty(U, i, j-1)

        left_bound_Tx(U, i::Int64, j::Int64)::Float64 = ( U[2, j+1] + U[2, j] - U[1, j+1] - U[1, j] ) / ( 2 * Δh )
        left_bound_Bx(U, i::Int64, j::Int64)::Float64 = ( U[2, j] + U[2, j-1] - U[1, j] - U[1, j-1] ) / ( 2 * Δh ) #left_bound_Tx(U, i, j-1)

        # Далее j несущественная переменная
        #  Производные на верхней границе области
        top_bound_Ry(U, i::Int64, j::Int64)::Float64 = ( U[i+1, size(U, 2)] + U[i, size(U, 2)] - U[i+1, size(U, 2)-1] - U[i, size(U, 2)-1] )  / ( 2 * Δh )
        top_bound_Ly(U, i::Int64, j::Int64)::Float64 = ( U[i, size(U, 2)] + U[i-1, size(U, 2)] - U[i, size(U, 2)-1] - U[i-1, size(U, 2)-1] )  / ( 2 * Δh ) #top_bound_Ry(U, i-1, j)

        top_bound_Rx(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[i+1, size(U, 2)] - U[i, size(U, 2)]) + U[i+1, size(U, 2)-1] - U[i, size(U, 2)-1] ) / ( 4 * Δh )
        top_bound_Lx(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[i, size(U, 2)] - U[i-1, size(U, 2)]) + U[i, size(U, 2)-1] - U[i-1, size(U, 2)-1] ) / ( 4 * Δh ) #top_bound_Rx(U, i-1, j)

        #  Производные на нижней границе области
        bottom_bound_Ry(U, i::Int64, j::Int64)::Float64 = ( U[i+1, 2] + U[i, 2] - U[i+1, 1] - U[i, 1] )  / ( 2 * Δh )
        bottom_bound_Ly(U, i::Int64, j::Int64)::Float64 = ( U[i, 2] + U[i-1, 2] - U[i, 1] - U[i-1, 1] )  / ( 2 * Δh ) #bottom_bound_Ry(U, i-1, j)

        bottom_bound_Rx(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[i+1, 1] - U[i, 1]) + U[i+1, 2] - U[i, 2] ) / ( 4 * Δh )
        bottom_bound_Lx(U, i::Int64, j::Int64)::Float64 = ( 3 * (U[i, 1] - U[i-1, 1]) + U[i, 2] - U[i-1, 2] ) / ( 4 * Δh ) #bottom_bound_Rx(U, i-1, j)


        # ∂U/∂t = ∇ ⋅ [ ∇J(U) ] + R
        # ∇J = (J_x, J_y)  -  векторная функция
        # Введены отдельные функции потоков для вычисления внутри области и на границе
        function J1_domain(U::Matrix{Float64}, J_x::Function, i::Int64, j::Int64)::Float64   # поток через правую сторону клетки
            return J_x(U, i, j, Rx, Ry)    # внутри области
        end

        function J2_domain(U::Matrix{Float64}, J_x::Function, i::Int64, j::Int64)::Float64   # поток через левую сторону клетки
            return J_x(U, i, j, Lx, Ly)    # внутри области
        end

        function J3_domain(U::Matrix{Float64}, J_y::Function, i::Int64, j::Int64)::Float64   # поток через верхнюю сторону клетки
            return J_y(U, i, j, Tx, Ty)    # внутри области
        end

        function J4_domain(U::Matrix{Float64}, J_y::Function, i::Int64, j::Int64)::Float64   # поток через нижнюю сторону клетки
            return J_y(U, i, j, Bx, By)    # внутри области
        end


        function J1_boundary(U::Matrix{Float64}, J_x::Function, i::Int64, j::Int64)::Float64   # поток через правую сторону клетки
            if (i == M)     # правая граница
                return zero(Float64)
            elseif (j == 1)     # нижняя граница
                return (1/2) * J_x(U, i, j, bottom_bound_Rx, bottom_bound_Ry)
            elseif (j == M)     # верхняя граница
                return (1/2) * J_x(U, i, j, top_bound_Rx, top_bound_Ry)
            else    # внутри области
                return J_x(U, i, j, Rx, Ry)
            end
        end

        function J2_boundary(U::Matrix{Float64}, J_x::Function, i::Int64, j::Int64)::Float64   # поток через левую сторону клетки
            if (i == 1)     # левая граница
                return zero(Float64)
            elseif (j == 1)     # нижняя граница
                return (1/2) * J_x(U, i, j, bottom_bound_Lx, bottom_bound_Ly)
            elseif (j == M)     # верхняя граница
                return (1/2) * J_x(U, i, j, top_bound_Lx, top_bound_Ly)
            else    # внутри области
                return J_x(U, i, j, Lx, Ly)
            end
        end

        function J3_boundary(U::Matrix{Float64}, J_y::Function, i::Int64, j::Int64)::Float64   # поток через верхнюю сторону клетки
            if (j == M)     # верхняя граница
                return zero(Float64)
            elseif (i == 1)     # левая граница
                return (1/2) * J_y(U, i, j, left_bound_Tx, left_bound_Ty)
            elseif (i == M)     # правая граница
                return (1/2) * J_y(U, i, j, right_bound_Tx, right_bound_Ty)
            else    # внутри области
                return J_y(U, i, j, Tx, Ty)
            end
        end

        function J4_boundary(U::Matrix{Float64}, J_y::Function, i::Int64, j::Int64)::Float64   # поток через нижнюю сторону клетки
            if (j == 1)     # нижняя граница
                return zero(Float64)
            elseif (i == 1)     # левая граница
                return (1/2) * J_y(U, i, j, left_bound_Bx, left_bound_By)
            elseif (i == M)     # правая граница
                return (1/2) * J_y(U, i, j, right_bound_Bx, right_bound_By)
            else    # внутри области
                return J_y(U, i, j, Bx, By)
            end
        end

    # внутренность области
    Threads.@threads for j = 2:(M-1)
        for i = 2:(M-1)
            new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 1 * ( (J1_domain(U, U_x, i, j) - J2_domain(U, U_x, i, j)) + (J3_domain(U, U_y, i, j) - J4_domain(U, U_y, i, j)) ) + Source_Term[i, j] )
        end
    end

    # граница области (левая и правая, без угловых точек)
    for i = [1, M]
        for j = 2:(M-1)
            new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 2 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
        end
    end

    if (isa(top_bc, Neumann) && isa(bottom_bc, Neumann))
        # граница области (верхняя и нижняя, без угловых точек)
        Threads.@threads for j = [1, M]
            for i = 2:(M-1)
                new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 2 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
            end
        end

        # углы области
        for i = [1, M], j = [1, M]
            new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 4 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
        end

    elseif (isa(top_bc, Neumann) && isa(bottom_bc, Dirichlet))
        # граница области (верхняя)
        for j = M
            for i = 2:(M-1)
                new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 2 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
            end
        end

        # углы области
        for i = [1, M], j = M
            new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 4 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
        end

    elseif (isa(top_bc, Dirichlet) && isa(bottom_bc, Neumann))
        # граница области (нижняя)
        for j = 1
            for i = 2:(M-1)
                new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 2 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
            end
        end

        # углы области
        for i = [1, M], j = 1
            new_U[i,j] = U[i,j] + Δt * ( (1 / Δh) * 4 * ( (J1_boundary(U, U_x, i, j) - J2_boundary(U, U_x, i, j)) + (J3_boundary(U, U_y, i, j) - J4_boundary(U, U_y, i, j)) ) + Source_Term[i, j] )
        end
    #elseif (isa(top_bc, Dirichlet) && isa(bottom_bc, Dirichlet))
    end

end
