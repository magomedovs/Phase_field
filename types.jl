abstract type Scheme_Type end
struct Explicit <: Scheme_Type end
struct Implicit <: Scheme_Type end

abstract type Boundary_condition_Type end
struct Neumann <: Boundary_condition_Type end
struct Dirichlet <: Boundary_condition_Type end

mutable struct Discretization
    scheme_type::Scheme_Type
    domain_side_length::Float64
    mesh_number::Int64
    time::Float64
    Δh::Float64
    Δt::Float64
    #coef::Float64
#=
    function Discretization(scheme_type, domain_side_length, mesh_number, time, Δh, Δt, coef)
        Δh = (domain_side_length) / (mesh_number - 1)
        if isa(scheme_type, Explicit)
            Δt = (Δh)^2 / (4 * coef)
        end

        new(scheme_type, domain_side_length, mesh_number, time, Δh, Δt, coef)
    end
=#
end

function create_Discretization(; scheme_type::Scheme_Type=Explicit(), domain_side_length::Float64=5.0, mesh_number::Int64=801, time::Float64=1.0)::Discretization
        Δh = (domain_side_length) / (mesh_number - 1)

        if isa(scheme_type, Explicit)
            coef = 1.4
            Δt = (Δh)^2 / (4 * coef)

        elseif isa(scheme_type, Implicit)
            coef = 1.0
            Δt = (Δh)^2 / (4 * coef)
        end
        new_struct = Discretization(scheme_type, domain_side_length, mesh_number, time, Δh, Δt)
        return new_struct
end
