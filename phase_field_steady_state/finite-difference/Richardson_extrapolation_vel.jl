using NLsolve
using LsqFit
using Plots

# Function for Richardson extrapolation
# A(h) = A₀ + C₁ * h^{1 * p} + C₂ * h^{2 * p} + O(h^{3 * p})
function extrapolate_func(h_array, A_array; p=2)
    if (length(h_array) != length(A_array))
        println("Arrays dimensions don't coincide!")
        return 0
    end

    h_mat = h_array.^(p * 0)
    for i=1:(length(A_array)-1)
        h_mat = [h_mat h_array.^(p * i)]
    end

    #println(h_mat)
    res = h_mat \ A_array
    return res
end

# Uploading data
computed_arr = []
for line in eachline("/Users/shamilmagomedov/Desktop/finite_diff_calculated_c.txt")
    append!(computed_arr, [map(x -> parse(Float64, x), split(line, " "))])
end

h_arr = [computed_arr[i][1] for i=1:length(computed_arr)]
c_arr = [computed_arr[i][2] for i=1:length(computed_arr)]

# Estimation of an exponent of the leading order step size error
# A1 should be computed on the coarse mesh
h1, h2, h3 = h_arr[5], h_arr[9], h_arr[10]
A1, A2, A3 = c_arr[5], c_arr[9], c_arr[10]

s2 = h1 / h2
s3 = h1 / h3

estim_func(k) = A2 - A3 + (A2 - A1) / (s2^k - 1) - (A3 - A1) / (s3^k - 1)
estim_func_prime(k) = -(A2 - A1) / (s2^k - 1)^2 * log(s2) * s2^k + (A3 - A1) / (s3^k - 1)^2 * log(s3) * s3^k

function f!(F, x)
    F[1] = estim_func(x[1])
end

plot(range(1, 15, length=Int(1e4)), x -> estim_func(x))

function j!(J, x)
    J[1, 1] = estim_func_prime(x[1])
end

sol = nlsolve(f!, [1.4]; ftol=1e-10)
#print("Solution zero : ", sol.zero)


# Extrapolation

#inds = [1, 2, 3, 4]
#inds = [1, 4, 7, 9]
#inds = [10, 11, 12, 13]
inds = [2, 3, 4, 5]

h_list = [h_arr[i] for i in inds]
c_list = [c_arr[i] for i in inds]
res_estim = extrapolate_func(h_list, c_list)


# Curve fitting
@. model(t, p) = (p[1] + p[2] * t^2 + p[3] * t^4)
p0 = res_estim[1:3]
fit = curve_fit(model, h_arr[1:end], (c_arr[1:end]), p0, g_tol=1e-16, x_tol = 1e-16, show_trace=true);
res = fit.param
c_extrap = res[1]
h2_coef = res[2]


# Plot of c(h) with extrapolated value of c
errors_arr = abs.(c_arr .- c_extrap) / c_extrap

plot(
    h_arr, errors_arr,
    label="|c(h) - c_ext| / c_ext", xlabel="h", ylabel="rel error", legend=:topleft, 
    axis=:log,
    yformatter=:scientific, shape=:circle
)

plot!(
    range(findmin(h_arr)[1], findmax(h_arr)[1], length=Int(1e5)),
    x -> (h2_coef * x^2 / c_extrap),
    label="$(round(h2_coef, digits=10)) / c_ext * h^2"
)

#savefig("/Users/shamilmagomedov/Desktop/c_plot_S_" * "$S" * "_eps_" * "$ϵ_0" * ".pdf")

#cov = estimate_covar(fit)
#se = stderror(fit)


#=
# Second approach https://en.wikipedia.org/wiki/Richardson_extrapolation
h_coarse = h_list[1]
h_fine = h_list[2]
coef = h_coarse / h_fine
A_coarse = c_list[1]
A_fine = c_list[2]

R = (coef^2 * A_fine - A_coarse) / (coef^2 - 1)
=#
