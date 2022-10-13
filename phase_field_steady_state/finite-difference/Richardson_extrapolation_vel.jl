using NLsolve
using LsqFit
using Plots

# Function for Richardson extrapolation of "c"
function extrapolate_c(h_list, c_list)
    if (length(h_list) != length(c_list))
        println("Arrays dimensions don't coincide!")
        return 0
    end

    h_mat = h_list.^(2*0)
    for i=1:(length(c_list)-1)
        h_mat = [h_mat h_list.^(2*i)]
    end

    println(h_mat)

    res = h_mat \ c_list
end

# Uploading data
computed_arr = []
for line in eachline("/Users/shamilmagomedov/Desktop/calculated_c.txt")
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
inds = [1, 4, 7, 9]
#inds = [10, 11, 12, 13]

h_list = [h_arr[i] for i in inds]
c_list = [c_arr[i] for i in inds]
res_estim = extrapolate_c(h_list, c_list)


# Curve fitting
@. model(t, p) = p[1] + p[2] * t^2 + p[3] * t^4;
p0 = res_estim[1:3]
fit = curve_fit(model, h_arr[1:end], c_arr[1:end], p0, x_tol = 1e-12, show_trace=true);
res = fit.param
c_extrap = res[1]
h2_coef = res[2]


#c_extrap = res_estim[1]
#h2_coef = res_estim[2]

#extrapolate_c([h_arr[10],h_arr[12],h_arr[14]], [c_arr[10],c_arr[12],c_arr[14]])
#extrapolate_c([h_arr[10],h_arr[12],h_arr[14], h_arr[16]], [c_arr[10],c_arr[12],c_arr[14], c_arr[16]])

# Plot of c(h) with extrapolated value of c
errors_arr = abs.(c_arr .- c_extrap) / c_extrap

plot(h_arr, errors_arr,
    label="|c(h) - c_ext| / c_ext", xlabel="h", ylabel="rel error", legend=:topleft, xaxis=:log, yaxis=:log, 
    yformatter=:scientific, shape=:circle)

plot!(range(findmin(h_arr)[1], findmax(h_arr)[1], length=Int(1e5)),
    x -> (h2_coef * x^2 / c_extrap), xaxis=:log, yaxis=:log,
     label="$(round(h2_coef, digits=10)) / c_ext * h^2")

#savefig("/Users/shamilmagomedov/Desktop/c_plot_S_" * "$S" * "_eps_" * "$ϵ_0" * ".pdf")

#cov = estimate_covar(fit)
#se = stderror(fit)

#=
plot(h_arr, c_arr,
    label="\"c\" computed for step size h", xlabel="1/h", ylabel="c", legend=:topleft, xaxis=:log, yaxis=:log, 
    yformatter=:scientific, shape=:circle)

myticks = [c_extrap; [c_arr[i] for i in range(1, length(c_arr), step=1)]]
myticks_str = map(x -> string(round(x, digits=10)), [c_extrap; [c_arr[i] for i in range(1, length(c_arr), step=1)]])

plot!(h_arr, fill(c_extrap, length(h_arr)), xaxis=:log, yaxis=:log,
   label="\"c\" from Richardson extrapolation")#, yticks = (myticks, myticks_str))

plot!(range(findmin(h_arr)[1], findmax(h_arr)[1], length=Int(1e5)),
 x -> (c_extrap + h2_coef * x^2), xaxis=:log, yaxis=:log,
  label="$(round(c_extrap, digits=10)) + $(round(h2_coef, digits=10)) * h^2")
=#

#savefig("/Users/shamilmagomedov/Desktop/c_plot_S_" * "$S" * "_eps_" * "$ϵ_0" * ".pdf")

#=
# Richardson extrapolation for "c"

h_list = 1 ./ [1e5, 2e5, 3e5]
c_list = [15.990274435396245, 15.990274340337136, 15.990274322773272]

h_mat = [h_list.^0 h_list.^2 h_list.^4]

res = h_mat \ c_list

# res[1] == 15.99027430873111
=#

#=
# Second approach https://en.wikipedia.org/wiki/Richardson_extrapolation
h_coarse = h_list[1]
h_fine = h_list[2]
coef = h_coarse / h_fine
A_coarse = c_list[1]
A_fine = c_list[2]

R = (coef^2 * A_fine - A_coarse) / (coef^2 - 1)
=#


# for h = 1e-5 * 1/4 --- c = 15.990274316504774
# for h = 1e-5 * 1/5 --- c = 15.99027431374056

#=
# Richardson extrapolation for "c"
h_list = 1 ./ [5e4, 1e5, 1.5e5]
c_list = [15.990274815616639, 15.990274435396245, 15.99027436499104]

h_mat = [h_list.^0 h_list.^2 h_list.^4]

res = h_mat \ c_list

# res[1] == 15.99027430866822
=#

#=
# Richardson extrapolation for "c"
h_list = 1 ./ [3e4, 4e4, 5e4]
c_list = [15.990275716896734, 15.990275100789168, 15.990274815616639]

h_mat = [h_list.^0 h_list.^2 h_list.^4]

res = h_mat \ c_list

# res[1] == 15.990274308638972
=#
