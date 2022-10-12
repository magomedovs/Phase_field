
# Function for saving matrices into the file.
# A string my_str will be appended to the end of the filename.
function save_two_matrices(phi, T; my_str="")
    modified_str = ""
    if (my_str != "")
        modified_str = string("_", my_str)
    end

    M = size(phi, 1)
    open(string("phi_steady", M, "_S", S, "_e", ϵ_0, modified_str, ".txt"), "w") do io
        writedlm(io, phi)
    end

    open(string("T_steady", M, "_S", S, "_e", ϵ_0, modified_str, ".txt"), "w") do io
        writedlm(io, T)
    end
end

function save_data(c)
    df = DataFrame("S" => S, "epsilon_0" => ϵ_0, "M" => M, "Domain_len" => H, "c" => c)
    CSV.write(string("data", "_S", S, "_e", ϵ_0, "_M", M, ".csv"), df)
end

# CSV.write("file_name.csv", df_name)
# input_file = CSV.File("file_name.csv")
# input_file_df = DataFrame(input_file)
