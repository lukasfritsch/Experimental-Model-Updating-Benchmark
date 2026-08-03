# helper function to transform the DataFrame
function unique_df_physical(samples)
    cols = [:z1, :z2, :z3, :z4, :z5, :z6, :z7]
    unique_samples = unique(samples[:, cols])

    z_mass = Matrix(unique_samples[:, [:z1, :z2, :z3]])
    M = permutedims(hcat([normal_to_simplex(y) for y in eachrow(z_mass)]...))

    z_k = Matrix(unique_samples[:, [:z4, :z5, :z6, :z7]])
    K = permutedims(hcat([normal_to_uniform(y) for y in eachrow(z_k)]...))

    return hcat([DataFrame(M, [:m1, :m2, :m3, :m4]), DataFrame(K, [:k1, :k2, :k3, :k4])]...)
end

# softmax function
softmax(ξ) = exp.(ξ) ./ sum(exp.(ξ))

# transform from standard normal to simplex (ℝ³ ↦ (0, 1)⁴)
normal_to_simplex(y) = 4 .* softmax(0.02 .* vcat(y, 0.0))

# transform from standard normal to uniform (ℝ⁴ ↦ (0.5, 1.5)⁴)
normal_to_uniform(x) = 0.5 .+ cdf.(Normal(), x)

function prior_to_physical(df)
    m = normal_to_simplex([df.z1, df.z2, df.z3])
    k = normal_to_uniform([df.z4, df.z5, df.z6, df.z7])

    return [m..., k...]
end

# log prior for M-H
function logprior(df)
    vec(
        sum(hcat(map(rv -> logpdf.(Normal(), df[:, rv.name]), prior)...); dims=2),
    )
end
