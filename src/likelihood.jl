function likelihood(df::DataFrame, data::Matrix{Float64})

    # Check if σ is also a parameter to infer
    if "σ" ∈ names(df)
        σ_scale = df.σ
    else
        σ_scale = 8 * ones(nrow(df))
    end

    lkl = zeros(size(df, 1))
    Σ = cov(data)
    Σinv = inv(Σ)
    logdetΣ = logdet(Σ)

    m, N = size(data)

    for (xi, row) in enumerate(eachrow(df))

        pred = row.f

        mse = sum([(datapoint .- pred)' * Σinv * (datapoint .- pred) for datapoint in eachrow(data)])
        lkl[xi] = -1 / 2 * (1 / σ_scale[xi] .^ 2 * mse + m * (N * log(σ_scale[xi] .^ 2) + logdetΣ + N * log(2π)))

    end

    return lkl
end
