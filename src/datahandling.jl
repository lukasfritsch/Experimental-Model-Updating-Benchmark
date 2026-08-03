function load_frequencies(filename::String)
    """
    Load frequency measurements from file.
    Reads columns 2-5 (the 4 natural frequencies).

    Returns: measurements × 4 matrix
    """
    lines = readlines(filename)
    data = []

    for line in lines
        # Split by whitespace and filter empty strings
        parts = split(strip(line))
        # Skip first column (measurement number), take columns 2-5
        freqs = parse.(Float64, parts[2:5])
        push!(data, freqs)
    end

    return permutedims(hcat(data...))  # transpose to get measurements × 4
end
