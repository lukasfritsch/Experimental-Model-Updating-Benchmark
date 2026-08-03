function dampedfourDofmodel(θ)
    # ------------------------------------------------------------------
    # Reference mass values
    # Measured mass of each block: 112.2 g = 0.1122 kg
    # ------------------------------------------------------------------
    m_ref = fill(0.1122, 4)             # [kg]
    ζ = [0.005979, 0.012586, 0.010997, 0.009148]

    # ------------------------------------------------------------------
    # Material and geometric properties (measured)
    # ------------------------------------------------------------------
    E = 2.1e11                          # Young's modulus [N/m^2]
    L = 47.5e-3                         # Storey height [m]
    h = 0.45e-3                         # Steel ruler thickness [m]
    b = 25.5e-3                         # Steel ruler width [m]

    # Second moment of area for rectangular cross-section
    I = b * h^3 / 12                    # [m^4]

    # ------------------------------------------------------------------
    # Reference stiffness per storey
    # Euler–Bernoulli beam in bending:
    # k = 12 E I / L^3
    # ------------------------------------------------------------------
    k_ref = fill(12 * E * I / L^3, 4)   # [N/m]

    # ------------------------------------------------------------------
    # Apply Bayesian scaling parameters
    # ------------------------------------------------------------------
    m = θ[1:4] .* m_ref                 # [kg]
    k = θ[5:8] .* k_ref                 # [N/m]

    # ------------------------------------------------------------------
    # Mass matrix
    # ------------------------------------------------------------------
    M = Diagonal(m)                     # [kg]

    # ------------------------------------------------------------------
    # Stiffness matrix (shear-building assumption, fixed base)
    # Factor 2 accounts for two parallel bending elements per storey
    # ------------------------------------------------------------------
    K = [
        2*(k[1] + k[2])   -2*k[2]           0.0              0.0;
        -2*k[2]           2*(k[2] + k[3])   -2*k[3]           0.0;
        0.0               -2*k[3]           2*(k[3] + k[4])  -2*k[4];
        0.0                0.0              -2*k[4]           2*k[4]
    ]                                   # [N/m]

    # ------------------------------------------------------------------
    # Generalized eigenvalue problem
    # K φ = ω² M φ
    # ------------------------------------------------------------------
    F = eigen(K,M)
    ω² = F.values
    Φ = F.vectors

    # Damping
    C = M * Φ * (2* diagm(ζ) * diagm(sqrt.(ω²))) * Φ' * M

    λ, _ = polyeig_quadratic(M, C, K)

    fd = imag(λ) ./ 2π

    return sort(fd[fd .> 0])
end

function polyeig_quadratic(M, C, K)
    n = size(M, 1)

    Z = zeros(n, n)
    # I = Matrix(I, n, n)

    A = [ -C  -K
           I   Z ]

    B = [  M   Z
           Z   I ]

    λ, X = eigen(A, B)
    return λ, X[1:n, :]   # physical eigenvectors
end
