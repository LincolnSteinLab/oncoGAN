namespace SimChA.IO;

// Parameters used in the MCMC sampling
// Input: 

public record MHParams(
    int NumIterations = 1, // Number of true samples
    double ThetaFitness = 1, // exponential multiplier for fitness
    bool MatchFitness = true // Match the fitness of the clone to a target fitness
);