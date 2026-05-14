namespace SimChA.IO;

public record EvoParams(
    double Acceptance = 0, 
    int MaxTries = 1 // Maximum number of tries to generate a new set of events in a time step
);
