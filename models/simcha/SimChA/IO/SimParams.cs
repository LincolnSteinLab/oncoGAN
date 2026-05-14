using System.Reflection;
using SimChA.Computation;
using SimChA.Data;

namespace SimChA.IO;

public record SimParams(
    int Seed = 0,
    string Assembly = "hg19",
    SexType Sex = SexType.Any,
    DistType RateDist = DistType.Uniform,
    double RateMean = 1,
    bool TetraploidStart = false,
    bool AutosomesOnly = false,
    MixtureType Mixture = MixtureType.Constant
);