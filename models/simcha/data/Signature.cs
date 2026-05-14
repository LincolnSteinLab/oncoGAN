using SimChA.Computation;
using SimChA.EventData;

namespace SimChA.Data;

[Serializable]
public record Signature(string Name, double Prob, List<CNEventPars> Events) : IHasProb;

