using SimChA.Computation;

namespace SimChA.EventData;

[Serializable]
public record CNEventPars(CNEventType Type, double Prob, double Frac = 0, double Frag = 0, string Signature = "") : IHasProb;
