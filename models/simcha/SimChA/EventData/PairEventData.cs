using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record PairEventData : BaseEventData
{
    public readonly int ContigIdA = -1;
    public readonly int ContigIdB = -1;
    public readonly long PosA = -1;
    public readonly long PosB = -1;
    public readonly bool Inverted;
    
    // Constructor used for Translocation
    public PairEventData(Random rnd, CNEventPars cnEventPars, int contigA, long lenA, int contigB, long lenB) : base(cnEventPars)
    {
        ContigIdA = contigA;
        PosA = Sampling.GetPos(rnd, lenA);
        ContigIdB = contigB;
        PosB = Sampling.GetPos(rnd, lenB);
        Inverted = rnd.CoinFlip();
    }
    
    public override void ApplyEvent(Karyotype kar)
        => kar.ApplyTranslocation(ContigIdA, ContigIdB, PosA, PosB, Inverted);
    
    public override string EventDesc()
        => $"contigA:{ContigIdA};contigB:{ContigIdB};posA:{PosA};posB:{PosB};dir:{Inverted}";
}