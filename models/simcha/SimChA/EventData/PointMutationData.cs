using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record PointMutationData : ContigEventData
{
    public long Location { get; }
    public Nucleotide Base { get; }

    public PointMutationData(Random rnd, CNEventPars CNEventPars, int contigId, long contigLen) 
        : base(CNEventPars, contigId, contigLen)
    {
        Location = Sampling.GetPos(rnd, contigLen);
        Base = Sampling.SampleBase(rnd);
    }
    
    public override void ApplyEvent(Karyotype kar)
    {
        kar.ApplyPointMutation(ContigId, Location, Base);
    }

    public override string EventDesc()
        => base.EventDesc() + $"location:{Location};base:{Base}";
}