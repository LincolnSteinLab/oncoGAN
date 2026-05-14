using Extreme.Statistics.Distributions;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record RigmaEventData : ContigEventData
{
    public long Start { get; }
    public List<long> StopsList { get; }
    
    public RigmaEventData(Random rnd, CNEventPars cnEventPars, int contigId, long contigLen) : base(cnEventPars, contigId, contigLen)
    {
        ContigId = contigId;
        long fragSize = Sampling.GetExpSeg(rnd, contigLen, cnEventPars.Frac);
        Start = Sampling.GetPos(rnd, contigLen - fragSize);
        int fracCount = GeometricDistribution.Sample(rnd, 1.0 / cnEventPars.Frag) + 1;
        StopsList = Enumerable.Range(0, fracCount).Select(_ => Sampling.GetExpSeg(rnd, contigLen, cnEventPars.Frag / cnEventPars.Frac)).ToList();
    }
    
    public override void ApplyEvent(Karyotype kar)
        => kar.ApplyRigma(ContigId, Start, StopsList);

    public override string EventDesc()
        => base.EventDesc() + $"start{Start};stops:{string.Join(",", StopsList)}";
}
