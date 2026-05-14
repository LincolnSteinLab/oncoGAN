using Extreme.Statistics.Distributions;
using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record TemplatedEventData : BaseEventData
{
    public List<(int id, long start, long len, bool dir)> Frags { get; } = new();
    
    public TemplatedEventData(Random rnd, CNEventPars cnEventPars, IReadOnlyList<(int id, long len)> seq) : base(cnEventPars)
    {
        int contigCount = GeometricDistribution.Sample(rnd, 1.0 / cnEventPars.Frag) 
                          + (cnEventPars.Type != CNEventType.TIBridge ? 1 : 2);
        
        for (int i = 0; i < Math.Min(contigCount, seq.Count); i++)
        {
            int id = seq[i].id;
            long contigLen = seq[i].len;
            // First segment of a bridge, or first and last on a chain do not have a length
            bool skipLen = i == 0 && cnEventPars.Type != CNEventType.TICycle ||
                           i == contigCount - 1 && cnEventPars.Type == CNEventType.TIChain;
            long fragLen = skipLen ? 0L : Sampling.GetExpSeg(rnd, contigLen, cnEventPars.Frac);
            long fragStart = Sampling.GetPos(rnd, contigLen - fragLen);
            bool dir = i == 0 || rnd.CoinFlip();
            Frags.Add((id, fragStart, fragLen, dir));
        }
    }

    public override void ApplyEvent(Karyotype kar)
    {
        if (EventType == CNEventType.TIBridge)
        {
            kar.ApplyTIBridge(Frags);
        }
        else if (EventType == CNEventType.TIChain)
        {
            kar.ApplyTIChain(Frags);
        }
        else if (EventType == CNEventType.TICycle)
        {
            kar.ApplyTICycle(Frags);
        }
    }

    public override string EventDesc()
        => string.Join(",", Frags.Select(x => $"({x.id},{x.start},{x.len},{Region.DirToStr(x.dir)})"));
}