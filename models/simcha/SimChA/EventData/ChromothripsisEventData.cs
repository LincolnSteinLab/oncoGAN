using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record ChromothripsisEventData : ContigEventData
{
    public List<long> StopsList { get; }
    public List<int> SelectionList { get; }
    
    public ChromothripsisEventData(Random rnd, CNEventPars cnEventPars, int contigId, long contigLen) : base(cnEventPars, contigId, contigLen)
    {
        ContigId = contigId;
        double chromothripsisLen =  contigLen * cnEventPars.Frac;
        int shardCount = Math.Max(1, contigLen > chromothripsisLen ? Sampling.GetFragCount(rnd, contigLen / chromothripsisLen) : 1);
        StopsList = Sampling.GetStopsForShards(rnd, contigLen, shardCount);
        int shardsKept = rnd.Next(1, StopsList.Count + 1);
        SelectionList = Enumerable.Range(0, shardCount).Shuffle(rnd).Take(shardsKept).ToList();
    }
    
    public override void ApplyEvent(Karyotype kar)
        => kar.ApplyChromothripsis(ContigId, StopsList, SelectionList);

    public override string EventDesc()
        => base.EventDesc() + $"stops:{string.Join(",", StopsList)};selection:{string.Join(",", SelectionList)}";
}