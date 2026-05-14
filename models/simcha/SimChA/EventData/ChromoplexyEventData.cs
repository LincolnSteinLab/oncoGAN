using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record ChromoplexyEventData : BaseEventData
{
    public List<int> ContigIds { get; } // Contigs that are involved in the event
    public List<List<long>> Stops { get; } // Breakpoints for each contig
    public List<int> Sequence { get; } // Sequence of shards (given by the number of pieces in each contig)
    public List<long> Breakpoints { get; } // Breakpoints for the whole event - shards are first joined then split again
    
    public ChromoplexyEventData(Random rnd, CNEventPars cnEventPars, IReadOnlyList<(int id, long len)> seq) : base(cnEventPars)
    {
        int contigCount = Math.Min(seq.Count, Sampling.GetChromoplexySiteCount(rnd));
        ContigIds = new List<int>();
        Stops = new List<List<long>>(); 
        long totalLen = 0L;
        int totalFrags = 0;
        for (int i = 0; i < contigCount; i++)
        {
            int partsCount = (int) Math.Min(Sampling.GetFragCount(rnd,  1 / cnEventPars.Frac), seq[i].len - 2);
            if (partsCount > 1)
            {
                totalLen += seq[i].len;
                totalFrags += partsCount;
                ContigIds.Add(seq[i].id);
                Stops.Add(Sampling.GetStopsForShards(rnd, seq[i].len, partsCount));
            }
        }
        Sequence = Enumerable.Range(0, totalFrags).Shuffle(rnd).ToList();
        int restShards = (int)Math.Max(Sampling.GetFragCount(rnd, cnEventPars.Frag), 1);
        Breakpoints = Sampling.GetStopsForShards(rnd, totalLen, restShards);
    }

    public override void ApplyEvent(Karyotype kar)
        => kar.ApplyChromoplexy(ContigIds, Stops, Sequence, Breakpoints);

    public override string EventDesc()
        => $"contigs:[{string.Join(",", ContigIds)}];" +
           $"stops:[{string.Join(",", Stops.Select(s => $"[{string.Join(",", s)}]"))}];" +
           $"sequence:[{string.Join(",", Sequence)}];" +
           $"breakpoints:[{string.Join(",", Breakpoints)}]";
}