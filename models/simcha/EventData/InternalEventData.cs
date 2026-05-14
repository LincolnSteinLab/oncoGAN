using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record InternalEventData : ContigEventData
{
    public long Start { get; }
    public long End { get; }

    // Constructor used for internal events
    public InternalEventData(Random rnd, CNEventPars CNEventPars, int contigId, long contigLen) : base(CNEventPars, contigId, contigLen)
    {   
        long segLen = Sampling.GetExpSeg(rnd, contigLen, CNEventPars.Frac);
        //Start = rnd.NextInt64(segLen, contigLen-segLen);
        Start = Sampling.GetPos(rnd, contigLen - segLen); 
        End = Start + segLen;
    }

    // Constructor for Centromere-bound events
    public InternalEventData(Random rnd, CNEventPars CNEventPars, int contigId, IEnumerable<(long start, long end)> centromeres, long contigLen) 
        : base(CNEventPars, contigId, contigLen)
    {
        var (start, end) = centromeres.Shuffle(rnd).First();
        var pos = rnd.NextInt64(start, end);
        long segLen = Sampling.GetExpSeg(rnd, contigLen, CNEventPars.Frac);
        if (rnd.CoinFlip())
        {
            Start = pos;
            End = Math.Min(pos + segLen, contigLen);
        }
        else
        {
            Start = Math.Max(0, pos - segLen);
            End = pos;
        }
    }

    public override void ApplyEvent(Karyotype kar)
    {
        switch (EventType)
        {
            case CNEventType.InternalDuplication:
            case CNEventType.CentromereBoundDuplication:
                kar.ApplyInternalDuplication(ContigId, Start, End);
                break;
            case CNEventType.InternalDeletion:
            case CNEventType.CentromereBoundDeletion:
                kar.ApplyInternalDeletion(ContigId, Start, End);
                break;
            case CNEventType.InternalInversion:
                kar.ApplyInternalInversion(ContigId, Start, End);
                break;
            case CNEventType.InvertedDuplication:
                kar.ApplyInvertedDuplication(ContigId, Start, End);
                break;
            default:
                throw new Exception($"Invalid event type {EventType} for InternalEventData");
        }
    }

    public override string EventDesc()
        => base.EventDesc() + $"start:{Start};end:{End}";
}