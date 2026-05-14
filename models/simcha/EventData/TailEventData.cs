using SimChA.Computation;
using SimChA.Data;
using SimChA.Simulation;

namespace SimChA.EventData;

public record TailEventData : ContigEventData
{
    public bool Direction { get; }
    public long Start { get; }
    
    // Constructor used for Tail CNEventPars
    public TailEventData(Random rnd, CNEventPars CNEventPars, int contigId, long contigLen) 
        : base(CNEventPars, contigId, contigLen)
    {
        long segLen = Sampling.GetExpSeg(rnd, contigLen, CNEventPars.Frac);
        Direction = rnd.CoinFlip();
        Start = Direction ? segLen : contigLen - segLen;
    }
    
    public TailEventData(Random rnd, CNEventPars CNEventPars, int contigId, IEnumerable<(long start, long end)> cents, long contigLen)
        : this(CNEventPars, contigId, ComputeArmParams(rnd, cents, contigLen), contigLen)
    { }
    
    private TailEventData(CNEventPars CNEventPars, int contigId, (long segLen, bool direction) arm, long contigLen)
        : base(CNEventPars, contigId, contigLen)
    {
        Direction = arm.direction;
        Start = Direction ? arm.segLen : contigLen - arm.segLen;
    }
    
    private static (long segLen, bool direction) ComputeArmParams(
        Random rnd, IEnumerable<(long start, long end)> cents, long contigLen)
    {
        var cent = cents.Shuffle(rnd).First();
        bool direction = rnd.CoinFlip();
        long length = direction ? cent.start : contigLen - cent.end;
        long breakpoint = Math.Max(1, rnd.NextInt64(cent.end - cent.start)); // Somewhere within the centromere, uniform 
        length += breakpoint;
        return (length, direction);
    }

    public override void ApplyEvent(Karyotype kar)
    {
        switch (EventType)
        {
            case CNEventType.TailDeletion:
            case CNEventType.ArmDeletion:
                kar.ApplyTailDeletion(ContigId, Start, Direction);
                break;
            case CNEventType.TailDuplication:
            case CNEventType.ArmDuplication:
                kar.ApplyTailDuplication(ContigId, Start, Direction);
                break;
            case CNEventType.BreakageFusionBridge:
                kar.ApplyBFB(ContigId, Start, Direction);
                break;
            default:
                throw new Exception($"Invalid event type {EventType} for TailEventData");
        }
    }
    
    public override string EventDesc()
        => base.EventDesc() + $"start:{(Direction ? 0 : Start)};end:{(Direction ? Start : Length)};";
}
