namespace SimChA.Data;

public class GenRange(long start, long end, string chrom)
{
    public long Start { get; protected set; } = start;
    public long End { get; protected set; } = end;
    public string Chrom { get; } = chrom;

    protected GenRange(GenRange other) : this(other.Start, other.End, other.Chrom) { }

    public override bool Equals(object? obj) 
        => obj is GenRange other && Start == other.Start && End == other.End && Chrom == other.Chrom;

    public void Revert()
    {
        long oldStart = Start;
        Start = -End;
        End = -oldStart;
    }
    
    public long Length => End - Start;
    
    public bool Forward => Start >= 0;

    public long AbsStart => Forward ? Start : -End;
    
    public long AbsEnd => Forward ? End : -Start;

    public static string DirToStr(bool dir) => dir ? ">" : "<";

    private string DirString => DirToStr(Forward);
    
    // True if this range is inside the other range
    public bool IsInsideOf(GenRange other) 
        => Chrom == other.Chrom && AbsStart >= other.AbsStart && AbsEnd <= other.AbsEnd;

    // True if this range shares at least one position with the other range
    public bool Overlaps(GenRange other)
        => Start < other.End && End > other.Start && Chrom == other.Chrom;
    
    public override string ToString() 
        => $"{DirString}{Chrom}[{AbsStart}:{AbsEnd})";
}
