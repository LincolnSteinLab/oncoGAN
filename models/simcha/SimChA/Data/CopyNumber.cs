namespace SimChA.Data;

public class CopyNumber(long start, long end, string chrom, int cnh1, int cnh2, int nsnvs)
    : GenRange(start, end, chrom)
{
    public int CNH1 { get; } = cnh1;
    public int CNH2 { get; } = cnh2;
    public int NSNVs { get; } = nsnvs;

    private static string NaIfNegStr(int val)
        => val < 0 ? "NA" : $"{val}";

    public static string Header()
        => "chrom\tstart\tend\tcn_a\tcn_b\tn_snvs";
    
    public string ToTSV()
        => string.Join('\t', Chrom, Start + 1, End, NaIfNegStr(CNH1), NaIfNegStr(CNH2), NaIfNegStr(NSNVs));
}
