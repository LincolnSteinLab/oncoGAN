namespace SimChA.Data;

public class Gene(long start, long end, string chrom, GeneLT listType, int geneId, double score)
    : GenRange(start, end, chrom)
{
    public GeneLT ListType { get; } = listType;
    public int GeneId { get; } = geneId; // Position within a list for my type
    public double Score { get; } = score;
}