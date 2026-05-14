namespace SimChA.Data;

public class GeneCount(Gene gene, int count = 0)
{
    public Gene Gene { get; } = gene;
    public int Count = count;
}