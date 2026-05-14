using SimChA.Data;

namespace SimChA.Computation;

public static class CopyNumbers
{
    public static Dictionary<string, List<int>> GetJointSegmentation(List<string> chromNames, List<Sample> samples)
    {
        var breaks = samples.Select(s => s.Karyotype.CalcBreaks()).ToList();
        var segmentation = chromNames.ToDictionary(
            chrom => chrom, 
            chrom => breaks.SelectMany(br => br.GetValueOrDefault(chrom, [])).ToHashSet().OrderBy(b => b).ToList());
        return segmentation;
    }

    public static List<CopyNumber> CalcCNs(IDictionary<string, List<int>> allBreaks, List<Contig> contigs)
    {
        var result = new List<CopyNumber>();
        foreach ((string chrom, var breaks) in allBreaks)
        {
            for (int i = 0; i < breaks.Count - 1; i++)
            {
                int start = breaks[i];
                int end = breaks[i + 1];
                var seg = new GenRange(start, end, chrom);
                var cns = contigs.Select(c => c.GetCNs(seg));
                (int cnA, int cnB, int nSNVs) = cns.Aggregate((CNA: 0, CNB: 0, SNV: 0), (acc, vals)
                    => (acc.CNA + vals.CNA, acc.CNB + vals.CNB, acc.SNV + vals.SNV));
                var cn = new CopyNumber(start, end, chrom, cnA, cnB, nSNVs);
                result.Add(cn);
            }
        }
        return result;
    }

    public static List<CopyNumber> CalcCNs(Karyotype karyotype, IDictionary<string, List<int>>? breaks = null) 
        => karyotype.CalcCNs(breaks ?? karyotype.CalcBreaks());

    public static double CalcPloidy(RefGen refGen, List<CopyNumber> copyNumbers, SexType sex)
        => 2 * copyNumbers.Select(c => c.Length * (c.CNH1 + c.CNH2)).Sum() / (float) refGen.SexGenomeLen[(int)sex];
}
