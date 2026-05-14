using System.Text;

namespace SimChA.Data;

// Lists prefixed with Sex hold values for all SexTypes
public class RefGen
{
    public string Name { get; }
    public Dictionary<string, int> ChrLengths { get; }
    private Dictionary<string, SexType> ChrSex { get; }
    public Dictionary<string, GenRange> Centromeres { get; }
    public string YChrName { get; }
    public string XChrName { get; }
    public List<List<Region>> SexGenome { get; }
    public List<long> SexGenomeLen { get; }
    public List<long> SexReferenceLen { get; }
    public List<List<string>> SexChromNames { get; }
    public List<string> AllChrNames => SexChromNames[(int)SexType.Male];
    public int AutosomesCount => SexChromNames[(int)SexType.Any].Count;
    private List<Dictionary<string, List<Gene>>> GeneChromMap { get; }
    public List<List<Gene[]>> SexGeneLists { get; }
    private Dictionary<string, StringBuilder>? GenContentsDict { get; }

    public RefGen(
        string name,
        Dictionary<string, int> chrLengths,
        Dictionary<string, SexType> chrSex,
        Dictionary<string, GenRange> centromeres,
        List<Dictionary<string, List<Gene>>> geneChromMap,
        Dictionary<string, StringBuilder>? genContentsDict = null)
    {
        Name = name;
        ChrLengths = chrLengths;
        ChrSex = chrSex;
        Centromeres = centromeres;
        GeneChromMap = geneChromMap;
        GenContentsDict = genContentsDict;

        YChrName = ChrSex.Where(pair => pair.Value == SexType.Male).Select(pair => pair.Key).FirstOrDefault("");
        XChrName = ChrSex.Where(pair => pair.Value == SexType.Female).Select(pair => pair.Key).FirstOrDefault("");
        SexChromNames =
        [
            ChrSex.Where(pair => pair.Value == SexType.Any).Select(pair => pair.Key).ToList(),
            ChrSex.Where(pair => pair.Value != SexType.Male).Select(pair => pair.Key).ToList(),
            ChrSex.Select(pair => pair.Key).ToList()
        ];
        SexGenome = Enum.GetValues(typeof(SexType)).Cast<SexType>().Select(t
                => CreateHaplotype(t, true).Concat(CreateHaplotype(t, false)).ToList())
            .ToList();
        SexGenomeLen = SexGenome.Select(genome => genome.Sum(chromReg => chromReg.Length)).ToList();
        SexReferenceLen = SexChromNames.Select(sexList
                => sexList.Sum(chrName => (long)chrLengths[chrName]))
            .ToList();
        SexGeneLists = SexChromNames.Select(chrs
                => GeneChromMap.Select(map
                        => CreateGeneList(chrs, map).ToArray())
                    .ToList())
            .ToList();
    }

    private static IEnumerable<Gene> CreateGeneList(List<string> chrs, Dictionary<string, List<Gene>> geneChromMap)
        => chrs.SelectMany(chrom => geneChromMap[chrom]).OrderBy(g => g.GeneId);

    public char[] GetGenContents(string chrom, int start, int length)
        => GenContentsDict == null
            ? ['N']
            : [..GenContentsDict[chrom].ToString(start, length)];

    private static Nucleotide CharToNucleotide(char c)
        => char.ToUpper(c) switch
        {
            'A' => Nucleotide.A,
            'C' => Nucleotide.C,
            'G' => Nucleotide.G,
            'T' => Nucleotide.T,
            _ => Nucleotide.N // Default case for invalid characters
        };

    public Nucleotide GetRefBase(string chrom, int location)
        => GenContentsDict == null
            ? Nucleotide.N
            : CharToNucleotide(GenContentsDict[chrom].ToString(location, 1)[0]);

    public Nucleotide GetRefBaseFromSeq(IEnumerable<string> seq, int location)
        => GenContentsDict == null
            ? Nucleotide.N
            : CharToNucleotide(seq.ElementAt(location)[0]);

    private List<string> ChrNamesForHap(SexType sex, bool firstHaplotype = true)
        => (firstHaplotype, sex) switch
        {
            (true, SexType.Male) => SexChromNames[(int)SexType.Female],
            (false, SexType.Male) => SexChromNames[(int)SexType.Any].Concat([YChrName]).ToList(),
            _ => SexChromNames[(int)sex]
        };

    private int GetExpCount(string chrom, SexType sexType)
    {
        if (chrom == XChrName)
        {
            return sexType switch
            {
                SexType.Female => 2,
                SexType.Male => 1,
                _ => 0
            };
        }

        if (chrom == YChrName)
        {
            return sexType == SexType.Male ? 1 : 0;
        }

        return 2;
    }

    public List<int[]> GetInitialGeneCounts(SexType sexType, bool keepEmpty)
    {
        List<int[]> res = [];

        // Filter chromosomes according to sexType
        var filteredGeneChromMap = GeneChromMap.Select(map =>
        {
            var filtered = map.Where(pair =>
                (sexType == SexType.Any && pair.Key != XChrName && pair.Key != YChrName) ||
                (sexType == SexType.Female && pair.Key != YChrName) ||
                (sexType == SexType.Male)
            ).ToDictionary(pair => pair.Key, pair => pair.Value);
            return filtered;
        });

        foreach (var genesPerChrom in filteredGeneChromMap)
        {
            int geneCount = genesPerChrom.Sum(pair => pair.Value.Count);
            int[] typeCounts = new int[geneCount];
            if (!keepEmpty)
            {
                foreach ((string chrom, var genes) in genesPerChrom)
                {
                    // In male the sex chromosomes have CN == 1, otherwise all are 2
                    int expCount = keepEmpty ? 0 : GetExpCount(chrom, sexType);
                    foreach (var gene in genes)
                    {
                        typeCounts[gene.GeneId] = expCount;
                    }
                }
            }
            res.Add(typeCounts);
        }

        return res;
    }

    private IEnumerable<Gene> GetChromGenes(string chrNo)
        => GeneChromMap.SelectMany(chrGenes => chrGenes[chrNo]);

    private Region GetRegion(string chrNo, bool isFirstHaplotype)
        => new(0, ChrLengths[chrNo], chrNo, isFirstHaplotype, null, GetChromGenes(chrNo).ToList());

    private IEnumerable<Region> CreateHaplotype(SexType sex, bool firstHap)
        => ChrNamesForHap(sex, firstHap).Select(chr => GetRegion(chr, firstHap));

    public IEnumerable<Gene> GetGenesBetween(string chrNo, int start, int end)
        => GeneChromMap.SelectMany(chrGenes => chrGenes[chrNo]
            .SkipWhile(g => g.Start < start)
            .TakeWhile(g => g.End <= end)
        );
}