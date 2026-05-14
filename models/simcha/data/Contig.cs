using SimChA.Computation;

namespace SimChA.Data;

public class Contig
{
    private long _length = -1;
    public long Length 
        => _length < 0 ? _length = RegionOps.CountLength(_regions) : _length;

    private List<SNV>? _snvs;
    public List<SNV> SNVs 
        => _snvs ??= _regions.SelectMany(r => r.SNVs).ToList();

    private List<Gene>? _genes;
    public List<Gene> Genes 
        => _genes ??= _regions.SelectMany(r => r.Genes).ToList();
    
    private List<Region> _regions;
    private List<Region> Regions
    {
        get => _regions;
        set
        {
            _regions = value;
            _length = -1;
            _snvs = null;
            _genes = null;
        }
    }

    public Contig()
    {
        _regions = [];
    }

    public Contig(IEnumerable<Region> regions) 
        => _regions = regions.ToList();

    public Contig(Contig other)
    {
        _regions = [..other._regions];
        _length = other._length;
        _snvs = other._snvs == null ? null : [..other._snvs];
        _genes =  other._genes == null ? null : [..other._genes];
    }

    public static Contig Concat(IEnumerable<Contig> contigs)
        => new(contigs.SelectMany(c => c.Regions));
    
    public bool Any()
        => Length > 0;
    
    public IEnumerable<string> GetSeq(RefGen refGen) 
        => Regions.Select(r => r.GetSeq(refGen)).ToList();
    
    public int CountRegions()
        => Regions.Count;

    public (int CNA, int CNB, int SNV) GetCNs(GenRange segRegion)
    {
        int cna = 0;
        int cnb = 0;
        int snvs = 0;
        foreach (var reg in Regions.Where(segRegion.IsInsideOf))
        {
            if (reg.Hap1)
            {
                cna += 1;
            }
            else
            {
                cnb += 1;
            }
            snvs += reg.CountSNVs(reg.Start, reg.End);
        }
        return (cna, cnb, snvs);
    }

    public Dictionary<string, List<int>> CalcBreaks()
    {
        Dictionary<string, List<int>> breaks = new();
        foreach (var region in Regions)
        {
            if (!breaks.ContainsKey(region.Chrom))
            {
                breaks[region.Chrom] = [];
            }
            breaks[region.Chrom].Add((int)region.AbsStart);
            breaks[region.Chrom].Add((int)region.AbsEnd);
        }
        return breaks;
    }
    

    public static string ToString(IEnumerable<Region> regions)
        => "[" + string.Join("~", regions.Select(r => r.ToString())) + "]";

    public override string ToString()
        => ToString(Regions);

    public List<string> GetRegionDescriptions()
        => Regions.Select(r => r.ToString()).ToList();

    public IEnumerable<Region> FindChrRegions(string chrNo)
        => Regions.Where(r => r.Chrom == chrNo);

    public void Clear()
    {
        Regions = [];
    }

    public void DeleteRange(long start, long end)
    { 
        Regions = RegionOps.DeleteRange(Regions, start, end);
    }

    public Contig Split(long pos, bool keepFirst)
    {
        var (first, second) = RegionOps.SplitRegions(Regions, pos);
        Regions = keepFirst ? first : second;
        return new Contig(keepFirst ? second : first);
    }

    public void Join(Contig other)
    {
        Regions = RegionOps.ConcatRegions(Regions, other.Regions);
    }

    public void InvertRange(long invStart, long invEnd)
    {
        var inverse = RegionOps.CopyRange(Regions, invStart, invEnd);
        RegionOps.Revert(inverse);
        var deleted = RegionOps.DeleteRange(Regions, invStart, invEnd);
        var (first, second) = RegionOps.SplitRegions(deleted, invStart);
        Regions = RegionOps.ConcatRegions([first, inverse, second]);
    }

    public void Revert()
    {
        RegionOps.Revert(Regions);
    }

    public void DuplicateRange(long start, long end)
    {
        // TODO: Should be replace with TakeUntil and TakeFrom
        var (_, second) = RegionOps.SplitRegions(Regions, start);
        var (first, _) = RegionOps.SplitRegions(Regions, end);
        Regions = RegionOps.ConcatRegions([first, second]);
    }

    public void Bridge(long pos, bool cutFront)
    {
        var (first, second) = RegionOps.SplitRegions(Regions, pos);
        if (cutFront)
        {
            var inverse = RegionOps.Copy(second);
            RegionOps.Revert(inverse);
            Regions = RegionOps.ConcatRegions(inverse, second);
        }
        else
        {
            var inverse = RegionOps.Copy(first);
            RegionOps.Revert(inverse);
            Regions = RegionOps.ConcatRegions(first, inverse);
        }
    }

    public IEnumerable<Contig> Scatter(List<long> locs)
    {
        var regions = RegionOps.Scatter(locs, Regions);
        return regions.Select(r => new Contig(r));
    }

    public void ScatterAndGather(List<long> locs, IEnumerable<int> indices)
    {
        var newRegions = RegionOps.Scatter(locs, Regions);
        Regions = RegionOps.Gather(newRegions, indices);
    }

    public Contig GetSubContig(long start, long end, bool inverse = false)
    {
        var ranges = RegionOps.CopyRange(Regions, start, end);
        if (inverse)
        {
            RegionOps.Revert(ranges);
        }
        return new Contig(ranges);
    }

    public void InsertContig(Contig other, long location)
    {
        if (location >= other.Length - 1)
        {
            AppendContig(other);
        }
        else
        {
            var (first, second) = RegionOps.SplitRegions(Regions, location);
            Regions = RegionOps.ConcatRegions([first, other.Regions, second]);
        }
    }

    public void AppendContig(Contig other)
    {
        Regions = RegionOps.ConcatRegions(Regions, other.Regions);
    }
    
    public void PointMutate(long location, Nucleotide oldNucleotide, Nucleotide newNucleotide)
    {
        RegionOps.PointMutateRegion(Regions, location, oldNucleotide, newNucleotide);
        _snvs = null;
    }

    public List<(long start, long end)> GetCentromeres(Dictionary<string, GenRange> centMap)
    {
        List<(long start, long end)> centromereList = [];
        long currentPos = 0;
        foreach (var reg in Regions)
        {
            var cent = centMap[reg.Chrom];
            if (cent.IsInsideOf(reg))
            {
                centromereList.Add(reg.Forward
                    ? (currentPos + cent.Start - reg.AbsStart, currentPos + cent.End - reg.AbsStart)
                    : (currentPos - cent.End + reg.AbsEnd, currentPos - cent.Start + reg.AbsEnd));
            }
            currentPos += reg.Length;
        }
        return centromereList;
    }

    public void MergeRegions()
    {
        Regions = RegionOps.MergeRegions(Regions);
    }
    
    public int CountGeneType(GeneLT geneType)
        => Genes.Count(g => g.ListType == geneType);
}
