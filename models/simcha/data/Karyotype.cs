using SimChA.Computation;
using SimChA.IO;

namespace SimChA.Data;

public class Karyotype
{
    private RefGen RefGen { get; }
    public SexType Sex { get; }
    public double FitnessVal { get; private set; }

    // NOTE: Empty contigs are retained in the list, but not reported. This way the initial indexing is preserved.
    private readonly List<Contig> _contigs;
    public List<int[]> GeneCounts { get; }

    public Karyotype(RefGen refGen, SexType sex)
    {
        RefGen = refGen;
        _contigs = refGen.SexGenome[(int) sex].Select(reg => new Contig([reg])).ToList();
        Sex = sex;
        GeneCounts = refGen.GetInitialGeneCounts(sex, false);
    }

    public Karyotype(Karyotype other)
    {
        RefGen = other.RefGen;
        _contigs = other._contigs.ConvertAll(ch => new Contig(ch));
        Sex = other.Sex;
        FitnessVal = other.FitnessVal;
        GeneCounts = other.GeneCounts.Select(geneCounts => (int[]) geneCounts.Clone()).ToList();
    }

    public Karyotype(RefGen refGen, List<Contig> contigs, SexType sex)
    {
        RefGen = refGen;
        _contigs = contigs;
        Sex = sex;
        GeneCounts = refGen.GetInitialGeneCounts(sex, true);
        foreach (var contig in _contigs)
        {
            AddGenes(contig);
        }
    }

    public int CountContigs()
        => _contigs.Count(c => c.Any());

    public long GenomeLen()
        => _contigs.Sum(c => c.Length);

    public IEnumerable<SNV> GetSNVs()
        => _contigs.SelectMany(c => c.SNVs);

    public Dictionary<string, List<int>> CalcBreaks()
    {
        var breakSets = RefGen.SexChromNames[(int) Sex].ToDictionary(c => c, c => new HashSet<int> {0, RefGen.ChrLengths[c]});
        foreach (var contig in _contigs)
        {
            foreach ((string chrom, var breaks) in contig.CalcBreaks())
            {
                breakSets[chrom].UnionWith(breaks);
            }
        }
        return breakSets.ToDictionary(k => k.Key, v => v.Value.OrderBy(i => i).ToList());
    }

    public List<CopyNumber> CalcCNs(IDictionary<string, List<int>> allBreaks)
        => CopyNumbers.CalcCNs(allBreaks, _contigs);

    public IEnumerable<int> ContigIds()
        => _contigs.Select((c, i) => (c, i)).Where(t => t.c.Any()).Select(t => t.i);


    public Dictionary<int, List<string>> GetRegionDescriptions()
        => ContigIds().ToDictionary(id => id, id => _contigs[id].GetRegionDescriptions());

    public override string ToString()
        => CountContigs() > 0 ? "[" + string.Join(";", _contigs) + "]" : "[]";

    // @CODY this should be made private, needs to update tests
    public IEnumerable<Region> FindChrRegions(string chrNo)
        => _contigs.SelectMany(c => c.FindChrRegions(chrNo));

    public long ContigLen(int contigId)
        => contigId < _contigs.Count ? _contigs[contigId].Length : 0;

    public List<(long start, long end)> GetCentromeres(int contigId)
        => _contigs[contigId].GetCentromeres(RefGen.Centromeres);

    private static (long start, long end) GetIndices(Contig contig, long position, bool fiveToThree)
        => fiveToThree ? (0, position) : (position, contig.Length);

    // @CODY this should be made private, needs to update test
    public Contig GetContig(int contigID)
        => _contigs[contigID];

    public IEnumerable<string> GetSeq()
        => _contigs.SelectMany(c => c.GetSeq(RefGen).Concat(new List<string> {"\n"}));
    
    public double UpdateFitness(RefGen refGen, FitParams fParams)
        => FitnessVal = Fitness.Calculate(this, refGen, fParams);

    public void ApplyTailDeletion(int contigID, long start, bool direction)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        (long tailStart, long tailEnd) = GetIndices(contig, start, direction);
        contig.DeleteRange(tailStart, tailEnd);
        AddGenes(contig);
    }

    public void ApplyTailDuplication(int contigID, long start, bool direction)
    {
        var contig = _contigs[contigID];
        (long tailStart, long tailEnd) = GetIndices(contig, start, direction);
        var newTail = new Contig(contig.GetSubContig(tailStart, tailEnd));
        _contigs.Add(newTail);
        AddGenes(newTail);
    }

    public void ApplyBFB(int contigID, long start, bool direction)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.Bridge(start, direction);
        AddGenes(contig);
    }

    public void ApplyContigDeletion(int contigID)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.Clear();
    }

    public void ApplyContigDuplication(int contigID)
    {
        var contig = _contigs[contigID];
        _contigs.Add(new Contig(contig));
        AddGenes(contig);
    }

    public void ApplyInternalDuplication(int contigID, long startPos, long endPos)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.DuplicateRange(startPos, endPos);
        AddGenes(contig);
    }

    public void ApplyInvertedDuplication(int contigID, long startPos, long endPos)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.DuplicateRange(startPos, endPos);
        contig.InvertRange(endPos, endPos + (endPos - startPos));
        AddGenes(contig);
    }

    public void ApplyInternalInversion(int contigID, long startPos, long endPos)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.InvertRange(startPos, endPos);
        AddGenes(contig);
    }

    public void ApplyInternalDeletion(int contigID, long startPos, long endPos)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.DeleteRange(startPos, endPos);
        AddGenes(contig);
    }

    // Translocation might invert based on the orientation of the holiday Junction https://en.wikipedia.org/wiki/Holliday_junction
    public void ApplyTranslocation(int contigA, int contigB, long posA, long posB, bool inverted)
    {
        var refContig = _contigs[contigA];
        RemoveGenes(refContig);
        var altContig = _contigs[contigB];
        RemoveGenes(altContig);
        if (inverted)
        {
            altContig.Revert();
        }

        var splitRef = refContig.Split(posA, true);
        var splitAlt = altContig.Split(posB, true);
        refContig.Join(splitAlt);
        AddGenes(refContig);
        altContig.Join(splitRef);
        AddGenes(altContig);
    }

    public void ApplyWGD()
    {
        _contigs.AddRange(_contigs.Select(ch => new Contig(ch)).ToList());
        DoubleGeneCounts();
    }

    public void ApplyChromothripsis(int contigID, List<long> stops, IEnumerable<int> selection)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        contig.ScatterAndGather(stops, selection);
        AddGenes(contig);
    }

    public void ApplyChromoplexy(List<int> contigIDs, List<List<long>> stops, IEnumerable<int> sequence,
        List<long> breakpoints)
    {
        var subContigs =
            Enumerable.Range(0, contigIDs.Count)
                .Select(i => _contigs[contigIDs[i]].Scatter(stops[i]))
                .SelectMany(x => x)
                .ToList();
        foreach (int id in contigIDs)
        {
            RemoveGenes(_contigs[id]);
            _contigs[id].Clear();
        }
        var ordered = Contig.Concat(sequence.Select(i => subContigs[i]));
        var newContigs = ordered.Scatter(breakpoints).ToList();
        for (int i = 0; i < newContigs.Count; i++)
        {
            if (i < contigIDs.Count)
            {
                int id = contigIDs[i];
                _contigs[id] = newContigs[i];
                AddGenes( _contigs[id] );
            }
            else {
                _contigs.Add(newContigs[i]);
                AddGenes( newContigs[i] );
            }
        }
    }

    public void ApplyPyrgo(int contigID, List<(long start, long len)> frags)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        long offset = 0;
        foreach ((long start, long len) in frags)
        {
            contig.DuplicateRange(start + offset, start + offset + len);
            offset += len;
        }
        AddGenes(contig);
    }

    // Fragments that do not return to the original chromosome
    public void ApplyTIChain(List<(int id, long start, long len, bool dir)> frags)
    {
        var template = new Contig();
        foreach (var frag in frags)
        {
            template.AppendContig(_contigs[frag.id].GetSubContig(frag.start, frag.start + frag.len));
        }
        _contigs.Add(template);
        AddGenes(template);
    }

    // First segment is the host, but there is no repetition
    public void ApplyTIBridge(List<(int id, long start, long len, bool dir)> frags)
    {
        var host = _contigs[frags[0].id];
        RemoveGenes(host);
        var template = new Contig();
        foreach (var frag in frags.Skip(1))
        {
            var contig = _contigs[frag.id].GetSubContig(frag.start, frag.start + frag.len, frag.dir);
            template.AppendContig(contig);
        }

        host.InsertContig(template, frags[0].start);
        AddGenes(host);
    }

    // First segment is the host, with repetition
    public void ApplyTICycle(List<(int id, long start, long len, bool dir)> frags)
    {
        var host = _contigs[frags[0].id];
        RemoveGenes(host);
        var template = new Contig();
        foreach (var newSection in frags.Select(frag 
                     => _contigs[frag.id].GetSubContig(frag.start, frag.start + frag.len, frag.dir)))
        {
            template.AppendContig(newSection);
        }
        host.InsertContig(template, frags[0].start);
        AddGenes(host);
    }

    public void ApplyRigma(int contigID, long rigmaStart, List<long> rigmaLens)
    {
        var contig = _contigs[contigID];
        RemoveGenes(contig);
        bool lastWasDeletion = false;
        foreach (long len in rigmaLens)
        {
            if (lastWasDeletion)
            {
                rigmaStart += len;
            }
            else
            {
                contig.DeleteRange(rigmaStart, rigmaStart + len);
            }

            lastWasDeletion = !lastWasDeletion;
        }
        AddGenes(contig);
    }

    public void ApplyPointMutation(int contigID, long location, Nucleotide newNucleotide)
    {
        // NOTE: PointMutations do not currently affect GeneCounts list
        var contig = _contigs[contigID];
        var oldNucleotide = RefGen.GetRefBaseFromSeq(contig.GetSeq(RefGen), (int) location);
        contig.PointMutate(location, oldNucleotide, newNucleotide);
    }

    public void MergeRegions()
    {
        _contigs.ForEach(c => c.MergeRegions());
    }
    
    private void RemoveGenes(Contig contig)
    {
        foreach (var gene in contig.Genes)
        {
            GeneCounts[(int) gene.ListType][gene.GeneId] -= 1;
        }
    }

    private void AddGenes(Contig contig)
    {
        foreach (var gene in contig.Genes)
        {
            GeneCounts[(int) gene.ListType][gene.GeneId] += 1;
        }
    }
    
    private void DoubleGeneCounts()
    {
        foreach (int[] listCount in GeneCounts)
        {
            for (int i = 0; i < listCount.Length; i++)
            {
                listCount[i] *= 2;
            }
        }
    }
}