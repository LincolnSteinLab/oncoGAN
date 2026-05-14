using System.Linq;
using SimChA.Data;

namespace SimChA.Computation;

public static class RegionOps
{
    private static void AddIfNotEmpty(ICollection<Region> regions, Region region)
    {
        if (region.Length > 0)
        {
            regions.Add(region);
        }
    }
    
    public static List<Region> Copy(List<Region> regions)
        => regions.ConvertAll(reg => new Region(reg));
    
    public static List<Region> DeleteRange(List<Region> regions, long start, long end)
    {
        var newRegions = new List<Region>();
        long seekPos = 0;
        foreach (var region in regions)
        {
            if (start >= seekPos + region.Length) // region before start
            {
                var newRegion = new Region(region);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else if (end <= seekPos) // region after end
            {
                var newRegion = new Region(region);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else if (start < seekPos && end > seekPos + region.Length) // Whole region inside deletion
            {
            }
            else if (end > seekPos + region.Length) // star inside, end outside of region
            {
                var newRegion = new Region(region);
                newRegion.ResizeBack(start - seekPos);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else if (start < seekPos) // start before the region, end inside the region
            {
                var newRegion = new Region(region);
                newRegion.ResizeFront(end - seekPos);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else // None coordinates inside of the region
            {
                var first = new Region(region);
                first.ResizeBack(start - seekPos);
                AddIfNotEmpty(newRegions, first);

                var second = new Region(region);
                second.ResizeFront(end - seekPos);
                AddIfNotEmpty(newRegions, second);
            }

            seekPos += region.Length;
        }

        return newRegions;
    }

    public static List<Region> CopyRange(List<Region> regions, long start, long end)
    {
        var newRegions = new List<Region>();
        long seekPos = 0;
        foreach (var region in regions)
        {
            if (start >= seekPos + region.Length) // region before start
            {
            }
            else if (end <= seekPos) // region after end
            {
                break;
            }
            else if (start < seekPos && end > seekPos + region.Length) // Whole region inside copy
            {
                var newRegion = new Region(region);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else if (end > seekPos + region.Length) // end outside of region
            {
                var newRegion = new Region(region);
                newRegion.ResizeFront(start - seekPos);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else if (start < seekPos) // start before the region
            {
                var newRegion = new Region(region);
                newRegion.ResizeBack(end - seekPos);
                AddIfNotEmpty(newRegions, newRegion);
            }
            else // None coordinates inside of the region
            {
                var newRegion = new Region(region);
                newRegion.ResizeBoth(start - seekPos, end - seekPos);
                AddIfNotEmpty(newRegions, newRegion);
            }
            seekPos += region.Length;
        }

        return newRegions;
    }

    public static (List<Region>, List<Region>) SplitRegions(List<Region> regions, long pos)
    {
        long seekPos = 0;
        var beforeRegions = new List<Region>();
        var afterRegions = new List<Region>();
        foreach (var region in regions)
        {
            if (pos > seekPos + region.Length) // region before pos
            {
                var newRegion = new Region(region);
                AddIfNotEmpty(beforeRegions, newRegion);
            }
            else if (pos <= seekPos) // region after pos
            {
                var newRegion = new Region(region);
                AddIfNotEmpty(afterRegions, newRegion);
            }
            else // split inside the region
            {
                var first = new Region(region);
                first.ResizeBack(pos - seekPos);
                AddIfNotEmpty(beforeRegions, first);
                
                var second = new Region(region);
                second.ResizeFront(pos - seekPos);
                AddIfNotEmpty(afterRegions, second);
            }

            seekPos += region.Length;
        }
        return (beforeRegions, afterRegions);
    }
    
    public static void PointMutateRegion(List<Region> regions, long location, 
        Nucleotide oldNucleotide, Nucleotide newNucleotide)
    {
        long seekPos = 0;
        foreach (var region in regions)
        {
            if (location >= seekPos && location < seekPos + region.Length)
            {
                region.AddSNV(location - seekPos, oldNucleotide, newNucleotide);
            }
            seekPos += region.Length;
        }        
    }

    public static void Revert(List<Region> regions)
    {
        foreach (var region in regions)
        {
            region.Revert();
        }
        regions.Reverse();
    }

    public static List<Region> ConcatRegions(IEnumerable<IEnumerable<Region>> listOfRegions)
        => listOfRegions.SelectMany(x => x).ToList();

    public static List<Region> ConcatRegions(IEnumerable<Region> first, IEnumerable<Region> second)
        => first.Concat(second).ToList();
    
    public static long GetLength(IEnumerable<Region> regions)
        => regions.Sum(r => r.Length);
    
    public static List<List<Region>> Scatter(List<long> locs, List<Region> regions)
    {
        if (locs.Count == 0) 
        {
            return [regions];
        }
        
        // First region
        var newRegions = new List<List<Region>> { CopyRange(regions, 0, locs[0]) };
        
        // Internal regions
        for (int i = 0; i < locs.Count - 1; i++)
        {
            long start = locs[i];
            long end = locs[i + 1];
            var copy = CopyRange(regions, start, end);
            newRegions.Add(copy);
        }
        // Last region
        newRegions.Add(CopyRange(regions, locs.Last(), CountLength(regions)));
        return newRegions;
    }
    
    public static List<Region> Gather(List<List<Region>> newRegions, IEnumerable<int> indices) 
        => ConcatRegions(indices.Select(i => newRegions[i]));
    
    public static List<Region> MergeRegions(List<Region> regions)
    {
        var newRegions = new List<Region>();
        for (int i = 0; i < regions.Count; i++)
        {
            var cur = regions[i];
            if (i == 0)
            {
                newRegions.Add(cur);
            }
            else
            {
                var last = newRegions[^1];
                if (last.Chrom == cur.Chrom && last.End == cur.Start)
                {
                    last.MergeWithNext(cur);
                }
                else
                {
                    newRegions.Add(cur);
                }
            }
        }
        return newRegions;
    }

    public static long CountLength(IEnumerable<Region> regions)
        => regions.Sum(r => r.Length);

}
