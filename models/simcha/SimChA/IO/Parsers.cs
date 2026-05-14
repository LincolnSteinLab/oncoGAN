using System.Globalization;
using System.Text.Json;
using System.Text;
using System.Text.RegularExpressions;
using SimChA.Data;

namespace SimChA.IO;

public static class Parsers
{
    public static SimChAConfig ParseSimParams(string serializedJSON)
    {
        SimChAConfig? res;
        var options = new JsonSerializerOptions
        {
            IncludeFields = true
        };
        try
        {
            res = JsonSerializer.Deserialize<SimChAConfig>(serializedJSON, options);
            if (res is null)
            {
                throw new Exception($"Could not parse the simulation parameters:\n{serializedJSON}");
            }
        }
        catch (JsonException)
        {
            throw new Exception($"Could not parse the simulation parameters:\n{serializedJSON}");
        }
        if (res.SimParams.Seed < 0)
        {
            return res with { SimParams = res.SimParams with { Seed = new Random().Next() } };
        }
        return res;
    }
    
    // Expected format is that there is a header and the columns contain:
    // SampleID, Chr, Start, End, CN hap1, CN hap2
    public static Dictionary<string, Karyotype> ParseCNAProfile(RefGen refGen, TextReader cnaFile, 
        bool autosomesOnly, bool zeroIndexed)
    {
        Dictionary<string, Karyotype> result = new();
    
        string? firstLine = cnaFile.ReadLine();
        if (firstLine == null)
        {
            throw new Exception("Fitness file is empty.");
        }
        if (firstLine.Split('\t').Length < 6)
        {
            throw new Exception("CNA file does not contain at least 6 columns.");
        }
        
        // Read lines and assign by samples
        Dictionary<string, List<(string chrom, int start, int end, int cnA, int cnB)>> sampleSegs = new();
        while (cnaFile.ReadLine() is { } line)
        {
            string[] lineSplit = line.Split('\t');
            string sampleId = lineSplit[0];
            if (!sampleSegs.TryGetValue(sampleId, out List<(string chrom, int start, int end, int cnA, int cnB)>? value))
            {
                value = ([]);
                sampleSegs[sampleId] = value;
                Console.Write($"Reading sample {sampleId}.".PadRight(80) + "\r");
            }
            string chrom = lineSplit[1];
            if (autosomesOnly && (chrom == refGen.YChrName || chrom == refGen.XChrName))
            {
                continue;
            }
            int start = zeroIndexed ? int.Parse(lineSplit[2]) :  int.Parse(lineSplit[2]) - 1;
            int end = int.Parse(lineSplit[3]);
            int cnA = (int) Math.Round(float.Parse(lineSplit[4]));
            int cnB = (int) Math.Round(float.Parse(lineSplit[5]));
            value.Add((chrom, start, end, cnA, cnB));
        }
        
        // Convert samples to karyotypes
        foreach ((string sampleId, var segs) in sampleSegs)
        {
            Console.Write($"Creating karyotype for sample {sampleId}.".PadRight(80) + "\r");

            List<(string chr, int start, int end, int cn)> hapA = [];
            List<(string chr, int start, int end, int cn)> hapB = [];
            bool chrYfound = false;
            bool chrXfound = false;

            foreach ((string chr, int start, int end, int cnA, int cnB) in segs)
            {
                hapA.Add((chr, start, end, cnA));
                hapB.Add((chr, start, end, cnB));
                chrYfound |= chr == "chrY";
                chrXfound |= chr == "chrX";
            }

            var regionsA = BuildHaplotypeRegions(hapA, true, refGen);
            var regionsB = BuildHaplotypeRegions(hapB, false, refGen);
            var newRegs = new List<Contig> { new(regionsA), new(regionsB) };
            var sexType = chrYfound ? SexType.Male : chrXfound ? SexType.Female : SexType.Any;
            result[sampleId] = new Karyotype(refGen, newRegs, sexType);
        }

        // Add the last sample
        return result;
    }

    static List<Region> BuildHaplotypeRegions(List<(string chr, int start, int end, int cn)> segments, bool isHapA, RefGen refGen)
    {
        var regions = new List<Region>();

        // Group by chromosome
        var chrGroups = segments.GroupBy(s => s.chr);

        foreach (var chrGroup in chrGroups)
        {
            string chr = chrGroup.Key;

            // Copy-counted intervals
            var intervals = chrGroup
                .Select(s => new { s.start, s.end, count = s.cn })
                .OrderBy(s => s.start)
                .ToList();

            // Keep track of how many copies are left per interval
            int[] counts = intervals.Select(x => x.count).ToArray();

            while (counts.Any(c => c > 0))
            {
                // Build a new layer
                int currentStart = -1;
                int currentEnd = -1;

                var usedIndexes = new List<int>();

                for (int pos = 0; pos < intervals.Count; pos++)
                {
                    if (counts[pos] > 0)
                    {
                        // Either start new region or extend it
                        if (currentStart == -1)
                        {
                            currentStart = intervals[pos].start;
                            currentEnd = intervals[pos].end;
                        }
                        else if (intervals[pos].start <= currentEnd)
                        {
                            // overlapping → extend
                            currentEnd = Math.Max(currentEnd, intervals[pos].end);
                        }
                        else
                        {
                            break; // non-overlapping: end the region
                        }

                        usedIndexes.Add(pos);
                    }
                    else if (currentStart != -1)
                    {
                        break; // as soon as we hit a gap, we break
                    }
                }

                if (currentStart != -1)
                {
                    // Form region
                    var genes = refGen.GetGenesBetween(chr, currentStart, currentEnd).ToList();
                    regions.Add(new Region(currentStart, currentEnd, chr, isHapA, null, genes));

                    // Decrement one copy from each contributing interval
                    foreach (int i in usedIndexes)
                    {
                        counts[i]--;
                    }
                }
                else
                {
                    break; // no more intervals to use
                }
            }
        }

        return regions;
    }
    
    public static Dictionary<string, List<Gene>> ParseGeneList(TextReader geneFile, List<string> chrNames, GeneLT type)
    {
        // Pre-initialization
        var geneList = chrNames.ToDictionary(c => c, _ => new List<Gene>());
        int listIndex = 0;
        string? firstLine = geneFile.ReadLine();
        if (firstLine == null)
        {
            throw new Exception("Gene file is empty.");
        }
        string[] columns = firstLine.Split('\t');
        if (firstLine.Split('\t').Length < 5)
        {
            throw new Exception("Gene file does not contain at least 5 columns.");
        }
        if (columns[0] != "chrom" 
            || columns[1] != "start" 
            || columns[2] != "end"
            || columns[3] != "name" 
            || columns[4] != "score")
        {
            throw new Exception("Gene file does not contain the expected header: chrom\tstart\tend\tname\tscore.");
        }
        
        while (geneFile.ReadLine() is { } line)
        {
            if (line == "")
            {
                continue;
            }

            string[] genString = line.Split('\t');
            string name = genString[3];
            double fitness = double.Parse(genString[4], CultureInfo.InvariantCulture.NumberFormat);
            string chrom = genString[0];
            // Convert to zero-based [start, end) index 
            int start = int.Parse(genString[1]) - 1;
            int end = int.Parse(genString[2]);
            var gene = new Gene(start, end, chrom, type, listIndex, fitness);
            geneList[chrom].Add(gene);
            listIndex += 1;
        }
        foreach (var pair in geneList)
        {
            pair.Value.Sort((g1, g2) => g1.Start.CompareTo(g2.Start));
        }
        return geneList;
    }

    public static List<CTreeNode> ParseClones(TextReader cloneStream, bool parseFitness, string sep)
    {
        const string idKey = "ID";
        const string parentIDKey = "ParentID";
        const string distanceKey = "Distance";
        const string fitnessKey = "Fitness";

        string firstLine = cloneStream.ReadLine() ?? throw new Exception("CloneIn file is empty.");
        var header = firstLine.Split(sep).Select(s => s.Trim()).ToList();
        var columns = new Dictionary<string, int> { { idKey, -1 }, { parentIDKey, -1 }, { distanceKey, -1 } };
        if (parseFitness)
        {
            columns.Add(fitnessKey, -1);
        }

        foreach (var column in columns)
        {
            int idx = header.IndexOf(column.Key);
            if (idx == -1) throw new Exception($"CloneIn file does not contain {column.Key} column.");
            columns[column.Key] = idx;
        }

        var clones = new List<CTreeNode>();
        while (cloneStream.ReadLine() is { } line)
        {
            var lineSplit = line.Split(sep).Select(s => s.Trim()).ToList();
            string id = lineSplit[columns[idKey]];
            string parentId = lineSplit[columns[parentIDKey]];
            int distance = int.Parse(lineSplit[columns[distanceKey]]);
            double fitness = parseFitness
                ? double.Parse(lineSplit[columns[fitnessKey]], CultureInfo.InvariantCulture.NumberFormat)
                : -1.0;
            var clone = new CTreeNode(id, parentId, distance, fitness);
            clones.Add(clone);
        }
        return clones;
    }

    public static Dictionary<string, (double, double, double, int)> ParseCloneComponents(TextReader fitnessStream)
    {
        var output = new Dictionary<string, (double, double, double, int)>();
        string? firstLine = fitnessStream.ReadLine();
        if (firstLine == null)
        {
            throw new Exception("Fitness file is empty.");
        }
        // Continue past the header
        while (fitnessStream.ReadLine() is { } line)
        {
            var lineSplit = line.Split("\t").Select(s => s.Trim()).ToList();
            string sampleName = lineSplit[0];
            double stressTerm = double.Parse(lineSplit[4], CultureInfo.InvariantCulture.NumberFormat);
            double tsg = double.Parse(lineSplit[5], CultureInfo.InvariantCulture.NumberFormat);
            double og  = double.Parse(lineSplit[6], CultureInfo.InvariantCulture.NumberFormat);
            double tsgOgTerm = og + tsg;
            double essTerm = double.Parse(lineSplit[7], CultureInfo.InvariantCulture.NumberFormat);
            // If the file includes data on how many chromosomal events the sample underwent
            int eventCount = lineSplit.Count >= 9 ? int.Parse(lineSplit[8]) : -1;
            output[sampleName] = (stressTerm, tsgOgTerm, essTerm, eventCount);
        }
        return output;
    }

    public static Dictionary<string, int> ParseEventCounts(TextReader sampleStream)
    {
        var output = new Dictionary<string, int>();
        string? firstLine = sampleStream.ReadLine();
        if (firstLine == null)
        {
            throw new Exception("Sample file is empty.");
        }
        // Continue past the header
        while (sampleStream.ReadLine() is { } line)
        {
            var lineSplit = line.Split("\t").Select(s => s.Trim()).ToList();
            string sampleName = lineSplit[0];
            int eventCount = int.Parse(lineSplit[1]);
            output.Add(sampleName, eventCount);
        }
        return output;
    }

    public static (Dictionary<string, int> chrLengths, Dictionary<string, SexType> chrSex) ParseChromosomes(string text)
    {
        IList<string> lines = text.Split("\n");
        Dictionary<string, int> chrLengths = new();
        Dictionary<string, SexType> chrSex = new();
        for (int index = 0; index < lines.Count; index++)
        {
            string line = lines[index];
            var lineSplit = line.Split("\t").Select(s => s.Trim()).ToList();
            string chrNo = lineSplit[0];
            int length = int.Parse(lineSplit[1]);
            chrLengths.Add(chrNo, length);
            var sexEnum = GetSexEnum(lines, lineSplit, index);
            chrSex.Add(chrNo, sexEnum);
        }
        return (chrLengths, chrSex);
    }

    // Set the centromeres to the boundaries of the centromere regions (given that p and q parts are separated)
    public static Dictionary<string, GenRange> ParseCentromeres(TextReader centromereFile)
    {   
        Dictionary<string, GenRange> cents = new();

        while (centromereFile.ReadLine() is { } line)
        {
            string[] lineSplit = line.Split('\t');
            string chrom = lineSplit[0];
            int start = int.Parse(lineSplit[1]);
            int end = int.Parse(lineSplit[2]);

            if (cents.ContainsKey(chrom))
            {
                var existing = cents[chrom];
                cents[chrom] = new GenRange(Math.Min(existing.Start, start), Math.Max(existing.End, end), chrom);
            }
            else
            {
                cents[chrom] = new GenRange(start, end, chrom);
            }
        }

        return cents;
    }

    private static SexType GetSexEnum(ICollection<string> lines, IReadOnlyList<string> lineSplit, int index)
    {
        if (lineSplit.Count <= 2)
        {
            return index switch
            {
                _ when index == lines.Count - 1 => SexType.Male,
                _ when index == lines.Count - 2 => SexType.Female,
                _ => SexType.Any
            };
        }

        string sexString = lineSplit[2];
        return Enum.Parse<SexType>(sexString);
    }
    
    public static IEnumerable<StringBuilder> ParseFasta(StreamReader fastaStream)
    {
        StringBuilder? sequence = null;
        while (fastaStream.ReadLine() is { } line)
        {
            if (line.StartsWith(";"))
            {
                continue;
            }
            if (line.StartsWith(">"))
            {
                if (sequence != null)
                {
                    yield return sequence;
                }
                const string pattern = "^>chr([1-9]|1[0-9]|2[0-2]|X|Y)$";
                var match = Regex.Match(line, pattern);
                if (match.Value == "")
                {
                    sequence = null;
                    continue;
                }
                string chrNo = match.Value[1..];
                Console.WriteLine($"Parsing the sequence for chr: " + chrNo);
                sequence = new StringBuilder("");
            }
            else
            {
                sequence?.Append(line);
            }
        }
        if (sequence != null)
        {
            yield return sequence;
        }
    }
}
