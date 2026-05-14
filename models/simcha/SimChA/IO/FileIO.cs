using System.Globalization;
using System.Text.Json;
using System.Text;
using SimChA.Data;
using SimChA.EventData;

namespace SimChA.IO;

public class FileIO
{
    // path
    public const string DATA_FOLDER = "data";
    
    // data
    private const string CHROMOSOMES_TSV = "chromosomes.tsv";
    private const string ESSENTIALS_TSV = "essentials.tsv";
    private const string OGS_TSV = "ogs.tsv";
    private const string TSGS_TSV = "tsgs.tsv";
    private const string GENOME_FASTA = "genome.fa";
    private const string CENTROMERES_TSV = "centromeres.tsv";
    
    // input
    private const string SIM_PARAMS_FILENAME = "sim_params.json";
    
    // output
    private const string SAMPLES_FILENAME = "samples.tsv";
    private const string CN_FILENAME = "copynumbers.tsv";
    private const string KARYOTYPES_FILENAME = "karyotypes.tsv";
    private const string CN_EVENTS_FILENAME = "events.tsv";
    private const string VCF_FILENAME = "vcf.tsv";
    
    private string Timestamp { get; }
    private string OutFolder { get; }

    public FileIO(string outFolder)
    {
        Timestamp = DateTime.Now.ToString("yy_MM_dd_HH_mm_ss");
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

        OutFolder = outFolder;
    }

    public bool CreateOutFolder()
    {
        try
        {
            if (Directory.Exists(OutFolder))
            {
                return false;
            }
            Directory.CreateDirectory(OutFolder);
            return true;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    }
    
    public void WriteCopyNumbers(Dictionary<string, IEnumerable<CopyNumber>> samples)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), CN_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var outputFile = new StreamWriter(outPath);
        outputFile.WriteLine("sample_id\t" + CopyNumber.Header());
        foreach ((string sampleID, var cns) in samples)
        {
            foreach (var cn in cns)
            {
                outputFile.WriteLine($"{sampleID}\t{cn.ToTSV()}");
            }
        }
    }

    public void WriteSimParams(SimChAConfig simChAConfig)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), SIM_PARAMS_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var file = new StreamWriter(outPath);
        var options = new JsonSerializerOptions { IncludeFields = true, WriteIndented = true };
        string jsonString = JsonSerializer.Serialize(simChAConfig, options);
        file.WriteLine(jsonString);
    }

    public void WriteKaryotypes(List<Sample> samples)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), KARYOTYPES_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var outputFile = new StreamWriter(outPath);
        outputFile.WriteLine("sample_id\tkaryotype");
        
        foreach (var sample in samples)
        {
            var kar = sample.Karyotype;
            outputFile.WriteLine($"{sample.SampleId}\t{kar}");
        }
    }

    public void WriteEvents(List<Sample> samples)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), CN_EVENTS_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var outputFile = new StreamWriter(outPath);
        outputFile.WriteLine("sample_id\t" + CNEventDesc.Header());
        foreach (var sample in samples)
        {
            foreach (var cnEvent in sample.Events)
            {   
                outputFile.WriteLine($"{sample.SampleId}\t{cnEvent.ToTSV()}");
            }
        }
    }
    
    public void WriteVCF(RefGen refGen, List<Sample> samples)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), VCF_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var outputFile = new StreamWriter(outPath);
        outputFile.WriteLine("##fileformat=VCFv4.3");
        outputFile.WriteLine("##source=SimChAV1.0");
        outputFile.WriteLine($"##reference=verily_{refGen.Name}_genome.fa");
        outputFile.WriteLine("#SAMPLEID\tCHROM\tPOS\tID\tREF\tALT");
        foreach (var sample in samples)
        {
            foreach (var snv in sample.Karyotype.GetSNVs())
            {
                var refBase = refGen.GetRefBase(snv.Chrom, (int)snv.Pos);
                // The VCF should *not* be aware of SNVs that didn't end up altering the location in the final karyotype
                if (refBase != snv.Alt)
                {
                    outputFile.WriteLine($"{sample.SampleId}\t{snv.Chrom}\t{snv.Pos}\t.\t{refBase}\t{snv.Alt}");
                }
            }
        }
    }

    public void WriteFasta(List<Sample> samples)
    {
        foreach (var sample in samples)
        {
            string outPath = Path.Combine(Path.GetFullPath(OutFolder), $"{sample.SampleId}_genome.fa");
            Console.WriteLine($"Writing to file {outPath}");
            using var outputFile = new StreamWriter(outPath);
            foreach (string region in sample.Karyotype.GetSeq())
            {
                outputFile.Write(region);
            }
        }
    }

    public void WriteSamples(List<SampleStat> sampleStats)
    {
        string outPath = Path.Combine(Path.GetFullPath(OutFolder), SAMPLES_FILENAME);
        Console.WriteLine($"Writing to file {outPath}");
        using var file = new StreamWriter(outPath);
        file.WriteLine(SampleStat.Header());

        foreach (var clone in sampleStats)
        {
            file.WriteLine(clone.ToString());
        }
    }

    public static (CTreeNode root, List<CTreeNode> tree) ReadCloneTree(string filePath, bool parseFitness)
    {
        string fileFullPath = Path.GetFullPath(filePath);
        string fileFormat = filePath[^3..];
        if (fileFormat != "tsv" && fileFormat != "csv")
        {
            throw new Exception($"File {filePath} should be a tsv or csv.");
        }
        string separator = fileFormat == "tsv" ? "\t" : ",";
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            var cloneFile = new StreamReader(fileFullPath);
            var tree = Parsers.ParseClones(cloneFile, parseFitness, separator);
            var allIds = new HashSet<string>(tree.Select(n => n.CloneId));
            var roots = tree.FindAll(n => n.ParentId == n.CloneId || !allIds.Contains(n.ParentId));
            // Normalize root to self-parented
            if (roots.Count == 1 && roots[0].ParentId != roots[0].CloneId)
            {
                var r = roots[0];
                var normalized = new CTreeNode(r.CloneId, r.CloneId, r.Distance, r.Fitness);
                int idx = tree.IndexOf(r);
                tree[idx] = normalized;
                roots[0] = normalized;
            }
            return roots.Count switch
            {
                > 1 => throw new Exception($"More than one ({roots.Count}) root nodes found in the clone tree {filePath}."),
                0 => throw new Exception($"No root node found in the clone tree  {filePath}."),
                _ => (roots[0], tree)
            };
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }

    private static Dictionary<string, StringBuilder> ReadFasta(List<string> allChrs, string folder)
    {
        string fileFullPath = Path.GetFullPath(Path.Combine(folder, GENOME_FASTA));
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            var fastaFile = new StreamReader(fileFullPath);
            var sequenceList =  Parsers.ParseFasta(fastaFile).ToList();
            return sequenceList.Select((seq, i) => new {i, seq}).ToDictionary(x => allChrs[x.i], x => x.seq);
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }

    public static SimChAConfig ReadSimChAConfig(string filePath)
    {
        string fileFullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            string serializedJSON = File.ReadAllText(fileFullPath);
            var config = Parsers.ParseSimParams(serializedJSON);
            if (config.SimParams == null)
            {
                throw new Exception("No simulation parameters found. Please set \"SimParams\" in the config JSON.");
            }
            if (config.FitParams == null)
            {
                throw new Exception("No fitness parameters found. Please set \"FitParams\" in the config JSON.");
            }
            return config;
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }

    private static List<Dictionary<string, List<Gene>>> MapGenesToChroms(string folder, Dictionary<string, SexType> chrSex)
    {
        List<Dictionary<string, List<Gene>>> geneLists = [];
        Dictionary<GeneLT, string> fileMap = new() {
            { GeneLT.TSG, TSGS_TSV },
            { GeneLT.OG, OGS_TSV },
            { GeneLT.Ess, ESSENTIALS_TSV }
        };
        for (int i = 0; i < fileMap.Count; i++)
        {
            string filePath = Path.Combine(folder, fileMap[(GeneLT) i]);
            string fileFullPath = Path.GetFullPath(filePath);
            if (!File.Exists(fileFullPath))
            {
                throw new Exception($"File {fileFullPath} does not exist");
            }
            try
            {
                var geneFile = new StreamReader(fileFullPath);
                var chrNames = chrSex.Select(pair => pair.Key).ToList();
                geneLists.Add(Parsers.ParseGeneList(geneFile, chrNames, (GeneLT) i));
            }
            catch (Exception e)
            {
                throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
            }
        }
        return geneLists;
    }

    public static List<Sample> ReadProfiles(RefGen refGen, string cnaProfile, bool autosomesOnly, bool zeroIndexed)
    {
        string fileFullPath = Path.GetFullPath(cnaProfile);
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            var cnaFile = new StreamReader(fileFullPath);
            var profiles = Parsers.ParseCNAProfile(refGen, cnaFile, autosomesOnly, zeroIndexed);
            var samples = new List<Sample>();
            foreach ((string sampleId, var karyotype) in profiles)
            {
                samples.Add(new Sample(sampleId, sampleId, karyotype));
            }
            return samples;
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }
    
    private static (Dictionary<string, int> chrLengths, Dictionary<string, SexType> chrSex) ReadChromosomes(string folder)
    {
        string fileFullPath = Path.GetFullPath(Path.Combine(folder, CHROMOSOMES_TSV));
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            string fileContent = File.ReadAllText(fileFullPath);
            return Parsers.ParseChromosomes(fileContent);
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }

    private static Dictionary<string, GenRange> ReadCentromeres(string folder)
    {
        string fileFullPath = Path.GetFullPath(Path.Combine(folder, CENTROMERES_TSV));
        if (!File.Exists(fileFullPath))
        {
            throw new Exception($"File {fileFullPath} does not exist");
        }
        try
        {
            var fileContent = new StreamReader(fileFullPath);
            return Parsers.ParseCentromeres(fileContent);
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse the file {fileFullPath}. Error {e.Message}");
        }
    }

    public static RefGen ReadGenRef(string dataFolder, string refName, string genesName, bool useVariants = false)
    {
        string dataFullPath = Path.GetFullPath(dataFolder);
        if (!Path.Exists(dataFullPath))
        {
            throw new ArgumentException($"Data folder does not exist: {dataFullPath}");
        }
        string assemblyFolder = Path.Combine(dataFullPath, refName); 
        if (!Path.Exists(assemblyFolder))
        {
            throw new ArgumentException($"Assembly folder does not exist: {assemblyFolder}");
        }
        string genesFolder = Path.Combine(Path.GetFullPath(assemblyFolder), genesName);
        if (!Path.Exists(genesFolder))
        {
            throw new ArgumentException($"Genes folder does not exist: {genesFolder}");
        }
        
        var (chrLengths, chrSex) = ReadChromosomes(assemblyFolder);
        var centromeres = ReadCentromeres(assemblyFolder);
        var allChrs = chrSex.Select(pair => pair.Key).ToList();
        var genContentsDict = useVariants ? ReadFasta(allChrs, assemblyFolder) : null;
        var geneChromMap = MapGenesToChroms(genesFolder, chrSex);
        return new RefGen(refName, chrLengths, chrSex, centromeres, geneChromMap, genContentsDict);
    }
}
