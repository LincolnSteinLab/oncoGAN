using System.Diagnostics;
using System.Globalization;
using CommandLine;
using SimChA.IO;
using SimChA.Simulation;
using SimChA.Data;
using SimChA.Computation;

// Configuration
Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
var cmdOptions = Parser.Default.ParseArguments<CmdOptions>(args);
cmdOptions.WithNotParsed(o =>
{
    Console.WriteLine("Exiting");
    o.ToList().ForEach(Console.WriteLine); // Write out errors
    Environment.Exit(1);
});
var watch = new Stopwatch();
watch.Start();

// Input
var options = cmdOptions.Value;

// Set working directory if RootFolder is specified
if (!string.IsNullOrEmpty(options.RootFolder))
{
    Environment.CurrentDirectory = options.RootFolder;
}

var selMode = options.SelectionMode;
var config = FileIO.ReadSimChAConfig(options.ConfigFile);
var rnd = new Random(config.SimParams.Seed);
var files = new FileIO(options.OutputPath);
files.CreateOutFolder();
var genRef = FileIO.ReadGenRef(FileIO.DATA_FOLDER, config.SimParams.Assembly, config.FitParams.GeneSet, options.ShouldParseGenome);
var simulator = Factory.GetSimulator(rnd, genRef, config, selMode);

// Construct samples
var samples = new List<Sample>();
if (options.Simulate)
{
    var validSigs = Factory.ValidateSignatures(config.Signatures);
    if (options.ExecMode == ExecMode.Repeats)
    {
        Console.WriteLine($"Creating {options.Repeats} samples:");
        for (int i = 0; i < options.Repeats; i++)
        {
            string sampleId = $"Sample_{i+1}";
            var node = new CTreeNode(sampleId, sampleId, -1, -1);
            var tree = new List<CTreeNode> {node};
            var newSample = simulator.Simulate(node, tree, validSigs);
            samples.Add(newSample[0]);
        }
    }
    else
    {
        var (root, tree) = FileIO.ReadCloneTree(options.CloneTreeFile, options.MHMode);
        Console.WriteLine($"Creating {tree.Count} samples from a tree:");
        samples = simulator.Simulate(root, tree, validSigs);
    }
}
else
{
    Console.WriteLine("Reading samples from data:");
    samples = FileIO.ReadProfiles(genRef, options.CNProfiles, config.SimParams.AutosomesOnly, options.ZeroIndexed);
}

// Score and segment samples
var sampleStats = new List<SampleStat>();
var sampleCNs = new Dictionary<string, IEnumerable<CopyNumber>>();
var jointSegmentation = options.WriteConsistentCNs 
    ? CopyNumbers.GetJointSegmentation(genRef.AllChrNames, samples) 
    : null;

Console.WriteLine("Analyzing samples:");
foreach (var sample in samples)
{
    Console.Write($"Analyzing sample {sample.SampleId}.".PadRight(80) + "\r");
    sampleStats.Add(SampleStat.GetSampleStat(sample, genRef, config.FitParams));
    if (options.CalcSegments)
    {
        sampleCNs[sample.SampleId] = CopyNumbers.CalcCNs(sample.Karyotype, jointSegmentation);
    }
}

// Output
Console.WriteLine();
files.WriteSimParams(config);
try
{
    files.WriteSamples(sampleStats);
    if (options.CalcSegments)
    {
        files.WriteCopyNumbers(sampleCNs);
    }
    if (options.WriteKaryotypes)
    {
        files.WriteKaryotypes(samples);
    }
    if (options.Simulate)
    {
        files.WriteEvents(samples);
    }
    if (options.WriteVariants)
    {
        files.WriteVCF(genRef, samples);
    }
    if (options.WriteFasta)
    {
        files.WriteFasta(samples);
    }
}
catch (Exception e)
{
    Console.WriteLine($"Failed to write data with exception {e.Message}. Stack: {e.StackTrace}");
    return e.HResult;
}

watch.Stop();
Console.WriteLine($"Total time: {TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds)}");
return 0;