using CommandLine;

namespace SimChA.IO;

public class CmdOptions
{
    [Option('O', "output", Required = false, Default = "./out", HelpText = "The path to the output files.")]
    public string OutputPath { get; set; }

    [Option('C', "config", Required = false, Default = "./configs/main_config.json", HelpText = "A json file with configuration of the experiment.")]
    public string ConfigFile { get; set; }

    [Option('T', "tree", Required = false, Default = "", HelpText = "A clone file that describes the tree to be built.")]
    public string CloneTreeFile { get; set; }
    
    [Option('R', "repeats", Required = false, Default = 1, HelpText = "Number of repeats to generate if the distance parameter is used.")]
    public int Repeats { get; set; }
    
    [Option('P', "cnprofiles", Required = false, Default = "", HelpText = "File with CNAs, will cause the program to write a scoring file.")]
    public string CNProfiles { get; set; }
    
    [Option('e', "evolution-mode", Required = false, Default = false, HelpText = "Flag to execute evolution mode. In this mode, events are selected in order to increase fitness.")]
    public bool EvolutionMode { get; set; }

    [Option('m', "mcmc-mode", Required = false, Default = false, HelpText = "The model will be run in MCMC mode. In this mode the fitness of the samples will be matched using the Metropolis-Hastings algorithm.")]
    public bool MHMode { get; set; }
    
    [Option('s', "segments", Required = false, Default = false, HelpText = "Write out copy numbers segments.")]
    public bool WriteCNs { get; set; }
    
    [Option('S', "consistent-segments", Required = false, Default = false, HelpText = "Write out copy number segments under a minimum consistent segmentation.")]
    public bool WriteConsistentCNs { get; set; }
    

    [Option('K', "karyotypes", Required = false, Default = false, HelpText = "Write out karyotypes of each sample.")]
    public bool WriteKaryotypes { get; set; }

    [Option('v', "variants", Required = false, Default = false, HelpText = "Write out VCF file with the variants of the final simulated karyotype. Requires data/<assembly>/genome.fa (see scripts/DownloadRefData.sh).")]
    public bool WriteVariants { get; set; }

    [Option('f', "fasta", Required = false, Default = false, HelpText = "Write out a FASTA file for each sample. Requires data/<assembly>/genome.fa (see scripts/DownloadRefData.sh). WARNING! Average file size is 6GB per sample.")]
    public bool WriteFasta { get; set; }

    [Option('z', "zero-index", Required = false, Default = false, HelpText = "Flag for zero-indexed input copy number profiles")]
    public bool ZeroIndexed { get; set; }
    
    [Option("root", Required = false, Default = ".", HelpText = "A path to the folder that will be considered root for relative paths. If not provided, the C# default will be used.")]
    public string RootFolder { get; set; }

    public ExecMode ExecMode
    {
        get
        {
            if (CloneTreeFile != "" && CNProfiles != "")
            {
                throw new Exception("Cannot run both tree and profiles at the same time.");
            }
            if (CloneTreeFile != "")
            {
                if (Repeats > 1)
                {
                    throw new Exception("Cannot run tree with repeats.");
                }
                return ExecMode.Tree;
            }
            if (CNProfiles != "")
            {
                if (Repeats > 1)
                {
                    throw new Exception("Cannot run profiles with repeats.");
                }
                return ExecMode.Profiles;
            }
            return ExecMode.Repeats;
        }
    }

    public SelectionMode SelectionMode
    {
        get
        {
            if (EvolutionMode && MHMode)
            {
                throw new Exception("Cannot run both evolution and MCMC mode at the same time.");
            }
            if (MHMode)
            {
                return SelectionMode.MetropolisHastings;
            }
            if (EvolutionMode)
            {
                return SelectionMode.Evolution;
            }
            return SelectionMode.MonteCarlo;
        }
    }
    
    public bool ShouldParseGenome 
        => WriteVariants || WriteFasta;
    
    public bool Simulate
        => ExecMode != ExecMode.Profiles;
    
    public bool CalcSegments
        => WriteCNs || WriteConsistentCNs;
}
