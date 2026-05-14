using SimChA.IO;
using SimChA.Computation;
namespace SimChA.Data;

public record SampleStat(
    string SampleId,
    string ParentId,
    SexType Sex,
    double Ploidy,
    double FitnessVal,
    double FitnessTarget,
    double Stress,
    double Tsg,
    double Og,
    double Ess,
    double StressRaw,
    double TsgRaw,
    double OgRaw,
    double EssRaw,
    int MutCount,
    string Mixture,
    int NumRejectedEvents)
{
    public static string Header() 
        => "sample_id" +
           "\tparent_id" +
           "\tsex" +
           "\tploidy" +
           "\tfitness" +
           "\tfitness_target" +
           "\tstress" +
           "\ttsg" +
           "\tog" +
           "\tess" +
           "\tstress_raw" +
           "\ttsg_raw" +
           "\tog_raw" +
           "\tess_raw" +
           "\tdist" +
           "\tmixture"+
           "\tn_rejected_events";
    
    public override string ToString() =>
        $"{SampleId}" +
        $"\t{ParentId}" +
        $"\t{Sex}" +
        $"\t{Ploidy}" +
        $"\t{FitnessVal}" +
        $"\t{FitnessTarget}" +
        $"\t{Stress}" +
        $"\t{Tsg}" +
        $"\t{Og}" +
        $"\t{Ess}" +
        $"\t{StressRaw}" +
        $"\t{TsgRaw}" +
        $"\t{OgRaw}" +
        $"\t{EssRaw}" +
        $"\t{MutCount}" +
        $"\t{Mixture}"+
        $"\t{NumRejectedEvents}";

    public static double CalcPloidy(Karyotype kar, RefGen refGen)
        => 2.0 * kar.GenomeLen() / refGen.SexGenomeLen[(int) kar.Sex];
    
    public static SampleStat GetSampleStat(Sample sample, RefGen refGen, FitParams fParams)
    {
        var kar = sample.Karyotype;

        double ploidy = CalcPloidy(kar, refGen);
        double stress_raw = Fitness.StressTerm(refGen.SexGenomeLen[(int) kar.Sex], kar.GenomeLen());
        var geneData = refGen.SexGeneLists[(int) kar.Sex];
        double tsg_raw = Fitness.TsgOgTerm(geneData[(int) GeneLT.TSG], kar.GeneCounts[(int) GeneLT.TSG]) ;
        double og_raw = Fitness.TsgOgTerm(geneData[(int) GeneLT.OG], kar.GeneCounts[(int) GeneLT.OG]);
        double ess_raw = Fitness.EssTerm(geneData[(int) GeneLT.Ess], kar.GeneCounts[(int) GeneLT.Ess]);
        double fitnessVal = 1 + stress_raw * fParams.Stress + (og_raw - tsg_raw) * fParams.TsgOg + ess_raw * fParams.Essentiality;
        
        int nRejectedEvents = sample.Events.Sum(e => e.NumRejections);
        double fitnessTarget = double.IsNaN(sample.FitnessTarget) ? kar.FitnessVal : sample.FitnessTarget;

        var res = new SampleStat(
            sample.SampleId, sample.ParentId, kar.Sex, ploidy, 
            fitnessVal, fitnessTarget, 
            stress_raw* fParams.Stress, tsg_raw * fParams.TsgOg, og_raw * fParams.TsgOg, ess_raw * fParams.Essentiality,
            stress_raw, tsg_raw, og_raw, ess_raw, 
            sample.Events.Count,  sample.MixtureString(), nRejectedEvents);
        return res;
    }
}
