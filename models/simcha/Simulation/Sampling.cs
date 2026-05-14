using SimChA.Computation;
using SimChA.EventData;
using MathNet.Numerics.Distributions;
using SimChA.Data;

namespace SimChA.Simulation;

public static class Sampling
{
    public static long GetNormSeg(Random rnd, long contigLen, double meanFrac) 
        => (long) Math.Clamp(Math.Round(contigLen * Normal.Sample(rnd, meanFrac, meanFrac / 3)), 1, contigLen);
    
    public static long GetExpSeg(Random rnd, long contigLen, long meanLen) 
        => (long) Math.Clamp(Math.Round(meanLen * Exponential.Sample(rnd, 1)), 1, contigLen);
    
    public static long GetExpSeg(Random rnd, long contigLen, double meanFrac) 
        => (long) Math.Clamp(Math.Round(contigLen * meanFrac * Exponential.Sample(rnd, 1)), 1, contigLen);
    
    public static long GetPos(Random rnd, long contigLen)
        => rnd.NextInt64(0, contigLen);
    
    public static int GetFragCount(Random rnd, double mean)
        => (int) Math.Max(1, Math.Round(Normal.Sample(rnd, mean, 1)));

    // https://ashpublications.org/blood/article/134/Supplement_1/3767/424006/Chromoplexy-and-Chromothripsis-Are-Important
    public static int GetChromoplexySiteCount(Random rnd)
        => rnd.NextSingle() switch
        {
            var n when n < .46 => 3,
            var n when n < .64 => 4,
            var n when n < .74 => 5,
            var n when n < .79 => 6,
            _ => 2
        };

    public static List<long> GetStopsForShards(Random rnd, long contigLen, int shardCount)
    {
        var stops = new List<long>();
        if (shardCount <= 1)
        {
            return stops;
        }
        for (int i = 1; i < shardCount; i++)
        {
            long newStop = GetPos(rnd, contigLen);
            stops.Add(newStop);
            stops.Sort();
        }
        return stops;
    }
    
    public static IList<double> CreateRandomMixture(Random rnd,  IList<double> concentrations)
        => concentrations.Count != 0 ? new Dirichlet(concentrations.ToArray(), rnd).Sample() : [];

    
    public static IList<double> ConcentrationsToProbabilities(Random rnd, List<double> concentrations, MixtureType mixType)
    {
        return mixType switch
        {
            MixtureType.Single => SelectFromMixture(rnd, concentrations),
            MixtureType.Constant => concentrations,
            MixtureType.Dirichlet => CreateRandomMixture(rnd, concentrations),
            _ => throw new ArgumentOutOfRangeException(nameof(mixType), mixType, null)
        };
    }
    
    public static List<double> SelectFromMixture(Random rnd, IList<double> concentrations)
    {
        int selected = rnd.PickRndIndex(concentrations);
        double[] mixture = new double[concentrations.Count];
        mixture[selected] = 1;
        return mixture.ToList();
    }
    
    public static double SampleContDist(Random rnd, DistType dist, double mean = 1)
    {
        return dist switch
        {
            DistType.Exponential => Exponential.Sample(rnd, 1 / mean),
            DistType.Normal => Normal.Sample(rnd, mean, 0.5),
            _ => mean
        };
    }

    public static int SampleDiscDist(Random rnd, DistType dist, double mean)
    {
        return dist switch
        {
            DistType.Geometric => Geometric.Sample(rnd, 1 / mean),
            DistType.Poisson => Poisson.Sample(rnd, mean),
              _ => (int) mean
        };
    }

    private static double GetParetoScale(double mean) 
        => 1.0/2.0*(-1.0 + Math.Sqrt(1.0 + 4.0*mean*mean));

    public static double SampleParetoLim(Random rnd, double mean)
    {
        double shape = 0.5;
        double scale = GetParetoScale(mean);
        for (int i = 0; i < 1000; i++) {
            double sample = Pareto.Sample(rnd, scale, shape);
            if (sample <= 1)
            {
                return sample;
            }
        } 
        return 1;
    }
    
    public static SexType GetSex(Random rnd, SexType sexType)
        => sexType switch
        {
            SexType.Any => rnd.CoinFlip() ? SexType.Male : SexType.Female,
            _ => sexType
        };

    public static Nucleotide SampleBase(Random rnd) 
        => (Nucleotide) rnd.Next(4);

    private static (int id, long len) SampleContigsByLength(Random rnd, Karyotype kar)
    {
        // Karyotype stores 0-length contigs for contig-ID-preservation, so we need to filter them out
        var contigIds = kar.ContigIds().Where(i => kar.ContigLen(i) > 0).ToList();
        long totalLength = contigIds.Sum(kar.ContigLen);
        var pArray = contigIds.Select(i => kar.ContigLen(i)/(1.0*totalLength)).ToList();
        int idSelected = contigIds.ToList()[rnd.PickRndIndex(pArray)];
        return (idSelected, kar.ContigLen(idSelected));
    }

    // Selects the contigs to be affected by the event
    private static IEnumerable<(int id, long len)> GetSeq(IEnumerable<(int id, long len)> contigs, CNEventType type, Karyotype kar) 
        => type switch {
            CNEventType.TIChain or CNEventType.TICycle or CNEventType.TIBridge or CNEventType.Chromoplexy => contigs,
            CNEventType.Translocation => contigs.Take(2),
            CNEventType.ArmDeletion or CNEventType.ArmDuplication or CNEventType.CentromereBoundDeletion or CNEventType.CentromereBoundDuplication =>
                [contigs.First(pair => kar.GetCentromeres(pair.id).Count != 0)], 
            _ => contigs.Take(1)
        };

    public static BaseEventData? GenerateCNEventData(Random rnd, Karyotype kar, CNEventPars cnEventPars)
    {
        IEnumerable<(int id, long len)> contigs = kar.ContigIds().Shuffle(rnd).Select(i => (i, kar.ContigLen(i)));
        List <(int id, long len)> seq;
        try
        {
            seq = GetSeq(contigs, cnEventPars.Type, kar).ToList();
            if (seq.Count == 0)
            {
                throw new InvalidOperationException("No suitable contig found");
            }
        } 
        catch (InvalidOperationException)
        {
            return null;
        }
        
        switch (cnEventPars.Type)
        {
            // Whole chromosome events
            case CNEventType.ChromDeletion:
            case CNEventType.ChromDuplication:
                return new ContigEventData(cnEventPars, seq[0].id, seq[0].len);
            
            case CNEventType.Pass:
            case CNEventType.Skip:
                return new BaseEventData(cnEventPars);
            
            case CNEventType.WholeGenomeDoubling:
                return new WGDEventData(cnEventPars);
            
            // Tail events
            case CNEventType.TailDeletion:
            case CNEventType.TailDuplication:
            case CNEventType.BreakageFusionBridge:
                return new TailEventData(rnd, cnEventPars, seq[0].id, seq[0].len);

            case CNEventType.ArmDeletion:
            case CNEventType.ArmDuplication:
                return new TailEventData(rnd, cnEventPars, seq[0].id, kar.GetCentromeres(seq[0].id), seq[0].len);
            
            case CNEventType.CentromereBoundDeletion:
            case CNEventType.CentromereBoundDuplication:
                return new InternalEventData(rnd, cnEventPars, seq[0].id, kar.GetCentromeres(seq[0].id), seq[0].len);

            // Internal events
            case CNEventType.InternalDuplication:
            case CNEventType.InternalDeletion:
            case CNEventType.InternalInversion:
            case CNEventType.InvertedDuplication:
                return new InternalEventData(rnd, cnEventPars, seq[0].id, seq[0].len);
            
            case CNEventType.Translocation:
                return seq.Count > 1
                    ? new PairEventData(rnd, cnEventPars, seq[0].id, seq[0].len, seq[1].id, seq[1].len)
                    : null;

            case CNEventType.Chromothripsis:
                return new ChromothripsisEventData(rnd, cnEventPars, seq[0].id, seq[0].len);

            case CNEventType.Chromoplexy:
                return new ChromoplexyEventData(rnd, cnEventPars, seq);

            case CNEventType.Pyrgo:
                return new PyrgoEventData(rnd, cnEventPars, seq[0].id, seq[0].len);

            case CNEventType.Rigma:
                return new RigmaEventData(rnd, cnEventPars, seq[0].id, seq[0].len);
            
            case CNEventType.TIChain:
            case CNEventType.TICycle:
            case CNEventType.TIBridge:
                return new TemplatedEventData(rnd, cnEventPars, seq);
            
            case CNEventType.SNV:
                var pointSeq = SampleContigsByLength(rnd, kar);
                return new PointMutationData(rnd, cnEventPars, pointSeq.id, pointSeq.len);

            default:
                throw new ArgumentOutOfRangeException(nameof(cnEventPars.Type), cnEventPars.Type, null);
        }
    }
}
