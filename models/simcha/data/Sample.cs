using SimChA.EventData;

namespace SimChA.Data;

public class Sample(
    string sampleId,
    string parentId,
    Karyotype kar,
    List<CNEventDesc>? events = null,
    Dictionary<string, double>? mixture = null,
    double fitnessTarget = double.NaN)
{
    public string SampleId { get; } = sampleId;
    public string ParentId { get; } = parentId;
    public Karyotype Karyotype { get; } = kar;
    public List<CNEventDesc> Events { get; } = events ?? [];
    public double FitnessTarget { get; } = fitnessTarget;
    private Dictionary<string, double> Mixture { get; } = mixture ?? new Dictionary<string, double>();

    public string MixtureString() 
        => string.Join(";", Mixture.Select(pair => $"{pair.Key}:{pair.Value:f4}"));
}