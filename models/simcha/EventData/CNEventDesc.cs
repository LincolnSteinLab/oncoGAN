namespace SimChA.EventData;

public record CNEventDesc(
    BaseEventData EventData, 
    int Depth, 
    double DeltaFitness = 0, 
    double TotalFitness = 0, 
    int NumRejections = 0, 
    string Signature = "",
    List<string>? RegionsGained = null,
    List<string>? RegionsLost = null)
{
    public static string Header()
        => "event_type" +
           "\tdepth" +
           "\tdescription" +
           "\tdelta_fitness" +
           "\ttotal_fitness" +
           "\tnum_rejections" +
           "\tsignature" +
           "\tregions_gained" +
           "\tregions_lost";

    public string ToTSV() =>
        $"{EventData.EventType}" +
        $"\t{Depth}" +
        $"\t{EventData.EventDesc()}" +
        $"\t{DeltaFitness:f4}" +
        $"\t{TotalFitness:f4}" +
        $"\t{NumRejections}" +
        $"\t{Signature}" +
        $"\t{FormatList(RegionsGained)}" +
        $"\t{FormatList(RegionsLost)}";

    private static string FormatList(List<string>? items)
        => "[" + (items is { Count: > 0 } ? string.Join(",", items) : "") + "]";
}