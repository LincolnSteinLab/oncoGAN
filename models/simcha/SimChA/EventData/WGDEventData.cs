using SimChA.Data;
namespace SimChA.EventData;

public record WGDEventData(CNEventPars CNEventPars) : BaseEventData(CNEventPars)
{
    public override void ApplyEvent(Karyotype kar)
    {
        kar.ApplyWGD();
    }

    public override string EventDesc()
        => "WGD";
}