using SimChA.Data;

namespace SimChA.EventData;

public record BaseEventData(CNEventPars CNEventPars)
{
    public CNEventType EventType => CNEventPars.Type;
    
    // TODO: This should be immutable, i.e. should return new Karyotype
    public virtual void ApplyEvent(Karyotype kar)
    {
    }

    public virtual string EventDesc()
        => "";
}