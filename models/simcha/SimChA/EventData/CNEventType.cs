using System.Text.Json.Serialization;

namespace SimChA.EventData;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum CNEventType
{
    ChromDeletion,
    ChromDuplication,
    TailDuplication,
    TailDeletion,
    InternalDeletion,
    InternalDuplication,
    InternalInversion,
    InvertedDuplication,
    Translocation,
    BreakageFusionBridge,
    WholeGenomeDoubling,
    Chromothripsis,
    Chromoplexy,
    TIChain,
    TICycle,
    TIBridge,
    Pyrgo,
    Rigma,
    SNV,
    ArmDeletion,
    ArmDuplication,
    CentromereBoundDuplication,
    CentromereBoundDeletion,
    Pass,
    Skip
    // Tyfonas
}