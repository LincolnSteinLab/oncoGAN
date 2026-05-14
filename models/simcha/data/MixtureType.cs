using System.Text.Json.Serialization;

namespace SimChA.Data;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum MixtureType
{
    Single,
    Dirichlet,
    Constant
}