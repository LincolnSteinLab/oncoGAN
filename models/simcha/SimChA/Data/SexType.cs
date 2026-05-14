using System.Text.Json.Serialization;

namespace SimChA.Data;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum SexType { Any = 0, Female = 1, Male = 2 }