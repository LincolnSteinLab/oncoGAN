// Created by Dr. Adam Streck, 2023, adam.streck@gmail.com

using System.Text.Json.Serialization;

namespace SimChA.Computation;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum DistType { Uniform, Exponential, Normal, Unit, Poisson, Geometric, Pareto }
