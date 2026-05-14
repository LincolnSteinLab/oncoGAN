// Created by Dr. Adam Streck, 2021, adam.streck@gmail.com

using SimChA.Data;

namespace SimChA.IO;

public record SimChAConfig(
    SimParams SimParams,
    FitParams FitParams,
    List<Signature>? Signatures = null,
    EvoParams? EvoParams = null,
    MHParams? MHParams = null);