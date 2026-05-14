// Created by Dr. Adam Streck, 2023, adam.streck@gmail.com

using SimChA.Computation;

namespace SimChA.IO;

public record FitParams
(
    double Stress = 0,
    double TsgOg = 0,
    double Essentiality = 0,
    string GeneSet = "Empty"
);
