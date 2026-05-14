﻿using System.Diagnostics;
using SimChA.Data;
using SimChA.EventData;
using SimChA.IO;

namespace SimChA.Simulation;

public static class Factory
{
    public static Simulator GetSimulator(Random rnd, RefGen refGen, SimChAConfig simChAConfig, SelectionMode selMode)
    {
        var sampleParams = simChAConfig.SimParams;
        var fitParams = simChAConfig.FitParams;
        switch (selMode)
        {
            case SelectionMode.MetropolisHastings:
                var mcParams = simChAConfig.MHParams ?? throw new Exception(
                    "Error: MCParams not set. Cannot perform MC sampling. Please set MCParams in the config file.");
                return new MHSimulator(rnd, refGen, sampleParams, fitParams, mcParams);

            case SelectionMode.Evolution:
                var evoParams = simChAConfig.EvoParams ?? throw new Exception(
                    "Error: EvoParams not set. Cannot perform evolution without evolution parameters. Please set EvoParams in the config file.");
                return new EvoSimulator(rnd, refGen, sampleParams, fitParams, evoParams);

            case SelectionMode.MonteCarlo:
                return new Simulator(rnd, refGen, sampleParams, fitParams);

            default:
                throw new Exception("Error: No selection mode set.");
        }
    }
    
    private static void ValidateEvent(CNEventPars cnEventPars)
    {
        switch (cnEventPars.Type)
        {
            case CNEventType.ChromDeletion:
            case CNEventType.ChromDuplication:
            case CNEventType.Pass:
            case CNEventType.WholeGenomeDoubling:
            case CNEventType.BreakageFusionBridge:
            case CNEventType.SNV:
            case CNEventType.ArmDeletion:
            case CNEventType.ArmDuplication:
            case CNEventType.Translocation:
                break;
                
            case CNEventType.InternalDeletion:
            case CNEventType.InternalDuplication:
            case CNEventType.InternalInversion:
            case CNEventType.InvertedDuplication:
            case CNEventType.Chromothripsis:
            case CNEventType.Chromoplexy:
            case CNEventType.CentromereBoundDeletion:
            case CNEventType.CentromereBoundDuplication:
            case CNEventType.TailDeletion:
            case CNEventType.TailDuplication:
                if (cnEventPars.Frac <= 0) 
                    throw new Exception($"Event {cnEventPars.Type} does not have a Frac parameter. E.g. \"Frac\": 0.1");
                break;
            
            case CNEventType.TIChain:
            case CNEventType.TIBridge:
            case CNEventType.TICycle:
            case CNEventType.Pyrgo:
            case CNEventType.Rigma:
                if (cnEventPars.Frac <= 0) 
                    throw new Exception($"Event {cnEventPars.Type} does not have a Frac parameter. E.g. \"Frac\": 0.1");
                if (cnEventPars.Frag <= 0) 
                    throw new Exception($"Event {cnEventPars.Type} does not have a Frag parameter. E.g. \"Frag\": 5");
                break;
            
            default:
                throw new ArgumentOutOfRangeException($"Unknown event type {cnEventPars.Type}");
        }
    }

    public static List<Signature> ValidateSignatures(List<Signature>? signatures)
    {
        if (signatures is null)
        {
            throw new Exception("No signatures found.");
        }
        var validSigs = new List<Signature>();
        
        foreach (var sig in signatures.Where(sig => sig.Prob > 0 && sig.Events.Any(e => e.Prob > 0)))
        {
            if (sig.Events is null || sig.Events.Count == 0)
            {
                throw new Exception($"Signature {sig.Name} does not have any events.");
            }
            double probSum = sig.Events.Sum(e => e.Prob);
            if (probSum <= 0)
            {
                throw new Exception($"Signature {sig.Name} has a total probability of {probSum}.");
            }
            foreach(var cnEventP in sig.Events)
            {
                ValidateEvent(cnEventP);
            }
            validSigs.Add(sig);
        }
        if (validSigs.Count == 0)
        {
            throw new Exception("No valid signatures found.");
        }
        return validSigs;
    }
    
    public static (List<CNEventPars>, Dictionary<string, double> mixture) MixSignatures(Random rnd, List<Signature> sigs, MixtureType mixType)
    {
        double probSum = sigs.Sum(x => x.Prob);
        var concentrations = sigs.Select(x => x.Prob / probSum).ToList();
        var probabilities = Sampling.ConcentrationsToProbabilities(rnd, concentrations, mixType);
        var mixture = new Dictionary<string, double>();
        var events = new List<CNEventPars>();
        for (int i = 0; i < sigs.Count; i++)
        {
            var sig = sigs[i];
            double sigProb = probabilities[i];
            mixture.Add(sig.Name, sigProb);
            if (!(sigProb > 0))
            {
                continue;
            }

            var selectedEvs = sig.Events.Where(ev => ev.Prob > 0).ToList();
            double evsProbSum = selectedEvs.Sum(ev => ev.Prob);
            events.AddRange(selectedEvs.Select(cnEventP => cnEventP with
            {
                Prob = cnEventP.Prob / evsProbSum * sigProb,
                Signature = sig.Name
            }));
        }
        return (events, mixture);    
    }

    public static List<CNEventPars> NormalizeEvents(List<CNEventPars> events)
    {
        double probSum = events.Sum(ev => ev.Prob);
        return events.Select(ev => ev with { Prob = ev.Prob/probSum }).ToList();
    }
}
