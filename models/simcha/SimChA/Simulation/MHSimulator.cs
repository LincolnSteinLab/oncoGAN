﻿using SimChA.Computation;
using SimChA.Data;
using SimChA.EventData;
using SimChA.IO;

namespace SimChA.Simulation;

public class MHSimulator(Random rnd, RefGen refGen, SimParams simParams, FitParams fitParams, MHParams mhParams)
    : Simulator(rnd, refGen, simParams, fitParams)
{
    private MHParams MHParams { get; } = mhParams;

    private List<(BaseEventData Event, string Signature)> InitEvents(Karyotype kar, int nMutations, List<CNEventPars> cnEventPs)
    {
        var eventPs = Enumerable.Range(0, nMutations).Select(_ => Rnd.PickRndElem(cnEventPs));
        return eventPs.Select(
            e => (Sampling.GenerateCNEventData(Rnd, kar, e)
                 ?? throw new Exception($"Failed to generate event data for {e}."), e.Signature)
        ).ToList();
    }

    private double Gaussian(double f, double ft)
        => Math.Exp(-Math.Pow(f - ft, 2.0)/(2*MHParams.ThetaFitness));

    private double AcceptProb(double newFit, double oldFit, double targetFitness)
        => Gaussian(newFit, targetFitness) / Gaussian(oldFit, targetFitness);

    private double AcceptProb(double newFit, double oldFit)
        => Math.Min(1, Math.Exp((newFit - oldFit)/MHParams.ThetaFitness));

    private double GetFitness(Karyotype kar, List<(BaseEventData Event, string Signature)> events)
    {
        foreach (var (eventData, _) in events)
        {
            eventData.ApplyEvent(kar);
        }
        return kar.UpdateFitness(RefGen, FitParams);
    }

    private List<(BaseEventData Event, string Signature)> GetNewProposal(List<CNEventPars> cnEventPs, Karyotype kar,
        List<(BaseEventData Event, string Signature)> oldEvents)
    {
        var proposedEvents = oldEvents.ToList();
        int index = Rnd.Next(proposedEvents.Count);
        var selectedP = Rnd.PickRndElem(cnEventPs);
        var newData = Sampling.GenerateCNEventData(Rnd, kar, selectedP);
        proposedEvents[index] = (newData ?? throw new Exception("Failed to generate new event data."), selectedP.Signature);
        return proposedEvents;
    }

    private List<(BaseEventData Event, string Signature)> GetEvents(List<CNEventPars> cnEventPs, Karyotype kar, int nEvents,
        double targetFitness)
    {
        if (nEvents == 0) return [];
        var currentEvents = InitEvents(kar, nEvents, cnEventPs);
        double currentFitness = GetFitness(new Karyotype(kar), currentEvents);
        
        double bestValue = MHParams.MatchFitness ?  double.PositiveInfinity : currentFitness;
        var bestEvents = new List<(BaseEventData Event, string Signature)>(currentEvents);

        for (int i = 0; i < MHParams.NumIterations; i++)
        {
            var proposedEvents = GetNewProposal(cnEventPs, kar, currentEvents);
            double proposedFitness = GetFitness(new Karyotype(kar), proposedEvents);

            double acceptProb = MHParams.MatchFitness
                ? AcceptProb(proposedFitness, currentFitness, targetFitness)
                : AcceptProb(proposedFitness, currentFitness);

            if (acceptProb > Rnd.NextDouble())
            {
                currentEvents = proposedEvents;
                // Update best events based on the mode
                if (MHParams.MatchFitness)
                {
                    double proposedDiff = Math.Abs(proposedFitness - targetFitness);
                    if (proposedDiff < bestValue)
                    {
                        bestValue = proposedDiff;
                        bestEvents = proposedEvents;
                    }
                }
                else
                {
                    if (proposedFitness > bestValue)
                    {
                        bestValue = proposedFitness;
                        bestEvents = proposedEvents;
                    }
                }
            }
        }
        return bestEvents;
    }

    protected override (Karyotype childKar, List<CNEventDesc> childEvs) SampleEvents(
        Karyotype parentKar,
        CTreeNode cnChild,
        List<CNEventPars> cnEventPs,
        int mutDepth)
    {
        var childKar = new Karyotype(parentKar);
        var childEvs = new List<CNEventDesc>();
        double targetFit = cnChild.Fitness;
        double oldFitness = childKar.FitnessVal;

        int eventCount = SampleEventCount(cnChild);
        var bestEvents = GetEvents(cnEventPs, new Karyotype(childKar), eventCount, targetFit);

        for (int evNo = 1; evNo <= eventCount; evNo++)
        {
            Console.Write($"\rSample {cnChild.CloneId}. Event {evNo}/{eventCount}.".PadRight(80));
            var (eventData, signature) = bestEvents[evNo - 1];
            var regionsBefore = childKar.GetRegionDescriptions();
            eventData.ApplyEvent(childKar);
            var regionsAfter = childKar.GetRegionDescriptions();
            var (rGained, rLost) = GetKaryotypeDiff(regionsBefore, regionsAfter);
            double newFitness = childKar.UpdateFitness(RefGen, FitParams);
            double dFit = newFitness - oldFitness;
            var newEv = new CNEventDesc(eventData, mutDepth + evNo, dFit, newFitness, Signature: signature,
                RegionsGained: rGained, RegionsLost: rLost);
            childEvs.Add(newEv);
            oldFitness = newFitness;
        }

        return (childKar, childEvs);
    }
}