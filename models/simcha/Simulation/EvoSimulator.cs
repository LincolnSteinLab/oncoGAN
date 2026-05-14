﻿using SimChA.Computation;
using SimChA.Data;
using SimChA.EventData;
using SimChA.IO;

namespace SimChA.Simulation;

public class EvoSimulator(Random rnd, RefGen refGen, SimParams simParams, FitParams fitParams, EvoParams evoParams)
    : Simulator(rnd, refGen, simParams, fitParams)
{
    private EvoParams EvoParams { get; } = evoParams;

    private (Karyotype newKar, BaseEventData eventData, int numTries, string signature) GetNewEvent(List<CNEventPars> cnEventPars, Karyotype currentKar)
    {
        for (int tryNo = 0; tryNo <= EvoParams.MaxTries; tryNo++)
        {
            if (SampleStat.CalcPloidy(currentKar, RefGen) > 32)
            {
                break;
            }
            var cnEventP = Rnd.PickRndElem(cnEventPars);
            var eventData = Sampling.GenerateCNEventData(Rnd, currentKar, cnEventP);
            if (eventData == null)
            {
                continue;
            }

            var proposedKar = new Karyotype(currentKar);
            eventData.ApplyEvent(proposedKar);
            double proposedFitness = proposedKar.UpdateFitness(RefGen, FitParams);
            if (Math.Exp(proposedFitness - currentKar.FitnessVal - EvoParams.Acceptance) > Rnd.NextDouble())
            {
                return (proposedKar, eventData, tryNo, cnEventP.Signature);
            }
        }
        return (currentKar, CreateSkipEvent(), EvoParams.MaxTries, "");
    }
    
    protected override (Karyotype childKar, List<CNEventDesc> childEvs) SampleEvents(
        Karyotype currentKar, 
        CTreeNode cnChild, 
        List<CNEventPars> cnEventPs, 
        int mutDepth)
    {
        var childEvs = new List<CNEventDesc>();
        int eventCount = SampleEventCount(cnChild);
        for (int evNo = 1; evNo <= eventCount; evNo++)
        {
            Console.Write($"\rSample {cnChild.CloneId}. Event {evNo}/{eventCount}.".PadRight(80));
            var regionsBefore = currentKar.GetRegionDescriptions();
            var (newKar, eventData, numTries, signature) = GetNewEvent(cnEventPs, currentKar);
            var regionsAfter = newKar.GetRegionDescriptions();
            var (rGained, rLost) = GetKaryotypeDiff(regionsBefore, regionsAfter);
            double newFit = newKar.FitnessVal;
            double dFit = newFit - currentKar.FitnessVal;
            var newEv = new CNEventDesc(eventData, mutDepth + evNo, dFit, newFit, numTries, signature,
                RegionsGained: rGained, RegionsLost: rLost);
            
            childEvs.Add(newEv);
            currentKar = newKar;
        }
        return (currentKar, childEvs);
    }
}