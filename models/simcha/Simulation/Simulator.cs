using SimChA.Computation;
using SimChA.Data;
using SimChA.EventData;
using SimChA.IO;

namespace SimChA.Simulation;

public class Simulator(Random rnd, RefGen refGen, SimParams simParams, FitParams fitParams)
{
    protected Random Rnd  { get; } = rnd;
    protected RefGen RefGen  { get; } = refGen;
    protected FitParams FitParams  { get; } = fitParams;
    protected SimParams SimParams { get; } = simParams;

    protected static BaseEventData CreateSkipEvent() 
        => new(new CNEventPars(CNEventType.Skip, 1));

    public List<Sample> Simulate(CTreeNode root, List<CTreeNode> cloneTree, List<Signature> sigs)
    {
        var (cnEventPs, sampleMixture) = Factory.MixSignatures(Rnd, sigs, SimParams.Mixture);
        var res = new List<Sample>();
        var sex = SimParams.AutosomesOnly ? SexType.Any : Sampling.GetSex(Rnd, SimParams.Sex);
        var rootKar = new Karyotype(RefGen, sex);
        if (SimParams.TetraploidStart)
        {
            rootKar.ApplyWGD();
        }
        rootKar.UpdateFitness(RefGen, FitParams);
        ApplyCNEventsRec(root, cloneTree, cnEventPs, sampleMixture, res, rootKar, 0);
        return res;
    }

    protected int SampleEventCount(CTreeNode node) 
        => node.Distance >= 0 ? node.Distance : Sampling.SampleDiscDist(Rnd, SimParams.RateDist, SimParams.RateMean);

    protected static (List<string> regionsGained, List<string> regionsLost) GetKaryotypeDiff(
        Dictionary<int, List<string>> regionsBefore, Dictionary<int, List<string>> regionsAfter)
    {
        var regionsGained = new List<string>();
        var regionsLost = new List<string>();
        
        // Regions from whole contigs that were gained
        foreach (var id in regionsAfter.Keys.Where(id => !regionsBefore.ContainsKey(id)))
            regionsGained.AddRange(regionsAfter[id]);
        
        // Regions from whole contigs that were lost
        foreach (var id in regionsBefore.Keys.Where(id => !regionsAfter.ContainsKey(id)))
            regionsLost.AddRange(regionsBefore[id]);
        
        // Regions changed within contigs that exist both before and after
        foreach (var id in regionsBefore.Keys.Intersect(regionsAfter.Keys))
        {
            var before = regionsBefore[id];
            var after = regionsAfter[id];
            var beforeCounts = before.GroupBy(r => r).ToDictionary(g => g.Key, g => g.Count());
            var afterCounts = after.GroupBy(r => r).ToDictionary(g => g.Key, g => g.Count());
            foreach (var key in afterCounts.Keys)
            {
                int diff = afterCounts[key] - (beforeCounts.TryGetValue(key, out int bc) ? bc : 0);
                for (int i = 0; i < diff; i++) regionsGained.Add(key);
            }
            foreach (var key in beforeCounts.Keys)
            {
                int diff = beforeCounts[key] - (afterCounts.TryGetValue(key, out int ac) ? ac : 0);
                for (int i = 0; i < diff; i++) regionsLost.Add(key);
            }
        }
        return (regionsGained, regionsLost);
    }

    protected virtual (Karyotype childKar, List<CNEventDesc> childEvs) SampleEvents(
        Karyotype parentKar, 
        CTreeNode cnChild, 
        List<CNEventPars> cnEventPs, 
        int mutDepth)
    {
        var childKar = new Karyotype(parentKar);
        var childEvs = new List<CNEventDesc>();
        int eventCount = SampleEventCount(cnChild);
        for (int evNo = 1; evNo <= eventCount; evNo++)
        {
            Console.Write($"Sample {cnChild.CloneId}. Event {evNo}/{eventCount}.".PadRight(80) + "\r");
            var eventP = Rnd.PickRndElem(cnEventPs);
            var eventData = Sampling.GenerateCNEventData(Rnd, childKar, eventP) ?? CreateSkipEvent();
            var regionsBefore = childKar.GetRegionDescriptions();
            eventData.ApplyEvent(childKar);
            var regionsAfter = childKar.GetRegionDescriptions();
            var (rGained, rLost) = GetKaryotypeDiff(regionsBefore, regionsAfter);
            var newEv = new CNEventDesc(eventData, mutDepth + evNo, Signature: eventP.Signature, 
                RegionsGained: rGained, RegionsLost: rLost);
            childEvs.Add(newEv);
        } 
        return (childKar, childEvs);
    }

    private void ApplyCNEventsRec(
        CTreeNode parent, 
        List<CTreeNode> cloneTree, 
        List<CNEventPars> cnEventPs,
        Dictionary<string, double> mixture,
        List<Sample> sampleList,
        Karyotype parentKar,
        int mutDepth)
    {
        var children = cloneTree.Where(c => c.ParentId == parent.CloneId).ToList();
        foreach (var child in children)
        {
            var (childKar, childEvs) = SampleEvents(parentKar, child, cnEventPs, mutDepth);
            var newClone = new Sample(child.CloneId,parent.CloneId, childKar, childEvs, mixture, child.Fitness);
            sampleList.Add(newClone);
            
            if (child.CloneId != parent.CloneId)
            {
                ApplyCNEventsRec(child, cloneTree, cnEventPs, mixture, sampleList, childKar, mutDepth + childEvs.Count);
            }
        }
    }
}
