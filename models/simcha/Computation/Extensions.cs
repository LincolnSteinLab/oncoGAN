namespace SimChA.Computation;

public static class Extensions
{
    public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source, Random rnd)
        => source.OrderBy(_ => rnd.Next());

    public static bool CoinFlip(this Random rnd)
        => rnd.Next(0, 2) == 0;
    
    // Biased coin, the higher the bias, the more likely it is to return true
    public static bool CoinFlip(this Random rnd, double bias)
        => rnd.NextDouble() < bias;
    
    public static int GetRandIndex<T>(this IEnumerable<T> source, Random rnd)
        => rnd.Next(source.Count());
    
    public static T GetRndElem<T>(this List<T> source, Random rnd)
        => source[rnd.Next(source.Count)];

    public static IEnumerable<int> GetRandIndices<T>(this IEnumerable<T> source, int count, Random rnd)
        => Enumerable.Range(0, source.Count()).Shuffle(rnd).Take(count);

    public static IEnumerable<T> GetValues<T>() where T : Enum 
        => Enum.GetValues(typeof(T)).Cast<T>();
    
    public static void ForEach<T>(this IEnumerable<T> sequence, Action<T> action)
    {
        foreach (var item in sequence)
        {
            action(item);
        }
    }
    public static int PickRndIndex(this Random rnd, IList<double> elems) 
    {
        double val = rnd.NextDouble();
        for (int i = 0; i < elems.Count; i++)
        {
            if (val < elems[i])
            {
                return i;
            }
            val -= elems[i];
        }
        return elems.Count - 1;
    }
    
    public static int PickRndIndex<T>(this Random rnd, IList<T> elems) where T : IHasProb
    {
        double val = rnd.NextDouble();
        for (int i = 0; i < elems.Count; i++)
        {
            if (val < elems[i].Prob)
            {
                return i;
            }
            val -= elems[i].Prob;
        }
        return elems.Count - 1;
    }
    
    public static T PickRndElem<T>(this Random rnd, List<T> elems) where T : IHasProb
        => elems[rnd.PickRndIndex(elems)];
}