namespace SimChA.Data;

public record CTreeNode(string CloneId, string ParentId, int Distance, double Fitness)
{
    public bool IsRoot => ParentId == CloneId;
}