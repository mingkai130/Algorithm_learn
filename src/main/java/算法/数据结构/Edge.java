package 算法.数据结构;

public class Edge {

    public int weight;
    public Node from, to;
    public Edge(int weight, Node from, Node to){
        this.weight = weight;
        this.from = from;
        this.to = to;
    }

}
