package 算法.数据结构;

import java.util.*;
public class tools {
    public static int[] a = new int[]{7, 8, 1, 2, 0};
    public static void main(String[] args) {
        // 数组
        int[] array = new int[5];
        // list
        ArrayList<Integer> ls = new ArrayList<>();
        // set
        HashSet<Integer> hs = new HashSet<>();
        // map
        HashMap<Integer, Integer> hm = new HashMap<>();
        // stack
        Stack<Integer> s = new Stack<>();
        // Queue
        Queue<Integer> q1 = new LinkedList<>();
        Queue<Integer> q2 = new PriorityQueue<>();
        // 堆
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
    }
    // 比较器
    public static Comparator<Integer> cm = new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o1 - o2;
        }
    };
}
