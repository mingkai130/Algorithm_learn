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

    // 交换某个 int 数组两个元素
    public static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    // 将某个 int 数组的部分逆置
    public static void reverse(int[] nums, int left, int right) {
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }
}
