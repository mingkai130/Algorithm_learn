package 算法;

import com.sun.source.tree.BinaryTree;

import java.util.*;

public class test {
    public static void main(String[] args) {
        System.out.println(Integer.MIN_VALUE); // -2147483648
        System.out.println(Integer.MAX_VALUE); //  2147483647

    }


    public static class LinkNode{
        public int val;
        public LinkNode next;
        public LinkNode(){
            this.val = 0;
        }
        public LinkNode(int value){
            this.val = value;
        }
    }
    // 树节点
    public static class TreeNode{
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int data){
            this.val = data;
        }
        public TreeNode(){
            this.val = 0;this.left = null;this.right = null;
        }
    }

    // 哈希表分为 hashset/hashmap
    public void hashes(){
        // hashmap
        HashMap<Integer, String> hm = new HashMap<>();
        hm.put(1, "first");
        hm.put(3, "third");
        hm.put(2, "second");
        hm.put(0, "sero");
        System.out.println(hm.get(3));
        System.out.println(hm);

        // hashset
        HashSet<Integer> hs = new HashSet<>();
        hs.add(0);
        hs.add(2);
        System.out.println(hs);
    }

    // 有序表结构 treemap
    public void trees(){
        TreeMap<Integer, String> tm = new TreeMap<>();
        tm.put(1, "one");
        tm.put(3, "three");
        tm.put(2, "two");
        System.out.println(tm);  // 自动根据 key 值排序
    }

    // 二叉树
    public static void Binary_tree(){

    }
    // 单链表遍历
    public static void LinkList(){
        LinkNode node1 = new LinkNode(),node2 = new LinkNode();
        node1.val = 1;node2.val = 2;
        node1.next = node2;
        LinkNode pointer = node1;
        while (pointer != null){
            System.out.println(pointer.val);
            pointer = pointer.next;
        }
    }
    // List 替换 int[]
    public static void List_test(){
        ArrayList<Integer> ls = new ArrayList<>();
        ls.add(2);ls.add(1);
        System.out.println(ls);
    }

    // 找树中两个节点的最低公共祖先（极度优化之后的代码）
    public static TreeNode lowestAncestor(TreeNode head,TreeNode o1,TreeNode o2){
        if (head == null || head == o1 || head == o2) {
            return head;
        }
        TreeNode left = lowestAncestor(head.left, o1, o2);
        TreeNode right = lowestAncestor(head.right, o1, o2);
        if (left != null && right != null){
            return head;
        }
        return left != null ? left : right;
    }
}
