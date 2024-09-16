package 算法;

import 算法.数据结构.ListNode;
import 算法.数据结构.tools.*;

import java.util.*;

import static 算法.数据结构.tools.*;

public class finished_codes_two {
    // 两两交换链表中的节点 想到递归了，但是不知道怎么写，看答案写出来
    public static ListNode swapPairs(ListNode head) {

        // 特殊情况处理
        if (head == null || head.next == null) return head;
        // 主思路 -- 递归实现
        ListNode ans = head.next;
        head.next = swapPairs(ans.next);
        ans.next = head;
        return ans;
    }
    // 两两交换链表中的节点  非递归解法 (手动模拟)
    public static ListNode swapPairs2(ListNode head) {

        // 特殊情况处理
        if (head == null || head.next == null) return head;
        // 主思路 -- 非递归实现

        ListNode left = head;
        ListNode right = left.next;
        ListNode temp = new ListNode();
        temp.next = head;
        ListNode ans = head.next;

        while(left.next != null){
            left.next = right.next;

            right.next = left;
            left = left.next;
            if (left == null)
                break;
            temp = right;
            temp.next.next  = left.next == null ? left : left.next;

            right = left.next;
        }

        return ans;
    }
    // K 个一组翻转链表 -- 递归思想
    public static ListNode reverseKGroup(ListNode head, int k) {

        // 特殊情况
        if (k == 1 || head == null || head.next == null) return head;

        // 主思路
        int count = 1;
        ListNode right = head;
        while (count < k) {
            right = right.next;
            count ++;
            if (right == null) return head;
        }
        // reverse from head to right

        ListNode stop = right.next;
        ListNode cur = head;
        ListNode pre = null;
        ListNode nxt = head.next;
        while (cur != stop){
            cur.next = pre;
            pre = cur;
            cur = nxt;
            if (nxt == null) continue;
            nxt = nxt.next;
        }
        head.next = reverseKGroup(stop, k);
        return right;
    }
    // 链表翻转 -- 头插法
    public static ListNode reverseList(ListNode head){
        // 特殊情况处理
        if (head == null || head.next == null) return head;
        // List 翻转
        ListNode cur = head;
        ListNode pre = null;
        ListNode nextNode = cur.next;
        while (cur != null){
            cur.next = pre;
            pre = cur;
            cur = nextNode;
            // 处理 nextNode 不能为null
            if (nextNode != null)
                nextNode = nextNode.next;
        }
        // 返回头节点，原来的 head 已经变成尾节点了
        return pre;
    }
    // 删除数组重复元素 -- 双指针
    public static int removeDuplicates(int[] nums) {
        // 特殊情况
        if (nums.length == 1) return 1;
        // main 先快排
        Arrays.sort(nums);
        int fast = 1, slow = 1;
        while (fast < nums.length){
            if (nums[fast] != nums[fast - 1]) {
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }

        return slow;
    }

    // 27. 移除元素 -- 数据结构学到的那个，直接数 K 个和value相等的，其余往前移k个
    public static int removeElement(int[] nums, int val) {

        // 数组先排序
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            nums[i - count] = nums[i];
            if (nums[i] == val) {
                count ++;
            }
        }
        return nums.length - count;
    }

    // 28. 找出字符串中第一个匹配项的下标  -- javaAPI一行，但是KMP不会写,也可以暴力解
    public static int strStr(String haystack, String needle) {
        if (haystack.contains(needle)) return haystack.indexOf(needle);
        return -1;
    }

    // 29. 两数相除 -- 不能用计算机的乘除
    public static int divide(int dividend, int divisor) {
        int sign = (dividend >> 31 ^ divisor >> 31) == 1 ? 1 : -1;
        // 位运算全部转换为负数计算，因为负数比正数大1，不易溢出
        dividend = (1 >> 31) | dividend;
        divisor = (1 >> 31) | divisor;
        // 转换为更大的
        double ans = (double) dividend / divisor;
        if (ans > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        if (ans < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        return (int) (sign == 1 ? -ans : ans);
    }

    // 30.暴力模拟手工匹配 -- 最后一个例子超时
    public static List<Integer> findSubstring(String s, String[] words) {

        int window_length = words.length * words[0].length();
        // 特殊非法情况处理
        if (window_length > s.length()) return new ArrayList<>();

        // main
        List<Integer> ans = new ArrayList<>();
        for (int i = 12; i < s.length() - window_length; i += 1) {

            Map<String, Integer> current_map = new HashMap<>();
            // words 入 map
            for (String word : words) {
                if (!current_map.containsKey(word)) current_map.put(word, 1);
                else current_map.put(word, current_map.getOrDefault(word, 0) + 1);
            }

            int j = i;
            for (; j < i + window_length; j += words[0].length()) {
                if (!current_map.containsKey(s.substring(j, j + words[0].length()))){
                    break;
                } else {
                    current_map.put(s.substring(j, j + words[0].length()), current_map.getOrDefault(s.substring(j, j + words[0].length()), 0) - 1);
                }
            }
            if (j == i + window_length) {
                Set<String> a = current_map.keySet();
                boolean t = true;
                for(Object k : current_map.keySet()) {
                    if (current_map.get(k) < 0) t = false;
                }
                if (t) ans.add(i);
            }
        }
        return ans;
    }

    // 31.下一个排列
    /*
    // 首先从后向前查找第一个顺序对 (i,i+1)，满足 a[i]<a[i+1]。这样「较小数」即为 a[i]。此时 [i+1,n) 必然是下降序列。
    //如果找到了顺序对，那么在区间 [i+1,n) 中从后向前查找第一个元素 j 满足 a[i]<a[j]。这样「较大数」即为 a[j]。
    //交换 a[i] 与 a[j]，此时可以证明区间 [i+1,n) 必为降序。我们可以直接使用双指针反转区间 [i+1,n) 使其变为升序
      */
    public static void nextPermutation(int[] nums) {
        // 找到从后往前第一个升序
        int i = nums.length - 1;
        for (; i  > 0; i--) {
            if (nums[i] > nums[i - 1])
                break;
        }
        // 特殊情况处理
        if ( i > 0){
            if (i == nums.length - 1) swap(nums, nums.length - 1, nums.length - 2);
            else{
                int j = nums.length - 1;
                for (; j > i; j--) {
                    if (nums[j] > nums[i - 1]){
                        // 换位
                        swap(nums, i - 1, j);
                        break;
                    }
                }
                if (j == i) swap(nums, i-1, i);
                // 让 (i, end) 升序
                reverse(nums, i, nums.length - 1);
            }
        }
        else
            // 直接原地换位
            reverse(nums, 0, nums.length - 1);
    }

    // 32.最长有效括号 自己暴力  -- 虽然对但是大例子超时
    public static int longestValidParentheses(String s) {
        if (s.isEmpty()) return 0;
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ')') continue;
            for (int j = s.length() - 1; j > i; j--) {
                if (s.charAt(j) == ')' && finished_codes.isValidkuohao(s.substring(i, j + 1)))
                {
                    ans = Math.max(ans, j - i + 1);
                    i = j + 1;
                }

            }
        }
        return ans;
    }
    // 32.最长有效括号 自己暴力  -- 虽然对但是大例子超时
    public static int longestValidParentheses_BP(String s) {
        if (s.isEmpty()) return 0;
        int ans = 0;
        // 定义一个 dp 数组，自动为全0
        int[] dp = new int[s.length()];

        for (int i = 1; i < dp.length; i++) {
            if (s.charAt(i) == ')') {
                // 形如 …………()
                if (s.charAt(i - 1) == '('){
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                }
                // 形如 ((…………)) -- 好难推导
                else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '('){
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                ans = Math.max(ans, dp[i]);
            }

        }
        return ans;
    }

    // 33.搜索旋转排序数组 -- 二分法递归解决 !! 这种不规律递增序列居然也能递归，只不过要分两种情况！
    // 二分的时候一定有一半是有序的！
    public static int search(int[] nums, int target) {
        int ans;
        int left = 0, right = nums.length - 1;
        ans = biSearch(nums, target, left, right);
        return ans;
    }
    public static int biSearch(int[] nums, int target, int left, int right){
        if (left > right) return -1;
        if (target == nums[left]) return left;
        if (target == nums[right]) return right;
        int mid = (left + right) / 2;
        if (target == nums[mid]) return mid;
        // 第一种情况,在左部分查找
        if (nums[mid] >= nums[left] && nums[mid] >= nums[right]){
            if (target > nums[left] && target < nums[mid]) return biSearch(nums, target, left, mid - 1);
            else return biSearch(nums, target, mid + 1, right);
        }
        else {
            if (target > nums[mid] && target < nums[right]) return biSearch(nums, target, mid + 1, right);
            else return biSearch(nums, target, left, mid - 1);
        }
    }

    // 34.在有序数组中查找元素的第一个和最后一个位置 -- 非递归二分查找，找到一个之后向两边扩散找边界
    public static int[] searchRange(int[] nums, int target) {
        int[] ans = new int[2]; ans[0] = -1;ans[1] = -1;
        // 特殊情况
        if (nums.length == 0 || target < nums[0] || target > nums[nums.length - 1])
            return new int[]{-1, -1};
        // 非递归二分查找
        int left = 0, right = nums.length - 1, mid = (left + right) / 2;
        while (left < right){
            if (nums[mid] == target) break;
            if (target > nums[mid]){
                left = mid + 1;
                mid = (left + right) / 2;
            }
            else if (target < nums[mid]){
                right = mid - 1;
                mid = (left + right) / 2;
            }
        }
        // 左右扩散找边界
        if (nums[mid] == target){
            int i = mid;
            for (; i > 0 ; i--) {
                if (nums[i - 1] != nums[i]) break;
            }
            ans[0] = i;
            for (i = mid; i < nums.length - 1; i++) {
                if (nums[i + 1] != nums[i]) break;
            }
            ans[1] = i;
        }
        return ans;
    }

    // 35.搜索插入位置 -- 直接二分查找 很easy
    public static int searchInsert(int[] nums, int target) {
        if (target <= nums[0]) return 0;
        int left = 0, right = nums.length - 1, mid = (left + right) >> 1;
        while (left <= right){
            if (nums[mid] == target) return mid;
            if (nums[mid] > target) {
                right = mid - 1;
                mid = (left + right) >> 1;
            }
            else {
                left = mid + 1;
                mid = (left + right) >> 1;
            }
        }
        return left;
    }

    // 36.有效的数独 -- 暴力 On方
    public static boolean isValidSudoku(char[][] board) {
        HashSet<Character> set = new HashSet<>();
        // 检查行是否合法
        for (int i = 0; i < 9; i++) {
            set.clear();
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.'){
                    if (set.contains(board[i][j]))
                        return false;
                    else {
                        set.add(board[i][j]);
                    }
                }
            }
        }
        set.clear();
        // 检查列是否合法
        for (int i = 0; i < 9; i++) {
            set.clear();
            for (int j = 0; j < 9; j++) {
                if (board[j][i] != '.'){
                    if (set.contains(board[j][i]))
                        return false;
                    else {
                        set.add(board[j][i]);
                    }
                }
            }
        }
        // 检查 3 * 3 是否合法
        for (int i = 0; i <= 6; i += 3) {
            for (int j = 0; j <= 6; j += 3) {
                set.clear();
                for (int k = i; k < i + 3; k++) {
                    for (int l = j; l < j + 3; l++) {
                        if (board[k][l] != '.') {
                            if (!set.contains(board[k][l]))
                                set.add(board[k][l]);
                            else {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        System.out.println(1);
        return true;
    }

}
