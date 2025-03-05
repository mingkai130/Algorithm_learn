package 算法;

import 算法.数据结构.ListNode;

import java.util.*;

import static 算法.数据结构.tools.*;

public class finished_codes2 {
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

    // 38.外观数列 -- 经过思考一次过！！正向模拟递归
    public static String countAndSay(int n) {
        if (n == 1) return "1";
        else {
            String ans = new String("1");
            for (int i = 1; i < n; i++) {
                ans = encode(ans);
            }
            return ans;
        }
    }
    public static String encode(String str){
        StringBuffer ans = new StringBuffer();
        for (int i = 0; i < str.length(); i++) {
            // 处理单个字符
            if (i == str.length() - 1 || str.charAt(i) != str.charAt(i + 1) && i + 1 < str.length()){
                ans.append('1');
                ans.append(str.charAt(i));
            }
            // 多个字符s
            else {
                int count = 1;
                while (i + 1 < str.length() && str.charAt(i) == str.charAt(i + 1) ){
                    count += 1;
                    i += 1;
                }
                ans.append(count);
                ans.append(str.charAt(i));
            }
        }
        return ans.toString();
    }

    // 39.组合总和 -- 想暴力回溯到叶节点，但是这个题目是每个数字都可以用无数次，要用dfs，这是答案dfs，不含剪枝操作  --- 重要模板
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> combine = new ArrayList<Integer>();
        int sum=0;
        dfs_back(candidates, target, ans, 0, combine, sum);
        return ans;
    }
    public static void dfs_back(int[] candidates, int target, List<List<Integer>> ans, int index, List<Integer> combine, int sum) {
        // index递归到最后一个candidate之后的那个，说明这条路走不通了
        if (index == candidates.length) return;
        // 找到一种组合
        if (target == 0) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        // 不选择当前值，直接跳过,index++, combine不变
        dfs_back(candidates, target, ans, index + 1, combine, sum);
        // 选择当前值，dfs继续，先把当前值加进来，dfs之后把它移出去，注意index不能变！
        if (target - candidates[index] >= 0) {
            combine.add(candidates[index]);
            dfs_back(candidates, target - candidates[index], ans, index, combine, sum);
            combine.removeLast();
        }
    }

    // 39.组合总和 -- 想暴力回溯到叶节点,自己试一下  --- 重要模板
    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {

        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> combine = new ArrayList<Integer>();
        dfs_back(candidates, target, ans, 0, combine, 0);
        return ans;
    }
    public static void dfs_back2(int[] candidates, int target, List<List<Integer>> ans, int index, List<Integer> combine, int sum) {
        // 此路不通
        if (index == candidates.length) return;
        // 找到答案
        if (sum == target) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        // 中间情况
        if (sum > target) return;
        // 不选择当前值
        dfs_back(candidates, target, ans, index + 1, combine, sum);
        // 选择当前值
        combine.add(candidates[index]);
        dfs_back(candidates, target, ans, index, combine, sum + candidates[index]);
        combine.removeLast();
    }

    // 40. 组合总和 II  一眼回溯  -- 不熟练……
    public static List<List<Integer>> combinationSum3(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();
        dfs_back_combine2(ans ,candidates, target, combine, 0, 0);
        return ans;
    }
    public static void dfs_back_combine2(List<List<Integer>> ans,int[] candidates, int target, List<Integer> combine, int index, int sum){

        // 找到答案
        if (sum == target) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        // 中间情况
        if (sum > target) return;
        for (int i = index; i < candidates.length; i++) {
            // 跳过重复的元素
            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }
            // 做选择
            combine.add(candidates[i]);
            // 递归：因为每个数字只能使用一次，所以 index 要传 i+1
            dfs_back_combine2(ans, candidates, target, combine, i + 1, sum + candidates[i]);
            // 撤销选择
            combine.remove(combine.size() - 1);
        }
    }

    // 41. 缺失的第一个正数 -- 哈希表解法很普通，答案用当前nums代替哈希表，很巧妙
    public static int firstMissingPositive(int[] nums) {
        Arrays.sort(nums);
        HashSet<Integer> hs = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            hs.add(nums[i]);
        }
        int i = 1;
        for (; i < Integer.MAX_VALUE; i++) {
            if (!hs.contains(i)) break;
        }
        return i;
    }

    // 43. 字符串相乘 -- 算法模拟大数乘法运算
    public static String multiply(String num1, String num2) {

        // 特殊情况处理
        if (num1.equals("0") || num2.equals("0")) return "0";
        if (num1.equals("1") || num2.equals("1")) return num1.equals("1")? num2:num1;
        // 主程序 -- 模拟乘法
        String ans= "";
        for (int i = num2.length() - 1; i >= 0 ; i--) {
            StringBuilder temp = new StringBuilder();
            int forward = 0;
            for (int j = num1.length() - 1; j >= 0; j--) {
                int multiply = (num1.charAt(j) - 48) * (num2.charAt(i) - 48) + forward;
                forward = multiply / 10;
                temp.insert(0, multiply % 10);
            }
            if (forward != 0) temp.insert(0,  forward);
            if (i < num2.length() - 1) for (int j = 0; j < num2.length() - 1 - i; j++) {
                temp.append("0");
            }
            ans = StringPlus(ans, temp.toString());
        }
        return ans;
    }
    // 算法模拟大数加法运算
    public static String StringPlus(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, add = 0;
        StringBuffer ans = new StringBuffer();
        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? num1.charAt(i) - '0' : 0;
            int y = j >= 0 ? num2.charAt(j) - '0' : 0;
            int result = x + y + add;
            ans.append(result % 10);
            add = result / 10;
            i--;
            j--;
        }
        ans.reverse();
        return ans.toString();
    }

    // 44.通配符匹配  -- 看了一眼就想动态规划，但是具体细节还没有很清楚，需要看答案
    public static boolean isMatch(String s, String p) {

        int m = s.length();
        int n = p.length();

        // dp[i][j] 表示字符串 s 的前 i 个字符和模式 p 的前 j 个字符是否能匹配
        boolean[][] dp = new boolean[m + 1][n + 1];

        // dp 数组边界处理
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '?')
                    dp[i][j] = dp[i - 1][j - 1];
                else if (p.charAt(j - 1) == s.charAt(i - 1))
                    dp[i][j] = dp[i - 1][j - 1];
                else if (p.charAt(j - 1) == '*')
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            }
        }

        return dp[m][n];
    }

    // 45. 跳跃游戏 II  答案从后往前找，挺厉害,虽然直观，但是时间复杂度比较高
    public static int jump(int[] nums) {
        int position = nums.length - 1;
        int ans = 0;
        while (position > 0){
            for (int i = 0; i < position; i++) {
                // 这个判断是核心，一定是 >=
                if (i + nums[i] >= position) {
                    ans ++;
                    position = i;
                    break;
                }
            }
        }
        return ans;
    }

    // 46.全排列   一眼回溯,就是不太熟练要多练！
    public static List<List<Integer>> permute(int[] nums) {
        ArrayList<List<Integer>> ans = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        backTrace(nums, ans, temp, 0);
        return ans;
    }
    public static void backTrace(int[] nums,ArrayList<List<Integer>> ans, ArrayList<Integer> temp, int level){
        // 到达最底层，加入答案集
        if (level == nums.length) ans.add(new ArrayList<>(temp));
        else{
            // for循环保证下一个位置可以放各种值！
            for (int i = 0; i < nums.length; i++) {
                // 直接用内置函数，不要用HashSet
                if (!temp.contains(nums[i])) {
                    temp.add(nums[i]);
                    backTrace(nums, ans, temp, level+1);
                    temp.removeLast();
                }
            }
        }
    }

    // 47.全排列 II  一眼回溯，要处理重复值
    public static List<List<Integer>> permute2(int[] nums) {
        ArrayList<List<Integer>> ans = new ArrayList<>();
        // 暂存某个答案
        ArrayList<Integer> temp = new ArrayList<>();
        // 暂存是否加过这个值
        ArrayList<Integer> used = new ArrayList<>();
        backTrace2(nums, ans, temp, 0, used);
        return ans;
    }
    public static void backTrace2(int[] nums, ArrayList<List<Integer>> ans, ArrayList<Integer> temp, int level, ArrayList<Integer> used){
        // 到达最底层，加入答案集
        if (level == nums.length && !ans.contains(temp))
            ans.add(new ArrayList<>(temp));
        else{
            // for循环保证下一个位置可以放各种值！
            for (int i = 0; i < nums.length; i++) {
                // 直接用内置函数，不要用HashSet
                if (!used.contains(i)) {
                    temp.add(nums[i]);
                    used.add(i);
                    backTrace2(nums, ans, temp, level+1, used);
                    temp.removeLast();
                    used.removeLast();
                }
            }
        }
    }

    // 48. 旋转图像  如果不用辅助数组的话半天没头绪; 看大佬：
    //用reverse将每一行数据进行倒排，倒排后的矩阵沿y=x对称分布，随后再用swap进行交换即可得到最终答案。
    public static void rotate(int[][] matrix) {
        // n*n 的二维矩阵
        int n = matrix.length;

        // 先reverse将每一行数据进行倒排
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = temp;
            }
        }
        // 沿 y=x 对称分布，直接swap
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < n - i; j++) {
                // 交换 matrix[i][j] 和 matrix[n-1-j][n-1-i]
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][n - 1 - i];
                matrix[n - 1 - j][n - 1 - i] = temp;
            }
        }
    }

    // 49. 字母异位词分组  自己暴力写，虽然答案对的但是超时
    public static List<List<String>> groupAnagrams(String[] strs) {

        // 特殊情况处理
        ArrayList<List<String>> ans = new ArrayList<>();
        if (strs.length == 0) return ans;
        List<String> cur = new ArrayList<>();
        boolean[] used = new boolean[strs.length];
        for (int i = 0; i < strs.length; i++) {
            if (!used[i])
            {
                cur.add(strs[i]);
                used[i] = true;
                for (int j = i + 1; j < strs.length; j++) {
                    if (CharacterMatch(strs[i], strs[j])) {
                        cur.add(strs[j]);
                        used[j] = true;
                    }
                }
            }
            if (!cur.isEmpty()) {
                ans.add(new ArrayList<>(cur));
                cur.clear();
            }
        }
        return ans;
    }
    public static boolean CharacterMatch(String a, String b){
        if (a.length() != b.length()) return false;
        HashMap<Character, Integer> map1 = new HashMap<>();
        HashMap<Character, Integer> map2 = new HashMap<>();
        for (int i = 0; i < a.length(); i++) {
            if (!map1.containsKey(a.charAt(i))){
                map1.put(a.charAt(i), 1);
            }
            else {
                map1.put(a.charAt(i), map1.get(a.charAt(i)) + 1);
            }
            if (!map2.containsKey(b.charAt(i))){
                map2.put(b.charAt(i), 1);
            }
            else {
                map2.put(b.charAt(i), map2.get(b.charAt(i)) + 1);
            }
        }
        return map1.equals(map2);
    }
    public static List<List<String>> groupAnagrams2(String[] strs) {

        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            // 字符串转为字符数组
            char[] array = str.toCharArray();
            // 排序
            Arrays.sort(array);
            String key = new String(array);
            // getOrDefault 方法，如果map有的话，得到key，否则得到一个空的list
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());

    }

    // 50. Pow(x, n)  -- 高精度幂次运算难点在于会溢出, 思路简单但超时
    public static double myPow(double x, int n) {
        if (n == 0) return 1;
        // 取指数的正负号
        int sign = -(1 & (n >> 31));
        System.out.println(sign);
        double ans = 1;
        for (int i = 1; i <= Math.abs(n); i++) {
            ans =  (ans * x);
        }

        return sign == 0 ? ans: 1 / (ans);
    }

    // 答案标准解答：快速幂
    public static double myPow2(double x, int n) {
        if (n == 0) return 1;
        double ans = myPowHelper(x, Math.abs(n));
        return n > 0? ans : 1 / (ans);
    }
    public static double myPowHelper(double x, int n){
        if (n == 0) return 1.0;
        double half = myPowHelper(x, n / 2);
        return n % 2 == 0? half * half : half * half * x;
    }

    // 53. 最大子数组和  暴力都不用试，肯定超时; ans 是动态规划
    public static int maxSubArray(int[] nums) {
        int answer = nums[0];

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int num = 1;num < nums.length;num++) {
            dp[num] = Math.max(nums[num], dp[num - 1] + nums[num]);
            answer = Math.max(dp[num], answer);
        }
        return answer;
    }

    // 54. 螺旋输出矩阵 -- 暴力仔细
    public static List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = n - 1, top = 0, bottom = m - 1;
        while (left <= right && top <= bottom) {
            // 从左到右
            for (int i = left; i <= right; i++) {
                ans.add(matrix[top][i]);
            }
            top++;
            // 从上到下
            for (int i = top; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            right--;
            // 从右到左
            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    ans.add(matrix[bottom][i]);
                }
            }
            bottom--;
            // 从下到上
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    ans.add(matrix[i][left]);
                }
            }
            left++;
        }
        return ans;
    }

    // 55. 跳跃游戏 -- 贪心，不断向右扩张
    public static boolean canJump(int[] nums) {

        int maxRight = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= maxRight){
                maxRight = Math.max(maxRight, i + nums[i]);
            }
            else return false;
            if (maxRight >= nums.length - 1) return true;
        }
        return false;
    }

    // 56. 合并区间, 思路易，但答案的格式需要注意！先排序很重要的
    public static int[][] merge(int[][] intervals) {

        // 特殊情况
        if (intervals.length == 0 || intervals.length == 1)
            return intervals;

        List<int[]> ans = new ArrayList<>();
        // 比较器，对区间按照第一个元素排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        for (int i = 0; i < intervals.length; i++) {
            // [L][R]
            int L = intervals[i][0], R = intervals[i][1];

            // 给答案中直接添加没有的区间
            if (ans.isEmpty() || ans.get(ans.size() - 1)[1] < L){
                ans.add(new int[] {L, R});
            }
            // 区间合并
            else {
                ans.get(ans.size() - 1) [1] = Math.max(ans.get(ans.size() - 1) [1], R);
            }

        }


        return ans.toArray(new int[ans.size()][]);
    }

    // 57.插入区间  分三种情况模拟，注意有交集的时候要仔细保存左右边界
    public static int[][] insert(int[][] intervals, int[] newInterval) {

        List<int[]> ans = new ArrayList<>();
        boolean placed = false;

        // 暂存重叠部分的边界
        int left = newInterval[0];
        int right = newInterval[1];
        for (int i = 0; i < intervals.length; i++) {
            // 无交集，直接将intervals[i] 加入ans
            if (intervals[i][1] < newInterval[0])
                ans.add(intervals[i]);
                // 无交集，如果之前没有加过 newInterval， 加入
            else if (intervals[i][0] > newInterval[1]) {
                if (!placed) {
                    ans.add(new int[] {left, right});
                    placed = true;
                }
                ans.add(intervals[i]);
            }
            // 有交集
            else {
                // 计算并集
                left = Math.min(left, intervals[i][0]);
                right = Math.max(right, intervals[i][1]);
            }
        }
        // 最终如果 newInterval 最大，加入
        if (!placed) {
            ans.add(new int[]{left, right});
        }
        return ans.toArray(new int[ans.size()][]);
    }

    // 58. 最后一个单词的长度 -- 易，一次通过
    public static int lengthOfLastWord(String s) {

        int ans = 0;
        boolean meetWord = false;
        for (int i = s.length() - 1 ; i >= 0 ; i--) {
            if (s.charAt(i) == ' ' && meetWord){
                break;
            }
            else if (s.charAt(i)!=' ') {ans ++; meetWord = true;}
        }

        return ans;
    }

    // 59. 螺旋矩阵 II  暴力模拟 一次通过，要注意每次转完 1/4 周，都要给对应的边界++/--
    public static int[][] generateMatrix(int n) {
        int[][] ans = new int[n][n];
        // 设置四个边界
        int top = 0, bottom = n - 1, left = 0, right = n - 1;

        int count = 1;

        while (left <= right && top <= bottom){
            // 左 -> 右
            for (int i = left; i <= right; i++) {
                ans[top][i] = count; count++;
            }
            top++;
            // 上 -> 下
            for (int i = top; i <= bottom; i++) {
                ans[i][right] = count; count++;
            }
            right--;
            // 右 -> 左
            if (bottom >= top)
                for (int i = right; i >= left; i--) {
                    ans[bottom][i] = count; count++;
                }
            bottom--;
            // 下 -> 上
            if (left <= right)
                for (int i = bottom; i >= top; i--) {
                    ans[i][left] = count; count++;
                }
            left++;
        }
        return ans;
    }

    // 61. 旋转链表 -- 自己暴力通过，不难；但是答案更巧妙，找到该链表的末尾节点，将
    //     其与头节点相连。这样就得到了闭合为环的链表。然后我们找到新链表的最后一个节点，将当前闭合为环的链表断开
    public static ListNode rotateRight(ListNode head, int k) {
        ListNode left = head, right = head;
        // 特殊情况处理
        if (head == null || head.next == null) return head;

            // 旋转链表
        else
        {       int count = 0; ListNode temp = head;

            while (temp!=null) {
                count++;    temp = temp.next;
            }
            count = k % count;
            while (count > 0){
                // 右指针找倒数第二个节点
                while (right.next.next != null) right = right.next;

                right.next.next = left;
                left = right.next;
                right.next = null;
                right = left;
                count--;
            }
        }
        return left;
    }

    // 62. 不同路径  一眼动态规划 ,f(i,j)=f(i−1,j)+f(i,j−1) 秒了
    public static int uniquePaths(int m, int n) {

        int[][] dp = new int[m + 1][n + 1];
        // dp 数组初始化
        for (int i = 0; i <= n; i++) {
            dp[1][i] = 1;
        }
        for (int i = 0; i <= m; i++) {
            dp[i][1] = 1;
        }
        for (int i = 2; i <= m; i++) {
            for (int j = 2; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i] [j - 1];
            }
        }
        return dp[m][n];
    }

    // 63. 不同路径 II  同样动态规划，有障碍物  思路很简单，dp数组更新的时候要注意
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.length;
        int m = obstacleGrid[0].length;
        int[][] dp = new int[n][m];

        for(int i = 0 ;i < m;i ++){
            if(obstacleGrid[0][i] == 1)break;
            dp[0][i] = 1;
        }
        for(int i = 0 ;i < n;i ++){
            if(obstacleGrid[i][0] == 1) break;
            dp[i][0] = 1;
        }


        for(int i = 1 ; i < n;i ++){
            for(int j = 1  ;j < m;j ++){
                if(obstacleGrid[i][j] == 0){
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }

        return dp[n-1][m-1];
    }

    // 64. 最小路径和   也是一眼动态规划 秒了
    public static int minPathSum(int[][] grid) {
        int m = grid.length + 1, n = grid[0].length + 1;

        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        // dp 数组初始化
        for (int i = 1; i < grid.length; i++) {
            dp[i][0] = grid[i][0] + dp[i - 1][0];
        }
        for (int i = 1; i < grid[0].length; i++) {
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        }
        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[0].length; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 2][n - 2];
    }
}
