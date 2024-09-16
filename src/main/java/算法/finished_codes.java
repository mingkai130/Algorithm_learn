package 算法;

import 算法.数据结构.ListNode;

import java.lang.annotation.Retention;
import java.util.*;

public class finished_codes {
    // 哈希表查找一个数组中某两个值相加等于target，并返回这两个值的下标（暴力解法太low了，这个哈希表更吊）
    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (mp.containsKey(target - nums[i])) {
                return new int[]{mp.get(target - nums[i]), i};
            } else {
                mp.put(nums[i], i);
            }
        }
        return new int[]{-1};
    }

    // 两个链表代表两个整数，返回两数之和的链表
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int sum = 0;
        ListNode head = new ListNode();
        ListNode cur = head;
        while (l1 != null || l2 != null) {

            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;  // 获取两个链表对应位置的值

            head.next = new ListNode((n1 + n2 + sum) % 10);
            head = head.next;
            sum = (n1 + n2 + sum) / 10;

            if (l1 != null)
                l1 = l1.next;
            if (l2 != null)
                l2 = l2.next;

            System.out.println(1);
        }
        // 最高位有可能进位
        if (sum > 0) {
            head.next = new ListNode(sum);
        }
        // 调整格式
        head = cur.next;
        cur.next = null;
        return head;
    }

    // 无重复字符的最长子串  O(n^2)
    public static int lengthOfLongestSubstring(String s) {
        // 用HashSet思路错误了(也算是暴力解法)
        if (s.isEmpty())
            return 0;
        if (s.length() == 1)
            return 1;
        HashSet<Character> hs = new HashSet<>();
        int max = 1;
        for (int j = 0; j < s.length(); j++) {
            for (int i = j; i < s.length(); i++) {
                if (hs.contains(s.charAt(i))) {
                    max = Math.max(max, hs.size());
                    hs.clear();
                    break;
                } else {
                    hs.add(s.charAt(i));
                    max = Math.max(max, hs.size());
                }
            }
        }
        return Math.max(max, hs.size());
    }

    // 无重复字符的最长子串--优化（滑动窗口思想）  O(n)
    public static int lengthOfLongestSubstring2(String s) {
        if (s.isEmpty())
            return 0;
        if (s.length() == 1)
            return 1;

        Map<Character, Integer> map = new HashMap<>();
        int left = 0, right = 0, max = 1;
        while (right < s.length()) {
            // 如果哈希表中没有right指向的那个元素
            if (!map.containsKey(s.charAt(right))) {
                max = Math.max(max, right - left + 1);
                map.put(s.charAt(right), right);
                right++;
            } else {  // 哈希表中有right所指向的元素   ---- 这里有问题

                int left_should_be = map.get(s.charAt(right)) + 1;
                for (; left < left_should_be; left++) {
                    map.remove(s.charAt(left));
                }
                if (left == right) {
                    map.clear();
                }
            }

        }
        return max;
    }

    // 寻找两个正序数组的中位数 暴力，O(m+n)，引入额外m+n的空间然后排序再输出
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        // 两个为空的情况
        if (nums1.length == 0 && nums2.length == 0)
            return 0;
            // 其中一个不为空的情况
        else {
            int m = Math.max(nums1.length, 0);
            int n = Math.max(nums2.length, 0);
            ArrayList<Integer> arr = new ArrayList<>();
            // 定义左右指针 和 当前运动了几个数字

            for (int i = 0; i < m; i++) {
                arr.add(nums1[i]);
            }
            for (int i = 0; i < n; i++) {
                arr.add(nums2[i]);
            }
            Collections.sort(arr);
            return (m + n) % 2 == 0 ? (double) (arr.get((m + n) / 2) + arr.get((m + n) / 2 - 1)) / 2 : arr.get((m + n) / 2);
        }


    }

    // 最长回文子串 (暴力解法) O(n^3)
    public static String longestPalindrome(String s) {
        // 特殊处理
        if (s.length() < 2) return s;
        // 定义一些参数
        int left = 0, right = 0;
        // 进入暴力循环
        for (int i = 0; i < s.length(); i++) {
            for (int j = i; j < s.length(); j++) {
                // 如果 i 到 j 是回文串, 根据情况更新
                if (is_returnable(s, i, j)) {
                    if ((j - i) > (right - left)) {
                        left = i;
                        right = j;
                    }
                }
            }
        }

        return s.substring(left, right + 1);
    }

    public static boolean is_returnable(String s, int start, int finish) {
        while (start <= finish) {
            if (s.charAt(start) != s.charAt(finish))
                return false;
            start++;
            finish--;
        }
        return true;
    }

    // 最长回文子串 (动态规划优化之后) O(n^2)
    public static String longestPalindrome2(String s) {

        if (s.length() < 2)
            return s;

        int len = s.length();
        int maxStart = 0;  //最长回文串的起点
        int maxEnd = 0;    //最长回文串的终点
        int maxLen = 1;  //最长回文串的长度

        boolean[][] bp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            bp[i][i] = true;
        }

        // 注意这里列 j 要从 1 开始！否则j-1就成-1了
        for (int j = 1; j < len; j++) {
            for (int i = 0; i < j; i++) {
                // 是回文,更新bp数组
                if ((j - i <= 2 || bp[i + 1][j - 1]) && s.charAt(i) == s.charAt(j)) {
                    bp[i][j] = true;
                    if (j - i + 1 > maxLen) {
                        maxLen = j - i + 1;
                        maxStart = i;
                        maxEnd = j;
                    }
                }
                // 否则不是回文串
                else bp[i][j] = false;
            }
        }
        return s.substring(maxStart, maxEnd + 1);
    }

    // 字符串 Z 字形变换    ------ 新知识：StringBuffer 可以用来存储字符或字符串
    public static String convert (String s, int numRows) {
        // 一行的时候 或者字符串s的长度过短 直接输出s
        if (numRows == 1 || s.length() < 2) return s;
        // 正题
        // 构建一个二维的保存字符的地方
        StringBuffer[] sb = new StringBuffer[numRows];
        for (int i = 0; i < sb.length; i++) {
            sb[i] = new StringBuffer();
        }
        // 主循环
        for (int i = 0, bool = 0, t = 0; i < s.length(); i++) {
            sb[t].append(s.charAt(i));
            // 控制方向
            if (t == 0) bool = 0;
            else if (t == numRows - 1) bool = 1;
            t += (bool == 0) ? 1 : -1;
        }
        // 构造答案
        StringBuffer ans = new StringBuffer();
        for (int i = 0; i < numRows; i++) {
            ans.append(sb[i]);
        }
        return ans.toString();
    }

    // 整数反转
    public static int reverse(int x) {
        // 绝对值小于 10 的特殊处理
        if (x < 10 && x > -10)
            return x;

        double ans = 0;
        while (x != 0){
            while(x != 0){
                int temp = x % 10;
                ans = ans * 10 + temp;
                x /= 10;
            }
            // 判断是否溢出
            if (ans > 2147483647 || ans < -2147483648)
                return 0;
        }
        // 用 double 算，最后转为 int
        return (int)ans;
    }

    // 字符串转为整数
    public static int myAtoi(String s) {

        // 去除前导空格
        if (s.isEmpty()) return 0;
        double ans = 0;
        int i = 0, sign = 1;
        while (i<s.length() && s.charAt(i) == ' ')
            i++;

        if (i >= s.length())
            return (int) (ans * sign);

        // 判断正负 (新方法，用一个sign记录正负号，返回 result * sign)
        if (s.charAt(i) == '-')
        {
            sign = -1;
            i++;
        }
        else if (s.charAt(i) == '+') i++;

        if (i >= s.length())
            return (int) (ans * sign);


        // 主循环，判断每个字符
        while (i < s.length()){
            // 遇到字符直接跳出主循环
            char cur = s.charAt(i);
            if (!(cur >= 48 && cur <= 57))
                break;
            // 否则记到数字里面
            ans = ans * 10 + ((int)cur-48);
            i++;
        }
        // 溢出判断
        if (ans > Integer.MAX_VALUE-1 && sign == 1) return Integer.MAX_VALUE - 1;
        else if (ans < Integer.MIN_VALUE && sign == -1) return Integer.MIN_VALUE;

        return (int) (ans * sign);
    }

    // 回文数 优化新思想
    public static boolean isPalindrome(int x) {

        // 特殊处理
        if (x < 0) return false; if (x == 0) return true;
        if (x % 10 == 0 && x != 0) return false;
        // 只反转数字的后一半

        int reversed = 0;
        while (x > reversed){
            reversed = (reversed * 10) + (x % 10);
            x = x / 10;
        }

        return x == reversed || x == reversed / 10;

    }

    // 整个则表达式匹配 -- 看答案要动态规划
    public static  boolean isMatch(String s, String p) {

        // 特殊情况
        if (s.equals(p)) return true;

        // 定义二维 bp 数组
        int m = s.length() + 1, n = p.length() + 1;
        boolean[][] bp = new boolean[m][n];
        // 两个空字符串总相等
        bp[0][0] = true;
        // bp数组初始化 a*b*c*……
        for (int i = 2; i < n; i+=2) {
            bp[0][i] = bp[0][i - 2] && p.charAt(i - 1) == '*';
        }
        // 主程序
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (p.charAt(j-1) == '*')
                {
                    bp[i][j] = bp[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        bp[i][j] = bp[i][j] || bp[i - 1][j];
                    }
                }
                else if (p.charAt(j-1) == '.')
                    bp[i][j] = bp[i-1][j-1];
                else
                    bp[i][j] = bp[i-1][j-1] && p.charAt(j-1)==s.charAt(i-1);

            }
        }

        return bp[s.length()][p.length()];
    }
    public static boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    // 盛最多水的容器 经典双指针,双指针代表的是:可以作为容器边界的所有位置的范围 ---- 自己也可以暴力两个for，但容易超时
    public static int maxArea(int[] height) {
        // 特殊值处理
        if (height.length == 2)
            return Math.min(height[0], height[1]);
        else if (height.length == 0) return 0;
        // 主程序
        int i = 0, j = height.length - 1;
        int max_water = 0;
        while (i < j){
            max_water = Math.max(Math.min(height[i], height[j]) * (j - i), max_water);
            if (height[i] < height[j]) i++;
            else j--;
        }
        return max_water;
    }

    // 整数转罗马数字 -- 暴力解太笨了
    public static String intToRoman(int num) {

        int[] value = {1000, 500, 100, 50, 10, 5, 1};
        String[] simble = {"M", "D", "C", "L", "X", "V", "I"};
        String ans = "";
        int mod = 10;
        // 主逻辑
        while (num > 0){
            // 获取每个十进制数字
            int temp = num % mod;
            num -= temp;

            // 若temp 为 4 或 9 开头的数字
            if (test4(temp)){
                String x = "";
                for (int i = value.length - 1;i >= 0; i--) {
                    if (value[i] > temp){
                        x = simble[i+1] + simble[i];
                        break;
                    }
                }
                ans = x + ans;
            }
            else if (test9(temp)){
                String x = "";
                for (int i = value.length - 1;i >= 0; i--) {
                    if (value[i] > temp){
                        x = simble[i+2] + simble[i];
                        break;
                    }
                }
                ans = x + ans;
            }
            else {
                String x = "";
                while (temp > 0) {
                    for (int i = 0; i < value.length; i++) {
                        if (value[i] <= temp) {
                            x = x + simble[i];
                            temp -= value[i];
                            break;
                        }
                    }
                }
                ans = x + ans;
            }
            mod *= 10;
        }
        return ans.toString();
    }
    public static boolean test4(int num){
        while (num > 10) {
            num /= 10;
        }
        if (num == 4) return true;
        return false;
    }
    public static boolean test9(int num){
        while (num > 10) {
            num /= 10;
        }
        if (num == 9) return true;
        return false;
    }

    // 整数转罗马数字 -- 优化算法
    public static String intToRoman2(int num) {
        // 这里使用答案的思想
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        // 首次自己用stringBuffer
        StringBuffer str = new StringBuffer();
        // 新思想哈哈哈，果然开动脑子了
        for (int i = 0; i < values.length; i++) {
            if (num >= values[i]){
                str.append(symbols[i]);
                num -= values[i];
                i--;
            }
            if (num == 0)
                break;
        }
        return str.toString();
    }

    // 罗马数字转整数
    public static int romanToInt(String s) {

        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] sign = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

        int ans = 0;
        for (int i = 0; i < sign.length; i++) {
            if (s.startsWith(sign[i])){
                ans += values[i];
                s = s.substring(sign[i].length());
                i--;
            }
        }
        return ans;
    }

    // 查找最长公共前缀 -- 写了个子函数暴力解       ！！撒花！！和答案思路一模一样！
    public static String longestCommonPrefix(String[] strs) {
        // 特殊情况处理
        if (strs.length == 1) return strs[0];
        String s = strs[0];
        for (int i = 1; i < strs.length; i++) {
            s = find_same(s, strs[i]);
        }
        return s;
    }
    public static String find_same(String str1, String str2){
        if (str1.isEmpty() || str2.isEmpty()) return "";
        for (int i = 0; i < Math.min(str1.length(), str2.length()); i++) {
            if (str1.charAt(i) != str2.charAt(i)){
                return str1.length()<str2.length()? str1.substring(0,i):str2.substring(0,i);
            }
            if (i == Math.min(str1.length()-1, str2.length()-1))
                return str1.length()<str2.length()? str1:str2;
        }
        return "";
    }

    // 表示对a从i往后排列，0~i-1 看作已经弄好的
    static ArrayList<Integer> result = new ArrayList<>();
    static int ans;
    public static int quanPaiLie(int[] a, int start, ArrayList<Integer> res){
        // 一种全排列查找成功
        if (start == a.length){
            for (int i = 0; i < a.length; i++) {
                result.add(a[i]);
            }
            ans++;
            System.out.println(result);
            result.clear();
        }

        for (int i = start; i < a.length; i++) {
            swap(a, start, i);
            quanPaiLie(a, start+1, result);
            swap(a, start, i);
        }
        return ans;
    }
    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // 汉诺塔问题真的太吊了，我的脑子真的无法承受~要仔细思考这个！
    static void Hanoi_Tower (int from ,int to, int other, int index){
        // 递归出口
        if (index == 1)
        {
            move(from, index, to);return;
        }
        // 程序核心
        Hanoi_Tower(from, other, to, index-1);
        move(from, index, to);
        Hanoi_Tower(other, to, from, index-1);

    }
    static void move(int from, int index, int to){
        System.out.println("把" + index + "从" + from +"移动到" + to);
    }


    // 01背包的暴力递归解答，其实就是暴力穷举出每一个物品拿还是不拿 ---- 二叉树思想
    static int[] weight = {8, 3, 4, 3};
    static int[] values = {9, 3, 4, 3};
    public static int bags_problem(int index, int max){

        // 递归出口
        if (index == weight.length)  return 0;
        // 主思想

        // 放不下就跳过
        if (weight[index] > max) return bags_problem(index+1, max);
            // 能放下就看放和不放哪个利润大
        else {
            int get_profit = values[index] + bags_problem(index + 1, max-weight[index]);
            int not_get_profit = bags_problem(index+1, max);
            return Math.max(get_profit, not_get_profit);
        }

    }
    static int knapSack(int W, int[] wt, int[] val, int n) {
        int[][] K = new int[n + 1][W + 1];

        // Build table K[][] in bottom up manner
        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                if (i == 0 || w == 0)
                    K[i][w] = 0;
                else if (wt[i - 1] <= w)
                    K[i][w] = Math.max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
                else
                    K[i][w] = K[i - 1][w];
            }
        }

        return K[n][W];
    }

    // 爬楼梯递归解决--思路 so easy 但是会超时
    public static int Stairs_num(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        return Stairs_num(n - 1) + Stairs_num(n - 2);
    }
    // 爬楼梯动态规划
    public static int Stairs_num2(int n) {
        // 特殊情况
        if (n == 0|| n == 1) return 1;
        // dp[n] 代表走到第 n 层有几种方法
        int[] dp = new int[n + 1];
        dp[0] = 1;dp[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }

    // 三数之和 -- 自己的暴力解虽然答案对，但是超时
    public static List<List<Integer>> threeSum(int[] nums) {
        // 整体思路是 a + b + c == 0  ==>  a + b = -c
        // -2, -1, -1, 0, 0, 0, 1, 1
        // 先快排排序
        Arrays.sort(nums);
        // 外层括号--整个容器
        List<List<Integer>> rts = new ArrayList<>();
        for (int i = 0; i < nums.length-2; i++) {
            for (int left = i + 1, right = nums.length-1; left < right ; ) {
                if (nums[i] + nums[left] + nums[right] == 0){
                    List<Integer> temp = new ArrayList<>();
                    temp.add(nums[i]);temp.add(nums[left]);temp.add(nums[right]);
                    if (!rts.contains(temp))
                        rts.add(temp);
                    left++;
                }
                else if (nums[i] + nums[left] + nums[right] < 0)
                    left ++;
                else
                    right --;
            }
        }
        return rts;
    }
    // 三数之和 -- 答案好巧妙！核心： i 固定时，零两个数一个要增大，另一个必需要减小，否则肯定和不为0
    // LeetCode: 题目涉及到一个数字增大时候会导致另一个数字的减小的时候可以用双指针思想
    // 相当于第二个指针和第三个指针移动的时候有一定的规律，可以刚好利用这个规律
    public static List<List<Integer>> threeSum2(int[] nums){

        // 排序后处理特殊情况，如果第一个数字大于0，不可能出现三个相加等于0
        Arrays.sort(nums);
        if (nums[0] > 0) return new ArrayList<>();

        // 返回这个ans
        // -2, -1, -1, 0, 0, 0, 1, 1
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            // 需要和上一次枚举的数不相同
            if (i > 0 && nums[i] == nums[i-1]) continue;

            int left = i + 1, right = nums.length-1;
            while (left < right){

                if (nums[i] + nums[left] + nums[right] == 0){
                    List<Integer> temp = new ArrayList<>();
                    temp.add(nums[i]);temp.add(nums[left]);temp.add(nums[right]);
                    ans.add(temp);
                    // left 加到下一个不同的数字, right 减到前一个不同的数字;
                    do {
                        left++;
                    }while (nums[left]==nums[left-1] && left < nums.length-1);
                    do {
                        right--;
                    }while (nums[right]==nums[right+1] && right > 0);
                }

                else if (nums[i] + nums[left] + nums[right] < 0){
                    do {
                        left++;
                    }while (nums[left]==nums[left-1] && left < nums.length-1);
                }

                else
                    do {
                        right--;
                    }while (nums[right]==nums[right+1]);
            }
        }

        return ans;
    }

    // 三数之和 -- 双指针
    public static int threeSumClosest(int[] nums, int target){

        // 特殊处理
        Arrays.sort(nums);
        if (nums.length == 3) return nums[0] + nums[1] + nums[2];

        // 主逻辑
        // 枚举第一个数
        int ans = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 2;) {
            // 枚举第二个指针
            int j = i + 1;
            int k = nums.length - 1;

            while (j < k) {
                // 更新ans
                if (Math.abs(target - (nums[i] + nums[j] + nums[k])) < Math.abs(target - ans))
                    ans = nums[i] + nums[j] + nums[k];
                // 双指针移动
                if (nums[i] + nums[j] + nums[k] > target)
                    k--;
                else if (nums[i] + nums[j] + nums[k] < target) {
                    j++;
                }
                else if (nums[i] + nums[j] + nums[k] == target)return nums[i] + nums[j] + nums[k];
            }
            i ++;
        }
        return ans;
    }

    // 电话号码的字母组合 -- 循环重数不定的时候用递归(回溯) 巧妙代替
    public static List<String> letterCombinations(String digits) {
        // 特殊处理
        if (digits.isEmpty()) return new ArrayList<>();

        // 最终返回结果result, 每个路径走完得到一个结果, 记为temp
        List<String> result = new ArrayList<>();
        String temp = "";

        // 制作电话号映射
        String[] phone = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

        // 回溯法可以通过递归的方式来帮我们去实现嵌套几个for循环
        // 其实本质就是一个递归算法
        tracingBack(phone, temp, 0, digits, result);

        return result;
    }
    public static void tracingBack (String[] phone, String temp, int layer, String digits, List<String> result){
        // 表示达到最后一个字符，可以将这个路径加到result里
        if (layer == digits.length()){
            result.add(temp);
        }
        // 走到中间某个数字
        else {
            int number = digits.charAt(layer) - 48;
            // 每个数字对应的字母需要遍历
            for (int i = 0; i < phone[number].length(); i++) {
                // 加入当前字符，进入下一层循环，恢复成原始状态
                tracingBack(phone, temp + phone[number].charAt(i), layer + 1, digits, result);

                /* 答案是这样：
                 *  temp += phone[number].charAt(i)
                 *  tracingBack……
                 *  temp -= phone[number].charAi(i)
                 * */
            }
        }
    }

    // 四数之和 -- 双指针, 有点思路了
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        // 特殊情况处理
        if (nums.length < 4) return new ArrayList<>();

        // 返回答案ans
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i < nums.length - 3; i++) {
            if (nums[i] + nums[i + 1] + nums[ i + 2] + nums[i + 3] > target) break;
            for (int j = i + 1; j < nums.length - 2; j++) {
                int left = j + 1, right = nums.length - 1;
                while (left < right){
                    long sum = (long)nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target){
                        List<Integer> temp = new ArrayList<>();
                        temp.add(nums[i]);temp.add(nums[j]); temp.add(nums[left]);temp.add(nums[right]);
                        if (!ans.contains(temp))
                            ans.add(temp);
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                    }
                    else if (sum < target) while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    else while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                }
            }
        }

        return ans;
    }

    // 删除链表的倒数第N个结点
    public  static ListNode removeNthFromEnd(ListNode head, int n) {

        if (head.next == null) return null;

        ListNode p = head;
        int count = 1;
        while (count < n) {
            p = p.next;
            count++;
        }
        ListNode left = head;
        while (p.next != null){
            p = p.next;
            left = left.next;
        }
        ListNode x = head;
        while (x.next != left && x != left) x = x.next;
        if (head == left) head = head.next;
        else x.next = x.next.next;
        return head;
    }

    // 有效的括号 -- 经典用栈匹配
    public static boolean isValidkuohao(String s) {

        Stack<Character> st = new Stack<>();
        // 特殊情况
        if (s.length() % 2 == 1) return false;

        // 主程序
        for (int i = 0; i < s.length(); i ++) {
            // 入栈
            if (s.charAt(i) == '(' || s.charAt(i) == '['|| s.charAt(i) == '{')
                st.push(s.charAt(i));
            else if (!st.isEmpty()){
                if (s.charAt(i) == ')' && st.peek() == '(') {
                    st.pop();
                }
                else if (s.charAt(i) == ']' && st.peek() == '[') {
                    st.pop();
                }
                else if (s.charAt(i) == '}' && st.peek() == '{') {
                    st.pop();
                }
                else return false;
            }
            else return false;
        }
        return st.isEmpty();
    }

    // 合并两个有序链表  -- 就最简单的归并思路
    public static ListNode mergeTwoLists(ListNode a, ListNode b) {
        if (a == null || b == null) {
            return a != null ? a : b;
        }
        ListNode head = new ListNode(0);
        ListNode tail = head, aPtr = a, bPtr = b;
        while (aPtr != null && bPtr != null) {
            if (aPtr.val < bPtr.val) {
                tail.next = aPtr;
                aPtr = aPtr.next;
            } else {
                tail.next = bPtr;
                bPtr = bPtr.next;
            }
            tail = tail.next;
        }
        tail.next = (aPtr != null ? aPtr : bPtr);
        return head.next;
    }

    // 括号生成 -- 暴力递归(二叉树暴力思想),用判断合法性函数,将合法的结果加入最终的解 -- 二叉树暴力从最后一层找答案
    public static List<String> generateParenthesis(int n) {

        // 新方法：调用自定义函数
        List<String> ans = new ArrayList<>();
        String cur = "";
        answer(ans, cur, 2 * n, 0);
        return ans;
    }
    public static void answer(List<String> ans,String current, int n, int layer){
        // 走到了最后一层，需要判断当前序列是否合法，若合法则加入ans集合
        if (layer == n){
            if (finished_codes.isValidkuohao(current))
                ans.add(current);
        }
        // 若是中间某层
        else {
            answer(ans, current + "(", n, layer + 1);
            answer(ans, current + ")", n, layer + 1);
        }
    }

    // 括号生成 -- 递归(回溯思想),构造整个字符串的过程中保证每一步是合法的
    public static List<String> generateParenthesis2(int n) {

        // 新方法：调用自定义函数
        List<String> ans = new ArrayList<>();
        String cur = "";
        // 起初左右括号都有 n 个
        answer2(ans, cur, n, n);
        return ans;
    }
    public static void answer2(List<String> ans, String current, int leftNumber, int rightNumber){
        // 走到了最后一层，需要判断当前序列是否合法，若合法则加入ans集合
        if (leftNumber == 0 && rightNumber == 0){
            ans.add(current);
            return;
        }
        // 若是中间某层
        if (leftNumber == rightNumber){
            // 如果剩余的左右括号相等，下一个只能放左括号
            answer2(ans, current + "(", leftNumber - 1, rightNumber);
        }
        else if (leftNumber < rightNumber){
            //剩余左括号小于右括号，下一个可以用左括号也可以用右括号
            if (leftNumber > 0){
                answer2(ans, current + "(", leftNumber - 1, rightNumber);
            }
            answer2(ans, current + ")", leftNumber, rightNumber - 1);
        }
    }

    // 合并 K 个升序链表 -- 暴力解法是利用之前那个 merge函数 暴力从第一个两两 merge 到最后一个，运行时间长
    public static ListNode mergeKLists(ListNode[] lists) {
        // 特殊处理
        if (lists.length == 0) return null;
        if (lists.length == 1) return lists[0];

        for (int i = 0; i < lists.length - 1; i++) {
            lists[i + 1] = mergeTwoLists(lists[i], lists[i + 1]);
        }

        return lists[lists.length - 1];
    }

    // 分治思想 两两合并直到最后一个链表   -- 递归思想
    public static ListNode merge(ListNode[] lists, int left, int right){
        if (left == right) {
            return lists[left];
        }
        if (left > right) {
            return null;
        }
        // 等效于 (l + r)/2
        int mid = (left + right) >> 1;
        // 递归思想
        return mergeTwoLists(merge(lists, left, mid), merge(lists, mid + 1, right));
    }
}

