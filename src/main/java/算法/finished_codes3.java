package 算法;

import java.util.*;

public class finished_codes3 {

    // 66. 加一 , 需要考虑进位情况 秒了
    public static int[] plusOne(int[] digits) {
        int[] ans = new int[digits.length + 1];

        int temp = 0;
        digits[digits.length - 1] += 1;
        for (int i = digits.length - 1; i >= 0 ; i--) {
            digits[i] += temp;

            if (digits[i] >= 10) {
                temp = 1;digits[i] %= 10;
            }else temp = 0;
        }
        if (temp != 0){
            ans[0] = 1;
            System.arraycopy(digits, 0, ans, 1, digits.length);
        }
        else return digits;

        return ans;
    }

    // 67. 二进制求和   自己暴力模拟太慢了
    public static String addBinary(String a, String b) {

        StringBuffer ans = new StringBuffer();
        int length= Math.max(a.length(), b.length()) - Math.min(a.length(), b.length());
        // 位数对齐
        if (a.length() >= b.length()) {
            while (length > 0){
                b = '0' + b;
                length --;
            }
        }
        else {
            while (length > 0){
                a = '0' + a;
                length --;
            }
        }
        // 模拟二进制加法
        int temp = 0;
        for (int i = Math.max(a.length(), b.length()) - 1; i >=0; i--) {
            // 字符转数字！
            int sum = (( a.charAt(i) - '0') + ( b.charAt(i) - '0') + temp);

            temp = sum / 2;
            ans.insert(0, sum % 2);
        }
        if (temp == 1) ans.insert(0, 1);
        return ans.toString();
    }
    // 答案好巧啊！
    public String addBinaryAns(String a, String b) {
        StringBuffer ans = new StringBuffer();
        int tmp = 0;
        for(int i = a.length()-1,j = b.length()-1;i>=0||j>=0;i--,j--){
            int sum = tmp;
            sum += i >= 0 ? a.charAt(i)-'0':0;
            sum += j >= 0 ? b.charAt(j)-'0':0;
            ans.append(sum%2);
            tmp = sum/2;
        }
        ans.append(tmp == 1 ?tmp :"");
        return ans.reverse().toString();
    }


    // 68. 文本左右对齐 -- 困难题, 不太会


    // 69. x 的平方根 , 自己暴力太sb了，看看人家答案多优雅，直接二分查找
    public static int mySqrt(int x) {
        if (x < 2)return x;
        int left = 0, right = x;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(x / mid == mid) {
                return mid;
            }
            else if(x / mid < mid) {
                right = mid - 1;
            }
            else {
                left = mid + 1;
            }
        }
        return right;
    }

    // 70. 爬楼梯  dp秒了
    public static int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;

        int[] dp = new int[n + 1];
        dp[1] = 1; dp[2] = 2;
        for (int i = 3; i <= n ; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    // 71. 简化路径 -- Linux 目录路径，暴力模拟失败
    public static String simplifyPath(String path) {
        Stack<String> ans = new Stack<>();
        // 暂存每个路径
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < path.length(); i++) {
            // 情况1
            char current = path.charAt(i);
            if (current == '/') {
                if(!sb.isEmpty() && !sb.toString().equals("..")) {
                    ans.push(sb.toString());
                    sb.delete(0, sb.length() );
                } else if (sb.toString().equals("..")) {
                    if (!ans.isEmpty()){
                        ans.pop();ans.pop();
                        sb.delete(0, sb.length());
                    }
                }
                if ( !ans.isEmpty()&& Objects.equals(ans.peek(), "/")) ans.pop();
                ans.push(String.valueOf(current));
            }
            // 情况2,字母
            else if (current >= 'A' && current <= 'z'){
                sb.append(current);
            }
            // 情况3, '.'-关键
            else {
                sb.append(current);
            }

            // 末尾处理
            if (i == path.length() - 1 && !sb.isEmpty()) ans.push(sb.toString());
        }


        return ans.toString();
    }
    // 看答案，先根据 '/' 分割字符串,后面微调
    public static String simplifyPath2(String path){
        Deque<String> stack = new ArrayDeque<String>();
        String[] words=path.split("/");
        for (String word : words) {
            if(word.isEmpty()||word.equals(".")){
                continue;
            }
            if (word.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
                continue;
            }
            stack.push(word);
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append("/").append(stack.pollLast());
        }
        return sb.isEmpty() ? "/" : sb.toString();
    }

    // 72. 编辑距离 不会，答案用动态规划  注意初始化！
    public static int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        if (word1.isEmpty() || word2.isEmpty()) return word1.length() + word2.length();
        // 初始化
        dp[0][0] = 0;
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= word2.length(); i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length() ; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1];
                else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1],Math.min(dp[i - 1][j],dp[i][j - 1])) + 1;
                }
            }
        }


        return dp[word1.length()][word2.length()];
    }

    // 73. 矩阵置零  自己暴力模拟，虽然能过但是打败 1% 同行，复杂度有点高
    public static void setZeroes(int[][] matrix) {
        HashSet<List<Integer>> hs = new HashSet<>();

        int m = matrix.length;
        int n = matrix[0].length;
        // 矩阵中 0 的位置入 Set 集合
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    List<Integer> temp = new ArrayList<>();temp.add(i);temp.add(j);
                    hs.add(temp);
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                List<Integer> temp = new ArrayList<>();temp.add(i);temp.add(j);
                // 如果这个位置是 0, 处理横竖
                if (hs.contains(temp)){
                    // 处理行
                    for (int k = 0; k < n; k++) {
                        matrix[i][k] = 0;
                    }
                    // 处理列
                    for (int k = 0; k < m; k++) {
                        matrix[k][j] = 0;
                    }
                }
            }

        }
    }

    // 74. 搜索二维矩阵  暴力模拟，二分查找秒了
    public static boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int line = -1;
        // 先找行
        for (int i = 0; i < m; i++) {
            if (target >= matrix[i][0] && target <= matrix[i][n - 1])
                line = i;
        }
        if (line == -1)
            return false;
        // 行内二分查找
        int left = 0, right = n - 1;
        while (left <= right){
            int mid = (left + right) / 2;
            if (matrix[line][mid] == target) return true;
            else if (target > matrix[line][mid]){
                left = mid + 1;
            }
            else right = mid - 1;
        }

        return false;
    }

    // 75. 颜色分类   荷兰三色旗问题  乱序数组最终为 0,0,0,0,1,1,1,1,2,2,2  这个题太细节了
    // 如果遇到0，不能简单地换位置，当 P0 < P1,如果简单交换，可能会把第一个1交换出去，导致答案错误
    public static void sortColors(int[] nums) {
        int n = nums.length;
        int p0 = 0, p1 = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                int temp = nums[i];
                nums[i] = nums[p1];
                nums[p1] = temp;
                ++p1;
            } else if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                if (p0 < p1) {
                    temp = nums[i];
                    nums[i] = nums[p1];
                    nums[p1] = temp;
                }
                ++p0;
                ++p1;
            }
        }
    }


    // 77.组合  一眼回溯，这里忘的差不多了   磕磕绊绊弄好了，但是还是不知道为啥通过
    public static List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backTrace(ans, temp, n, k, 1);

        return ans;
    }
    public static void backTrace(List<List<Integer>> ans, List<Integer> current, int n, int k, int cur) {
        // 答案加入集合
        if (current.size() == k) ans.add(new ArrayList<>(current));

            // 中间层
        else {
            for (int i = (current.isEmpty()?0:current.getLast()) + 1; i <= n; i++) {
                current.add(i);
                backTrace(ans, current, n, k, cur + 1);
                current.removeLast();
            }


        }
    }


    // 78. 子集   给一个 int[]，返回所有可能的子集   这里答案用 dfs！太牛逼了，index 表示下标位置   哎好巧明，自己想不到
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();

        dfs(ans, temp, nums, 0);

        return ans;
    }
    public static void dfs(List<List<Integer>> ans, List<Integer> temp, int[] nums, int index){
        // index 走到最后一位
        if (index == nums.length) ans.add(new ArrayList<>(temp));
            // 中间位置只分为选择和不选择当前数字
        else {
            // 选择
            temp.add(nums[index]);
            dfs(ans, temp, nums, index + 1);
            temp.removeLast();
            // 不选择
            dfs(ans, temp, nums, index + 1);
        }
    }

}
