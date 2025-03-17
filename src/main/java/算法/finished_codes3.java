package 算法;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;
import java.util.Stack;

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





}
