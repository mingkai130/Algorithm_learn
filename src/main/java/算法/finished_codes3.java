package 算法;

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
}
