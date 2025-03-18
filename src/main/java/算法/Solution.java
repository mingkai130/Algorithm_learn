package 算法;

import javax.imageio.event.IIOReadProgressListener;
import javax.sound.sampled.ReverbType;
import javax.swing.*;
import javax.swing.plaf.basic.BasicTreeUI;
import javax.swing.text.EditorKit;
import javax.swing.text.Element;
import javax.swing.text.StyledEditorKit;
import javax.xml.stream.FactoryConfigurationError;
import java.awt.*;
import java.awt.event.KeyListener;
import java.awt.font.NumericShaper;
import java.io.FileReader;
import java.lang.management.BufferPoolMXBean;
import java.lang.reflect.AnnotatedArrayType;
import java.lang.reflect.Array;
import java.math.BigInteger;
import java.text.StringCharacterIterator;
import java.time.Instant;
import java.time.chrono.IsoChronology;
import java.time.temporal.Temporal;
import java.util.*;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.BiConsumer;
import java.util.logging.Level;

import 算法.数据结构.*;
import 算法.*;


class Solution {

    public static void main(String[] args) {
        int[][] ints = new int[][]{{1,3,1},{1,5,1},{4,2,1}};
        int[] ints2 = new int[]{1,2,3};
        int[][] booleanInts = new int[][]{{0,1,1, 0},{3,4,5,6},{1,3,1,5}};
//        String[] strs = new String[]{"eat", "tea", "tan", "ate", "nat", "bat"};

//        ListNode head = new ListNode(1);
//        head.add(2);head.add(3);head.add(4);head.add(5);


        System.out.println(subsets(ints2));
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


