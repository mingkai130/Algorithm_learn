package 算法;

import javax.sound.sampled.ReverbType;
import javax.swing.*;
import javax.swing.plaf.basic.BasicTreeUI;
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
import java.util.*;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import 算法.*;


class Solution {

    public static void main(String[] args) {
        int[][] ints = new int[][]{{1, 3},{6, 9}};
        int[] ints2 = new int[]{2,5};
//        String[] strs = new String[]{"eat", "tea", "tan", "ate", "nat", "bat"};

        System.out.println(Arrays.deepToString(insert(ints, ints2)));
    }

    // 57.插入区间
    public static int[][] insert(int[][] intervals, int[] newInterval) {

        List<int[]> ans = new ArrayList<>();




        return ans.toArray(new int[ans.size()][]);
    }
    // -1：不覆盖    1：左覆盖  0：全覆盖  2：右覆盖
    public static int isIntervaled(int[][] contents, int[][] one, int index){

        if (contents[index][1] <= one[0])
    }

}
