package 算法;

import javax.sound.sampled.ReverbType;
import javax.swing.plaf.basic.BasicTreeUI;
import javax.swing.text.StyledEditorKit;
import java.awt.event.KeyListener;
import java.awt.font.NumericShaper;
import java.lang.management.BufferPoolMXBean;
import java.lang.reflect.Array;
import java.util.*;
import java.util.List;

import 算法.*;

class Solution {

    public static void main(String[] args) {

    }

    // 38.外观数列
    public String countAndSay(int n) {
        if (n == 1) return "1";
        else {
            return countAndSay(n - 1);
        }
    }

}