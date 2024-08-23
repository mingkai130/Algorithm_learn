package org.example;


import com.sun.source.tree.Tree;

import java.math.BigInteger;
import java.util.*;

// 基础课程
public class class1 {
    public static void main(String[] args) {

        int a = 8;
        System.out.println(a>>1);

    }


    // 两数异或运算操作换位
    public void swap(int a, int b){
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    }
    public void input(){
        int a;
        Scanner sc = new Scanner(System.in);
        a = sc.nextInt();
    }


}