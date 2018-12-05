# -*- coding: utf-8 -*-

# マハラノビス距離（ユークリッド距離）

import sys
import os
import numpy as np
import random

# クラス数，学習データ数，テストデータ数，特徴数
class_num = 3
train_num = 100
test_num = 100
size = 4

# 学習データ，テストデータ，平均ベクトル，分散共分散行列，行列式
train_vec = np.zeros((class_num,train_num,size), dtype=np.float64)
test_vec = np.zeros((class_num,test_num,size), dtype=np.float64)
ave_vec = np.zeros((class_num,size), dtype=np.float64)
var = np.zeros((class_num,size,size), dtype=np.float64)
det = np.zeros(class_num, dtype=np.float64)

# 平均値,標準偏差の設定
d = np.zeros((class_num,size), dtype=np.int32)
s = np.zeros((class_num,size), dtype=np.float64)
for i in range(class_num):
    for j in range(size):
        d[i][j] = random.randint(1,5)
        s[i][j] = random.uniform(0,2)

# 学習データの生成
for i in range(class_num):
    for j in range( train_num ):
        for k in range( size ):
            train_vec[i][j][k] = np.random.normal( d[i][k] , s[i][k] )

# テストデータの生成
for i in range(class_num):
    for j in range( test_num ):
        for k in range( size ):
            test_vec[i][j][k] = np.random.normal( d[i][k] , s[i][k] )

# 平均ベクトル
for i in range( class_num ):
    ave_vec[i] = np.mean( train_vec[i] , axis=0 )
print( "\n [ 平均ベクトル ]" )
print( ave_vec )

for i in range(class_num):
    # 分散共分散行列
    v = np.cov( train_vec[i] , rowvar=0 , bias=1 )

    # 行列式
    det[i] = np.linalg.det(v)
    if det[i] != 0.0:
        # 行列式が0でない場合→分散共分散行列の逆行列を計算
        var[i] = np.linalg.inv(v)
        print( "\n [ 検算 ]" )
        print( np.dot( var[i] , v ) )
    else:
        # 行列式が0の場合→分散共分散行列を単位行列とする
        I = np.identity(size)
        var[i]=I

# 混合行列
result_e = np.zeros((class_num,class_num), dtype=np.int32)
result_m = np.zeros((class_num,class_num), dtype=np.int32)
for i in range(class_num):
    for j in range(0,test_num):
        # ユークリッド距離（SSD）
        min_val = float('inf')
        ans = 0
        for k in range(class_num):
            aT = np.reshape( (test_vec[i][j]-ave_vec[k]) , (1,size) )
            a = np.reshape( (test_vec[i][j]-ave_vec[k]) , (size,1) )
            dist = np.dot( aT , a )

            if dist < min_val:
                min_val = dist
                ans = k
        result_e[i][ans] +=1

        # マハラノビス距離
        min_val = float('inf')
        ans = 0
        for k in range(class_num):
            aT = np.reshape( (test_vec[i][j]-ave_vec[k]) , (1,size) )
            a = np.reshape( (test_vec[i][j]-ave_vec[k]) , (size,1) )
            dist = np.dot( np.dot( aT , var[k] ) , a )

            if dist < min_val:
                min_val = dist
                ans = k
        result_m[i][ans] +=1

print( "\n [混合行列（ユークリッド距離）]" )
print( result_e )
print( "\n 正解数 ->" ,  np.trace(result_e) )

print( "\n [混合行列（マハラノビス距離）]" )
print( result_m )
print( "\n 正解数 ->" ,  np.trace(result_m) )

    
    
    
