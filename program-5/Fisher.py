# -*- coding: utf-8 -*-

# フィッシャーの線形判別（顔画像）

import sys
import os
import numpy as np
from PIL import Image

# クラス数
class_num = 2

# 画像の大きさ
size = 8

# 学習データ
train_num = 80

# テストデータ
test_num = 100-train_num

# 学習データ，平均ベクトル（クラス），平均ベクトル（全データ）
train_vec = np.zeros((class_num,train_num,size*size), dtype=np.float64)
ave_vec = np.zeros((class_num,size*size), dtype=np.float64)
all_ave_vec = np.zeros((size*size), dtype=np.float64)

# クラス内分散共分散行列，クラス間分散共分散行列
Sw = np.zeros((size*size,size*size), dtype=np.float64)
Sb = np.zeros((size*size,size*size), dtype=np.float64)

# 固有値，固有ベクトル
lamda = np.zeros(size*size, dtype=np.float64)
eig_vec = np.zeros((size*size,size*size), dtype=np.float64)

# fig以下の画像を削除（MS-Windows）
os.system("del /Q fig\*")

# 学習データの読み込み
dir = [ "Male" , "Female" ]
for i in range(class_num):
    for j in range(1,train_num+1):
        train_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(train_file).convert('L')
        resize_img = work_img.resize((size, size))
        train_vec[i][j-1] = np.asarray(resize_img).astype(np.float64).flatten()
        
# 平均ベクトル（各クラス）
for i in range(class_num):
    ave_vec[i] = np.mean( train_vec[i] , axis=0 )

    # 平均顔の保存（各クラス）
    ave_img = Image.fromarray(np.uint8(np.reshape(ave_vec[i],(size,size))))
    ave_file = "fig/" + str(dir[i]) + "-ave.png"
    ave_img.save(ave_file)

# 平均ベクトル（全データ）
all_ave_vec = np.mean( ave_vec , axis=0 )

# 平均顔の保存（全データ）
ave_img = Image.fromarray(np.uint8(np.reshape(all_ave_vec,(size,size))))
ave_file = "fig/all-ave.png"
ave_img.save(ave_file)

# クラス内分散共分散
for i in range(class_num):   
    Sw += np.cov( train_vec[i] , rowvar=0 , bias=1 )

# クラス間分散共分散
for i in range(class_num):
    a = np.reshape( ave_vec[i] - all_ave_vec , (size*size,1) )
    Sb += np.dot( a , a.T )

# クラス内分散共分散の逆行列
Sw_1 = np.linalg.inv( Sw )

V = np.dot( Sw_1 , Sb )
print( " Rank -> " , np.linalg.matrix_rank(V) )

# 固有値分解
lamda , eig_vec = np.linalg.eig( V )

# 変換行列
D = 1
A = np.reshape( eig_vec[:,0:D].real , (size*size,D) )

'''
# 固有値展開を行わない場合
a = np.reshape( ave_vec[0] - ave_vec[1] , (size*size,1) )
A = np.dot( Sw_1 , a )
'''

# 変換後の各クラスの平均値
m = np.zeros((class_num,D), dtype=np.float64)
for i in range(class_num):
    a = np.reshape( ave_vec[i] , (size*size,1) )
    m[i] = np.dot( A.T , a ).flatten()
print( m )

# 混合行列
result = np.zeros((class_num,class_num), dtype=np.int32)
for i in range(class_num):
    for j in range(train_num,101):
        # テストデータの読み込み
        pat_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(pat_file).convert('L')
        resize_img = work_img.resize((size, size))
        pat_vec = np.reshape( np.asarray(resize_img).astype(np.float64) , (size*size,1) )

        # 変換行列によって変換
        y = np.dot( A.T , pat_vec ).flatten()

        # 変換後の平均値による最近傍法
        min_val = float('inf')
        ans = 0
        for k in range(class_num):
            dist = np.dot( (y-m[k]).T , y-m[k] )
                     
            if dist < min_val:
                min_val = dist
                ans = k

        result[i][ans] +=1
        print( i , j , "->" , ans )

# 混合行列の出力
print( "\n [混合行列]" )
print( result )
print( "\n 正解数 ->" ,  np.trace(result) )
            
            
            
