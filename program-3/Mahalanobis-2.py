# -*- coding: utf-8 -*-

# マハラノビス距離（MNIST）

# > pip install matplotlib

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 縦，横の大きさ
size = 14

# 学習データ数（テストデータ数）
train_num = 100

# 学習データ，平均ベクトル，固有値，固有ベクトル
train_vec = np.zeros((10,train_num,size*size), dtype=np.float64)
ave_vec = np.zeros((10,size*size), dtype=np.float64)
lamda = np.zeros((10,size*size), dtype=np.float64)
eig_vec = np.zeros((10,size*size,size*size), dtype=np.float64)

# fig以下の画像を削除（MS-Windows）
os.system("del /Q fig\*")

# 学習データの読み込み
for i in range(10):
    for j in range(1,train_num+1):
        train_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
        work_img = Image.open(train_file).convert('L')

        # 画像の大きさを変更
        resize_img = work_img.resize((size, size))
        train_vec[i][j-1] = np.asarray(resize_img).astype(np.float64).flatten()
      
for i in range(10):
    # 平均ベクトル
    ave_vec[i] = np.mean( train_vec[i] , axis=0 )

    # 平均画像の保存
    ave_img = Image.fromarray(np.uint8(np.reshape(ave_vec[i],(size,size))))
    ave_file = "fig/" + str(i) + "-ave.png"
    ave_img.save(ave_file)

for i in range(10):
    # 分散共分散
    cov = np.cov( train_vec[i] , rowvar=0 , bias=1 )
    
    # 固有値分解
    lamda[i] , eig_vec[i] = np.linalg.eig( cov )

    # 固有ベクトルの表示
    for j in range(5):
        a = np.reshape( eig_vec[i][:,j].real , (size,size) )
        plt.imshow(a , interpolation='nearest')
        plt.colorbar()
        file = "fig/eigen-" + str(i) + "-" + str(j) + ".png"
        plt.savefig(file)
        plt.clf()

    # 検算
    print( np.dot( eig_vec[i][:,0] , eig_vec[i][:,1]) )
    print( np.dot( eig_vec[i][:,1] , eig_vec[i][:,1] ) )

# 混合行列
result = np.zeros((10,10), dtype=np.int32)

# 利用する固有ベクトルの個数
D = 60
for i in range(10):
    for j in range(1,train_num+1):
        # テストデータの読み込み
        pat_file = "mnist/test/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
        work_img = Image.open(pat_file).convert('L')
        resize_img = work_img.resize((size, size))
        pat_vec = np.asarray(resize_img).astype(np.float64).flatten()

        min_val = float('inf')
        ans = 0
        for k in range(10):
            dist = 0
            # マハラノビス距離（近似）
            for l in range(0,D):
                e = eig_vec[k][:,l].real
                a = pat_vec-ave_vec[k]
                dist += ( np.dot( e , a ) ) ** 2 / lamda[k][l]
                
            if dist < min_val:
                min_val = dist
                ans = k

        result[i][ans] +=1
        print( i , j , "->" , ans )

# 混合行列の出力
print( "\n [混合行列]" )
print( result )
print( "\n 正解数 ->" ,  np.trace(result) )
            
            
            
