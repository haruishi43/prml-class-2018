# -*- coding: utf-8 -*-

# 固有顔

# > pip install matplotlib

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 2

# 画像の大きさ
size = 32

# 学習データ
train_num = 90

# テストデータ
test_num = 100-train_num

# 学習データ，平均ベクトル
train_vec = np.zeros((class_num,train_num,size*size), dtype=np.float64)
ave_vec = np.zeros((class_num,size*size), dtype=np.float64)

# 分散共分散行列，固有値，固有ベクトル
Sw = np.zeros((class_num,size*size,size*size), dtype=np.float64)
lamda = np.zeros((class_num,size*size), dtype=np.float64)
eig_vec = np.zeros((class_num,size*size,size*size), dtype=np.float64)

# 係数の個数の入力
D = int( input( " D? > " ) )

# fig以下の画像を削除（MS-Windows）
os.system("del /Q fig\*")

# データの読み込み
dir = [ "Male" , "Female" ]
for i in range(class_num):
    for j in range(1,train_num+1):
        train_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(train_file).convert('L')
        resize_img = work_img.resize((size, size))
        train_vec[i][j-1] = np.asarray(resize_img).astype(np.float64).flatten()
        
# 平均ベクトル
for i in range(class_num):
    ave_vec[i] = np.mean( train_vec[i] , axis=0 )
    
    # 平均顔の保存
    ave_img = Image.fromarray(np.uint8(np.reshape(ave_vec[i],(size,size))))
    ave_file = "fig/" + str(dir[i]) + "-ave.png"
    ave_img.save(ave_file)

for i in range(class_num):
    # クラス内分散共分散
    Sw[i] = np.cov( train_vec[i] , rowvar=0 , bias=1 )

    # 固有値分解
    lamda[i] , eig_vec[i] = np.linalg.eig( Sw[i] )

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

    for j in range(train_num,101):
        # テストデータの読み込み
        pat_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(pat_file).convert('L')
        resize_img = work_img.resize((size, size))
        src_vec = np.reshape( np.asarray(resize_img).astype(np.float64) , (size*size,1) )

        # 係数ベクトル
        c = np.zeros((size*size), dtype=np.float64)
        for k in range(size*size):
            a = np.resize( ave_vec[i] , (size*size,1) )
            c[k] = np.dot( eig_vec[i][:,k].real.T , ( src_vec - a ))

        # 復元
        restore_vec = np.zeros((size*size), dtype=np.float64)
        for k in range(0,D):
            restore_vec += c[k] * eig_vec[i][:,k].real
        restore_vec += ave_vec[i]

        # 画像の描画
        plt.figure()

        # 元画像の表示
        plt.subplot(1,2,1)
        plt.imshow(np.asarray(resize_img).astype(np.float64),cmap='gray')
        plt.title( "Original Image" )
        
        # 復元画像の表示
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(restore_vec,(size,size)),cmap='gray')

        # 画像の保存
        plt.title( "Restore Image( " + str(D) + ")" )
        file = "fig/" + dir[i] + "-" + str(j) + "-result.png"
        plt.savefig(file)
        plt.close()
        
