# -*- coding: utf-8 -*-

# ヘッブの学習（主成分分析）

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 2

# 画像の大きさ
size = 16

# 学習データ
train_num = 100

# 学習データ
train_vec = np.zeros((class_num,train_num,size*size), dtype=np.float64)

# 出力の個数（固有ベクトルの個数）
output_size = 5

# 入力の個数
input_size = size*size

# 重みの初期化
weight = np.random.uniform( -0.5 , 0.5, (output_size,input_size) )

# 重みの変更値
d_weight = np.zeros( (output_size,input_size) )

# 学習回数，学習係数
LOOP = 1000
alpha = 0.01

# fig以下の画像を削除（MS-Windows）
os.system("del /Q fig\*")

# 学習データの読み込み
dir = [ "Male" , "Female" ]
for i in range(class_num):
    for j in range(1,train_num+1):
        # グレースケール画像として読み込み→大きさの変更→numpyに変換，ベクトル化
        train_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(train_file).convert('L')
        resize_img = work_img.resize((size, size))
        train_vec[i][j-1] = np.asarray(resize_img).astype(np.float64).flatten()

        # ノルムを1に正規化
        train_vec[i][j-1] = train_vec[i][j-1] / np.linalg.norm(train_vec[i][j-1]) 

# 学習
for o in range(class_num):
    for loop in range(LOOP):
        print( loop )
        
        for t in range(0,train_num):

            # 出力値の計算
            e = train_vec[o][t].reshape( (input_size,1) )
            V = np.dot( weight , e )
            #print( V.shape )

            # 重みの更新値の計算
            for i in range( output_size ):    
                for j in range( input_size ):

                    sum_o = 0
                    for k in range(i+1):
                        sum_o += V[k][0] * weight[k][j]
                        
                    d_weight[i][j] = alpha * V[i][0] * ( e[j][0] - sum_o )

            # 重みの更新
            weight += d_weight

    # 重みベクトルの画像化
    for j in range(output_size):
        a = np.reshape( weight[j], (size,size) )
        plt.imshow(a , interpolation='nearest')
        plt.colorbar()
        file = "fig/weight-" + dir[o] + "-" + str(j) + ".png"
        plt.savefig(file)
        plt.clf()


    # 重みベクトルの保存
    filename = "weight-pca-" + dir[o] + ".txt"
    f = open( filename , "w" )
    for i in range( output_size ):    
        for j in range( input_size ):
            f.write( str( weight[i][j] ) + "\n" )
    f.close()

    # 検算
    for i in range(output_size):
        print( np.dot( weight[1].T , weight[i] ) )



