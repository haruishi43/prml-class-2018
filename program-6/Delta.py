# -*- coding: utf-8 -*-

# デルタルール（MNIST）

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 10

# 画像の大きさ
size = 14

# 学習データ数
train_num = 100

# 学習データ
train_vec = np.zeros((class_num,train_num,size*size+1), dtype=np.float64)

# 重み
weight = np.random.uniform(-0.5,0.5,(class_num,size*size+1))
one = np.array([1])

# 学習回数，学習係数
LOOP = 100
alpha = 0.05

# 学習データの読み込み
def Read_train_data():
    for i in range(class_num):
        for j in range(1,train_num+1):
            # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
            train_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
            work_img = Image.open(train_file).convert('L')
            resize_img = work_img.resize((size, size))
            temp = np.asarray(resize_img).astype(np.float64).flatten()

            # 入力値の合計を1とする
            temp = temp / np.sum( temp )

            # 閾値に対応する入力値（1で固定）を追加
            train_vec[i][j-1] = np.hstack( (temp , one ) )

# 学習
def Train():
    for loop in range(LOOP):
        error = 0
        for t in range(class_num*train_num):
            # ランダムにj番目の数字iを選択
            i = np.random.randint(0,class_num)
            j = np.random.randint(0,train_num)

            # 教師信号の設定
            teach = np.zeros(class_num, dtype=np.float64)
            teach[i] = 1
        
            for k in range(class_num):
                # 出力値の計算
                out = np.dot( weight[k], train_vec[i][j] )
                
                # 重みの修正
                weight[k] -= alpha * ( out - teach[k] ) * train_vec[i][j]

                # 誤差二乗和の計算
                error += ( out - teach[k] ) * ( out - teach[k] )

        # 誤差二乗和の出力
        print( loop , " Error : " , error )

    # 重みの画像化
    for i in range(class_num):
        temp = weight[i][0:size*size]
        a = np.reshape( temp , (size,size) )
        plt.imshow(a , interpolation='nearest')
        file = "fig/weight-" + str(i) + ".png"
        plt.savefig(file)
        plt.close()

    # 重みの保存
    with open("weight.txt", mode='w') as f:
        for i in range(class_num):
            for j in range(size*size+1):
                f.write(str(weight[i][j])+"\n")

# 重みの読み込み
def Load_Weight():
    with open("weight.txt", mode='r') as f:
        for i in range(class_num):
            for j in range(size*size+1):
                weight[i][j] = float( f.readline().strip() )

def Predict():
    # 混合行列
    result = np.zeros((class_num,class_num), dtype=np.int32)
    for i in range(class_num):
        for j in range(1,train_num+1):
            # テストデータの読み込み
            pat_file = "mnist/test/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"

            # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
            work_img = Image.open(pat_file).convert('L')
            resize_img = work_img.resize((size, size))
            temp = np.asarray(resize_img).astype(np.float64).flatten()

            # 入力値の合計を1とする
            temp = temp / np.sum( temp )

            # 閾値に対応する入力値（1で固定）を追加
            pat_vec = np.hstack( ( temp , one ) )

            # 出力値の計算
            out = np.dot( weight , np.resize( pat_vec , (size*size+1,1) ) )
            
            # 予測
            ans = np.argmax( out )

            result[i][ans] +=1
            print( i , j , "->" , ans )

    print( "\n [混合行列]" )
    print( result )
    print( "\n 正解数 ->" ,  np.trace(result) )

if __name__ == '__main__':
    
    argvs = sys.argv
    
    # 引数がtの場合
    if argvs[1] == "t":

        # 学習データの読み込み
        Read_train_data()

        # 学習
        Train()

    # 引数がpの場合
    elif argvs[1] == "p":

        # 重みの読み込み
        Load_Weight()

        # テストデータの予測
        Predict()
