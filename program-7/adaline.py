# -*- coding: utf-8 -*-

# adaline（MNIST）

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 10

# 画像の大きさ
size = 14
feature = size * size

# 学習データ数
train_num = 100

# データ
data_vec = np.zeros((class_num,train_num,feature), dtype=np.float64)

# 学習係数
alpha = 0.1

# 出力層
class Outunit:
    def __init__(self, n, m): # m 出力層の個数，n 一つ前の層の個数
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(n,m))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,m)

    def Propagation(self, x): # x 入力値
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値（活性化関数なし）
        self.out = self.u

    def Error(self, t): # t 教師信号
        # 誤差
        delta = self.out - t

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename): # filename 保存したいファイル名
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)

        # 重みの画像化
        for i in range(class_num):
            a = np.reshape( self.w[:,i] , (size,size) )
            plt.imshow(a , interpolation='nearest')
            file = "dat/weight-" + str(i) + ".png"
            plt.savefig(file)
            plt.close()
        
    def Load(self, filename): # filename ロードしたいファイル名
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work['w']
        self.b = work['b']

# データの読み込み
def Read_data( flag ):

    dir = [ "train" , "test" ]
    for i in range(class_num):
        for j in range(1,train_num+1):
            # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
            train_file = "mnist/" + dir[ flag ] + "/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
            work_img = Image.open(train_file).convert('L')
            resize_img = work_img.resize((size, size))
            data_vec[i][j-1] = np.asarray(resize_img).astype(np.float64).flatten()
            
            # 入力値の合計を1とする
            data_vec[i][j-1] = data_vec[i][j-1] / np.sum( data_vec[i][j-1] )

# 予測
def Predict():

    # 重みのロード
    outunit.Load( "dat/adaline.npz" )

    # 混合行列
    result = np.zeros((class_num,class_num), dtype=np.int32)
    
    for i in range(class_num):
        for j in range(0,train_num):
            # 入力データ
            input_data = data_vec[i][j].reshape(1,feature)

            # 伝播
            outunit.Propagation( input_data )

            # 教師信号
            teach = np.zeros( (1,class_num) )
            teach[0][i] = 1

            # 予測
            ans = np.argmax( outunit.out[0] )

            result[i][ans] +=1
            print( i , j , "->" , ans )

    print( "\n [混合行列]" )
    print( result )
    print( "\n 正解数 ->" ,  np.trace(result) )

# 学習
def Train():

    # エポック数
    epoch = 100

    for e in range( epoch ):
        error = 0.0
        for i in range(class_num):
            for j in range(0,train_num):
                # 入力データ
                rnd_c = np.random.randint(class_num)
                rnd_n = np.random.randint(train_num)
                input_data = data_vec[rnd_c][rnd_n].reshape(1,feature)

                # 伝播
                outunit.Propagation( input_data )

                # 教師信号
                teach = np.zeros( (1,class_num) )
                teach[0][rnd_c] = 1 

                # 誤差
                outunit.Error( teach )

                # 重みの修正
                outunit.Update_weight()

                # 誤差二乗和
                error += np.dot( ( outunit.out - teach ) , ( outunit.out - teach ).T )
        print( e , "->" , error )

    # 重みの保存
    outunit.Save( "dat/adaline.npz" )

if __name__ == '__main__':

    # 出力層のコンストラクター
    outunit = Outunit( feature , class_num )
    
    argvs = sys.argv

    # 引数がtの場合
    if argvs[1] == "t":

        # 学習データの読み込み
        flag = 0
        Read_data( flag )

        # 学習
        Train()

    # 引数がpの場合
    elif argvs[1] == "p":

        # テストデータの読み込み
        flag = 1
        Read_data( flag )

        # テストデータの予測
        Predict()

