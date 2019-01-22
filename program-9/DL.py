# -*- coding: utf-8 -*-

# ネットワークの多層化（MNIST）

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

# シグモイド関数
def Sigmoid( x ):
    return 1 / ( 1 + np.exp(-x) )

# シグモイド関数の微分
def Sigmoid_( x ):
    return ( 1-Sigmoid(x) ) * Sigmoid(x)

# ReLU関数
def ReLU( x ):
    return np.maximum( 0, x )

# ReLU関数の微分
def ReLU_( x ):
    return np.where( x > 0, 1, 0 )

# ソフトマックス関数
def Softmax( x ):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

# 出力層
class Outunit:
    def __init__(self, m, n):
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(m,n))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,n)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値（シグモイド関数）
        self.out = Sigmoid( self.u )

    def Error(self, t):
        # 誤差
        f_ = self.out * ( 1 - self.out )
        #f_ = Sigmoid_( self.u )
        delta = ( self.out - t ) * f_

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        # 前の層に伝播する誤差
        self.error = np.dot(delta, self.w.T) 

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename):
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)
        
    def Load(self, filename):
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work['w']
        self.b = work['b']

# 中間層
class Hunit:
    def __init__(self, m, n):
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(m,n))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,n)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値（ReLU関数）
        self.out = ReLU( self.u )

    def Error(self, p_error):
        # 誤差
        f_ = ReLU_( self.u )
        delta = p_error * f_

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        # 前の層に伝播する誤差
        self.error = np.dot(delta, self.w.T) 

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename):
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)
        
    def Load(self, filename):
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
    outunit.Load( "dat/BP-out.npz" )
    hunit1.Load( "dat/BP-hunit1.npz" )
    hunit2.Load( "dat/BP-hunit2.npz" )
    hunit3.Load( "dat/BP-hunit3.npz" )
    hunit4.Load( "dat/BP-hunit4.npz" )
    hunit5.Load( "dat/BP-hunit5.npz" )
    hunit6.Load( "dat/BP-hunit6.npz" )

    # 混合行列
    result = np.zeros((class_num,class_num), dtype=np.int32)
    
    for i in range(class_num):
        for j in range(0,train_num):
            # 入力データ
            input_data = data_vec[i][j].reshape(1,feature)

            # 伝播
            hunit1.Propagation( input_data )
            hunit2.Propagation( hunit1.out )
            hunit3.Propagation( hunit2.out )
            hunit4.Propagation( hunit3.out )
            hunit5.Propagation( hunit4.out )
            hunit6.Propagation( hunit5.out )
            outunit.Propagation( hunit6.out )

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
    epoch = 200

    for e in range( epoch ):
        error = 0.0
        for i in range(class_num):
            for j in range(0,train_num):
                # 入力データ
                rnd_c = np.random.randint(class_num)
                rnd_n = np.random.randint(train_num)
                input_data = data_vec[rnd_c][rnd_n].reshape(1,feature)

                # 伝播
                hunit1.Propagation( input_data )
                hunit2.Propagation( hunit1.out )
                hunit3.Propagation( hunit2.out )
                hunit4.Propagation( hunit3.out )
                hunit5.Propagation( hunit4.out )
                hunit6.Propagation( hunit5.out )
                outunit.Propagation( hunit6.out )

                # 教師信号
                teach = np.zeros( (1,class_num) )
                teach[0][rnd_c] = 1 

                # 誤差
                outunit.Error( teach )
                hunit6.Error( outunit.error )
                hunit5.Error( hunit6.error )
                hunit4.Error( hunit5.error )
                hunit3.Error( hunit4.error )
                hunit2.Error( hunit3.error )
                hunit1.Error( hunit2.error )

                # 重みの修正
                outunit.Update_weight()
                hunit6.Update_weight()
                hunit5.Update_weight()
                hunit4.Update_weight()
                hunit3.Update_weight()
                hunit2.Update_weight()
                hunit1.Update_weight()
                
                # 誤差二乗和
                error += np.dot( ( outunit.out - teach ) , ( outunit.out - teach ).T )
        print( e , "->" , error )

    # 重みの保存
    outunit.Save( "dat/BP-out.npz" )
    hunit1.Save( "dat/BP-hunit1.npz" )
    hunit2.Save( "dat/BP-hunit2.npz" )
    hunit3.Save( "dat/BP-hunit3.npz" )
    hunit4.Save( "dat/BP-hunit4.npz" )
    hunit5.Save( "dat/BP-hunit5.npz" )
    hunit6.Save( "dat/BP-hunit6.npz" )
    
if __name__ == '__main__':

    # 中間層の個数
    hunit_num1 = 256
    hunit_num2 = 128
    hunit_num3 = 64
    hunit_num4 = 32
    hunit_num5 = 32
    hunit_num6 = 32

    # 中間層のコンストラクター
    hunit1 = Hunit( feature , hunit_num1 )
    hunit2 = Hunit( hunit_num1 , hunit_num2 )
    hunit3 = Hunit( hunit_num2 , hunit_num3 )
    hunit4 = Hunit( hunit_num3 , hunit_num4 )
    hunit5 = Hunit( hunit_num4 , hunit_num5 )
    hunit6 = Hunit( hunit_num5 , hunit_num6 )

    # 出力層のコンストラクター
    outunit = Outunit( hunit_num6 , class_num )
    
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

