# -*- coding: utf-8 -*-

# オートエンコーダ

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 10

# 画像の大きさ
size = 32
feature = size * size

# 学習データ数
train_num = 200

# データ
data_vec = np.zeros((class_num, train_num, 3, feature), dtype=np.float64)

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
        self.w = np.random.uniform(-0.5, 0.5, (m,n))

        # 閾値
        self.b = np.random.uniform(-0.5, 0.5, n)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値
        self.out = Sigmoid(self.u)

    def Error(self, t):
        # 誤差
        f_ = Sigmoid_(self.u)
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

        # 出力値
        self.out = Sigmoid(self.u)

    def Error(self, p_error):
        # 誤差
        f_ = Sigmoid_(self.u)
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


def Read_data(flag):
    '''
    Import cifar-10 data
    '''

    dir = ["train" , "test"]
    dir1 = ["airplane" , "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]
    
    for i in range(class_num):
        for j in range(0, train_num):
            # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
            train_file = "../cifar-10/" + dir[flag] + "/" + dir1[i] + "/" + str(j) + ".png"
            work_img = Image.open(train_file).convert('RGB')
            resize_img = work_img.resize((size, size))

            work = np.resize(np.asarray(resize_img).astype(np.float64), (size,size, 3))
            
            data_vec[i][j][0] = work[:,:,0].flatten()
            data_vec[i][j][1] = work[:,:,1].flatten()
            data_vec[i][j][2] = work[:,:,2].flatten()


def Train():
    '''
    Training
    '''

    # エポック数
    epoch = 1000

    for e in range(epoch):
        error = 0.0
        for i in range(class_num):
            for j in range(0, train_num):
                rnd_c = np.random.randint(class_num)
                rnd_n = np.random.randint(train_num)

                # Input
                r_data = data_vec[rnd_c][rnd_n][0].reshape(1, feature) / 255
                g_data = data_vec[rnd_c][rnd_n][1].reshape(1, feature) / 255
                b_data = data_vec[rnd_c][rnd_n][2].reshape(1, feature) / 255

                # Label
                r_teach = data_vec[rnd_c][rnd_n][0].reshape(1, feature) / 255
                g_teach = data_vec[rnd_c][rnd_n][1].reshape(1, feature) / 255
                b_teach = data_vec[rnd_c][rnd_n][2].reshape(1, feature) / 255

                # Forward
                r_hunit.Propagation(r_data)
                r_outunit.Propagation(r_hunit.out)
                g_hunit.Propagation(g_data)
                g_outunit.Propagation(g_hunit.out)
                b_hunit.Propagation(b_data)
                b_outunit.Propagation(b_hunit.out)

                # Backprop
                r_outunit.Error(r_teach)
                r_hunit.Error(r_outunit.error)
                g_outunit.Error(g_teach)
                g_hunit.Error(g_outunit.error)
                b_outunit.Error(b_teach)
                b_hunit.Error(b_outunit.error)

                # Update weights
                r_outunit.Update_weight()
                r_hunit.Update_weight()
                g_outunit.Update_weight()
                g_hunit.Update_weight()
                b_outunit.Update_weight()
                b_hunit.Update_weight()

                r_error = np.dot((r_outunit.out - r_teach), (r_outunit.out - r_teach).T)
                g_error = np.dot((g_outunit.out - g_teach), (g_outunit.out - g_teach).T)
                b_error = np.dot((b_outunit.out - b_teach), (b_outunit.out - b_teach).T)
                error += (r_error + g_error + b_error)
                
        print(e, "->", error)

    # 重みの保存
    r_outunit.Save("data/r-hw-out.npz")
    g_outunit.Save("data/g-hw-out.npz")
    b_outunit.Save("data/b-hw-out.npz")
    r_hunit.Save("data/r-hw-hunit.npz")
    g_hunit.Save("data/g-hw-hunit.npz")
    b_hunit.Save("data/b-hw-hunit.npz")


def Predict():
    '''
    Prediction
    '''

    # 重みのロード
    r_outunit.Load("data/r-hw-out.npz")
    g_outunit.Load("data/g-hw-out.npz")
    b_outunit.Load("data/b-hw-out.npz")
    r_hunit.Load("data/r-hw-hunit.npz")
    g_hunit.Load("data/g-hw-hunit.npz")
    b_hunit.Load("data/b-hw-hunit.npz")

    # 混合行列
    #result = np.zeros((class_num, class_num), dtype=np.int32)
    
    for i in range(class_num):
        for j in range(0, train_num):
            # Input
            r_data = data_vec[i][j][0].reshape(1, feature) / 255
            g_data = data_vec[i][j][1].reshape(1, feature) / 255
            b_data = data_vec[i][j][2].reshape(1, feature) / 255

            # Label
            r_teach = data_vec[i][j][0].reshape(1, feature) / 255
            g_teach = data_vec[i][j][1].reshape(1, feature) / 255
            b_teach = data_vec[i][j][2].reshape(1, feature) / 255

            # 伝播
            r_hunit.Propagation(r_data)
            r_outunit.Propagation(r_hunit.out)
            g_hunit.Propagation(g_data)
            g_outunit.Propagation(g_hunit.out)
            b_hunit.Propagation(b_data)
            b_outunit.Propagation(b_hunit.out)

            if j < 1:
                # original
                original = np.zeros((3, size*size), dtype=np.float64)
                original[0] = data_vec[i][j][0]
                original[1] = data_vec[i][j][1]
                original[2] = data_vec[i][j][2]

                original_vec = np.resize(original, (3, size, size))
                original_vec = np.transpose(original_vec, (1, 2, 0))

                original_image = Image.fromarray(np.uint8(original_vec))

                # prediction
                prediction = np.zeros((3, size*size), dtype=np.float64)
                prediction[0] = r_outunit.out
                prediction[1] = g_outunit.out
                prediction[2] = b_outunit.out

                prediction_vec = np.resize(prediction * 255, (3, size, size))
                prediction_vec = np.transpose(prediction_vec, (1, 2, 0))

                prediction_image = Image.fromarray(np.uint8(prediction_vec))

                # 画像の描画
                plt.figure()

                # 元画像の表示
                plt.subplot(1,2,1)
                plt.imshow(original_image)
                plt.title( "Original Image" )
                
                # 復元画像の表示
                plt.subplot(1,2,2)
                plt.imshow(prediction_image)

                # 画像の保存
                plt.title( "Decode Image(" + str(i) + "," + str(j) + ")" )
                file = "fig/decode-" + str(i) + "-" + str(j) + "-result.png"
                plt.savefig(file)
                plt.close()

    # 結合係数の画像化
    g_size = 10
    plt.figure(figsize=(g_size, g_size))
    
    count = 1
    for i in range(hunit_num):

        plt.subplot(g_size,g_size,count)
        plt.imshow(np.reshape(r_hunit.w[:,i], (size, size)), cmap='gray')
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        count += 1

    file = "fig/r_hunit-weight.png"
    plt.savefig(file)
    plt.close()

        
if __name__ == '__main__':

    # 中間層の個数
    hunit_num = 100

    # 中間層のコンストラクター
    r_hunit = Hunit(feature, hunit_num)
    g_hunit = Hunit(feature, hunit_num)
    b_hunit = Hunit(feature, hunit_num)

    # 出力層のコンストラクター
    r_outunit = Outunit(hunit_num, feature)
    g_outunit = Outunit(hunit_num, feature)
    b_outunit = Outunit(hunit_num, feature)

    argvs = sys.argv

    # 引数がtの場合
    if argvs[1] == "t":

        # 学習データの読み込み
        flag = 0
        Read_data(flag)

        # 学習
        Train()

    # 引数がpの場合
    elif argvs[1] == "p":

        # os.system("del fig/*.png")
        
        # テストデータの読み込み
        flag = 1
        Read_data(flag)

        # テストデータの予測
        Predict()
