# -*- coding: utf-8 -*-
#
# CNNによるクラス分類（MNIST）
#

# chainer のインストール
!curl https://colab.chainer.org/install | sh -

# 必要なライブラリイのインストール
!pip install h5py
!pip install pillow

# drive のマウント(認証が必要)
from google.colab import drive
drive.mount('/content/gdrive')

# データの展開
!unzip -q  '/content/gdrive/My Drive/data/mnist.zip'
!ls

import sys
import numpy as np
import chainer
from chainer import cuda
from chainer import Function
from chainer import report
from chainer import training
from chainer import utils
from chainer import Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList, cuda
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from PIL import Image

# chainerのバージョン
print('GPU availability:', chainer.cuda.available)
print('cuDNN availablility:', chainer.cuda.cudnn_enabled)
chainer.print_runtime_info()

# CPU，GPUの表示
!cat /proc/cpuinfo
!cat /proc/driver/nvidia/gpus/0000:00:04.0/information

# クラス数
class_num = 10

# 画像の大きさ
XSIZE = 28
YSIZE = 28

# 学習データ数
train_num = 100

# データ（GPUの場合，単精度の方が早い場合が多いです）
data_vec = np.zeros((class_num,train_num,YSIZE,XSIZE), dtype=np.float32)

# 学習のパラメータ
batchsize=10
n_epoch=10
n_train=class_num*train_num

# データの読み込み
def Read_data( flag ):

    dir = [ "train" , "test" ]
    for i in range(class_num):
        print( i )
        for j in range(1,train_num+1):
            # グレースケール画像で読み込み→大きさの変更→numpyに変換
            train_file = "mnist/" + dir[ flag ] + "/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
            work_img = Image.open(train_file).convert('L')
            data_vec[i][j-1]  = np.asarray(work_img).astype(np.float32)
            
            # データの正規化
            data_vec[i][j-1] = data_vec[i][j-1] / 255.0
    print( "----" )

# CNN
class CNN(chainer.Chain):
    # 畳み込みネットワークの設定
    def __init__(self):
        super(CNN, self).__init__(
            # 畳み込み層の設定
            conv1 = L.Convolution2D(1, 64, 3, stride=1, pad=1),
            conv2 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv3 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv4 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv5 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv6 = L.Convolution2D(64, 64, 3, stride=1, pad=1),

            # 全結合層の設定
            full1 = L.Linear(64*7*7,100),
            full2 = L.Linear(100, 10)
        )

    # 損失関数
    def __call__(self, x, y):
        # ソフトマックスクロスエントロピー誤差
        return F.softmax_cross_entropy(self.fwd(x), y)

    # 畳み込みネットワーク
    def fwd(self, x):
        # 畳み込み→畳み込み→プーリング
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2,stride=2,pad=0)

        # 畳み込み→畳み込み→プーリング
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, ksize=2,stride=2,pad=0)

        # 畳み込み→畳み込み
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))

        # 全結合層
        h = F.relu(self.full1(h))
        out = self.full2(h)
        return out

# 学習
def Train():
    # Adamによる更新
    optimizer = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer.setup(model)
    # 正則化
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
    
    for epoch in range(n_epoch):
        error = 0.0
        for i in range(0, n_train, batchsize):
            x = np.zeros((batchsize, 1, YSIZE, XSIZE), dtype=np.float32)
            y = np.zeros(batchsize, dtype=np.int32)

            # バッチの作成
            for j in range(batchsize):
                rnd_c = np.random.randint(class_num)
                rnd = np.random.randint(train_num)
                x[j,0,:,:] = data_vec[rnd_c][rnd]
                y[j] = rnd_c 

            # 入力データ
            xt = Variable(x)
            yt = Variable(y)

            # 勾配の初期化→伝播，誤差の計算→逆伝播→パラメータの更新
            model.zerograds()
            loss = model( xt , yt )
            error += loss.data
            loss.backward()
            optimizer.update()

            if i != 0 and i % 100 == 0:
               print( ' (' + str( epoch ) + ',' + str( i ) + ') : ' + str( error ) )
               error = 0.0

    # パラメータの保存
    serializers.save_hdf5("/content/gdrive/My Drive/data/model-CNN-gpu.h5", model)

# 予測
def Predict():

    # パラメータのロード
    serializers.load_hdf5("/content/gdrive/My Drive/data/model-CNN-gpu.h5", model)
    x = np.zeros((1, 1, YSIZE, XSIZE), dtype=np.float32)
    y = np.zeros( 1, dtype=np.int32)

    # 混合行列
    result = np.zeros((class_num,class_num), dtype=np.int32)
    
    for i in range(class_num):
        for j in range(train_num):
            # 入力データ
            x[0,0,:,:] = data_vec[i][j]
            y[0] = i
            
            xt = Variable(x)
            yt = Variable(y)
            
            # 予測
            predict = model.fwd( xt )
            ans = np.argmax( predict.data[0] )

            # 混合行列
            result[i][int(ans)] += 1

    print( "\n [混合行列]" )
    print( result )
    print( "\n 正解数 ->" ,  np.trace(result) )

# モデルの設定
model = CNN()

# データの読み込み
flag = 0
Read_data( flag )

# 学習
Train()

# データの読み込み
flag = 1
Read_data( flag )

# 予測
Predict()
