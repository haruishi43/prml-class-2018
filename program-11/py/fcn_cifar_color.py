# -*- coding: utf-8 -*-
#
# FCNによるカラー化(cifar）
#

# chainer のインストール
!curl https://colab.chainer.org/install | sh -

# 必要なライブラリイのインストール
!pip install h5py
!pip install pillow
!pip install matplotlib

# Google drive のマウント（認証が必要）
from google.colab import drive
drive.mount('/content/gdrive')

# データの展開
!unzip -q  '/content/gdrive/My Drive/data/cifar-10.zip'
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
import matplotlib.pyplot as plt

# chainerのバージョンの確認
print('GPU availability:', chainer.cuda.available)
print('cuDNN availablility:', chainer.cuda.cudnn_enabled)
chainer.print_runtime_info()

# CPU，GPUの確認
!cat /proc/cpuinfo
!cat /proc/driver/nvidia/gpus/0000:00:04.0/information

# クラス数
class_num = 10

# 画像の大きさ
XSIZE = 32
YSIZE = 32

# 学習データ数
train_num = 200

# データ
data_vec = np.zeros((class_num,train_num,1,YSIZE,XSIZE), dtype=np.float32)
teach_vec = np.zeros((class_num,train_num,3,YSIZE,XSIZE), dtype=np.float32)

# 学習のパラメータ
batchsize=10
n_epoch=100
n_train=class_num * train_num

def normalize_img(x):
    # -1 以上を1，0以下を0とする
    return np.float32(0 if x<0 else (1 if x>1 else x))
n_img=np.vectorize(normalize_img)

# データの読み込み
def Read_data( flag ):

    dir = [ "train" , "test" ]
    dir1 = [ "airplane" , "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck" ]
    for i in range(class_num):
        print( i )
        for j in range(0,train_num):
            # グレースケール画像で読み込み
            train_file = "cifar-10/" + dir[ flag ] + "/" + dir1[i] + "/" + str(j) + ".png"
            work_img = Image.open(train_file).convert('L')
            
            # numpyに変換
            temp = np.asarray(work_img).astype(np.float32)

            # 入力値の正規化
            data_vec[i][j][0] = temp / 255.0
          
            # RGB画像で読み込み
            work_img = Image.open(train_file).convert('RGB')

            # numpyに変換
            temp = np.asarray(work_img).astype(np.float32)

            # (32,32,3)→(3,32,32)に変換
            temp = np.transpose(temp, (2,0,1))
            
            # 入力値の正規化
            teach_vec[i][j] = temp / 255.0
    print( " ---- " )

# CNN
class CNN(chainer.Chain):
    # 畳み込みネットワークの設定
    def __init__(self):
        super(CNN, self).__init__(
            # 畳み込み層の設定
            conv1 = L.Convolution2D(1, 128, 3, stride=1, pad=1),
            conv2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv5 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv6 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv7 = L.Convolution2D(128, 3, 3, stride=1, pad=1),
            
            # バッチノーマライゼーション
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(128),
            bn4 = L.BatchNormalization(128),
            bn5 = L.BatchNormalization(128),
            bn6 = L.BatchNormalization(128),
            bn7 = L.BatchNormalization(3),
        )

    # 損失関数
    def __call__(self, x, y):
        # 誤差二乗和
        return F.mean_squared_error(self.fwd(x), y)

    # 畳み込みネットワーク
    def fwd(self, x):
        # 畳み込み
        # 大きさ 1
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, ksize=2,stride=2,pad=0)
        
        # 大きさ 1/2
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, ksize=2,stride=2,pad=0)
        
        # 大きさ 1/4
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pooling_2d(h, ksize=2,stride=2,pad=0)

        # 大きさ 1/8
        h = F.relu(self.bn4(self.conv4(h)))
        
        # 逆畳み込み
        h = F.unpooling_2d(h ,2, cover_all=False)

        # 大きさ 1/4
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.unpooling_2d(h ,2, cover_all=False)
        
        # 大きさ 1/2
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.unpooling_2d(h ,2, cover_all=False)
        
        # 大きさ 1
        h = F.relu(self.conv7(h))
        return h

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
            y = np.zeros((batchsize, 3, YSIZE, XSIZE), dtype=np.float32)

            # バッチの作成
            for j in range(batchsize):
                rnd_c = np.random.randint(class_num)
                rnd = np.random.randint(train_num)
                x[j,0,:,:] = data_vec[rnd_c][rnd]
                y[j,:,:,:] = teach_vec[rnd_c][rnd]
                
            # 入力データ
            xt = Variable(cuda.to_gpu(x))
            yt = Variable(cuda.to_gpu(y))

            # 勾配の初期化→伝播，誤差の計算→逆伝播→パラメータの更新
            model.zerograds()
            loss = model( xt , yt )
            error += loss.data
            loss.backward()
            optimizer.update()
        print( str( epoch ) +  ' : ' + str( error ) )

    # パラメータの保存
    serializers.save_hdf5("/content/gdrive/My Drive/data/model-color-cifar.h5", model)

# 予測
def Predict():

    # パラメータのロード
    serializers.load_hdf5("/content/gdrive/My Drive/data/model-color-cifar.h5", model)
    x = np.zeros((1, 1, YSIZE, XSIZE), dtype=np.float32)
    y = np.zeros((1, 3, YSIZE, XSIZE), dtype=np.float32)

    for i in range(class_num):
        for j in range(train_num):
            #　入力データ
            x[0,0,:,:] = data_vec[i][j]
            y[0,:,:,:] = teach_vec[i][j]

            xt = Variable(cuda.to_gpu(x))
            yt = Variable(cuda.to_gpu(y))

            # 予測
            predict = cuda.to_cpu( ( model.fwd( xt ) ).data )
            
            if j < 1:
                # 画像の描画
                plt.figure()
                
                # グレースケール画像の表示
                plt.subplot(1,3,1)
                plt.imshow(data_vec[i][j][0],vmin=0, vmax=1,cmap="gray")
                plt.title( "Grayscale Image" )
                
                plt.subplot(1,3,2)
                # (3,32,32) -> (32,32,3)に変換
                work = np.transpose( teach_vec[i][j] , (1,2,0) )
                plt.imshow(work,vmin=0, vmax=1)
                plt.title( "Original Image" )
                
                # 復元画像の表示
                plt.subplot(1,3,3)
                # (1,3,32,32) -> (3,32,32)に変換
                work = np.reshape(predict,(3,YSIZE,XSIZE))
                # (3,32,32) -> (32,32,3)に変換
                work = np.transpose( work , (1,2,0) )
                work = n_img( work )
                plt.imshow(work,vmin=0, vmax=1)

                # 画像の保存
                plt.title( "Decode Image(" + str(i) + "," + str(j) + ")" )
                file = "/content/gdrive/My Drive/data/result/color-" + str(i) + "-" + str(j) + "-result.png"
                print( file )
                plt.savefig(file)
                plt.close()

# GPU
xp = cuda.cupy
cuda.get_device(0).use()
model = CNN()
model.to_gpu()

# データの読み込み
flag = 0
Read_data( flag )

# 学習
Train()

flag = 1    

# データの読み込み
Read_data( flag )

# 予測
Predict()

!nvidia-smi

