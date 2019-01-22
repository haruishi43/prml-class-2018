# -*- coding: utf-8 -*-

# 階層型ニューラルネットワーク

# chainer のインストール
!curl https://colab.chainer.org/install | sh -

# 他のパッケージのインストール
!pip install h5py

# drive のマウント（認証が必要）
from google.colab import drive
drive.mount('/content/gdrive')

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

# chainerのバージョン
print('GPU availability:', chainer.cuda.available)
print('cuDNN availablility:', chainer.cuda.cudnn_enabled)
chainer.print_runtime_info()

# CPU，GPUの確認
!cat /proc/cpuinfo
!cat /proc/driver/nvidia/gpus/0000:00:04.0/information

# データ
DATA = 100
class_num = 3
data = np.zeros( (DATA,5), dtype=np.float32 )
teach = np.zeros( DATA, dtype=np.int32 )

# データの読み込み
def Read_data():
  count = 0
  for line in open('/content/gdrive/My Drive/data/data-1.csv', 'r' ):
    work = line[:-1].split(',')
    data[count] = work[0:5]
    teach[count] = work[5:].index( '1' )
    count+=1

# ネットワークの設定
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            # ネットワークの構造の設定
            l1=L.Linear(5,128),    #  入力層のニューロン数5個，中間層1のニューロン数は128個
            l2=L.Linear(128,128),  #  中間層1のニューロン数128個，中間層2のニューロン数は128個
            l3=L.Linear(128,3),    #  中間層2のニューロン数128個，出力層のニューロン数は3個
        )
    
    # 損失関数
    def __call__(self,x,y):
        # ソフトマックスクロスエントロピー誤差
        return F.softmax_cross_entropy(self.fwd(x), y)

    # 順伝播
    def fwd(self,x):
         h1 = F.relu(self.l1(x))
         h2 = F.relu(self.l2(h1))
         h3 = self.l3(h2)
         return h3

# 学習
def Train():
    # # 更新方法はAdam
    optimizer = optimizers.Adam() 
    optimizer.setup(model)

    # バッチ数，エポック数の設定
    n_epoch = 500
    batch_size = 10
    n_train = DATA

    # 学習
    for epoch in range(n_epoch):
        error = 0.0
        for i in range(0, n_train, batch_size):
            x = np.zeros((batch_size,5),dtype=np.float32)
            y = np.zeros( batch_size ,dtype=np.int32)

            # バッチの作成
            for j in range(batch_size):
                rnd = np.random.randint(DATA)
                x[j,:]=data[rnd,:]
                y[j]=teach[rnd]
            
            # 入力データ
            xt = Variable(cuda.to_gpu(x))
            yt = Variable(cuda.to_gpu(y))

            # 勾配の初期化→誤差の計算→逆伝播→パラメータの更新
            model.zerograds()
            loss = model(xt,yt)
            error += loss.data
            loss.backward()
            optimizer.update()

        print( epoch , ":" , error )
        
    # パラメータの保存
    serializers.save_hdf5( "/content/gdrive/My Drive/data/model.h5" , model )

# 予測
def Predict():
    
    # パラメータのロード
    serializers.load_hdf5( "/content/gdrive/My Drive/data/model.h5" , model )
    
    # 混合行列
    result = np.zeros( (class_num,class_num) , dtype=np.int )
    
    x = np.zeros((1,5),dtype=np.float32)
    y = np.zeros( 1 ,dtype=np.int32)
    for i in range(DATA):
      # 入力データ
      x[0,:]=data[i,:]
      y[0]=teach[i]
      
      xt = Variable(cuda.to_gpu(x))
      yt = Variable(cuda.to_gpu(y))
      
      # 予測
      predict=model.fwd(xt)
      
      # 結果の出力
      ans = np.argmax( predict.data[0] )
      print( xt.data[0] , ":" , predict.data[0] , "->" , ans , "[" , yt.data[0] , "]" )
      
      result[int(yt.data[0])][int(ans)] +=1
    
    print( "\n [混合行列]" )
    print( result )
    print( "\n 正解数 ->" ,  np.trace(result) )

# モデルの設定(GPU)
gpu_device = 0
cuda.get_device(gpu_device).use()
model = MLP()
model.to_gpu()

# データの読み込み
Read_data()

# 学習
Train()

# データの読み込み
Read_data()

# 予測
Predict()

!nvidia-smi

