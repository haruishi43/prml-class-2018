# -*- coding: utf-8 -*-
#
# MNISTのダウンロード
#

# chainer のインストール
!curl https://colab.chainer.org/install | sh -

# 必要なライブラリイのインストール
!pip install pillow

# drive のマウント(認証が必要)
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
import matplotlib.pyplot as plt

# MNISTのダウンロード
train, test = chainer.datasets.get_mnist()
train_x, train_y = train._datasets
test_x, test_y= test._datasets

# 学習データの表示
print( train_x.shape , train_y.shape )
print( train_y[0] )
plt.imshow(np.reshape(train_x[0],(28,28)),cmap="gray")
plt.show()

# テストデータの表示
print( test_x.shape , test_y.shape )
print( test_y[0] )
plt.imshow(np.reshape(test_x[0],(28,28)),cmap="gray")
plt.show()

