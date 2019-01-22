# -*- coding: utf-8 -*-
#
# cifarのダウンロード
#

# chainer のインストール
!curl https://colab.chainer.org/install | sh -

# 必要なライブラリイのインストール
!pip install pillow
!pip install matplotlib

# Google drive のマウント（認証が必要）
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
from PIL import Image
import matplotlib.pyplot as plt

train_data, test_data = chainer.datasets.cifar.get_cifar10()

print( len( train_data ) , len( test_data ) )

train_x , train_y = train_data[1]
print( train_y )

# (3,32,32) -> (32,32,3)に変換
plt.imshow(np.transpose( train_x , (1,2,0) ),vmin=0, vmax=1)
plt.show()

test_x , test_y = test_data[3]
print( test_y )

# (3,32,32) -> (32,32,3)に変換
plt.imshow(np.transpose( test_x , (1,2,0) ),vmin=0, vmax=1)
plt.show()

