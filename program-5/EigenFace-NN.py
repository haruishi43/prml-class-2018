# -*- coding: utf-8 -*-

# 固有顔によって次元圧縮→最近傍法による性別推定

# > pip install matplotlib

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# クラス数
class_num = 2

# 画像の大きさ
size = 32

# 学習データ
train_num = 80

# テストデータ
test_num = 100-train_num

# 学習データ，平均ベクトル
train_vec = np.zeros((train_num*class_num,size*size), dtype=np.float64)
ave_vec = np.zeros(size*size, dtype=np.float64)

# 分散共分散行列，固有値，固有ベクトル
Sw = np.zeros((size*size,size*size), dtype=np.float64)
lamda = np.zeros((size*size), dtype=np.float64)
eig_vec = np.zeros((size*size,size*size), dtype=np.float64)

# 係数の個数の入力
D = int( input( " D? > " ) )

# 係数ベクトル
c = np.zeros((train_num*2,size*size), dtype=np.float64)

# fig以下の画像を削除（MS-Windows）
os.system("del /Q fig\*")

# データの読み込み
dir = [ "Male" , "Female" ]
for i in range(class_num):
    for j in range(1,train_num+1):
        train_file = "face/" + dir[i] + "/" + str(j) + ".png"
        work_img = Image.open(train_file).convert('L')
        resize_img = work_img.resize((size, size))
        train_vec[i*train_num+j-1] = np.asarray(resize_img).astype(np.float64).flatten()
        
# 平均ベクトル
ave_vec = np.mean( train_vec , axis=0 )
    
# 平均顔の保存
ave_img = Image.fromarray(np.uint8(np.reshape(ave_vec,(size,size))))
ave_file = "fig/ave.png"
ave_img.save(ave_file)

# クラス内分散共分散
Sw = np.cov( train_vec , rowvar=0 , bias=1 )

# 固有値分解
lamda , eig_vec = np.linalg.eig( Sw )

# 固有ベクトルの表示
for j in range(10):
    a = np.reshape( eig_vec[:,j].real , (size,size) )
    plt.imshow(a , interpolation='nearest')
    plt.colorbar()
    file = "fig/eigen-" + str(j) + ".png"
    plt.savefig(file)
    plt.close()
        
# 検算
print( np.dot( eig_vec[:,0] , eig_vec[:,1]) )
print( np.dot( eig_vec[:,1] , eig_vec[:,1] ) )
