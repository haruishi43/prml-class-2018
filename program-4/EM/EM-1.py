# -*- coding: utf-8 -*-

# EMアルゴリズム（混合正規分布(二変量)）

import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# 正規分布の個数
K = 4

# 特徴数（このプログラムはD=2のみ）
D = 2

# 正規分布一つあたりのデータ数
P = 50

# 全データ数
N = K*P

# 正規分布（多変量）
def mnd(x, ave, cov):
    a = np.sqrt( np.linalg.det(cov)*(2*np.pi)**cov.ndim)
    b = np.reshape( x-ave , (D,1) )
    bT = np.reshape( x-ave , (1,D) )
    c = -0.5 * np.dot( np.dot( bT , np.linalg.inv(cov) ) , b )
    return np.exp(c)/a

#平均，分散共分散行列
average = np.zeros( (K,D) , dtype=np.float32 )
cov = np.zeros( (K,D,D) , dtype=np.float32 )

# 平均，標準偏差の設定
for k in range(K):
    for d in range(D):
        a = (np.random.rand() - 0.5) * 20.0
        average[k][d] = a

    for i in range(D):
        for j in range(D):
            s = np.random.randint( 1 , 5 )
            if i == j:
                cov[k][i][j] = s

# データ（全データ数，特徴数）
data = np.zeros( (N,D) , dtype=np.float32 )

# データの生成
for k in range(K):
    for p in range(P):
        data[k*P+p][0],data[k*P+p][1] = np.random.multivariate_normal(average[k],cov[k]).T

# グラフの描画
plt.figure()

# 散布図をプロットする
for n in range(N):
    plt.scatter(data[n][0],data[n][1],color="red")

# ラベル
plt.xlabel("x",size=10)
plt.ylabel("y",size=10)
plt.title( "MND" )

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)

# グラフの表示
plt.show()
plt.clf()

# 初期化(Q値，平均，分散共分散，重み）
Q = np.zeros((N, K),dtype=np.float64)
e_average = np.zeros((K,D),dtype=np.float64)
e_cov = np.zeros((K,D,D),dtype=np.float64)
e_lamda = np.zeros(K,dtype=np.float64)

for k in range(K):
    e_average[k] = random.choice(data)
    e_cov[k] = (np.cov(data , rowvar=0 , bias=1) )
    e_lamda[k] = 1/K

print( "平均 -> " , e_average )
print( "分散 -> " , e_cov )
print( "重み -> " , e_lamda )

# EMアルゴリズムにより，パラメータQ値（Q），平均ベクトル
#（e_average），分散共分散行列（e_cov），重み（e_lamda）
# を推定しなさい





# Q値が最大のクラスターを表示
for i in range(N):
    ans = np.argmax( Q[i,:] )
    print( i , ans )

# グラフの描画
plt.figure()

plt.subplot(2,1,1)

# 散布図をプロットする
plot_c = [ "red" , "orange" , "blue" , "green" , "pink" ]
plot_l = [ "class-1" , "class-2" , "class-3" , "class-4" , "clss-5" ]
for k in range(K):
    plt.scatter(data[k*P:(k+1)*P,0],y=data[k*P:(k+1)*P,1],color=plot_c[k],label=plot_l[k])

for k in range(K):
    plt.figtext((average[k][0]+20)/40,(average[k][1]+20)/40/2+0.5,plot_l[k],size=10)

# ラベル
plt.xlabel('x',size=10)
plt.ylabel('y',size=10)

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)

# 散布図をプロットする
for i in range(N):
    ans = np.argmax( Q[i,:] )
    plt.scatter(data[i][0],data[i][1],color=plot_c[ans])

for k in range(K):
    plt.figtext((e_average[k][0]+20)/40,(e_average[k][1]+20)/40/2,plot_l[k],size=10)

# ラベル
plt.xlabel('x',size=10)
plt.ylabel('y',size=10)

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)

# 保存
plt.savefig("EM-result.png")
plt.show()
plt.clf()
