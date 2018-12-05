# -*- coding: utf-8 -*-

# EMアルゴリズム（混合正規分布(一変量））

import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st

# 正規分布の個数
K = 3

# 正規分布一つあたりのデータ数
P = 500

# 全データ数
N = K*P

# 繰り返し回数
LOOP = 10

# データ，平均，標準偏差
data = []
average = []
sigma = []

# 平均，標準偏差の設定
for k in range(K):
    a = (np.random.rand() - 0.5) * 10.0
    s = np.random.rand() * 3.0
    average = np.append( average , a )
    sigma = np.append( sigma , s )
print( average , sigma )

# K個の正規分布を生成
for k in range(K):
    data = np.append( data , np.random.normal(average[k],sigma[k],P) ) 

# ヒストグラムの表示
plt.figure()
plt.hist(data,bins=100,normed=True)
plt.title("K-Gaussian")
plt.xlim([-10,10])
plt.show()
plt.clf()

# Q値，平均値，標準偏差の初期化
Q = np.zeros((N, K),dtype=np.float64)
e_average = np.zeros(K,dtype=np.float64)
e_sigma = np.zeros(K,dtype=np.float64)
e_lamda = np.zeros(K,dtype=np.float64)
for k in range(K):
    e_average[k] = random.choice(data)
    e_sigma[k] = np.sqrt(np.var(data))
    e_lamda[k] = 1/K
 
# EMアルゴリズム
for loop in range(LOOP):

    # E-step
    for i in range(N):
        temp = np.zeros(K,dtype=np.float64)
        sum_t = 0
        for k in range(K):
            temp[k] = st.norm.pdf( data[i] , e_average[k] , e_sigma[k] )
            sum_t += ( temp[k] * e_lamda[k] )

        for k in range(K):
            Q[i][k] = ( temp[k] * e_lamda[k] ) / sum_t

    # M-step
    # 重みの更新
    for k in range(K):
            sum_Q = 0
            for i in range(N):
                sum_Q += Q[i][k]
            e_lamda[k] = sum_Q / N
    print( loop , ":lamda -> " , e_lamda )

    # 平均値の更新
    new_average = np.zeros(K,dtype=np.float32)
    for k in range(K):
        sum_q = 0
        sum_q1 = 0
        for i in range(N):
            sum_q += Q[i][k] * data[i]
            sum_q1 += Q[i][k]
        new_average[k] = sum_q / sum_q1
    print( loop , ":average -> " , new_average )

    # 標準偏差の更新
    new_sigma = np.zeros(K,dtype=np.float32)
    for k in range(K):
        sum_q = 0
        sum_q1 = 0
        for i in range(N):
            sum_q += Q[i][k] * (data[i]-e_average[k])**2
            sum_q1 += Q[i][k]
        new_sigma[k] = np.sqrt( sum_q / sum_q1 )
    print( loop , ":sigma -> " , new_sigma )
    print( " ----- " )

    # 平均値，標準偏差を更新
    e_average = new_average.copy()
    e_sigma = new_sigma.copy()

# 推定した正規分布
result_gaussian = np.zeros((K, N),dtype=np.float64)

# 推定した混合正規分布
result_gaussian_mixture = np.zeros(N,dtype=np.float64)

# グラフの表示
line = np.linspace(-10,10,N)

# 予測した混合正規分布の表示
for k in range(K):
    for i in range(N):
        result_gaussian[k][i] = st.norm.pdf(line[i], e_average[k], e_sigma[k])
        result_gaussian_mixture[i] +=e_lamda[k] *  result_gaussian[k][i]

# 元データの表示
plt.figure()
plt.hist(data,bins=100,normed=True)

# 混合正規分布の表示
plt.plot(line, result_gaussian_mixture )
plt.title("Gaussian Mixture")
plt.show()
plt.clf()
