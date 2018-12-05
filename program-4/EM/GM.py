# -*- coding: utf-8 -*-

# 混合正規分布のグラフ表示

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# 合成する正規分布の個数
K = 3

# データ数
N = 100
line = np.linspace(-10, 10, N)

# 平均，標準偏差，重みを格納する配列
average = []
sigma = []
lamda =[]

# 平均，標準偏差，重みの設定
for k in range(K):
    a = (np.random.rand() - 0.5) * 10.0
    s = np.random.rand() * 3.0
    l = np.random.randint(1,10)
    average = np.append( average , a )
    sigma = np.append( sigma , s )
    lamda = np.append( lamda , l )

sum_l = np.sum( lamda )
for k in range(K):
    lamda[k] = lamda[k] / sum_l
print( average , sigma , lamda )

# 正規分布を格納する配列
gaussian = np.zeros((K, N),dtype=np.float64)

# 混合正規分布を格納する配列
gaussian_mixture = np.zeros(N,dtype=np.float64)
for k in range(K):
    for n in range(N):
        # 正規分布
        gaussian[k][n] = st.norm.pdf(line[n], average[k], sigma[k])

        # 混合正規分布
        gaussian_mixture[n] += lamda[k] * gaussian[k][n]

# K個の正規分布の表示
plt.figure()
plt.subplot(2,1,1)
for k in range(K):
    plot_t = "lamda={:6.4f}".format( lamda[k] ) 
    plt.plot(line, gaussian[k],label=plot_t)
    plt.legend()
    plt.title("Gaussian")

# 混合正規分布の表示
plt.subplot(2,1,2)
plt.plot(line, gaussian_mixture )
plt.title("Gaussian Mixture")
plt.tight_layout() 
plt.show()
plt.clf()
