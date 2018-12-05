# -*- coding: utf-8 -*-

# 単純ベイズ決定則

import sys
import os
import numpy as np

# データ
data = []

answer = [ "yes" , "no" ]

# 事前確率
pp = { "yes":0 , "no":0 }

# 条件付き確率
cp = { "晴れ｜yes":0 , "晴れ｜no":0 ,
      "曇り｜yes":0 , "曇り｜no":0 ,
      "雨｜yes":0 , "雨｜no":0 ,
      "暑い｜yes":0 , "暑い｜no":0 ,
      "適温｜yes":0 , "適温｜no":0 ,
      "寒い｜yes":0 , "寒い｜no":0 ,
      "高い｜yes":0 , "高い｜no":0 ,
      "適当｜yes":0 , "適当｜no":0 ,
      "低い｜yes":0 , "低い｜no":0 ,
      "あり｜yes":0 , "あり｜no":0 ,
      "なし｜yes":0 , "なし｜no":0
      }
      
# 学習データの読み込み
f = open( "Bayes-train.csv" , "r" )
count = 0
for line in f:
    # 先頭行を無視する
    if count == 0:
        count += 1
        continue
    
    data.append( line.rstrip().split( "," ) )
    count += 1
f.close()

# 天気,気温,湿度,講義,頭痛
# "yes" "no" の頻度
for s in data:
    pp[s[4]] +=1

# 条件付き確率
for s in data:
    for i in range(4):
        work = s[i] + "｜" +s[4]
        cp[ work ] += 1 / pp[ s[4] ]

    '''
    # 天気
    work = s[0] + "｜" +s[4]
    cp[ work ] += 1 / pp[ s[4] ]
    
    # 気温
    work = s[1] + "｜" +s[4]
    cp[ work ] += 1 / pp[ s[4] ]

    # 湿度
    work = s[2] + "｜" +s[4]
    cp[ work ] += 1 / pp[ s[4] ]

    # 講義
    work = s[3] + "｜" +s[4]
    cp[ work ] += 1 / pp[ s[4] ]
    '''
    
# 事前確率
for i in pp.keys():
    pp[ i ] = pp[i] / len(data)

print( " [ 事前確率 ]" )
for i in pp.keys():
    print( "{:^10} : {:6.4f}".format( i , pp[i] ) )

# 条件付き確率
print( "\n [ 条件付き確率 ]" )
for i in cp.keys():
    print( "{:^10} : {:6.4f}".format( i , cp[i] ) )

# テストデータの読み込み
f = open( "Bayes-test.csv" , "r" )
count = 0
print( "\n [ 事後確率 ]" )
for line in f:
    # 先頭行を無視する
    if count == 0:
        count += 1
        continue

    work = line.rstrip().split( "," )

    # 事後確率を求める
    predict = np.ones( 2 , np.float32 )
    for i in range(2):
        for j in range(4):
            temp = work[j] + "｜" + answer[i]
            predict[i] = predict[i] * cp[ temp ] 
        predict[i] = predict[i] * pp[answer[i]]

    # 事後確率の出力
    s = ""
    for i in range(2):
        s = answer[ i ] + "｜"
        for j in range(4):
            s += work[j] + " "
        print( "{:>18}: {:10.8f}".format( s , predict[ i ] ) )

    # 事後確率最大の結果の出力
    ans = np.argmax( predict )
    print( answer[ ans ] )
    count+=1
f.close()
