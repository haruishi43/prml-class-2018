#coding:utf-8

# エルマンネット（語系列課題学習）

import sys
import os
import numpy as np

# 学習係数
alpha = 0.01

# シグモイド関数
def Sigmoid( x ):
    return 1 / ( 1 + np.exp(-x) )

# シグモイド関数の微分
def Sigmoid_( x ):
    return ( 1-Sigmoid(x) ) * Sigmoid(x)

# ReLU関数
def ReLU( x ):
    return np.maximum( 0, x )

# ReLU関数の微分
def ReLU_( x ):
    return np.where( x > 0, 1, 0 )

# ソフトマックス関数
def Softmax( x ):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

# Tanh関数
def Tanh( x ):
    return ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )

# 出力層
class Outunit:
    def __init__(self, n, m):
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(n,m))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,m)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # ソフトマックス関数
        self.out = Softmax( self.u )

    def Error(self, t):
        # 誤差
        f_ = 1
        delta = ( self.out - t ) * f_

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        # 前の層に伝播する誤差
        self.error = np.dot(delta, self.w.T)

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename):
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)
        
    def Load(self, filename):
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work['w']
        self.b = work['b']

# 中間層
class Hunit:
    def __init__(self, n, m):
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(n,m))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,m)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値（恒等関数）
        self.out = Sigmoid( self.u )

    def Error(self, p_error):
        # 誤差
        f_ = ( 1 - self.out ) * self.out
        delta = p_error * f_

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        # 前の層に伝播する誤差
        self.error = np.dot(delta, self.w.T)

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename):
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)
        
    def Load(self, filename):
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work['w']
        self.b = work['b']

# 埋め込み層
class Embed:
    def __init__(self, n, m):
        # 重み
        self.w = np.random.uniform(-0.5,0.5,(n,m))

        # 閾値
        self.b = np.random.uniform(-0.5,0.5,m)

    def Propagation(self, x):
        self.x = x

        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b

        # 出力値（恒等関数）
        self.out = self.u

    def Error(self, p_error):
        # 誤差
        f_ = 1
        delta = p_error * f_

        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        # 前の層に伝播する誤差
        self.error = np.dot(delta, self.w.T)

    def Update_weight(self):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, filename):
        # 重み，閾値の保存
        np.savez(filename, w=self.w, b=self.b)
        
    def Load(self, filename):
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work['w']
        self.b = work['b']

# 単語id -> 単語
def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

# コーパスの読み込み
def Read_Corpus():
    
    # ファイルのオープン
    work = []
    with open('corpus-200.txt',"r",encoding="utf-8-sig") as f:
        for line in f:
            # 改行を削除
            work.append( line.rstrip("\n") )
            
    # 文字列に変換
    work1 = " ".join(work)

    # 空白で分割
    text = work1.strip().split()
    
    # 辞書の作成
    # word{"単語"} -> 単語id
    # sentence = [ 単語id , 単語id , ・・・ , 単語id ]
    word = {}
    sentence = np.ndarray((len(text),), dtype=np.int32)
    for i, w in enumerate(text):
        if w not in word:
            word[w] = len(word)
        sentence[i] = word[w]
    return word , sentence

# 学習
def Train():

    # エポック数
    EPOCH = 100
    for e in range(EPOCH):
        # 文脈層の初期化
        context = np.zeros( (1,hunit_num) )

        # 誤差二乗和
        error = 0.0
        for s in range(len(sentence)):
            
            if sentence[s] == eos_id: 
                # 文脈層の初期化
                context = np.zeros( (1,hunit_num) )
                continue
            
            # 入力層への入力（one-hot vector）
            input_data = np.zeros( (1,feature) )
            input_data[0][ sentence[s] ] = 1

            # 埋め込み層への入力
            embed.Propagation( input_data )

            # 埋め込みと文脈層の値を結合
            hunit_in = np.hstack((embed.out,context))

            # 中間層への入力
            hunit.Propagation( hunit_in )

            # 現在の中間層の値を次の文脈層の値とする
            context = hunit.out

            # 出力層の入力
            outunit.Propagation( hunit.out )

            # 教師信号（one-hot vector）
            teach = np.zeros( (1,feature) )
            teach[0][ sentence[s+1] ] = 1

            # 誤差の計算
            outunit.Error( teach )
            hunit.Error( outunit.error )
            work = np.reshape( hunit.error[0,0:embed_num] , (1,embed_num ) )
            embed.Error( work )
            
            # 重みの修正
            outunit.Update_weight()
            hunit.Update_weight()
            embed.Update_weight()

            # 誤差二乗和の計算
            error += np.dot( ( outunit.out[0] - teach[0] ) , ( outunit.out[0] - teach[0] ) )
        print( e , "->" , error )

    # 重みの保存
    outunit.Save( "dat/Elman-out.npz" )
    hunit.Save( "dat/Elman-hunit.npz" )
    embed.Save( "dat/Elman-embed.npz" )

# 予測
def Predict():

    # 重みのロード
    outunit.Load( "dat/Elman-out.npz" )
    hunit.Load( "dat/Elman-hunit.npz" )
    embed.Load( "dat/Elman-embed.npz" )

    # 文脈層の初期化
    context = np.zeros( (1,hunit_num) )
    
    for s in range(len(sentence)):
        
        if sentence[s] == eos_id:
            print( "------" )

            # 文脈層の初期化
            context = np.zeros( (1,hunit_num) )
            continue
        
        # 入力層への入力
        input_data = np.zeros( (1,feature) )
        input_data[0][ sentence[s] ] = 1

        # 埋め込み層への入力
        embed.Propagation( input_data )

        # 埋め込みと文脈層の値を結合
        hunit_in = np.hstack((embed.out,context))

        # 中間層への入力
        hunit.Propagation( hunit_in )

        # 現在の中間層の値を次の文脈層の値とする
        context = hunit.out

        # 出力層への入力
        outunit.Propagation( hunit.out )

        # 予測
        predict = np.argmax( outunit.out[0] )

        # 予測結果の出力
        print( get_key_from_value(word, sentence[s] ) , "->" , get_key_from_value(word, sentence[s+1] ) , "[" , get_key_from_value(word, predict) + " ]" )

if __name__ == '__main__':

    # コーパスの読み込み
    word , sentence = Read_Corpus()
    eos_id = word[ '<eos>' ]
    print( eos_id )
    
    # 埋め込み層の個数
    embed_num = 100

    # 中間層の個数
    hunit_num = 100

    # 特徴数
    feature = len( word )

    # 埋め込み層，中間層のコンストラクター
    embed = Embed( feature , embed_num )
    hunit = Hunit( embed_num+hunit_num , hunit_num  )

    # 出力層のコンストラクター
    outunit = Outunit( hunit_num , feature )

    argvs = sys.argv

    # 引数がtの場合
    if argvs[1] == "t":

        # 学習
        Train()

    elif argvs[1] == "p":

        # 予測
        Predict()

        
        
