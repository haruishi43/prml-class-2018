# -*- coding: utf-8 -*-

# テンプレートマッチング(numpy）
#
# > pip install numpy
# > pip install pillow

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# 探索画像の読み込み
search_file = "search.jpg"
search_img = Image.open(search_file).convert('RGB')

# numpyに変換(Y,X,channel)
search = np.asarray(search_img).astype(np.float32)
print( search.shape )
search_width = search.shape[1]
search_height = search.shape[0]
print( search_width , search_height )

# テンプレートの読み込み
template_file = "template.jpg"
template_img = Image.open(template_file).convert('RGB')

# numpyに変換(Y,X,channel)
template = np.asarray(template_img).astype(np.float32)
print( template.shape )
template_width = template.shape[1]
template_height = template.shape[0]
print( template_width , template_height )

# テンプレートマッチング
min_val = float('inf')
ans_x = 0
ans_y = 0
for y in range(0,search_height-template_height,5):
    print(y)
    for x in range(0,search_width-template_width,5):
        sum = 0.0
        # SSDの計算

        '''
        # numpyを利用しない場合
        for i in range(3):
            for yy in range(template_height):
                for xx in range(template_width):
                    sum += ( search[y+yy][x+xx][i] - template[yy][xx][i] ) * ( search[y+yy][x+xx][i] - template[yy][xx][i] )
        '''
        
        s = search[y:y+template_height,x:x+template_width,0:3].flatten()
        t = template.flatten()
        sum = np.dot( (t-s).T , (t-s) )
        
        # 最小値を記憶
        if min_val > sum:
            min_val = sum
            ans_x = x
            ans_y = y

# 枠の描画
draw = ImageDraw.Draw(search_img)
draw.rectangle((ans_x, ans_y, ans_x+template_width, ans_y+template_height), outline=(255,0,0))

# 結果の保存
search_img.save("result.jpg")

# 結果の表示
search_img.show()
                    



