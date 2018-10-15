# -*- coding: utf-8 -*-

# ステレオマッチング
#
# > pip install numpy
# > pip install pillow

import sys
import os
import numpy as np
from PIL import Image

# 左画像の読み込み
left_file = "left1.jpg"
left_img = Image.open(left_file).convert('L')

# numpyに変換 -> (Y,X,channel)
left = np.asarray(left_img).astype(np.float32)
print( left.shape )
left_width = left.shape[1]
left_height = left.shape[0]
print( left_width , left_height )

# 右画像の読み込み
right_file = "right1.jpg"
right_img = Image.open(right_file).convert('L')

# numpyに変換 -> (Y,X,channel)
right = np.asarray(right_img).astype(np.float32)
print( right.shape )
right_width = right.shape[1]
right_height = right.shape[0]
print( right_width , right_height )

# 視差マップ
result = np.ones((left_height, left_width))

# 探索領域の大きさ
search_size = 21 // 2

# テンプレートの大きさ
template_size = 21 // 2 
for y in range(0,left_height,1):
    print(y)
    for x in range(0,left_width,1):

        ans_x = 0
        ans_y = 0
        min_val = float( 'inf' )

        # 左画像の座標(x,y）を中心としたテンプレートに類似した領域を
        # 右画像から検索し，その座標（ans_x,ans_y）を求めなさい
                
        result[y,x]=(x-ans_x)*(x-ans_x)+(y-ans_y)*(y-ans_y)

min = np.min( result )
max = np.max( result )
result = (result-min)/(max-min) * 255

result_img = Image.fromarray(np.uint8(result))
result_img.save('result.jpg')
            
        
                    



