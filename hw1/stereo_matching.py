import sys
import os
import numpy as np
from PIL import Image


def get_grayscale_image(img_path):
    img = Image.open(img_path).convert('L')
    numpy_img = np.asarray(img).astype(np.float32)
    return numpy_img


def save_image(img, img_name):
    img = Image.fromarray(np.uint8(img))
    img.save(img_name)


def stereo_matching(r_img, l_img, kernel_size, search_size):
    # variables
    l_width = l_img.shape[1]
    l_height = l_img.shape[0]
    r_width = r_img.shape[1]
    r_height = r_img.shape[0]
    half_kernel = int(kernel_size // 2)
    half_search = int(search_size // 2)

    # initialize result
    result = np.zeros((l_height, l_width))

    for y in range(half_kernel, l_height-half_kernel, 1):
        print(".", end="", flush=True)
        for x in range(half_kernel, l_width-half_kernel, 1):
            ans_x = 0
            ans_y = 0
            min_val = float('inf')

            # get template area
            t = l_img[y-half_kernel:y+half_kernel,
                     x-half_kernel:x+half_kernel]
            t = t.flatten()

            # template matching algorithm
            # assumption: patch is near the template...
            #FIXME: how can i make this faster?
            for y_ in range(y-half_search, y+half_search, 1):
                if y_-half_kernel < 0 or y_+half_kernel > r_height:
                    continue
                for x_ in range(x-half_search, x+half_search, 1):
                    if x_-half_kernel < 0 or x_+half_kernel  > r_width:
                        continue

                    s = r_img[y-half_kernel:y+half_kernel,
                              x_-half_kernel:x_+half_kernel]
                    s = s.flatten()
                    
                    # ssd
                    sum_ = np.dot((t - s).T , (t - s))

                    if min_val > sum_:
                        min_val = sum_
                        ans_x = x_
                        ans_y = y

            result[y,x] = (x-ans_x) * (x-ans_x) + (y-ans_y) * (y-ans_y)

    # post process results (normalize to image)
    min = np.min( result )
    max = np.max( result )
    result = (result-min)/(max-min) * 255

    return result


if __name__ == '__main__':

    l_file = "left.jpg"
    r_file = "right.jpg"

    l_img = get_grayscale_image(l_file)
    r_img = get_grayscale_image(r_file)

    template_size = 21
    search_size = 21
    
    # stereo matching
    result = stereo_matching(r_img, l_img, template_size, search_size)

    # save result
    save_image(result, 'result.jpg')
