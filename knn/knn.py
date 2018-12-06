import sys
import os
import argparse

import numpy as np
from PIL import Image


def get_train_img(args):

    train_num = args.train_num
    train_img = np.zeros((10,train_num,28,28), dtype=np.float32)

    train_root = os.path.join(args.data, 'train')
    assert os.path.exists(train_root), f'ERROR: cannot find training data at {train_root}'

    # loading MNIST training images
    for i in range(10):
        for j in range(1,train_num+1):
            train_file = os.path.join(train_root, f'{str(i)}', f'{str(i)}_{str(j)}.jpg')
            train_img[i][j-1] = np.asarray(Image.open(train_file).convert('L')).astype(np.float32)

    return train_img


def k_nearest(args, train_img):

    # variables
    k = args.k

    # initialize confusion matrix
    result = np.zeros((10,10), dtype=np.int32)

    test_root = os.path.join(args.data, 'test')
    assert os.path.exists(test_root), f'ERROR: cannot find testing data at {test_root}'

    for i in range(10):
        for j in range(1,101):
           
            # test data
            pat_file = os.path.join(test_root, f'{str(i)}', f'{str(i)}_{str(j)}.jpg')
            pat_img = np.asarray(Image.open(pat_file).convert('L')).astype(np.float32)
            
            # feature vector
            p = pat_img.flatten()
            
            #FIXME: bad!!!
            ans = 0
            dist_arr = [float('inf')] * k
            num_arr = [float('inf')] * k

            # collect k closest numbers:
            for num in range(10):
                for l in range(1, args.train_num+1):

                    # feature vector
                    t = train_img[num][l-1].flatten()

                    # ssd
                    dist = np.dot( (t-p).T , (t-p) )

                    for ind, d in enumerate(dist_arr):
                        if d > dist:
                            # insert to list
                            dist_arr.insert(ind, dist)
                            num_arr.insert(ind, num)

                            # pop last item
                            del dist_arr[-1]
                            del num_arr[-1]
                            break
            
            # find number that has the most votes
            ans_arr = [0]*10
            for n in range(10):
                count = num_arr.count(n)
                ans_arr[n] = count
            
            #print(ans_arr)
            max_value = max(ans_arr)
            ans = ans_arr.index(max_value)

            result[i][ans] +=1
            print(f'{i} {j}: {ans}')

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='K-Nearest')
    parser.add_argument('--k', type=int, default=1,
                        help='k value')
    parser.add_argument('--data', default='../mnist/',
                        help='root to data path (mnist)')
    parser.add_argument('--train-num', type=int, default=100,
                        help='number of training data')
    args = parser.parse_args()

    assert args.k > 0, 'k should be greater than 0'
    assert args.train_num > 0, 'train_num should be greater than 0'

    # number of training data used for comparison
    train_img = get_train_img(args)

    # initialize confusion matrix
    result = k_nearest(args, train_img)

    print("\n [ Confusion Matrix ]")
    print( result )
    print("\n Number of correct predictions: " , np.trace(result))