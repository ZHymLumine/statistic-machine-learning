import time
import numpy as np
from collections import defaultdict


featureNum = 10
dataList = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
dataList = np.mat(dataList)
x = [0 for i in range(dataList.shape[0])]


if __name__ == '__main__':
    print(dataList)
    print(x)
    for i in range(dataList.shape[0]):
        print(dataList[i, 0])