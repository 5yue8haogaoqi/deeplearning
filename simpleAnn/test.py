# -*- encoding:utf-8 -*-

import numpy as np
from layers import  *

with open('data/train.npy', 'rb') as f:
    data = np.load(f)
    print data.shape

    print data[0].shape
    print data[1].shape

    print data[0][0][0]
    print type(data[1])


a = np.array([np.array([1,1,2]),np.array([1,1,2])])
print a[0]