#   -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
"""
this is a dataset for minist
there is two parts for minist data
u can use the part index when init
"""
class minstData:

    def __init__(self, test_number=2000, batch_size=100):
        self.data = pd.read_csv('../data/train.csv')
        self.test_number = test_number
        self.batch_size = batch_size
        self.pos = 0

        self.output = np.array(self.data["label"], dtype=np.float32).reshape(-1,1)
        self.input = np.array(self.data.iloc[:,1:],dtype=np.float32)
        """
        transform input by normalization
        transform output to onehotcode
        """
        self.input = pp.MinMaxScaler().fit_transform(self.input)
        self.output = pp.OneHotEncoder().fit_transform(self.output)
        self.test_input = self.input[:test_number]
        self.test_output = self.output[:test_number]
        self.train_input = self.input[test_number+1:]
        self.train_output = self.output[test_number+1:]
    """
    fetch the batch_size data
    batch_size is decided by the init param batch_size
    """
    def nex_batch(self):
        if ((self.pos + self.batch_size) <= self.train_input.shape[0]):
            return self.train_input[self.pos:self.pos+self.batch_size],self.train_output[self.pos:self.pos+self.batch_size]
        else:
            self.pos = 0
            self.nex_batch()


#minst = minstData()
#print (len(minst.nex_batch()))