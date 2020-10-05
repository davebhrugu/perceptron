import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

df = pd.read_csv("C:\\Users\\daveb\\Downloads\\heart.csv")
df.head(14)

label = df['target']
df = df.drop(['target'],axis=1)

class ptron:
    def __init__(self,threshold,no_inputs,learning_rate):
        self.weights = np.random.randn(no_inputs-1)
        self.threshold = threshold
        self.learning_rate = learning_rate
        print(self.weights) 
    
    def prd (self,inputs):
        z = np.dot(inputs,self.weights)
        
        if z>0:
            return 1
        else:
            return -1
    
    def trn (self,inputs,labels):
        for j in range(self.threshold):
            for i in range(len(labels)):
                delta_w =  self.learning_rate * (labels[i] - self.prd(inputs[i])) * inputs[i]
                self.weights = self.weights + delta_w
        print('Training Done')


c = ptron(1000,14,0.05)
c.trn(df.values,label)