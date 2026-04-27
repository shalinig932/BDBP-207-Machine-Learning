#Write a program to partition a dataset (simulated data for regression)  into two parts,
# based on a feature (BP) and for a threshold,
# t = 80. Generate additional two partitioned datasets based on different threshold values of t = [78, 82].

import numpy as np
import pandas as pd

#--------------------
#simulate dataset
#--------------------
np.random.seed(42)
n=100
data=pd.DataFrame({'BP':np.random.randint(1,100,n),
                  'Age':np.random.randint(1,100,n),
                   'Cholestrol':np.random.randint(1,100,n),
                   'Target':np.random.randn(n) * 10 + 15})
print('original dataset:',data)
print(data.head())

#-------------------
#partition
#-------------------

def partition(df,threshold):
    left=df[df['BP']<=threshold]
    right=df[df['BP']>threshold]
    print('left partition:', left)
    print('right partition:', right)
    return left,right


#---------------------
#threshold setting
#---------------------


left_80,right_80=partition(data,80)
left_78,right_78=partition(data,78)
left_82,right_82=partition(data,82)
