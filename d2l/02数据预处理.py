#L2. 数据预处理
import pandas as pd
import numpy as np
import torch
import os

os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file,'w')as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data=pd.read_csv(data_file)

#为了处理缺失数据，典型的方法包括插值和删除，这里我们将考虑插值
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]#iloc索引
inputs=inputs.fillna(inputs.mean())#用均值填充缺失值
print(inputs)
#对于inputs中的类别值或离散值，我们将“NaN”视为一个类别
# inputs=pd.get_dummies(inputs,dummy_na=True)
# print(inputs)
