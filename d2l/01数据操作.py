import torch
import numpy as np
import os
import pandas as pd
import sys
#L1. 数据操作
x=torch.arange(12)
x=x.reshape(3,4)
#x.shape形状
#x.numel数量
#torch.zeros(2,3,4)
#torch.ones(2,3,4)
X=torch.arange(12,dtype=torch.float32).reshape(3,4)
Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((X,Y),dim=0)
torch.cat((X,Y),dim=1)
#X==Y通过 逻辑运算符 构建二元张量相等处取true
X.sum()#所有元素求和
#形状不同时，广播机制
A=torch.arange(3).reshape(3,1)
B=torch.arange(2).reshape(1,2)
A+B
#索引
#X[-1]-1表示选择最后一个元素
X[1,2]=9
X[0:2,:]=12#赋值  
#内存开销
before=id(Y)#case1
Y=Y+X
id(Y)==before#返回False因为内存地址改变
#case2
Z=torch.zeros_like(Y)
before=id(Z)
Z[:]=X+Y
id(Z)==before#返回True因为Z的内存地址不变
#即可以通过X[:]=X+Y或者X+=Y来减少内存开销
#转换为numpy
A=X.numpy()
B=torch.tensor(A)
#type(A):numpy.ndarry
#type(B):torch.Tensor
# #转换为其他python对象
A=X.numpy()
B=torch.tensor(A)
#X+=Y
#X.add_(Y)
#X=X+Y
#X=torch.tensor([1.0,2,4,8])
#Y=torch.tensor([2,2,2,2])
#X/Y
#torch.exp(X)
#torch.stack((X,Y))
#torch.norm(X)

