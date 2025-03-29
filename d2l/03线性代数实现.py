import torch

x=torch.tensor([3.0])
y=torch.tensor([2.0])
print(x+y,x*y,x/y,x**y)#加减乘方

x1=torch.arange(4)#通过索引来访问张量的元素
len(x1),x1.shape,x1.size(),x1[-1],x1[1:3]
#len返回张量的长度，shape返回张量的形状，size返回张量的长度，索引从0开始
A=torch.arange(20).reshape(5,4)
A.T#矩阵A的转置
B=torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B==B.T)#判断是否为对称矩阵
A=torch.arange(20,dtype=torch.float32).reshape(5,4)
B=A.clone()#分配新内存，复制A的值
print(A,A+B)

#两个矩阵按元素乘法成为哈达玛积：A*B
x=torch.arange(4,dtype=torch.float32)
print(x,x.sum())#求和, sum()返回一个标量，表示张量的所有元素的和
A.mean(axis=0)#沿轴0（行）求均值
A.sum(axis=1)/A.shape[1]#沿轴1（列）求均值

#非降维求和
A.sum(axis=1,keepdims=True)#keepdims=True保持轴数不变

#通过广播将A除以A的和
A/A.sum(axis=1,keepdims=True)
#点积

torch.dot(x,y)
x1=torch.arange(4)
y1=torch.ones(4,dtype=torch.float32)
print(x1,y1,torch.dot(x1,y1))#点积
print(torch.sum(x1*y1))#点积:按元素乘法，然后求和

#torch.mv(A,x)#矩阵A和向量x的矩阵-向量积
#torch.mm(A,B)#矩阵A和B的矩阵-矩阵乘法
#torch.norm(x)#L2范数
#torch.norm(x).item()#返回标量
#torch.abs(x)#绝对值
#torch.abs(u).sum()#L1范数：绝对值求和
#torch.norm(torch.ones(3,4))#F范数：矩阵元素的平方和再开根号
