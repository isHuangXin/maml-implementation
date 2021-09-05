import torch
import numpy as np
from torch.nn import functional as F
from torch import nn

print("hx-maml-pytorch")
print(torch.__version__)

# 关于lambda的的使用
update_lr = 0.1
grad = [1, 2, 3]
fast_weights = [4, 5, 6]
print(list(map(lambda p: p[1] - update_lr * p[0], zip(grad, fast_weights))))
print("done!")

# numpy实现矩阵的点乘和叉乘
#  a = [ [1, 2],
#        [3, 4] ]
#
#  b = [ [5, 6],
#        [7, 8] ]
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"矩阵a和矩阵b点乘:\n{a * b}")
print(f"矩阵a和矩阵b点乘:\n{np.dot(a, b)}")
print("done!")


# 关于torch.eq()的使用
print(torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]])))
print(torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]])).sum())
print(torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]])).sum().item())
print("done!")

# 关于列表前加*号的使用
param = [1, 2, 3, 4, 5, 6]
print(param[:4])
print(*param[:4])
print(torch.ones(*param[:3]))
print(torch.ones(*param[:3]).shape)
print("done!")

# 关于F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])的使用
inputs = torch.randn(1, 4, 5, 5)
print(inputs)
weights = torch.randn(4, 8, 3, 3)
print(weights)
result = F.conv_transpose2d(inputs, weights, padding=1)
print(result)

print(f"inputs.shape: {inputs.shape}")
print(f"weights.shape: {weights.shape}")
print(f"result.shape: {result.shape}")
print("done!")

# 自己测试PyTorch里的F.conv_transpose2d反卷积
"""
首先: 
    1.卷积公式为 output=[(input-kernel + 2*padding) / stride] + 1, 其中[]为向下取整
    2.反卷积可以看成是卷积的逆过程,反卷积为 output' = (input' - 1) * stride + kernel - 2*padding
    3.当stride为偶数时: 
"""
# 测试用例1:  当stride为偶数时
X = torch.Tensor([[[[1, 2],
                    [3, 4]]]])
C = torch.Tensor([[[[5, 6],
                    [7, 8]]]])
Y = F.conv_transpose2d(X, C, stride=2, padding=0)
print(f"X.shape: {X.shape}")
print(f"C.shape: {C.shape}")
print(f"Y.shape: {Y.shape}")
print(f"Y: {Y}")
print("测试用例1 done!")

# 测试用例2: 当stride为奇数时
X = torch.Tensor([[[[1, 2],
                    [3, 4]]]])
C = torch.Tensor([[[[5, 6],
                    [7, 8]]]])
Y = F.conv_transpose2d(X, C, stride=1, padding=0)
print(f"X.shape: {X.shape}")
print(f"C.shape: {C.shape}")
print(f"Y.shape: {Y.shape}")
print(f"Y: {Y}")
print("测试用例2 done!")

