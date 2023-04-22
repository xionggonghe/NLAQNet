import torch
import torch.nn as nn
from  quaternion_ops import *
import math
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import numpy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QuaternionSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        out_inp = quaternion_sigmoid(inp)
        return out_inp
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        out_inp = quaternion_sigmoidBackward(inp)
        return out_inp


class QSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = QuaternionSigmoid.apply(x)
        return out


class QuaternionRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        out_inp = quaternion_relu(inp)
        return out_inp
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        out_inp = quaternion_reluBackward(inp)
        return out_inp


class QRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = QuaternionRelu.apply(x)
        return out



class Qmaxpool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, kernel_size, stride):
        out, ker = quaternion_pool(inp, kernel_size, stride)
        ctx.save_for_backward(inp, ker)
        return out

    @staticmethod
    def backward(ctx, grad):
        inp, ker, = ctx.saved_tensors
        out_inp = quaternion_poolBackward(inp, ker, grad)
        # o1 = out_inp.numpy()
        return out_inp, None, None

class Qmaxpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride


    def forward(self, x):
        out = Qmaxpool.apply(x, self.kernel_size, self.stride)
        return out



if __name__=='__main__':
    # print("start !! ")
    x = torch.randn([8, 16, 8, 8])
    # x = torch.arange(start=0, end=8*4*8*8, step=1, out=None, dtype=float)
    x = x.reshape(8, 16, 8, 8)
    x.requires_grad = True
    x1 = x.detach().numpy()
    # print("x: ", x.shape)
    pool = Qmaxpool2d((2, 2), 2)
    y = pool(x)
    y.backward(y.clone().detach())
    # y = x.reshape(8, 4, 8, 8)
    # grad = x.grad_fn
    # print("x grad: ", x.grad_fn)
    # print("y grad: ", y.grad_fn)
    y1 = y.detach().numpy()
    # grad1 = grad.numpy()
    # print("y: ", y.shape)
    # print("grad: ", grad1.shape)




# QRelu = QRelu()
#
#
# x = torch.randn([3,2,3,4],  requires_grad=True)
# print("x:", x)
# out = 2*x+1
# out = QRelu(out)
#
# print("out: ", out)
# out.backward(torch.ones_like(x), retain_graph=True)
# print(f"\n Grad call\n{x.grad}")
# out.backward(torch.ones_like(x), retain_graph=True)
# print(f"\nSecond call\n{x.grad}")
# print("x", x)
# print("x.grad:", x.grad)




