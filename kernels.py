import torch
from pykeops.torch import Vi, Vj, Genred
import os
import numpy as np

os.environ['PYKEOPS_VERBOSE'] = '0'

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'

def SumKernel(*kernels):
    def K(*args):
        return sum(k(*args) for k in kernels)
    return K

def SumOfGaussKernel(sigmas):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    D2 = x.sqdist(y)
    K = 0
    for sigma in sigmas:
        gamma = 1 / (sigma * sigma)
        K += (-D2 * gamma).exp()

    return (K * b).sum_reduction(axis=1)

def GaussExpKernel(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    gamma_s = 2 / 0.5**2
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (gamma_s*(u * v).sum()).exp()

    return (K * b).sum_reduction(axis=1)

def GaussKernel(sigma):
   x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
   gamma = 1 / (sigma * sigma)
   D2 = x.sqdist(y)
   K = (-D2 * gamma).exp()
   return (K * b).sum_reduction(axis=1)

def GaussKernelGenred(): 
    return Genred(
        f"Exp(-IntInv(25)*SqDist(X,Y))",
        [
            "X = Vi(3)",   # x: shape (N, 3)
            "Y = Vj(3)",   # y: shape (M, 3)
        ],
        reduction_op="Sum",
        axis=1,
    )

def EnergyKernel():
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    D = x.sqdist(y).sqrt()
    return (-D).sum_reduction(axis=1)

def EnergyKernelGenred():
    return Genred(
        "-Sqrt(SqDist(X,Y))",
        [
            "X = Vi(3)",   # corresponds to argument 0
            "Y = Vj(3)",   # corresponds to argument 1
        ],
        reduction_op="Sum",
        axis=1,
    )    

def GaussLinKernel(sigma):
    oos2 = 1/sigma**2
    
    def K(x,y,u,v,b):
        # calculates the similarity based on the Euclidean distance between points  
        # x and y in a high-dimensional space
        Kxy = torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))
        # computes the squared dot product between corresponding vectors u and v 
        # associated with points x and y, respectively. This part captures the 
        # similarity in directions (e.g., surface normals) at the points
        Sxy = torch.sum(u[:,None,:]*v[None,:,:],dim=2)**2
        
        # composite kernel
        return (Kxy*Sxy)@b
    
    return K

def GaussCurrentsKernel(sigma):
    oos2 = 1/sigma**2
    
    def K(x,y,u,v,b):
        # calculates the similarity based on the Euclidean distance between points  
        # x and y in a high-dimensional space
        Kxy = torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))
        # computes the dot product between corresponding vectors u and v 
        # associated with points x and y, respectively. This part captures the 
        # similarity in directions (e.g., surface normals) at the points
        Sxy = torch.sum(u[:,None,:]*v[None,:,:],dim=2)
        
        # composite kernel
        return (Kxy*Sxy)@b
    
    return K