"""
main.py
=======
High-level entry points for surface and point-cloud matching via LDDMM.
 
Typical usage
-------------
    from lddmm_toolbox import MatchSurface, MatchPoints
 
    result = MatchSurface(VS, FS, VT, FT, p0, sigmaV=20, sigmaW=20)
    result = MatchPoints(VS, VT, sigmaW=20)
 
Each function returns an LDDMM_def object that exposes .shoot() and .flow().
"""

import torch
from .config import torchdtype, torchdeviceId
from .kernels import GaussLinKernel, GaussExpKernel, EnergyKernel, GaussKernel, GaussCurrentsKernel, GaussKernelGenred, EnergyKernelGenred
from .losses import lossVarifoldSurf, lossMeas
from .optimization import LDDMM_def, LDDMM_Optimize

# ──────────────────────────────────────────────
# Surface matching
# ──────────────────────────────────────────────

def MatchSurface(VS, FS, VT, FT, p0, 
                 sigmaV=20, sigmaW=20, 
                 method=LDDMM_Optimize, 
                 niter=10, gamma=0.1, nt=10):
    """
    Match source surface (VS, FS) to target surface (VT, FT) using the
    varifold (GaussLin) data attachment.
 
    Parameters
    ----------
    VS, VT  : (N, 3) vertex arrays
    FS, FT  : (M, 3) face-connectivity arrays
    p0      : initial momentum tensor
    sigmaV  : kernel bandwidth for the varifold loss
    sigmaW  : kernel bandwidth for the velocity field
    method  : optimisation routine (default: LDDMM_Optimize)
    niter   : L-BFGS iterations
    gamma   : regularisation weight
    nt      : number of integration time steps
    """
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
    FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)

    dataloss = lossVarifoldSurf(FS, VT, FT, GaussLinKernel(sigma=sigmaV))

    return method(p0, q0, dataloss, sigmaW, niter=niter, gamma=gamma, nt=nt)    

def MatchSurfaceExp(VS, FS, VT, FT, p0, 
                    sigmaV=20, sigmaW=20, 
                    method=LDDMM_Optimize, 
                    niter=10, gamma=0.1, nt=10):
    """
    Match source surface to target using the exponential varifold (GaussExp) kernel.
    """
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
    FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)

    dataloss = lossVarifoldSurf(FS, VT, FT, GaussExpKernel(sigma=sigmaV))

    return method(p0, q0, dataloss, sigmaW, niter=niter, gamma=gamma, nt=nt)   

def MatchSurfaceCurrents(VS, FS, VT, FT, p0, 
                         sigmaV=20, sigmaW=20, 
                         method=LDDMM_Optimize, 
                         niter=10, gamma=0.1, nt=10):
    """
    Match source surface to target using the currents (GaussCurrents) kernel.
    """
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
    FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)

    dataloss = lossVarifoldSurf(FS, VT, FT, GaussCurrentsKernel(sigma=sigmaV))

    return method(p0, q0, dataloss, sigmaW, niter=niter, gamma=gamma, nt=nt) 

# ──────────────────────────────────────────────
# Point-cloud matching
# ──────────────────────────────────────────────

def MatchPoints(VS, VT, 
                sigmaW=20, 
                method=LDDMM_Optimize, 
                niter=10, gamma=0.1, nt=10):     
    """
    Match source point cloud VS to target VT using the energy-distance kernel.
 
    Parameters
    ----------
    VS, VT  : array-like (N, 3) – source and target point clouds
    sigmaW  : velocity-kernel bandwidth
    method  : optimisation routine (default: LDDMM_Optimize)
    """   
    VS = torch.tensor(VS, dtype=torchdtype, device=torchdeviceId)
    VT = torch.tensor(VT, dtype=torchdtype, device=torchdeviceId)
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)

    dataloss = lossMeas(VT, EnergyKernelGenred())
    p0 = torch.zeros(q0.shape, device=torchdeviceId, requires_grad=True)

    return method(p0, q0, dataloss, sigmaW, niter=niter, gamma=gamma, nt=nt)

def MatchPointsGauss(VS, VT, 
                     sigmaV=5, sigmaW=20, 
                     method=LDDMM_Optimize, 
                     niter=10, gamma=0.1, nt=10):  
    """
    Point-cloud matching using the Gaussian kernel.
    """      
    VS = torch.tensor(VS, dtype=torchdtype, device=torchdeviceId)
    VT = torch.tensor(VT, dtype=torchdtype, device=torchdeviceId)
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)

    dataloss = lossMeas(VT, GaussKernelGenred())
    p0 = torch.zeros(q0.shape, device=torchdeviceId, requires_grad=True)

    return method(p0, q0, dataloss, sigmaW, niter=niter, gamma=gamma, nt=nt)