import torch
from .config import torchdeviceId, torchdtype
from .kernels import GaussExpKernel, EnergyKernel, SumOfGaussKernel, GaussKernel
from .losses import LDDMMloss
from .shooting import Shooting, Flow

# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────

class LDDMM_def:
    """
    Stores the result of an LDDMM optimisation:
    initial momentum, initial positions, and velocity kernel.
    """
    def __init__(self, p0, q0, Kv):
        self.init_mom = p0
        self.init_pos = q0
        self.kernel = Kv

    def shoot(self, deltat=1.0, init_mom=None, nt=10):
        """Return the full geodesic trajectory."""
        mom = init_mom if init_mom is not None else self.init_mom

        return Shooting(mom, self.init_pos, self.kernel, deltat=deltat, nt=nt)
    
    def flow(self, x0, nt=10):
        """Transport arbitrary points x0 along the optimised geodesic."""
        x0 = torch.tensor(x0, dtype=torchdtype, device=torchdeviceId)

        return Flow(x0, self.init_mom, self.init_pos, self.kernel, nt=nt)
    
# ──────────────────────────────────────────────
# Generic L-BFGS optimiser wrapper
# ──────────────────────────────────────────────

def Optimize(loss, x, niter=10):
    optimizer = torch.optim.LBFGS([x])
    losses = []
   
    print('Performing optimization...')
    for i in range(niter):
        print("iteration ",i+1,"/",niter)

        def closure():
            optimizer.zero_grad()
            L = loss(x)
            losses.append(L.item())
            L.backward()

            return L
        
        optimizer.step(closure)
    
    print('Optimization complete\n')

    return losses

# ──────────────────────────────────────────────
# LDDMM optimisation entry points
# ──────────────────────────────────────────────

def LDDMM_Optimize(p0, q0,dataloss, sigma, niter=10, gamma=0.1, nt=10):    
    """
    LDDMM optimisation using a sum-of-Gaussians velocity kernel.
 
    Parameters
    ----------
    p0        : initial momentum  (requires_grad=True)
    q0        : initial positions (requires_grad=True)
    dataloss  : data attachment loss callable
    sigma     : kernel bandwidth (scalar or list)
    niter     : number of L-BFGS iterations
    gamma     : regularisation weight
    nt        : number of time steps
 
    Returns
    -------
    LDDMM_def containing the optimised deformation.
    """
    sigma = torch.tensor([sigma], dtype=torchdtype, device=torchdeviceId)
    Kv = SumOfGaussKernel(sigma)
    loss = LDDMMloss(Kv, dataloss, gamma=gamma, nt=nt)
    
    # Perform optimization
    Optimize(lambda p0 : loss(p0,q0), p0, niter=niter)
            
    return LDDMM_def(p0,q0,Kv)

def LDDMM_Optimize_points(p0, q0,dataloss,sigma, niter=10, gamma=0.1, nt=10):    
    """
    LDDMM optimisation using the Gaussian velocity kernel.
    """
    sigma = torch.tensor([sigma], dtype=torchdtype, device=torchdeviceId)

    Kv = GaussKernel(sigma)
    loss = LDDMMloss(Kv, dataloss, gamma=gamma, nt=nt)
    
    # Perform optimization
    Optimize(lambda p0 : loss(p0,q0), p0, niter=niter)
            
    return LDDMM_def(p0,q0,Kv)