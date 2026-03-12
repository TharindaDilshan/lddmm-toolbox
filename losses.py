import torch
from .shooting import Hamiltonian, Shooting

# ──────────────────────────────────────────────
# LDDMM total loss
# ──────────────────────────────────────────────

def LDDMMloss(K, dataloss, gamma=0.1, nt=10):
    """
    Full LDDMM loss = regularisation term + data attachment.
 
    Parameters
    ----------
    K        : velocity kernel
    dataloss : callable VS -> scalar
    gamma    : regularisation weight
    nt       : number of time steps
    """
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K, nt=nt)[-1]
        
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)

    return loss

# ──────────────────────────────────────────────
# Data attachment: varifold (surfaces)
# ──────────────────────────────────────────────

def lossVarifoldSurf(FS, VT, FT, K):
    """
    Varifold loss between a deforming source surface and a fixed target surface.
 
    Parameters
    ----------
    FS : (nF_S, 3) source face connectivity
    VT : (nV_T, 3) target vertex positions
    FT : (nF_T, 3) target face connectivity
    K  : oriented kernel (e.g. GaussLinKernel or GaussExpKernel)
    """
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()

        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT*K(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)

        return (
            cst
            + (LS * K(CS, CS, NSn, NSn, LS)).sum()
            - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()
        )

    return loss

# ──────────────────────────────────────────────
# Data attachment: measure (point clouds)
# ──────────────────────────────────────────────

def lossMeas(VT, K):
    """
    Measure-based loss between a deforming source point cloud and a fixed target.
 
    Parameters
    ----------
    VT : (nT, 3) target point positions
    K  : unoriented kernel (e.g. EnergyKernel or GaussKernel)
    """
    nT = VT.shape[0]
    cst = K(VT, VT).sum()/nT**2
    def loss(VS):
        nS = VS.shape[0]

        return cst + K(VS, VS).sum()/nS**2 - 2*K(VS, VT).sum()/(nS*nT)
    
    return loss