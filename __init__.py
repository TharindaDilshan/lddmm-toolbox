"""
lddmm_toolbox
=============
Large Deformation Diffeomorphic Metric Mapping (LDDMM) utilities.
 
Public API
----------
Entry points (surface & point matching):
    MatchSurface, MatchSurfaceExp, MatchSurfaceCurrents, MatchPoints
 
Optimisation:
    LDDMM_Optimize, LDDMM_Optimize_points, LDDMM_def
 
Kernels:
    SumOfGaussKernel, GaussKernel, EnergyKernel,
    GaussLinKernel, GaussExpKernel, GaussCurrentsKernel
 
Losses:
    LDDMMloss, lossVarifoldSurf, lossMeas
 
Shooting / integration:
    Shooting, Flow, RalstonIntegrator, Hamiltonian, HamiltonianSystem
"""

from .kernels import (
    GaussExpKernel,
    EnergyKernel,
    GaussKernel,
    GaussLinKernel,
    GaussKernelGenred,
    EnergyKernelGenred,
    SumOfGaussKernel,
    GaussCurrentsKernel,
    SumKernel,
)

from .shooting import (
    RalstonIntegrator,
    Hamiltonian,
    HamiltonianSystem,
    Shooting,
    Flow,
)

from .losses import (
    LDDMMloss,
    lossVarifoldSurf,
    lossMeas,
)

from .optimization import (
    LDDMM_def,
    Optimize,
    LDDMM_Optimize,
    LDDMM_Optimize_points,
)

from .main import (
    MatchSurface,
    MatchSurfaceExp,
    MatchSurfaceCurrents,
    MatchPoints,
    MatchPointsGauss,
)