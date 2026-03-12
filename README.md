# LDDMM Toolbox

**Large Deformation Diffeomorphic Metric Mapping** — a PyTorch + KeOps toolkit for diffeomorphic registration of 3D surfaces and point clouds.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Package Structure](#2-package-structure)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Entry Point Functions](#5-entry-point-functions)
6. [The LDDMM_def Result Object](#6-the-lddmm_def-result-object)
7. [Kernels Reference](#7-kernels-reference)
8. [Loss Functions](#8-loss-functions)
9. [Shooting & Flow](#9-shooting--flow)
10. [Optimisation](#10-optimisation)
11. [Building a Custom Pipeline](#11-building-a-custom-pipeline)
12. [Tips & Troubleshooting](#12-tips--troubleshooting)

---

## 1. Overview

The LDDMM Toolbox provides a modular Python implementation of Large Deformation Diffeomorphic Metric Mapping — a framework for computing smooth, invertible deformations between shapes. It supports registration of triangulated surfaces and 3D point clouds using a geodesic shooting approach built on Hamiltonian mechanics.

The toolbox is built on top of **PyTorch** for automatic differentiation and L-BFGS optimisation, and **PyKeOps** for scalable kernel computations that run efficiently on both CPU and GPU.

**Key capabilities:**
- Surface-to-surface registration using varifold or currents data attachment
- Point cloud registration using energy-distance or Gaussian measure losses
- Geodesic shooting and flow of arbitrary points through optimised deformations
- Modular kernel library: choose or combine kernels for velocity fields and data attachment
- Automatic GPU acceleration via PyKeOps when CUDA is available

---

## 2. Package Structure

Drop the `lddmm_toolbox/` folder anywhere on your Python path.

```
lddmm_toolbox/
├── __init__.py        # Re-exports the full public API
├── config.py          # Device and dtype configuration (CPU/GPU auto-detection)
├── kernels.py         # All kernel functions for velocity fields and data attachment
├── shooting.py        # Ralston integrator, Hamiltonian mechanics, Shooting and Flow
├── losses.py          # LDDMMloss, varifold loss, measure-based loss
├── optimization.py    # LDDMM_def result class, Optimize wrapper, LDDMM_Optimize routines
└── main.py            # High-level entry points: MatchSurface*, MatchPoints*
```

---

## 3. Installation

### Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- PyKeOps ≥ 2.1
- NumPy

### Install dependencies

```bash
pip install torch torchvision
pip install pykeops
pip install numpy
```

### Add the toolbox to your project

```bash
# Option A — copy the folder next to your script
cp -r lddmm_toolbox/ my_project/

# Option B — add the parent directory to PYTHONPATH
export PYTHONPATH="/path/to/parent:$PYTHONPATH"
```

> **GPU support:** If a CUDA-capable GPU is detected, the toolbox automatically runs on `cuda:0`. No manual configuration is required — `config.py` handles device selection at import time.

---

## 4. Quick Start

### Surface registration

```python
from lddmm_toolbox import MatchSurface
import torch

# VS, FS: source vertices (N×3) and faces (M×3) as tensors
# VT, FT: target vertices and faces
p0 = torch.zeros(VS.shape, requires_grad=True)

result = MatchSurface(VS, FS, VT, FT, p0,
                      sigmaV=20,   # varifold kernel bandwidth
                      sigmaW=20,   # velocity kernel bandwidth
                      niter=50,
                      gamma=0.1)

# Get the deformed source vertices
p_final, q_final = result.shoot()[-1]
```

### Point cloud registration

```python
from lddmm_toolbox import MatchPoints
import numpy as np

VS = np.load('source_points.npy')   # shape (N, 3)
VT = np.load('target_points.npy')   # shape (M, 3)

result = MatchPoints(VS, VT, sigmaW=15, niter=30)

# Get the deformed source positions
trajectory = result.shoot()
deformed   = trajectory[-1][1]
```

### Flowing new points through an existing deformation

```python
# After optimisation, apply the deformation to unseen points
new_points = np.array([[1.0, 2.0, 3.0], ...])
flow_traj  = result.flow(new_points, nt=10)
deformed_new = flow_traj[-1][0]
```

---

## 5. Entry Point Functions

All entry points live in `main.py` and are exported from the package root.

### 5.1 `MatchSurface`

Registers a source surface to a target using the **GaussLin varifold** kernel. Suitable for unoriented surfaces — orientation ambiguity is handled by the squared dot-product.

```python
result = MatchSurface(VS, FS, VT, FT, p0,
                      sigmaV=20, sigmaW=20,
                      method=LDDMM_Optimize,
                      niter=10, gamma=0.1, nt=10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VS` | — | Source vertex positions, tensor `(N, 3)` |
| `FS` | — | Source face connectivity, integer tensor `(M, 3)` |
| `VT` | — | Target vertex positions, tensor `(N, 3)` |
| `FT` | — | Target face connectivity, integer tensor `(M, 3)` |
| `p0` | — | Initial momentum, tensor `(N, 3)`, `requires_grad=True` |
| `sigmaV` | `20` | Bandwidth for the varifold data attachment kernel |
| `sigmaW` | `20` | Bandwidth for the velocity field kernel |
| `method` | `LDDMM_Optimize` | Optimisation routine to use |
| `niter` | `10` | Number of L-BFGS iterations |
| `gamma` | `0.1` | Regularisation weight (higher = smoother deformation) |
| `nt` | `10` | Number of time integration steps |

---

### 5.2 `MatchSurfaceExp`

Same as `MatchSurface` but uses the **GaussExp varifold** kernel, which takes the exponential of the normal dot-product rather than its square. Can give smoother gradients in some cases.

```python
result = MatchSurfaceExp(VS, FS, VT, FT, p0, sigmaV=20, sigmaW=20, ...)
```

---

### 5.3 `MatchSurfaceCurrents`

Registers surfaces using the **currents** representation (`GaussCurrents` kernel). Currents use the plain dot-product — not its square — so they are sensitive to surface orientation. Faces should be consistently oriented.

```python
result = MatchSurfaceCurrents(VS, FS, VT, FT, p0, sigmaV=20, sigmaW=20, ...)
```

---

### 5.4 `MatchPoints`

Registers a source 3D point cloud to a target using the **energy-distance** kernel as data attachment. Accepts numpy arrays or tensors.

```python
result = MatchPoints(VS, VT,
                     sigmaW=20,
                     method=LDDMM_Optimize,
                     niter=10, gamma=0.1, nt=10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VS` | — | Source point cloud, array or tensor `(N, 3)` |
| `VT` | — | Target point cloud, array or tensor `(M, 3)` |
| `sigmaW` | `20` | Bandwidth for the velocity field kernel |
| `method` | `LDDMM_Optimize` | Optimisation routine |
| `niter` | `10` | Number of L-BFGS iterations |
| `gamma` | `0.1` | Regularisation weight |
| `nt` | `10` | Number of time steps |

---

## 6. The `LDDMM_def` Result Object

All entry points return an `LDDMM_def` instance that stores the optimised initial momentum, initial positions, and velocity kernel. It exposes two methods:

### `.shoot(deltat=1.0, init_mom=None, nt=10)`

Runs geodesic shooting and returns the full time-discretised trajectory as a list of `(p, q)` tuples. The last element is the final deformed configuration.

```python
trajectory = result.shoot()
p_final, q_final = trajectory[-1]   # final momentum and positions
```

You can supply a different initial momentum to re-shoot along a rescaled geodesic:

```python
p_scaled = result.init_mom * 0.5
trajectory = result.shoot(init_mom=p_scaled)
```

### `.flow(x0, nt=10)`

Transports an arbitrary set of points `x0` through the optimised deformation. `x0` can be a numpy array or tensor.

```python
new_pts = np.array([[x1,y1,z1], [x2,y2,z2], ...])
flow_traj    = result.flow(new_pts, nt=10)
transported  = flow_traj[-1][0]   # final positions of new_pts
```

---

## 7. Kernels Reference

Kernels serve two distinct roles: **(1) velocity field kernels** control the smoothness of the deformation, and **(2) data attachment kernels** measure the similarity between shapes.

### 7.1 Velocity field kernels

| Function | Use Case | Arguments | Notes |
|----------|----------|-----------|-------|
| `SumOfGaussKernel(sigmas)` | Multi-scale velocity field | `sigmas`: list of bandwidths | Recommended for most tasks |
| `GaussKernel()` | Single fixed-scale velocity field | *(none)* | Sigma=5, fixed. Used in `LDDMM_Optimize_points` |

### 7.2 Data attachment kernels — surfaces

| Function | Use Case | Arguments | Notes |
|----------|----------|-----------|-------|
| `GaussLinKernel(sigma)` | Varifold — unoriented surfaces | `sigma`: position bandwidth | Squared dot-product; orientation-invariant |
| `GaussExpKernel(sigma)` | Varifold — exponential weighting | `sigma`: position bandwidth | Exp of dot-product |
| `GaussCurrentsKernel(sigma)` | Currents — oriented surfaces | `sigma`: position bandwidth | Dot-product; orientation-sensitive |

### 7.3 Data attachment kernels — point clouds

| Function | Use Case | Notes |
|----------|----------|-------|
| `EnergyKernel()` | Energy-distance metric | `k(x,y) = -‖x−y‖`. Default for `MatchPoints` |
| `GaussKernel()` | Gaussian measure loss | Sigma=5, fixed. Used in `MatchPoints_temp` |

> **Choosing bandwidths:** Set `sigmaV` (data attachment) to the typical scale of local features you want to match, and `sigmaW` (velocity) to roughly the scale of the global deformation. Values in the range 10–50 (in the coordinate units of your data) are common for anatomical shapes.

---

## 8. Loss Functions

These can be used directly if you need a custom registration pipeline.

### `LDDMMloss(K, dataloss, gamma=0.1, nt=10)`

Combines the regularisation energy (Hamiltonian at time 0) with a data attachment term evaluated at the end of the geodesic.

```python
loss = LDDMMloss(K, dataloss, gamma=0.1, nt=10)
L    = loss(p0, q0)
```

### `lossVarifoldSurf(FS, VT, FT, K)`

Computes the varifold discrepancy between a deforming source surface and a fixed target. The target constant term is precomputed once for efficiency.

```python
dataloss = lossVarifoldSurf(FS, VT, FT, GaussLinKernel(sigma=20))
L = dataloss(VS_deformed)   # called each iteration
```

### `lossMeas(VT, K)`

Computes the measure-based (MMD) discrepancy between a deforming source point cloud and a fixed target.

```python
dataloss = lossMeas(VT, EnergyKernel())
L = dataloss(VS_deformed)
```

---

## 9. Shooting & Flow

Lower-level functions available for custom workflows, all in `shooting.py`.

### `Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator(), deltat=1.0)`

Integrates the Hamiltonian system forward in time from `(p0, q0)` under velocity kernel `K`. Returns a list of `nt+1` states `(p, q)`.

### `Flow(x0, p0, q0, K, nt=10, deltat=1.0, Integrator=RalstonIntegrator())`

Simultaneously shoots the geodesic `(p0, q0)` and transports `x0` along the induced velocity field. Useful for applying deformations to landmarks or meshes not included in the optimisation.

### `RalstonIntegrator()`

Returns a second-order Runge-Kutta (Ralston method) integrator for tuple-valued ODE systems. Used internally by `Shooting` and `Flow`.

### `Hamiltonian(K)` / `HamiltonianSystem(K)`

Low-level utilities. `Hamiltonian(K)` returns `H(p,q) = 0.5 * p^T K(q,q) p`. `HamiltonianSystem(K)` returns the right-hand side `(-dH/dq, dH/dp)` via automatic differentiation.

---

## 10. Optimisation

### `LDDMM_Optimize`

The default optimisation routine used by all `MatchSurface*` and `MatchPoints` functions. Builds a sum-of-Gaussians velocity kernel, assembles the LDDMM loss, and runs L-BFGS.

```python
deformation = LDDMM_Optimize(p0, q0, dataloss, sigma,
                              niter=10, gamma=0.1, nt=10)
```

### `LDDMM_Optimize_points`

Variant that uses the fixed `GaussKernel` (sigma=5) as the velocity kernel. Intended for point-cloud tasks where the default bandwidth is appropriate.

### `Optimize(loss_fn, x, niter=50)`

Generic L-BFGS wrapper. Accepts any scalar loss callable and a tensor to optimise. Returns a list of loss values per iteration.

```python
losses = Optimize(loss_fn, x, niter=50)
```

---

## 11. Building a Custom Pipeline

The modular design lets you mix and match components freely. The example below performs surface registration with a multi-scale velocity kernel and a currents data attachment:

```python
import torch
from lddmm_toolbox.kernels      import SumOfGaussKernel, GaussCurrentsKernel
from lddmm_toolbox.losses       import LDDMMloss, lossVarifoldSurf
from lddmm_toolbox.optimization import Optimize, LDDMM_def
from lddmm_toolbox.config       import torchdtype, torchdeviceId

# Multi-scale velocity kernel
sigmas = torch.tensor([5.0, 15.0, 30.0], dtype=torchdtype, device=torchdeviceId)
Kv = SumOfGaussKernel(sigmas)

# Currents data attachment
dataloss = lossVarifoldSurf(FS, VT, FT, GaussCurrentsKernel(sigma=10.0))

# Assemble and optimise
loss = LDDMMloss(Kv, dataloss, gamma=0.05, nt=20)
p0   = torch.zeros(VS.shape, device=torchdeviceId, requires_grad=True)
q0   = VS.clone().detach().requires_grad_(True)
Optimize(lambda p: loss(p, q0), p0, niter=100)

result = LDDMM_def(p0, q0, Kv)
```

---

## 12. Tips & Troubleshooting

### Deformation is too rigid / not converging
- Increase `sigmaW` — a larger bandwidth gives a more global, flexible deformation
- Decrease `gamma` to allow larger deformations
- Increase `niter` or `nt`

### Deformation is too irregular / folds
- Decrease `sigmaW` or increase `gamma`
- Use `SumOfGaussKernel` with multiple scales to improve stability

### Out of memory on GPU
- Sub-sample your meshes to reduce the number of vertices/points
- Reduce `nt` — fewer time steps use less memory during backpropagation

### KeOps compilation on first run
PyKeOps compiles CUDA/C++ kernels on first use — this is expected and only happens once per kernel configuration. Set `PYKEOPS_VERBOSE=1` to monitor compilation progress.

### Surface normals and orientation
- `GaussLinKernel` and `GaussExpKernel` are orientation-invariant (varifold)
- `GaussCurrentsKernel` is orientation-sensitive — ensure consistent face winding before use

---

*LDDMM Toolbox · PyTorch + KeOps · 3D Shape Registration*
