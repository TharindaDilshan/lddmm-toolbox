import os
import torch

os.environ['PYKEOPS_VERBOSE'] = '0'

# ── PyTorch device / dtype ────────────────────
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

# ── KeOps counterparts ────────────────────────
KeOpsdeviceId = torchdeviceId.index  
KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'