import torch.nn as nn
import copy

def clones(module, N):
    "N개의 동일한 Layer를 구성"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])