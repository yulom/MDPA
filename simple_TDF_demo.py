import torch
import torch.nn as nn
from protoLoss import TDF_Loss
# Example usage
cls_num = 4
features_dim = 1000
batch_size = 64

# Generate random target output probabilities and prototypes
random_matrix = torch.rand(batch_size, cls_num)
prototypes = nn.Parameter(torch.randn(cls_num, features_dim), requires_grad=True)

# Compute TDF loss
LOSS = TDF_Loss(random_matrix, prototypes)

print(LOSS)