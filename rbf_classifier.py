#import statements 
import torch
import torch.nn as nn
import torch.nn.functional as F
         

"""RBF Classifier"""

class RBFClassifier(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        self.register_buffer("prototypes", prototypes)  # shape: [10, 84]
        

    def forward(self, x):  # x shape: [batch_size, 84]
        x = F.normalize(x, p=2, dim=1)  # Normalize input
        protos = F.normalize(self.prototypes, p=2, dim=1)  # Normalize prototypes
        dists = torch.cdist(x, protos, p=2) ** 2    # shape: [batch, 10]
        return dists  