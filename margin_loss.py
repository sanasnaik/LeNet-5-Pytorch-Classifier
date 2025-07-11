#import statements 
import torch
import torch.nn as nn

class LeNetRBFMarginLoss(nn.Module):
    def __init__(self, j=0.1):
        super().__init__()
        self.j = j

    def forward(self, dists, labels):
        """
        dists: [batch, 10] — squared distances from RBF layer
        labels: [batch] — true labels
        """
        batch_size = dists.size(0)

        # Get distance to correct class for each sample
        target_dists = dists[torch.arange(batch_size), labels]

        # Mask for incorrect classes
        mask = torch.ones_like(dists).bool()
        mask[torch.arange(batch_size), labels] = False
        incorrect_dists = dists[mask].view(batch_size, -1)

        # Compute log term
        j_tensor = torch.tensor(-self.j, device=incorrect_dists.device)
        log_term = torch.log(torch.exp(j_tensor) + torch.sum(torch.exp(-incorrect_dists), dim=1))

        # Final loss
        loss = target_dists + log_term
        return torch.mean(loss)
