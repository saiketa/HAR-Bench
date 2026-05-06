import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature=0.2):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, zis, zjs):
        """
        zis, zjs: [B, D]
        """
        batch_size = zis.shape[0]

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        representations = torch.cat([zis, zjs], dim=0)  # [2B, D]
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2B, 2B]

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # [2B]

        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=zis.device)).float()
        logits = similarity_matrix / self.temperature
        logits_masked = logits * mask + (-1e9) * (1 - mask)

        labels = torch.arange(2 * batch_size, device=zis.device)
        labels = (labels + batch_size) % (2 * batch_size)

        loss = self.criterion(logits_masked, labels)
        return loss