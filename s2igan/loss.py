import torch
from torch import nn


class MatchingLoss(nn.Module):
    def __init__(self, beta: float = 10):
        """
        from this paper: https://arxiv.org/abs/1711.10485
        beta is gamma_3
        """
        super().__init__()
        self.beta = beta
        self.eps = 1e-9
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, labels):
        bs = labels.shape[0]

        mask = self.create_mask(labels)
        sim = self.cosine_sim(x, y)
        sim = sim + mask.log()

        diag_label = torch.autograd.Variable(torch.LongTensor(list(range(bs))))
        if torch.cuda.is_available():
            diag_label = diag_label.to("cuda:0")
        loss_0 = self.criterion(sim, diag_label)
        loss_1 = self.criterion(sim.T, diag_label)

        return loss_0 + loss_1

    def create_mask(self, labels):
        mask = torch.ne(labels.view(1, -1), labels.view(-1, 1))
        mask = mask.fill_diagonal_(1)
        mask = mask.to(dtype=torch.float, device=labels.device)
        return mask

    def cosine_sim(self, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        norm_x = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, ord=2, dim=-1, keepdim=True)

        num = torch.bmm(x, y.transpose(1, 2))
        den = torch.bmm(norm_x, norm_y.transpose(1, 2))

        sim = self.beta * (num / den.clamp(min=1e-8))

        return sim.squeeze()


class DistinctiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, cls_x, cls_y, labels):
        return self.crit(cls_x, labels) + self.crit(cls_y, labels)


class SENLoss(nn.Module):
    def __init__(self, beta: int = 10):
        super().__init__()
        self.matching_loss = MatchingLoss(beta)
        self.distinctive_loss = DistinctiveLoss()

    def forward(self, x, y, cls_x, cls_y, labels):
        match_loss = self.matching_loss(x, y, labels)
        dist_loss = self.distinctive_loss(cls_x, cls_y, labels)
        return match_loss.detach(), dist_loss.detach(), match_loss + dist_loss

class SEDLoss(nn.Module):
    def __init__(self, beta: int = 10):
        super().__init__()
        self.distinctive_loss = DistinctiveLoss()

    def forward(self, cls_x, cls_y, labels):
        dist_loss = self.distinctive_loss(cls_x, cls_y, labels)
        return dist_loss.detach() + dist_loss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, x_mean, x_logvar):
        # Compute kl divergence loss
        kl_div = torch.mean(x_mean.pow(2) + x_logvar.exp() - 1 - x_logvar)

        return kl_div


class RSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels):
        return (
            self.crit(R1, one_labels.long())
            + self.crit(R2, zero_labels.long())
            + self.crit(R3, two_labels.long())
            + self.crit(R_GT_FI, zero_labels.long())
        )
