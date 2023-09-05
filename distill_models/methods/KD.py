import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(torch.nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/abs/1503.02531
    """

    # y_s: outputs of the student network, shape: [batch_size, num_classes]
    # y_t: outputs of the teacher network, shape: [batch_size, num_classes]
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss

def KD(t):
    kd = DistillKL(t)
    return kd

# Test

# if __name__ == '__main__':
#     kd_loss = DistillKL(4)
#
#     ys = torch.randn(32, 10)
#     yt = torch.randn(32, 10)
#     loss = kd_loss(ys, yt)
#     print(loss)
