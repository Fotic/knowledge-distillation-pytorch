import torch
import torch.nn as nn
import torch.nn.functional as F


class L2_dist(torch.nn.Module):
    """Computes the Euclidean Distance"""

    # y_s: outputs of the student network, shape: [batch_size, num_classes]
    # y_t: outputs of the teacher network, shape: [batch_size, num_classes]

    def __init__(self, p):
        super(L2_dist, self).__init__()
        self.p = p

    def forward(self, y_s, y_t):
        l = torch.nn.PairwiseDistance(p=self.p)
        l2 = l(y_s, y_t)
        loss = torch.sqrt(torch.sum(l2))
        return loss

def L2():
    l2 = L2_dist(2)
    return l2

# test
# if __name__ == '__main__':
#     kd_loss = L2(2)
#
#     ys = torch.randn(32, 10)
#     yt = torch.randn(32, 10)
#     loss = kd_loss(ys, yt)
#     print(loss)
