import torch
import torch.nn as nn


class FitNets(nn.Module):
    """
    Fitnets: hints for thin deep nets, ICLR 2015
    https://arxiv.org/abs/1412.6550
    """

    # f_s: features of the student network, shape: [batch_size, s_dimensions]
    # f_t: features of the teacher network, shape: [batch_size, t_dimensions]

    def __init__(self):
        super(FitNets, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.criterion(f_s, f_t)
        return loss


def FitNet():
    f = FitNets()
    return f

# Test
# if __name__ == '__main__':
#     kd_loss = FitNets()
#
#     fs = torch.randn(32, 1024)
#     ft = torch.randn(32, 1024)
#     loss = kd_loss(fs, ft)
#     print(loss)
