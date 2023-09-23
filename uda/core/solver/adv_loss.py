import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdversarialLoss(nn.Module):
    r"""Domain adversarial loss from `Adversarial Discriminative Domain Adaptation (CVPR 2017)
    <https://arxiv.org/pdf/1702.05464.pdf>`_.
    Similar to the original `GAN <https://arxiv.org/pdf/1406.2661.pdf>`_ paper, ADDA argues that replacing
    :math:`\text{log}(1-p)` with :math:`-\text{log}(p)` in the adversarial loss provides better gradient qualities.
    Inputs:
        - domain_pred (tensor): predictions of domain discriminator
        - domain_label (str, optional): whether the data comes from source or target.
          Must be 'source' or 'target'. Default: 'source'
    Shape:
        - domain_pred: :math:`(minibatch,)`.
        - Outputs: scalar.
    """

    def __init__(self):
        super(DomainAdversarialLoss, self).__init__()

    def forward(self, domain_pred, domain_label='source'):
        assert domain_label in ['source', 'target']
        if domain_label == 'source':
            return F.binary_cross_entropy(domain_pred, torch.ones_like(domain_pred).to('cuda:0'))
        else:
            return F.binary_cross_entropy(domain_pred, torch.zeros_like(domain_pred).to('cuda:0'))
