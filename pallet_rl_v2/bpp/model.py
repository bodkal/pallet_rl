"""Policy / value networks (Sec. V-B).

Both are a stack of 6 convolutional layers with 128 planes and ReLU.
Policy head emits W x L x b x (k+1) logits (every Cartesian combination of
location, item and orientation); they are masked with the action mask before
the softmax.  The value network ends in a linear layer with tanh, so
v_psi(s) in [-1,1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def trunk(cin, planes=128, n=6):
    layers = []
    c = cin
    for _ in range(n):
        layers += [nn.Conv2d(c, planes, 3, padding=1, bias=False),
                   nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        c = planes
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    def __init__(self, cin, M, planes=128, n=6):
        super().__init__()
        self.body = trunk(cin, planes, n)
        self.head = nn.Conv2d(planes, M, 1)

    def forward(self, x):
        return self.head(self.body(x))            # (B,M,W,L)


class ValueNet(nn.Module):
    def __init__(self, cin, W, L, planes=128, n=6):
        super().__init__()
        self.body = trunk(cin, planes, n)
        self.red = nn.Sequential(nn.Conv2d(planes, 16, 1), nn.BatchNorm2d(16),
                                 nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(16 * W * L, 256), nn.ReLU(inplace=True),
                                nn.Linear(256, 1))

    def forward(self, x):
        z = self.red(self.body(x)).flatten(1)
        return torch.tanh(self.fc(z)).squeeze(-1)  # (B,) in [-1,1]


class Nets(nn.Module):
    def __init__(self, cfg, planes=128, n=6, with_value=True):
        super().__init__()
        self.policy = PolicyNet(cfg.CIN, cfg.M, planes, n)
        self.value = ValueNet(cfg.CIN, cfg.W, cfg.L, planes, n) if with_value else None
        self.cfg_dict = cfg.as_dict()


class Evaluator:
    """Batched masked-policy evaluation used by MCTS and by the rollouts."""

    def __init__(self, net, cfg, device, amp=True):
        self.net, self.cfg, self.device = net, cfg, device
        self.amp = amp and device.type == 'cuda'

    @torch.inference_mode()
    def probs(self, feat, mask_flat):
        """feat (n,CIN,W,L) float32 np ; mask_flat (n,A) bool np -> (n,A) float32 np"""
        t = torch.from_numpy(feat).to(self.device, non_blocking=True)
        if self.amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = self.net.policy(t)
        else:
            lg = self.net.policy(t)
        lg = lg.float().reshape(t.shape[0], -1)
        m = torch.from_numpy(mask_flat).to(self.device, non_blocking=True)
        lg = lg.masked_fill(~m, -1e30)
        return torch.softmax(lg, dim=1).cpu().numpy()
