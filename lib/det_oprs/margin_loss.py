import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, m: float =0.5, s: float=64.0):
        super(ArcFace, self).__init__()
        self.scale = s
        self.margin = m
        self.cos_margin = math.cos(m)
        self.sin_margin = math.sin(m)
        self.min_cos_theta = math.cos(math.pi - m)
        self.easy_margin = False
        
    def forward(self, feats, w, labels: torch.Tensor):
        assert feats.ndim == 3, f'Expected feats dim=3, but got feats dim={feats.ndim}'
        assert w.ndim == 2, f'Expected w dim=2, but got w dim={w.ndim}'
        feats = F.normalize(feats, dim=-1).reshape(-1, feats.shape[-1])
        w = F.normalize(w, dim=-1)
        logits = torch.mm(feats, w.t())
        cos_theta = logits.clamp(-1,1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-5)
        assert torch.isfinite(sin_theta).all(), 'logit inf'
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin #cos(theta+m)       
        if self.easy_margin:
            final_target_logit = torch.where(cos_theta>0, cos_theta_m, cos_theta)
        else:
            final_target_logit = torch.where(cos_theta>self.min_cos_theta, cos_theta_m, cos_theta - sin_theta*self.margin)
        
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, labels[:,None], 1)
        assert mask.sum() == mask.shape[0]
        final_target_logit = mask*final_target_logit + (1-mask)*cos_theta
        final_target_logit *= self.scale
        final_loss = F.cross_entropy(final_target_logit, labels)        
        
        return final_loss

class MagFace(nn.Module):
    def __init__(self, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, scale=32.):
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin     
        self.scale = scale
        self.easy_margin = False
    
    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)
    
    def _margin(self, x):
        """gen ada_margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, x, w, labels):
        x = F.normalize(x, dim=-1).reshape(-1, x.shape[-1])
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self._margin(x_norm)
        cos_m = torch.cos(ada_margin)
        sin_m = torch.sin(ada_margin)

        cos_theta = torch.mm(x, F.normalize(w, dim=-1).t())
        cos_theta = cos_theta.clamp(-1,1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2) + 1e-5)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cut_off = torch.cos(math.pi-ada_margin)
            cos_theta_m = torch.where(cos_theta>cut_off, cos_theta_m, cos_theta - sin_theta*ada_margin)
        
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, labels[:,None], 1)
        assert mask.sum() == mask.shape[0]
        output = (mask * cos_theta_m + (1-mask) * cos_theta) * self.scale
        loss = F.cross_entropy(output, labels) + self.calc_loss_G(x_norm)
        return loss