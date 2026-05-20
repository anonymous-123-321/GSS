from torch import nn
import torch
import torch.nn.functional as F
import math
class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, y):
        raise NotImplementedError()


class L2Loss(DistanceLoss):
    def forward(self, x, y):
        return (x - y).pow(2).sum(1).pow(0.5)



def compute_kl_divergence2(mu_specific, sigma_specific, mu_general, sigma_general, margin_ratio=2.0, eps=1e-6):
    var_S = (sigma_specific ** 2) + eps
    var_G = (sigma_general ** 2) + eps

    scaled_var_S = margin_ratio * var_S


    log_term = torch.log(var_G / scaled_var_S)
    var_term = scaled_var_S / var_G

    
    
    distance_term = ((mu_specific - mu_general) ** 2) / var_G

    
    kl_div = 0.5 * (log_term + var_term + distance_term - 1.0)
    var_reg = 0.05 * var_G.mean()


    mu_video_norm = torch.norm(mu_specific, p=2, dim=-1).mean().detach()
    mu_box_norm = torch.norm(mu_general, p=2, dim=-1).mean().detach()
    mu_distance = torch.norm(mu_specific - mu_general, p=2, dim=-1).mean().detach()

    stats = {
        "var_video": var_S.mean().detach(),
        "var_ratio": (var_S.mean()/var_G.mean() ).detach(),

        "mu_video_norm": mu_video_norm,
        "mu_box_norm": mu_box_norm,
        "mu_distance": mu_distance

    }

    return kl_div.mean(dim=-1).mean()+var_reg ,stats
