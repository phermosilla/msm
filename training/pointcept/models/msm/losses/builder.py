
from pointcept.utils.registry import Registry

SS_LOSSES = Registry("ss_losses")

def build_ss_loss(cfg):
    return SS_LOSSES.build(cfg)
