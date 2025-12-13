import os
import sys
import weakref
import torch

from tensorboardX import SummaryWriter

import wandb

class SummaryLogger:

    def __init__(self, cfg):
        self.tensorboard_obj = SummaryWriter(cfg.save_path)
        self.wandb_obj = wandb.init(
            project=cfg.project_name,
            name = cfg.exp_name,
            group = cfg.group_name,
            config=cfg)
        self.set_metrics = set()
        wandb.define_metric("custom_step")
        
    def add_scalar(self, key, value, epoch):
        self.tensorboard_obj.add_scalar(key, value, epoch)

        if not key in self.set_metrics:
            wandb.define_metric(key, step_metric="custom_step")
            self.set_metrics.add(key)

        self.wandb_obj.log({key: value, "custom_step":epoch})