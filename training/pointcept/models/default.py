import torch
import torch.nn as nn

from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorSample(nn.Module):
    def __init__(self, backbone=None, criteria=None, num_samples = 1024):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_samples = num_samples

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]        

        # train
        if self.training:
            selected = torch.randint(0, input_dict["segment"].shape[0], (self.num_samples,))
            batch_ids = offset2batch(input_dict['offset'])[selected]
            points = input_dict["grid_coord"][selected]
            segments = input_dict["segment"][selected]
            input_dict["sample_grid_coord"] = points
            input_dict["sample_offset"] = batch2offset(batch_ids)

            seg_logits = self.backbone(input_dict)
            loss = self.criteria(seg_logits, segments)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            seg_logits = self.backbone(input_dict)
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            seg_logits = self.backbone(input_dict)
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


#@MODELS.register_module()
#class DefaultClassifier(nn.Module):
#    def __init__(
#        self,
#        backbone=None,
#        criteria=None,
#        num_classes=40,
#        backbone_embed_dim=256,
#    ):
#        super().__init__()
#        self.backbone = build_model(backbone)
#        self.criteria = build_criteria(criteria)
#        self.num_classes = num_classes
#        self.backbone_embed_dim = backbone_embed_dim
#        self.cls_head = nn.Sequential(
#            nn.Linear(backbone_embed_dim, 256),
#            nn.BatchNorm1d(256),
#            nn.ReLU(inplace=True),
#            nn.Dropout(p=0.5),
#            nn.Linear(256, 128),
#            nn.BatchNorm1d(128),
#            nn.ReLU(inplace=True),
#            nn.Dropout(p=0.5),
#            nn.Linear(128, num_classes),
#        )
#
#    def forward(self, input_dict):
#        feat = self.backbone(input_dict)
#        cls_logits = self.cls_head(feat)
#        if self.training:
#            loss = self.criteria(cls_logits, input_dict["category"])
#            return dict(loss=loss)
#        elif "category" in input_dict.keys():
#            loss = self.criteria(cls_logits, input_dict["category"])
#            return dict(loss=loss, cls_logits=cls_logits)
#        else:
#            return dict(cls_logits=cls_logits)
        

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes

    def forward(self, input_dict):
        cls_logits = self.backbone(input_dict)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
