import torch.nn as nn
import torch

from collections import OrderedDict

import spconv.pytorch as spconv

from .utils import offset2batch, MODELS, build_model

@MODELS.register_module("MSM3D_EncDec")
class MSM3D_EncDec(nn.Module):

    def __init__(self, 
                 encoder=None,
                 decoder = None,
                 encoder_weight_path=None,
                 encoder_weight_prefix=None,
                 encoder_weight_new_prefix=None,
                 decoder_weight_path=None,
                 decoder_weight_prefix=None,
                 decoder_weight_new_prefix=None,
                 freeze_encoder=False,
                 use_point_coords=False):
        super().__init__()

        self.encoder_weight_path = encoder_weight_path
        self.encoder_weight_prefix = encoder_weight_prefix
        self.encoder_weight_new_prefix = encoder_weight_new_prefix
        self.decoder_weight_path = decoder_weight_path
        self.decoder_weight_prefix = decoder_weight_prefix
        self.decoder_weight_new_prefix = decoder_weight_new_prefix
        self.freeze_encoder = freeze_encoder
        self.use_point_coords = use_point_coords

        self.encoder = build_model(encoder)
        if not self.encoder_weight_path is None:
            self.load_paramteres(self.encoder, 
                                 self.encoder_weight_path,
                                 self.encoder_weight_prefix,
                                 self.encoder_weight_new_prefix)
        
        if not decoder is None:
            self.decoder = build_model(decoder)
        else:
            self.decoder = None
        if not self.decoder_weight_path is None:
            self.load_paramteres(self.decoder, 
                                 self.decoder_weight_path,
                                 self.decoder_weight_prefix,
                                 self.decoder_weight_new_prefix)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


    def load_paramteres(self, model, path, prefix, new_prefix):
        param_names = model.state_dict()
        saved_dict = torch.load(path)
        state_dict_key = "state_dict"
        if not state_dict_key in saved_dict:
            state_dict_key = "model"
        weight = OrderedDict()
        for key, value in saved_dict[state_dict_key].items():
            new_key = key
            if not prefix is None:
                if not new_prefix is None:
                    if key.startswith(prefix):
                        new_key = key.replace(prefix, new_prefix)
                else:
                    if key.startswith(prefix):
                        new_key = key.replace(prefix, "")
            else:
                if not new_prefix is None:
                    new_key = new_prefix+key
                else:
                    new_key = key
            if new_key in param_names:
                weight[new_key] = value
            #else:
            #    print("unable to load", new_key)
        model.load_state_dict(weight, strict=True)


    def forward(self, input_dict):

        if self.use_point_coords:
            x = (input_dict["coord"], input_dict["feat"], input_dict["offset"].to(torch.int32))
        else:
            discrete_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]

            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 96).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1)
        
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    _, x_list = self.encoder(x)
        else:
            _, x_list = self.encoder(x)

        if "sample_grid_coord" in input_dict:
            dec_input_dict = {}
            dec_input_dict["grid_coord"] = input_dict["sample_grid_coord"]
            dec_input_dict["offset"] = input_dict["sample_offset"]
        else:
            dec_input_dict = input_dict

        if self.decoder is None:
            x = x_list[0].features
        else:
            x = self.decoder(x_list, dec_input_dict)
        return x
