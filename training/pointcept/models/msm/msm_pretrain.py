import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import pointops

import numpy as np

import spconv.pytorch as spconv

from torch_geometric.utils import scatter

from ..builder import MODELS, build_model
from .losses.builder import SS_LOSSES, build_ss_loss 
from .utils import CosineScheduler, match_3d_coords

from pointcept.models.utils import offset2batch, batch2offset

PRINT_TIMING = False
        
@MODELS.register_module()
class MSMPreTrain(nn.Module):
    def __init__(self, 
                 backbone = None,
                 patch_proj_head = None,
                 loss = None,
                 momentum = 0.999,
                 wd = 0.005,
                 total_iters = 0,
                 mask_ratio = 0.5,
                 match_ks = [4],
                 match_radius = [0.04],
                 project_teacher_patchs = True):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.match_ks = match_ks
        self.match_radius = match_radius
        self.project_teacher_patchs = project_teacher_patchs

        self.student = build_model(backbone)
        self.teacher = build_model(backbone)
        
        self.student_patch_proj = torch.nn.ModuleList()
        if self.project_teacher_patchs:
            self.teacher_patch_proj = torch.nn.ModuleList()
        for cur_patch_proj_head in patch_proj_head:
            self.student_patch_proj.append(build_model(cur_patch_proj_head))
            if self.project_teacher_patchs:
                self.teacher_patch_proj.append(build_model(cur_patch_proj_head))
        
        self.loss = build_ss_loss(loss)

        self.empty_voxel_feat_list = torch.nn.ParameterList()
        for cur_channels in reversed(backbone['channels']):
            self.empty_voxel_feat_list.append(torch.nn.Parameter(
                    torch.randn(1, cur_channels, requires_grad=True)))

        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()):
            param_k.data = param_q.data
            param_k.requires_grad = False

        if self.project_teacher_patchs:
            for cur_patch_proj_head_it in range(len(patch_proj_head)):
                for param_q, param_k in zip(
                    self.student_patch_proj[cur_patch_proj_head_it].parameters(), 
                    self.teacher_patch_proj[cur_patch_proj_head_it].parameters()):
                    param_k.data = param_q.data
                    param_k.requires_grad = False

        if not isinstance(momentum, float):
            self.momentum_scheduler = CosineScheduler(
                base_value = momentum.base_value, 
                final_value = momentum.final_value, 
                total_iters = total_iters)
            self.momentum = momentum.base_value
        else:
            self.momentum_scheduler = None
            self.momentum = torch.tensor(momentum)

        if not isinstance(wd, float):
            self.wd_scheduler = CosineScheduler(
                base_value = wd.base_value, 
                final_value = wd.final_value, 
                total_iters = total_iters)
            self.wd = wd.base_value
        else:
            self.wd_scheduler = None
            self.wd = torch.tensor(wd)

    def update_momentum_model(self):
        if not self.momentum_scheduler is None:
            self.momentum = self.momentum_scheduler.__getitem__()

        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        if self.project_teacher_patchs:
            for cur_patch_proj_head_it in range(len(self.student_patch_proj)):
                for param_q, param_k in zip(
                    self.student_patch_proj[cur_patch_proj_head_it].parameters(), 
                    self.teacher_patch_proj[cur_patch_proj_head_it].parameters()):
                    param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        return self.momentum 


    def mask_crop(self, crop_dict):
        num_down_layers = len(self.teacher.down)
        batch = offset2batch(crop_dict["offset"])
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), 
             crop_dict["grid_coord"].int()//(2**num_down_layers)], dim=1).contiguous()
        unique_indexs, inverse_indices = torch.unique(indices, return_inverse=True, dim=0)
        mask_unique = torch.rand(unique_indexs.shape[0], device=indices.device) > self.mask_ratio
        mask_unique = mask_unique[inverse_indices]
        new_mask_dict = {
            "origin_coord": crop_dict["origin_coord"][mask_unique],
            "grid_coord" : crop_dict["grid_coord"][mask_unique],
            "feat" : crop_dict["feat"][mask_unique],
            "offset" : batch2offset(batch[mask_unique])
        }
        return new_mask_dict, mask_unique


    def forward(self, input_dict):

        ######## Train
        if self.training:

            ################# TIME
            if PRINT_TIMING:
                start = torch.cuda.Event(enable_timing=True)
                event_1 = torch.cuda.Event(enable_timing=True)
                event_2 = torch.cuda.Event(enable_timing=True)
                event_3 = torch.cuda.Event(enable_timing=True)
                event_4 = torch.cuda.Event(enable_timing=True)
                event_5 = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            ################# FTIME

            #### Define the different dictionaries.
            crop_dict_1 = {
                "origin_coord": input_dict["view_1_origin_coord"], 
                "grid_coord" : input_dict["view_1_grid_coord"],
                "feat" : input_dict["view_1_feat"],
                "offset" : input_dict["view_1_offset_crop"]}
            mask_crop_dict_1, mask_crop_1 = self.mask_crop(crop_dict_1)
            full_dict_1 = {
                "origin_coord": input_dict["view_1_full_origin_coord"],
                "grid_coord" : input_dict["view_1_full_grid_coord"],
                "feat" : input_dict["view_1_full_feat"],
                "offset" : input_dict["view_1_offset_full"]}
            crop_dict_2 = {
                "origin_coord": input_dict["view_2_origin_coord"],
                "grid_coord" : input_dict["view_2_grid_coord"],
                "feat" : input_dict["view_2_feat"],
                "offset" : input_dict["view_2_offset_crop"]}
            mask_crop_dict_2, mask_crop_2 = self.mask_crop(crop_dict_2)
            full_dict_2 = {
                "origin_coord": input_dict["view_2_full_origin_coord"],
                "grid_coord" : input_dict["view_2_full_grid_coord"],
                "feat" : input_dict["view_2_full_feat"],
                "offset" : input_dict["view_2_offset_full"]}
            
            ################# TIME
            if PRINT_TIMING:
                event_1.record()
            ################# FTIME
            
            #### Compute teacher features.
            self.teacher = self.teacher.eval()
            if self.project_teacher_patchs:
                for cur_patch_proj_head_it in range(len(self.student_patch_proj)):
                    self.teacher_patch_proj[cur_patch_proj_head_it].eval()
            with torch.no_grad():

                # Process backbone full view 1.
                _, teacher_feat_list_1 = self.teacher(full_dict_1)

                # Compute local feature full view 1.
                if self.project_teacher_patchs:
                    teacher_feat_list_proj_1 = []
                    for cur_feat_iter in range(len(teacher_feat_list_1)):
                        teacher_feat_list_proj_1.append(teacher_feat_list_1[cur_feat_iter].replace_feature(
                            self.teacher_patch_proj[cur_feat_iter](teacher_feat_list_1[cur_feat_iter].features)))
                else:
                    teacher_feat_list_proj_1 = teacher_feat_list_1

                # Process backbone full view 2.
                _, teacher_feat_list_2 = self.teacher(full_dict_2)

                # Compute local feature full view 2.
                if self.project_teacher_patchs:
                    teacher_feat_list_proj_2 = []
                    for cur_feat_iter in range(len(teacher_feat_list_2)):
                        teacher_feat_list_proj_2.append(teacher_feat_list_2[cur_feat_iter].replace_feature(
                            self.teacher_patch_proj[cur_feat_iter](teacher_feat_list_2[cur_feat_iter].features)))
                else:
                    teacher_feat_list_proj_2 = teacher_feat_list_2

            ################# TIME
            if PRINT_TIMING:
                event_2.record()
            ################# FTIME

            #### Compute student features.
            self.student = self.student.train()
            for cur_patch_proj_head_it in range(len(self.student_patch_proj)):
                self.student_patch_proj[cur_patch_proj_head_it].train()

            # Process baskbone crop view 1.     
            sparse_crop_1 = self.student.create_sparse_tensor(crop_dict_1)
            sparse_mask_crop_1 = self.student.create_sparse_tensor(mask_crop_dict_1)
            enc_feats_1 = self.student.forward_encoder_masked(sparse_mask_crop_1)
            _, student_feat_list_1 = self.student.forward_decoder_unmasked(
                enc_feats_1, sparse_crop_1, self.empty_voxel_feat_list)

            # Compute local feature crop view 1.
            student_feat_list_proj_1 = []
            for cur_feat_iter in range(len(student_feat_list_1)):
                student_feat_list_proj_1.append(student_feat_list_1[cur_feat_iter].replace_feature(
                    self.student_patch_proj[cur_feat_iter](student_feat_list_1[cur_feat_iter].features)))
                
            # Process baskbone crop view 2.     
            sparse_crop_2 = self.student.create_sparse_tensor(crop_dict_2)
            sparse_mask_crop_2 = self.student.create_sparse_tensor(mask_crop_dict_2)
            enc_feats_2 = self.student.forward_encoder_masked(sparse_mask_crop_2)
            _, student_feat_list_2 = self.student.forward_decoder_unmasked(
                enc_feats_2, sparse_crop_2, self.empty_voxel_feat_list)
            
            # Compute local feature crop view 2.
            student_feat_list_proj_2 = []
            for cur_feat_iter in range(len(student_feat_list_2)):
                student_feat_list_proj_2.append(student_feat_list_2[cur_feat_iter].replace_feature(
                    self.student_patch_proj[cur_feat_iter](student_feat_list_2[cur_feat_iter].features)))
            
            ################# TIME
            if PRINT_TIMING:
                event_3.record()
            ################# FTIME

            #### Compute origin coord hierarchy.
            with torch.no_grad():
                def compute_orig_coord_list(in_orig_coord, cur_feature_list):
                    orig_coord_list = [in_orig_coord]
                    for cur_feat_iter in range(len(cur_feature_list)-1):
                        index_full = cur_feature_list[cur_feat_iter].indices[:,1:]//2
                        index_full = torch.cat((
                            cur_feature_list[cur_feat_iter].indices[:,0].reshape((-1,1)),
                            index_full), -1)
                        aux_indices, _ = match_3d_coords(
                            cur_feature_list[cur_feat_iter+1].indices, index_full)
                        new_orig_coords = scatter(
                            orig_coord_list[-1],
                            aux_indices.to(torch.int64),
                            dim = 0, 
                            dim_size = cur_feature_list[cur_feat_iter+1].indices.shape[0], 
                            reduce = 'mean')
                        orig_coord_list.append(new_orig_coords)
                    return orig_coord_list

                teacher_orig_coord_1 = compute_orig_coord_list(
                    full_dict_1["origin_coord"], teacher_feat_list_proj_1)
                teacher_orig_coord_2 = compute_orig_coord_list(
                    full_dict_2["origin_coord"], teacher_feat_list_proj_2)
                student_orig_coord_1 = compute_orig_coord_list(
                    crop_dict_1["origin_coord"], student_feat_list_proj_1)
                student_orig_coord_2 = compute_orig_coord_list(
                    crop_dict_2["origin_coord"], student_feat_list_proj_2)
            
            ################# TIME
            if PRINT_TIMING:
                event_4.record()
            ################# FTIME

            def match_contrastive_pair(
                view1_coord, view1_batch_ids, 
                view2_coord, view2_batch_ids, 
                max_k, max_radius):
                view1_offset = batch2offset(view1_batch_ids)
                view2_offset = batch2offset(view2_batch_ids)
                index, distance = pointops.knn_query(
                    max_k,
                    view2_coord.float(),
                    view2_offset.int(),
                    view1_coord.float(),
                    view1_offset.int(),
                )
                if max_k > 1:
                    index = torch.cat(
                        [
                            torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                            .view(-1, 1, 1)
                            .expand(-1, max_k, 1),
                            index.view(-1, max_k, 1),
                        ],
                        dim=-1,
                    )[distance.squeeze(-1) < max_radius]
                    _, count = index[:, 0].unique(return_counts=True)
                    select = (
                        torch.cumsum(count, dim=0)
                        - torch.randint(count.max(), count.shape, device=count.device) % count
                        - 1
                    )
                    index = index[select]
                else:
                    index = torch.cat(
                        [
                            torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                            .view(-1, 1),
                            index.view(-1, 1),
                        ],
                        dim=-1,
                    )[distance.squeeze(-1) < max_radius]
                return index
            

            teacher_patch_features_loss_1 = []
            teacher_patch_features_loss_2 = []
            student_patch_features_loss_1 = []
            student_patch_features_loss_2 = []
            patch_batch_ids_1 = []
            patch_batch_ids_2 = []
            enc_feats_1.reverse()
            enc_feats_2.reverse()
            for cur_feat_iter in range(len(teacher_feat_list_proj_1)):

                # Get patches view 1.
                _, is_empty = match_3d_coords(
                    enc_feats_1[cur_feat_iter].indices,
                    student_feat_list_proj_1[cur_feat_iter].indices)
                
                cur_student_feats = student_feat_list_proj_1[cur_feat_iter].features[is_empty]
                cur_student_batch_ids = student_feat_list_proj_1[cur_feat_iter].indices[is_empty, 0]
                cur_student_orig_pos = student_orig_coord_1[cur_feat_iter][is_empty]
                cur_student_sort_idxs = torch.argsort(cur_student_batch_ids)
                cur_student_feats = cur_student_feats[cur_student_sort_idxs].contiguous()
                cur_student_batch_ids = cur_student_batch_ids[cur_student_sort_idxs].contiguous()
                cur_student_orig_pos = cur_student_orig_pos[cur_student_sort_idxs].contiguous()

                cur_teacher_feats = teacher_feat_list_proj_2[cur_feat_iter].features
                cur_teacher_batch_ids = teacher_feat_list_proj_2[cur_feat_iter].indices[:, 0]
                cur_teacher_orig_pos = teacher_orig_coord_2[cur_feat_iter]
                cur_teacher_sort_idxs = torch.argsort(cur_teacher_batch_ids)
                cur_teacher_feats = cur_teacher_feats[cur_teacher_sort_idxs].contiguous()
                cur_teacher_batch_ids = cur_teacher_batch_ids[cur_teacher_sort_idxs].contiguous()
                cur_teacher_orig_pos = cur_teacher_orig_pos[cur_teacher_sort_idxs].contiguous()

                match_idxs = match_contrastive_pair(
                    cur_student_orig_pos, cur_student_batch_ids, 
                    cur_teacher_orig_pos, cur_teacher_batch_ids,
                    self.match_ks[cur_feat_iter], self.match_radius[cur_feat_iter])

                student_patch_features_loss_1.append(cur_student_feats[match_idxs[:, 0]])
                teacher_patch_features_loss_1.append(cur_teacher_feats[match_idxs[:, 1]])
                patch_batch_ids_1.append(cur_student_batch_ids[match_idxs[:, 0]])
                
                # Get patches view 2.
                _, is_empty = match_3d_coords(
                    enc_feats_2[cur_feat_iter].indices,
                    student_feat_list_proj_2[cur_feat_iter].indices)
                
                cur_student_feats = student_feat_list_proj_2[cur_feat_iter].features[is_empty]
                cur_student_batch_ids = student_feat_list_proj_2[cur_feat_iter].indices[is_empty, 0]
                cur_student_orig_pos = student_orig_coord_2[cur_feat_iter][is_empty]
                cur_student_sort_idxs = torch.argsort(cur_student_batch_ids)
                cur_student_feats = cur_student_feats[cur_student_sort_idxs].contiguous()
                cur_student_batch_ids = cur_student_batch_ids[cur_student_sort_idxs].contiguous()
                cur_student_orig_pos = cur_student_orig_pos[cur_student_sort_idxs].contiguous()

                cur_teacher_feats = teacher_feat_list_proj_1[cur_feat_iter].features
                cur_teacher_batch_ids = teacher_feat_list_proj_1[cur_feat_iter].indices[:, 0]
                cur_teacher_orig_pos = teacher_orig_coord_1[cur_feat_iter]
                cur_teacher_sort_idxs = torch.argsort(cur_teacher_batch_ids)
                cur_teacher_feats = cur_teacher_feats[cur_teacher_sort_idxs].contiguous()
                cur_teacher_batch_ids = cur_teacher_batch_ids[cur_teacher_sort_idxs].contiguous()
                cur_teacher_orig_pos = cur_teacher_orig_pos[cur_teacher_sort_idxs].contiguous()

                match_idxs = match_contrastive_pair(
                    cur_student_orig_pos, cur_student_batch_ids, 
                    cur_teacher_orig_pos, cur_teacher_batch_ids,
                    self.match_ks[cur_feat_iter], self.match_radius[cur_feat_iter])

                student_patch_features_loss_2.append(cur_student_feats[match_idxs[:, 0]])
                teacher_patch_features_loss_2.append(cur_teacher_feats[match_idxs[:, 1]])
                patch_batch_ids_2.append(cur_student_batch_ids[match_idxs[:, 0]])
                
            ################# TIME
            if PRINT_TIMING:
                event_5.record()
            ################# FTIME

            #### Compute loss.
            loss_dict = self.loss(
                [student_patch_features_loss_1, student_patch_features_loss_2],
                [teacher_patch_features_loss_1, teacher_patch_features_loss_2],
                [patch_batch_ids_1, patch_batch_ids_2])
            

            #### Update momentum model and weight decay.
            if input_dict['update_weights']:
                with torch.no_grad():
                    self.update_momentum_model()
                    if not self.wd_scheduler is None:
                        self.wd = self.wd_scheduler.__getitem__()

            #### Add weight decay and momentum to loss dictionary.
            loss_dict["wd"] = self.wd
            loss_dict["momentum"] = self.momentum

            ################# TIME
            if PRINT_TIMING:
                end.record()
                torch.cuda.synchronize()
                print("Data", start.elapsed_time(event_1))
                print("Teacher", event_1.elapsed_time(event_2))
                print("Student", event_2.elapsed_time(event_3))
                print("Original Coords", event_3.elapsed_time(event_4))
                print("Matching", event_4.elapsed_time(event_5))
                print("Loss", event_5.elapsed_time(end))
                print()
            ################# FTIME

            return loss_dict
        
        ######## Validation.
        else:
            
            # Compute deep features.
            cur_dict = {
                "grid_coord" : input_dict["grid_coord"],
                "feat" : input_dict["feat"],
                "offset" : input_dict["offset"]}  
            return self.student(cur_dict)[-1]