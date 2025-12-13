import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from .builder import SS_LOSSES

@SS_LOSSES.register_module()
class SSHReconLoss(nn.Module):
    def __init__(
        self,
        local_weights = [1.0]):
        super(SSHReconLoss, self).__init__()
        self.local_weights = local_weights
    

    def local_reconstruction_loss(self, student, teacher):
        norm_teacher_local = F.normalize(teacher.to(torch.float32), 2, -1, eps=1e-6)
        norm_student_local = F.normalize(student.to(torch.float32), 2, -1, eps=1e-6)
        local_patch_loss = F.smooth_l1_loss(
            norm_student_local, norm_teacher_local, reduction='sum')
        local_patch_loss = local_patch_loss/student.shape[0]
        return local_patch_loss
    

    def forward(self, 
                patch_student,
                patch_teacher,
                patch_batch_ids):
        
        #### Local loss.
        
        # Compute patch loss.
        patch_loss = 0.0
        for cur_patch_iter in range(len(patch_teacher)): 
            cur_teacher_feats = patch_teacher[cur_patch_iter]
            cur_student_feats = patch_student[cur_patch_iter]
            for cur_patch_level_iter in range(len(cur_teacher_feats)):
                # Rec loss.
                patch_loss = patch_loss + \
                    self.local_weights[cur_patch_level_iter]*self.local_reconstruction_loss(
                    cur_student_feats[cur_patch_level_iter].to(torch.float32),
                    cur_teacher_feats[cur_patch_level_iter].to(torch.float32))
                # Cov loss.
                norm_cur_student = F.normalize(
                    cur_student_feats[cur_patch_level_iter].to(torch.float32), 2, -1, eps=1e-6)
                cov = torch.matmul(norm_cur_student.T, norm_cur_student)
                scalar =  cov.shape[1] / (norm_cur_student.shape[0] * norm_cur_student.shape[0] * 0.05)
                I = torch.eye(cov.shape[1], device=cov.device)
                

        patch_loss = patch_loss / float(len(patch_teacher) * len(self.local_weights))


        return {"loss" : patch_loss}