import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameAugment(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        temperature=0.2, 
        alpha=0.5,    # Alignment weight
        device='cuda'
    ):
        super(FrameAugment, self).__init__()
        
        # Initialize attributes
        self.seq_len = seq_len
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Template noise matrix
        self.noise_template = torch.randn(1, seq_len, seq_len, device=device)

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        
        # Expand noise template to batch size
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)
        
        # Compute alignment-aware augmenting path
        augmenting_path = self.compute_augmenting_path(mixing_matrix, feature)
        
        # Apply augmentation
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix, feature):
        # Generate Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(mixing_matrix) + EPS) + EPS)
        
        # Normalize Gumbel noise with softmax
        normalized_gumbel = F.softmax(gumbel_noise / self.temperature, dim=-1)
        
        # Compute alignment matrix using cosine similarity
        feature_flat = feature.reshape(feature.size(0), feature.size(1), -1)
        alignment_matrix = torch.einsum('bij,bkj->bik', feature_flat, feature_flat)  # Cosine similarity
        alignment_matrix = F.softmax(alignment_matrix, dim=-1)
        
        # Combine normalized Gumbel noise with alignment matrix
        augmenting_path = self.alpha * normalized_gumbel + (1 - self.alpha) * alignment_matrix
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        # Apply the augmenting path with matrix multiplication
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        return augmented_feature

class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameAugment(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=0.2, 
            device='cuda'
        )

    def forward(self, x):
        ret = {}

        augment_frame = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        # Final outputs
        ret["x"] = x
        ret["features"] = augment_frame
        ret["dummy"] = torch.tensor(0.0, device=x.device)
        ret["total_loss"] = ret["dummy"]

        return ret