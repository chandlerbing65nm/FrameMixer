import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameMixup(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        temperature=0.2, 
        frame_reduction_ratio=None, 
        device='cuda'
        ):
        super(FrameMixup, self).__init__()
        self.seq_len = seq_len
        self.frame_reduction_ratio = frame_reduction_ratio
        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len

        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)  # [1, reduced_len, seq_len]
        self.temperature = temperature

    def forward(self, feature):

        batch_size, seq_len, feat_dim = feature.size()  # feature: [batch_size, seq_len, feat_dim]
        noise_template = self.noise_template.expand(batch_size, -1, -1)  # [batch_size, reduced_len, seq_len]
        augmenting_path = self.compute_augmenting_path(noise_template)  # [batch_size, reduced_len, seq_len]

        augmented_feature = self.apply_augmenting(feature, augmenting_path)  # [batch_size, reduced_len, feat_dim]
        return augmented_feature  # [batch_size, reduced_len, feat_dim]

    def compute_augmenting_path(self, noise_template):
        mu, sigma = 0, 1  # mean and standard deviation for Gaussian
        gaussian_noise = torch.normal(mu, sigma, size=noise_template.size(), device=noise_template.device)  # [batch_size, reduced_len, seq_len]
        return F.softmax(gaussian_noise / self.temperature, dim=-2)  # [batch_size, reduced_len, seq_len]

    def apply_augmenting(self, feature, augmenting_path):
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)  # [batch_size, reduced_len, feat_dim]
        return augmented_feature  # [batch_size, reduced_len, feat_dim]


class FrameMixer(nn.Module):
    def __init__(self, in_t_dim, in_f_dim, temp=0.2, Tr=0.0):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        self.temperature=temp 
        self.frame_reduction_ratio = None if Tr == 0.0 else Tr
        
        self.frame_augment = FrameMixup(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=self.temperature, 
            frame_reduction_ratio=self.frame_reduction_ratio,
            device='cuda'
        )

    def forward(self, x):
        ret = {}

        augment_frame = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        return augment_frame