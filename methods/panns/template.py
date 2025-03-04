import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchaudio
import librosa

from methods.panns.pytorch_utils import *
from methods.panns.models import *

from specaug.diffres.frontend import DiffRes
from specaug.fma.frontend import FMA
from specaug.specmix.frontend import SpecMix

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class SpecAugmenter(nn.Module):
    def __init__(self, sample_rate, hop_size, duration, mel_bins, args):
        super(SpecAugmenter, self).__init__()

        self.args = args
        self.training = True  # Will be updated based on the model's training state

        # Parse SpecAugment parameters
        specaugment_params = list(map(int, args.specaugment_params.split(',')))
        self.specaugment = SpecAugmentation(
            time_drop_width=specaugment_params[0],
            time_stripes_num=specaugment_params[1],
            freq_drop_width=specaugment_params[2],
            freq_stripes_num=specaugment_params[3]
        )

        # Parse DiffRes parameters
        diffres_params = args.diffres_params.split(',')
        dimension_reduction_rate = float(diffres_params[0])
        learn_pos_emb = diffres_params[1].lower() == 'true'
        self.diffres = DiffRes(
            in_t_dim=int((sample_rate / hop_size) * duration) + 1,
            in_f_dim=mel_bins,
            dimension_reduction_rate=dimension_reduction_rate,
            learn_pos_emb=learn_pos_emb
        )

        # Parse SpecMix parameters
        specmix_params = list(map(float, args.specmix_params.split(',')))
        self.specmix = SpecMix(
            prob=specmix_params[0],
            min_band_size=int(specmix_params[1]),
            max_band_size=int(specmix_params[2]),
            max_frequency_bands=int(specmix_params[3]),
            max_time_bands=int(specmix_params[4])
        )

        # Parse FrameMixer parameters
        framemixer_params = args.framemixer_params.split(',')
        self.fma = FMA(
            in_t_dim=int((sample_rate / hop_size) * duration) + 1,
            in_f_dim=mel_bins,
            temp=float(framemixer_params[0]),
            Tr=float(framemixer_params[1])
        )

    def forward(self, x):
        spec_aug = self.args.spec_aug
        output_dict = {}

        if spec_aug == 'diffres':
            x = x.squeeze(1)
            ret = self.diffres(x)
            guide_loss = ret["guide_loss"]
            x = ret["feature"].unsqueeze(1)
            output_dict['diffres_loss'] = guide_loss

        elif spec_aug == 'fma':
            x = x.squeeze(1)
            x = self.fma(x)
            x = x.unsqueeze(1)

        elif spec_aug == 'specaugment':
            if self.training:
                x = self.specaugment(x)

        elif spec_aug == 'specmix':
            if self.training:
                x, rn_indices, lam = self.specmix(x)
                output_dict['rn_indices'] = rn_indices
                output_dict['mixup_lambda'] = lam

        return x, output_dict
        
class PANNS_CNN6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_CNN6, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args
        self.duration = args.target_duration

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )

        # SpecAugmenter
        self.spec_augmenter = SpecAugmenter(
            sample_rate=sample_rate,
            hop_size=hop_size,
            duration=self.duration,
            mel_bins=mel_bins,
            args=args
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)  # Assuming 512 is embedding size

        self.bn = nn.BatchNorm2d(mel_bins)
        self.bn2 = nn.BatchNorm2d(mel_bins)
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        # Select the frontend based on args.frontend
        x = self.apply_frontend(input)

        # Precomputed features alignment and normalization
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = self.bn(x)         # Apply batch normalization
        x = x.transpose(1, 3)

        # Apply SpecAugmentations
        self.spec_augmenter.training = self.training  # Update training state
        x, aug_output = self.spec_augmenter(x)
        x_aug = x

        # Base model processing
        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_transfer(embedding)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding,
            'augmented': x_aug
        }

        if self.training and aug_output:
            output_dict.update(aug_output)

        return output_dict

    def apply_frontend(self, input):
        """
        Select and apply the appropriate frontend based on args.frontend.
        """
        if self.args.frontend == 'logmel':
            # Logmel frontend
            x = self.spectrogram_extractor(input)  # (batch, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch, 1, time_steps, freq_bins)
        elif self.args.frontend == 'lfcc':
            # LFCC frontend
            x = self.spectrogram_extractor(input)  # (batch, 1, time_steps, freq_bins)
            x = x.squeeze(1)  # (batch, time_steps, freq_bins)

            # Apply LFCC transformation
            lfcc_transform = torchaudio.transforms.LFCC(
                sample_rate=self.sample_rate,
                n_filter=self.mel_bins,  # Match the number of linear filter bins to mel_bins
                f_min=self.fmin,
                f_max=self.fmax,
                n_lfcc=self.mel_bins,  # Number of LFCC coefficients
                log_lf=True,
                speckwargs={
                    'n_fft': self.window_size,
                    'hop_length': self.hop_size,
                    'win_length': self.window_size,
                }
            ).to(input.device)  # Ensure LFCC is on the correct device
            x = lfcc_transform(x)
            # Reduce complex-valued tensor to magnitude
            if x.size(-1) == 2:  # If the last dimension is for real and imaginary parts
                x = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)  # (batch, time_steps, n_mfcc)
            x = x.unsqueeze(1)  # (batch, 1, time_steps, lfcc_bins)
        elif self.args.frontend == 'mfcc':
            # MFCC frontend
            x = self.spectrogram_extractor(input)  # (batch, 1, time_steps, freq_bins)
            x = x.squeeze(1)  # (batch, time_steps, freq_bins)

            # Ensure window tensor is on the correct device
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.mel_bins,
                log_mels=True,
                melkwargs={
                    'n_fft': self.window_size,
                    'hop_length': self.hop_size,
                    'n_mels': self.mel_bins,
                    'f_min': self.fmin,
                    'f_max': self.fmax
                }
            ).to(input.device)  # Move MFCC transform to the correct device
            x = mfcc_transform(x)
            # Reduce complex-valued tensor to magnitude
            if x.size(-1) == 2:  # If the last dimension is for real and imaginary parts
                x = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)  # (batch, time_steps, n_mfcc)
            x = x.unsqueeze(1)  # (batch, 1, time_steps, mfcc_bins)
        else:
            raise ValueError(f"Unsupported frontend type: {self.frontend}")

        # import ipdb; ipdb.set_trace() 
        # print(x.shape)

        return x


class PANNS_CNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_CNN14, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args
        self.duration = args.target_duration

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )

        # SpecAugmenter
        self.spec_augmenter = SpecAugmenter(
            sample_rate=sample_rate,
            hop_size=hop_size,
            duration=self.duration,
            mel_bins=mel_bins,
            args=args
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)  # Assuming 2048 is embedding size

        self.bn = nn.BatchNorm2d(mel_bins)
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch, time_steps, mel_bins)

        # Precomputed features alignment and normalization
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = self.bn(x)         # Apply batch normalization
        x = x.transpose(1, 3)

        # Apply SpecAugmentations
        self.spec_augmenter.training = self.training  # Update training state
        x, aug_output = self.spec_augmenter(x)

        # Base model processing
        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_transfer(embedding)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
        }

        if self.training and aug_output:
            output_dict.update(aug_output)

        return output_dict

class PANNS_RESNET22(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_RESNET22, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = ResNet22(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args
        self.duration = args.target_duration

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )
        
        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)  # Assuming 512 is embedding size
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict

class PANNS_MOBILENETV1(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None):

        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_MOBILENETV1, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args
        self.duration = args.target_duration

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Step 2: Optionally store the custom modules (but do not apply them yet)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # SpecAugmenter
        self.spec_augmenter = SpecAugmenter(
            sample_rate=sample_rate,
            hop_size=hop_size,
            duration=self.duration,
            mel_bins=mel_bins,
            args=args
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, num_classes, bias=True)  # Assuming 512 is embedding size
        
        self.bn = nn.BatchNorm2d(mel_bins)
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch, time_steps, mel_bins)

        # Precomputed features alignment and normalization
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = self.bn(x)         # Apply batch normalization
        x = x.transpose(1, 3)

        # Apply SpecAugmentations
        self.spec_augmenter.training = self.training  # Update training state
        x, aug_output = self.spec_augmenter(x)

        x = self.base.features(x)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_transfer(embedding)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
        }

        if self.training and aug_output:
            output_dict.update(aug_output)

        return output_dict

class PANNS_MOBILENETV2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None):

        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_MOBILENETV2, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args
        self.duration = args.target_duration

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = MobileNetV2(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Step 2: Optionally store the custom modules (but do not apply them yet)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # SpecAugmenter
        self.spec_augmenter = SpecAugmenter(
            sample_rate=sample_rate,
            hop_size=hop_size,
            duration=self.duration,
            mel_bins=mel_bins,
            args=args
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, num_classes, bias=True)  # Assuming 512 is embedding size
        
        self.bn = nn.BatchNorm2d(mel_bins)
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch, time_steps, mel_bins)

        # Precomputed features alignment and normalization
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = self.bn(x)         # Apply batch normalization
        x = x.transpose(1, 3)

        # Apply SpecAugmentations
        self.spec_augmenter.training = self.training  # Update training state
        x, aug_output = self.spec_augmenter(x)

        x = self.base.features(x)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_transfer(embedding)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
        }

        if self.training and aug_output:
            output_dict.update(aug_output)

        return output_dict

class PANNS_WAVEGRAM_CNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, freeze_base=False
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_WAVEGRAM_CNN14, self).__init__()
        audioset_classes_num = 527

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = Wavegram_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)  # Assuming 512 is embedding size
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)


        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


def main():
    # Create a dummy input
    batch_size = 8  # Define the batch size
    seq_length = 32000  # Sequence length (e.g., could be the length of an audio clip)
    
    # Dummy input tensor (batch_size, seq_length)
    dummy_input = torch.randn(batch_size, seq_length)

    # Instantiate the model
    model = PANNS_CNN6(
        sample_rate=128000, 
        window_size=2048, 
        hop_size=1024, 
        mel_bins=64, 
        fmin=50, 
        fmax=None, 
        num_classes=4  # Example number of classes for the classification task
    )

    # Load pretrained weights (if available)
    pretrained_checkpoint_path = "./weights/Cnn6_mAP=0.343.pth"
    
    # Check if the path to the pretrained model exists and load
    if os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained weights from {pretrained_checkpoint_path}")
        model.load_from_pretrain(pretrained_checkpoint_path)
    else:
        print(f"Pretrained model not found at {pretrained_checkpoint_path}. Skipping weight loading.")


    # Set the model to evaluation mode for testing (optional if no training-specific layers like BatchNorm)
    model.eval()

    # Forward pass through the model with the dummy input
    with torch.no_grad():  # No gradient calculation is needed for inference
        output_dict = model(dummy_input)

    # Retrieve and print outputs
    clipwise_output = output_dict['clipwise_output']
    embedding = output_dict['embedding']

    print(f"Clipwise output shape: {clipwise_output.shape}")  # Expected: (batch_size, classes_num)
    print(f"Embedding shape: {embedding.shape}")  # Expected: (batch_size, embedding_size)

if __name__ == "__main__":
    main()