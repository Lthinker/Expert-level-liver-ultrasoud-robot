from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import importlib.util
import sys
import cv2

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class USSimpleImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path( 
            zarr_path, keys=['img', 'action','force_state']) 
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed) # e.g., val_mask.shape (206,), contains True and False
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon, # e.g., horizon=16
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'], 
            'force_state': self.replay_buffer['force_state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample['img'].astype(np.float32),-1,1)/255 #

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'force_state': sample['force_state'].astype(np.float32) # T, 3
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
class USSimpleImageDatasetDomainRam3(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            DomainRamPath = '/Data3/lzhdata3/DomainRam',
            use_domain_randomization = False,
            mask_path = '/Data3/lzhdata3/DomainRam/default_convex_mask.npz',
            batchUniform = False,
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path( 
            zarr_path, keys=['img', 'action','force_state']) 
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed) # e.g., val_mask.shape (206,), contains True and False
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon, # e.g., horizon=16
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batchUniform = batchUniform
        self.use_domain_randomization = use_domain_randomization
        if self.use_domain_randomization:
            domain_ram_module = import_from_path(
                "DomainRam3Plus", 
                DomainRamPath
            )
            # 获取类
            FixedMaskConvexDomainRandomization = domain_ram_module.FixedMaskConvexDomainRandomization
            self.domain_randomizer = FixedMaskConvexDomainRandomization(mask_path)
            print("域随机化已启用")
        else:
            self.domain_randomizer = None
            print("域随机化已禁用")
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'force_state': self.replay_buffer['force_state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)
    def _apply_domain_randomization(self, image,batchUniform):
        import torch
        
        # image shape: (T, C, H, W)
        T, C, H, W = image.shape
        augmented_images = []
        
        for t in range(T):
            for c in range(C):
                single_channel = torch.tensor(image[t, c], dtype=torch.float32)   
                augmented_channel = self.domain_randomizer.apply_all_augmentations(single_channel)
                augmented_images.append(augmented_channel.numpy())
        
        # (T, C, H, W) 
        augmented_image = np.array(augmented_images).reshape(T, C, H, W)
        
        return augmented_image
    def _sample_to_data(self, sample):
        image = np.moveaxis(sample['img'].astype(np.float32),-1,1)/255 # 这里只有action和image
        image = np.mean(image,axis=1,keepdims=True)  # 16 1 400 400
        
        if self.domain_randomizer is not None:
            image = self._apply_domain_randomization(image,batchUniform=self.batchUniform)
        else:
            pass
        B, C, H, W = image.shape
        target_size = 226
        resized_images = np.zeros((B, C, target_size, target_size), dtype=np.float32)
        for i in range(B):
            for c in range(C):
                resized_images[i, c] = cv2.resize(image[i, c], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        # image = resized_images

        image = np.concatenate([resized_images]*3,axis=1)
        data = {
            'obs': {
                'image': image, 
                'force_state': sample['force_state'].astype(np.float32) # T, 3
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
