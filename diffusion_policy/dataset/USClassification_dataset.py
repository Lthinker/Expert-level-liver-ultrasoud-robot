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
import zarr

class CreateClassificationSampler():
    def __init__(self, data, mask, episode_ends):
        self.data = data
        self.indexes = []
        for ii in range(0,len(episode_ends)):
            if mask[ii] == 0:
                continue
            if ii == 0:
                epibegin = 0
                epiend = episode_ends[ii]
            else:
                epibegin = episode_ends[ii-1]
                epiend = episode_ends[ii]
            self.indexes.append(np.arange(epibegin,epiend))
        self.indexes = np.concatenate(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def sample_sequence(self,idx):
        realindex = self.indexes[idx]
        returndata = {}
        for key in self.data.keys():
            returndata[key] = self.data[key][realindex]
        return returndata


class USSimpleClassificationDataset(BaseImageDataset):
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
        self.img_store = zarr.open(zarr_path,'r') 
        self.data = self.img_store['data']['img'][:]
        self.label = np.expand_dims(self.img_store['data']['label'][:],-1)
        self.episode_ends = self.img_store['meta']['episode_ends'][:]
        split = int(len(self.episode_ends)*0.8)
        val_mask = np.zeros(len(self.episode_ends),dtype=bool);val_mask[split:] = True
        train_mask = ~val_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        
        # 先造一个train sampler
        self.sampler = CreateClassificationSampler({'img':self.data,'label':self.label},train_mask,self.episode_ends)
        num_positive = (self.label[:self.episode_ends[split]] == 1).sum().item()  
        num_negative = (self.label[:self.episode_ends[split]] == 0).sum().item() 
        self.pos_weight = num_negative / num_positive

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = CreateClassificationSampler({'img':self.data,'label':self.label},~self.train_mask,self.episode_ends)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample['img'].astype(np.float32),-1,0)/255 
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
            },
            'label': sample['label'].astype(np.float32) # T, 3
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data