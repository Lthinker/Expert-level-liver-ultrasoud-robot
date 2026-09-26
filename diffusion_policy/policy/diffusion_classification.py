from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D,ControlNet_ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import robotcontrol
import time
import importlib
import math3d as m3d 
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicyForce
import sys
sys.path.append('efficient-kan/src/')
from efficient_kan import KANLinear

class NormClassifer(nn.Module):
    def __init__(self):
        super(NormClassifer, self).__init__()
        self.fc1 = KANLinear(64, 64)
        self.fc2 = KANLinear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.fc1(x)) + x
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class FocalLoss(nn.Module):  
    def __init__(self, alpha=1, gamma=2, reduce=True, size_average=True):  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.reduce = reduce  
        self.size_average = size_average  

    def forward(self, inputs, targets):  
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  
        pt = torch.exp(-BCE_loss)  # 通过取负的损失指数计算概率  

        # 计算 Focal Loss  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  
        
        if self.reduce:  
            return F_loss.mean() if self.size_average else F_loss.sum()  
        
        return F_loss 

class DiffusionUnetClassification(DiffusionUnetHybridImagePolicyForce):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            position_img_fusion=None,
            # parameters passed to step
            **kwargs):
        super().__init__(shape_meta=shape_meta,  
                         noise_scheduler=noise_scheduler,  
                         horizon=horizon,  
                         n_action_steps=n_action_steps,  
                         n_obs_steps=n_obs_steps,  
                         num_inference_steps=num_inference_steps,  
                         obs_as_global_cond=obs_as_global_cond,  
                         crop_shape=crop_shape,  
                         diffusion_step_embed_dim=diffusion_step_embed_dim,  
                         down_dims=down_dims,  
                         kernel_size=kernel_size,  
                         n_groups=n_groups,  
                         cond_predict_scale=cond_predict_scale,  
                         obs_encoder_group_norm=obs_encoder_group_norm,  
                         eval_fixed_crop=eval_fixed_crop,  
                         **kwargs) 

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            # lzh add force
            global_cond_dim = obs_feature_dim * n_obs_steps + 6 * n_obs_steps # force

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        # if 'resume' in kwargs:
        #     model.load_state_dict(torch.load(kwargs['resume']['checkpoint_path'])['state_dicts']['model'])
        # Position Image Fusion Class, PIFUsion
        if not (position_img_fusion is None):
            if isinstance(position_img_fusion['class_path'], str):  
                module_name, class_name = position_img_fusion['class_path'].rsplit(".", 1)  
                module = importlib.import_module(module_name)  
                PIFusion = getattr(module, class_name)  
                self.PIFusion = PIFusion(input_dim=position_img_fusion['input_dim'],output_dim=position_img_fusion['output_dim'],hidden_dim=position_img_fusion['hidden_dim'])
            else:  
                assert(0)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.NormClassifer = NormClassifer()
        
        if 'loss_fn' in kwargs:
            module_name, class_name = kwargs['loss_fn']['class_path'].rsplit(".", 1)  
            self.class_name = class_name
            module = importlib.import_module(module_name)  
            criterion = getattr(module, class_name)
            self.criterion = criterion(**kwargs['loss_fn']['args'])  
        
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def set_criterion(self,pos_weight):
        if self.class_name == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            pass
    def rel2abs(self, nactions,n_obs_steps): 
        bs, T, Da = nactions.shape
        abs_actions = torch.zeros_like(nactions)
        abs_actions[:,:,:6] = nactions[:,:,:6] 

        abs_actions[:,n_obs_steps-1,6:] = 0 

        for bb in range(0,bs):
            current_pose_6d = robotcontrol.rvec2quat(np.zeros(6))
            for tt in range(n_obs_steps,T):
                operation = nactions[bb, tt-1, 6:].cpu().numpy()
                
                operation[:3] = operation[:3] 
                operation[3:] = operation[3:] 
                current_pose_6d = robotcontrol.update_pose_diffusionpolicy(operation = operation, current_pose_6d = current_pose_6d)
                pose_next = robotcontrol.quat2rec(current_pose_6d)
                euler = robotcontrol.quat_to_euler(current_pose_6d[3:])
                pose_next = np.array([pose_next[0],pose_next[1],pose_next[2],euler[0],euler[1],euler[2]])
                abs_actions[bb, tt, 6:] = torch.tensor(pose_next)

        for bb in range(0,bs):
            current_pose_6d = robotcontrol.rvec2quat(np.zeros(6))
            for tt in range(n_obs_steps-2,0-1,-1): 
                operation = nactions[bb, tt, 6:].cpu().numpy() 
                # 数据单位转换  
                operation[:3] = operation[:3] 
                operation[3:] = operation[3:] 
                current_pose_6d = robotcontrol.update_pose_diffusionpolicy_inverse(operation = operation, current_pose_6d = current_pose_6d)
                # 单元测试看看能不能变回来
                pose_next = robotcontrol.quat2rec(current_pose_6d)
                euler = robotcontrol.quat_to_euler(current_pose_6d[3:])
                pose_next = np.array([pose_next[0],pose_next[1],pose_next[2],euler[0],euler[1],euler[2]])
                abs_actions[bb, tt, 6:] = torch.tensor(pose_next)
        return abs_actions
    
    def compute_loss(self, batch):
       
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs']) 
        target = batch['label']
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = nobs
            nobs_features = self.obs_encoder(this_nobs) 
            pred = self.NormClassifer(nobs_features)
        
        loss = self.criterion(pred, target)        
        return loss
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            assert 'past_action' not in obs_dict 
       
            if 'obs' in obs_dict:
                nobs = self.normalizer.normalize(obs_dict['obs'])
            else:
                nobs = self.normalizer.normalize(obs_dict) 
            # build input
            device = self.device
            dtype = self.dtype
            batch_size = nobs['image'].shape[0]

            # handle different ways of passing observation
            local_cond = None
            global_cond = None
            # abs_actions 
            if self.obs_as_global_cond: 
                # reshape B, T, .. to B*T
                this_nobs = nobs
                nobs_features = self.obs_encoder(this_nobs) 

            result = {
                'pred_label': (pred>0.5).float(),
                'pred': pred
            }
            return result