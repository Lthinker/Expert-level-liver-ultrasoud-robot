from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
import dill
logger = logging.getLogger(__name__)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed # dsed=128
        if global_cond_dim is not None: # global_cond_dim=132
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:])) # now, cond_dim=260, all_dims = [2, 512, 1024, 2048], [(2, 512), (512, 1024), (1024, 2048)]

        local_cond_encoder = None
        if local_cond_dim is not None: # 这里pushT的时候是None
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1] # this is 2048
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, # 2048, 2048, 260
                kernel_size=kernel_size, n_groups=n_groups, # 5, 8
                cond_predict_scale=cond_predict_scale # True
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, # 2048, 2048, 260
                kernel_size=kernel_size, n_groups=n_groups, # 5, 8
                cond_predict_scale=cond_predict_scale # True
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out): # [(2, 512), (512, 1024), (1024, 2048)]
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ])) # 所以最后一层应该就没有downsampled而是直接出2048, downsample要从forward里面看一下nn.Conv1d(dim, dim, 3, 2, 1)

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): # Unet上升支，和前述代码一致
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity() # nn.ConvTranspose1d(dim, dim, 4, 2, 1)
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim) # zh: batch size, sequenze length, features
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps) # [64,128]  # bs 128

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1) # [64,260]
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder # 这里隐约就是2个residual block，resnet一个，resnet2一个
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample # [64,2,16]
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature) # s1 x: [64,2,16] -> [64,512,16] s2 [64,512,8] -> [64,1024,8] s3 [64,1024,4] -> [64,2048,4] 
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature) # s1 [64,512,16] -> [64,512,16] s2 [64,1024,8] -> [64,1024,8] s3 [64,2048,4] -> [64,2048,4] 
            h.append(x)
            x = downsample(x) # downsample在第三维发生 s1 [64,512,16] -> [64,512,8] s2 [64,1024,8] -> [64,1024,4] s3 [64,2048,4] -> [64,2048,4] 

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature) # [64,260] and [64,2048,4],x的形状一直保持[64,2048,4]

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1) # U net残差链接 s1 [64,2048,4] concat之后是 [64,4096,4]
            x = resnet(x, global_feature) # 压缩到[64,1024,4]
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature) # s1 [64,1024,4] -> [64,1024,4]
            x = upsample(x) # s1 [64,1024,4] -> [64,1024,8] s2 [64,512,8] -> [64,512,16]

        x = self.final_conv(x) # [64,512,16] -> [64,2,16]

        x = einops.rearrange(x, 'b t h -> b h t')
        return x   # 预测的噪声
    
    # yxs 0820
    def forward_no_cfg(self):
        self.eval()

        acc_mean = self.forward()
        acc_mean = torch.reshape(acc_mean,())
        acc_mean = torch.split(acc_mean,)[0]   
        return acc_mean

class ControlNet_ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=3, n_groups=8, cond_predict_scale=False,basecheckpoint=None,freeze_base=True):
        super().__init__()
        print("ControlNet_ConditionalUnet1D")
        print("basechekpoint",basecheckpoint)
        print("freeze_base",freeze_base)
        self.basenet = ConditionalUnet1D(input_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale)
        self.controlnet = ConditionalUnet1D(input_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale)
        if basecheckpoint is not None:
            # debug
            payload = torch.load(open(basecheckpoint, 'rb'), pickle_module=dill)
            state_dict = payload['state_dicts']['ema_model']  
            
            model_state_dict = {}  
            for key, value in state_dict.items():  
                if key.startswith('model.'):  
                    # 移除前缀 'model.'  
                    new_key = key.replace('model.', '', 1)  
                    model_state_dict[new_key] = value  
            load_result_base = self.basenet.load_state_dict(model_state_dict, strict=False) 
            load_result_conl = self.controlnet.load_state_dict(model_state_dict, strict=False) 
            assert(len(load_result_base.missing_keys) == 0 and len(load_result_base.unexpected_keys) == 0 and len(load_result_conl.missing_keys) == 0 and len(load_result_conl.unexpected_keys) == 0)
        ### -------------------------------------- ###
        #         Insert zero conv module
        ### -------------------------------------- ###     
        self.zero_convs = nn.ModuleList([])
        for idx, (resnet, resnet2, downsample) in enumerate(self.basenet.down_modules):
            self.zero_convs.append(self.make_zero_conv(resnet2.out_channels))
        self.zero_convs.append(self.make_zero_conv(self.basenet.mid_modules[-1].out_channels))

    def make_zero_conv(self, channels):
        return zero_module(nn.Conv1d(channels, channels, 1, padding=0))
    def forward(self, 
            sample: torch.Tensor, 
            control_input: torch.Tensor,
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim) # zh: batch size, sequenze length, features
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')
        control_input = einops.rearrange(control_input, 'b h t -> b t h')
        ### -------------------------------------- ###
        #                Condition
        ### -------------------------------------- ###
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.basenet.diffusion_step_encoder(timesteps) # [64,128]

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1) # [64,260]
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            # 这里没改，但其实感觉condition可以分出去
            assert 0
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder # 这里隐约就是2个residual block，resnet一个，resnet2一个
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        ### -------------------------------------- ###
        #                Denoise
        ### -------------------------------------- ###
        ### -------------------------------------- ###
        #       Base network down sample
        ### -------------------------------------- ###
        x = sample # [64,2,16]
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.basenet.down_modules):
            x = resnet(x, global_feature) # s1 x: [64,2,16] -> [64,512,16] s2 [64,512,8] -> [64,1024,8] s3 [64,1024,4] -> [64,2048,4] 
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature) # s1 [64,512,16] -> [64,512,16] s2 [64,1024,8] -> [64,1024,8] s3 [64,2048,4] -> [64,2048,4] 
            h.append(x)
            x = downsample(x) # downsample在第三维发生 s1 [64,512,16] -> [64,512,8] s2 [64,1024,8] -> [64,1024,4] s3 [64,2048,4] -> [64,2048,4] 

        for mid_module in self.basenet.mid_modules:
            x = mid_module(x, global_feature) # [64,260] and [64,2048,4],x的形状一直保持[64,2048,4]
        h.append(x)

        ### -------------------------------------- ###
        #       Control network down sample
        ### -------------------------------------- ###
        zeroconvcount = 0
        x = control_input # [64,2,16]
        h_control = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.controlnet.down_modules):
            x = resnet(x, global_feature) # s1 x: [64,2,16] -> [64,512,16] s2 [64,512,8] -> [64,1024,8] s3 [64,1024,4] -> [64,2048,4] 
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature) # s1 [64,512,16] -> [64,512,16] s2 [64,1024,8] -> [64,1024,8] s3 [64,2048,4] -> [64,2048,4] 
            h_control.append(self.zero_convs[zeroconvcount](x));zeroconvcount+=1
            x = downsample(x) # downsample在第三维发生 s1 [64,512,16] -> [64,512,8] s2 [64,1024,8] -> [64,1024,4] s3 [64,2048,4] -> [64,2048,4] 

        for mid_module in self.controlnet.mid_modules:
            x = mid_module(x, global_feature) # [64,260] and [64,2048,4],x的形状一直保持[64,2048,4]
        h_control.append(self.zero_convs[zeroconvcount](x));zeroconvcount+=1
        
        ### -------------------------------------- ###
        #       Control network up sample
        ### -------------------------------------- ###        
        x_middle = h.pop() + h_control.pop()
        x = x_middle
        for idx, (resnet, resnet2, upsample) in enumerate(self.basenet.up_modules):
            x = torch.cat((x, h.pop()+h_control.pop()), dim=1) # U net残差链接 s1 [64,2048,4] concat之后是 [64,4096,4]
            x = resnet(x, global_feature) # 压缩到[64,1024,4]
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.basenet.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature) # s1 [64,1024,4] -> [64,1024,4]
            x = upsample(x) # s1 [64,1024,4] -> [64,1024,8] s2 [64,512,8] -> [64,512,16]

        x = self.basenet.final_conv(x) # [64,512,16] -> [64,2,16]

        x = einops.rearrange(x, 'b t h -> b h t')
        return x