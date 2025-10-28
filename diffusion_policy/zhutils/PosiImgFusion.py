from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Module
import math
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
import sys

sys.path.append('efficient-kan/src/efficient_kan/')
from kan import KANLinear

def positional_encoding(coords, freq_bands):  
    N, D = coords.shape  
    coords = coords.unsqueeze(-1)  # [N, D, 1]  
    coords_freq = coords * freq_bands  # [N, D, num_freqs]  
    coords_sin = torch.sin(2 * math.pi * coords_freq)  # [N, D, num_freqs]  
    coords_cos = torch.cos(2 * math.pi * coords_freq)  # [N, D, num_freqs]  
    encoded = torch.cat([coords_sin, coords_cos], dim=-1)  # [N, D, num_freqs * 2]  
    encoded_coords = encoded.view(N, -1)  # [N, D * num_freqs * 2]  
    return encoded_coords  

class AttnFusionKANForce(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_heads=8,n_obs_steps=5):  
        super(AttnFusionKANForce, self).__init__()  
        self.n_obs_steps = n_obs_steps
        self.max_freq = 10  
        self.num_freqs = 6  
        self.hidden_dim = hidden_dim  
        self.num_heads = num_heads  
        self.posi_mapping_fimg = KANLinear(2 * 6 * self.num_freqs, input_dim)  
        self.posi_mapping_fforce = KANLinear(2 * 6 * self.num_freqs, input_dim)  
        self.force_mapping = KANLinear(2 * 6 * self.num_freqs, input_dim)  
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)  
        self.fc = nn.Sequential(  
            KANLinear(input_dim, hidden_dim),  
            nn.ReLU(),  
            KANLinear(hidden_dim, output_dim)  
        ) 
        self.forcefc = nn.Sequential(  
            KANLinear(input_dim*2, hidden_dim),  
            nn.ReLU(),  
        )  
        self.forcefc2 = nn.Sequential(  
            KANLinear(n_obs_steps*64, n_obs_steps * 6),  
        )  
        
    def forward(self, ImageFeature, PosiFeature, Force):  
        batch_size, seq_len, _ = ImageFeature.shape  
        Posi = PosiFeature.reshape([-1,6])
        freq_bands = 2 ** torch.linspace(0, self.max_freq, steps=self.num_freqs,device=Posi.device)  # num_freq * Posi数目（6） * 2
        PEencoding = positional_encoding(coords=Posi, freq_bands=freq_bands)
        PEencoding_fimg = self.posi_mapping_fimg(PEencoding)
        PEencoding_fimg = PEencoding_fimg.reshape([PosiFeature.shape[0],PosiFeature.shape[1],-1])
        PEencoding_fforce = self.posi_mapping_fforce(PEencoding)
        Force = Force.reshape([-1,6])
        freq_bands = 2 ** torch.linspace(0, self.max_freq, steps=self.num_freqs,device=Force.device)
        Forceencoding = positional_encoding(coords=Force, freq_bands=freq_bands)
        Forceencoding = self.force_mapping(Forceencoding)
        Force_posi = torch.cat([PEencoding_fforce, Forceencoding],dim=-1)
        Force_posi = self.forcefc(Force_posi) + Forceencoding
        Force_posi = Force_posi.reshape([PosiFeature.shape[0],-1])
        Force_posi = self.forcefc2(Force_posi)
        query = PEencoding_fimg  # 24 5 64
        key = ImageFeature  # 24 5 64
        value = ImageFeature 
        attn_output, attn_weights = self.attention(query, key, value) 
        fusion_output = attn_output + ImageFeature  
        output = self.fc(fusion_output) 
        return output, Force_posi
    
    def forward_Memory(self, ImageFeature, PosiFeature, Force):  
        batch_size, seq_len, _ = ImageFeature.shape  
        assert(batch_size == 1)
        Posi = PosiFeature.reshape([-1,6])
        freq_bands = 2 ** torch.linspace(0, self.max_freq, steps=self.num_freqs,device=Posi.device)  # num_freq * Posi数目（6） * 2
        PEencoding = positional_encoding(coords=Posi, freq_bands=freq_bands)
        PEencoding_fimg = self.posi_mapping_fimg(PEencoding)
        PEencoding_fimg = PEencoding_fimg.reshape([PosiFeature.shape[0],PosiFeature.shape[1],-1])
        PEencoding_fforce = self.posi_mapping_fforce(PEencoding)
        Force = Force.reshape([-1,6])
        freq_bands = 2 ** torch.linspace(0, self.max_freq, steps=self.num_freqs,device=Force.device)
        Forceencoding = positional_encoding(coords=Force, freq_bands=freq_bands)
        Forceencoding = self.force_mapping(Forceencoding)

        # Choose the best feature
        query = PEencoding_fimg  
        key = ImageFeature  
        value = ImageFeature 
        attn_output, attn_weights = self.attention(query, key, value)  
        key_weight = attn_weights.sum(axis=1)
        top_vals, top_indices = torch.topk(key_weight, k=self.n_obs_steps, dim=-1)
        top_indices,_ = top_indices.sort()
        query = PEencoding_fimg[:,top_indices[0]]
        key = ImageFeature[:,top_indices[0]]
        value = ImageFeature[:,top_indices[0]]
        attn_output, attn_weights = self.attention(query, key, value)  
        fusion_output = attn_output + ImageFeature[:,top_indices[0]]  
        output = self.fc(fusion_output)  
        Force_posi = torch.cat([PEencoding_fforce[top_indices[0]], Forceencoding[top_indices[0]]],dim=-1)
        Force_posi = self.forcefc(Force_posi) + Forceencoding[top_indices[0]]
        Force_posi = Force_posi.reshape([PosiFeature.shape[0],-1])
        Force_posi = self.forcefc2(Force_posi)
        return output, Force_posi, top_indices[0]
    
