import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from math import sqrt

class EmbeddingLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding =  nn.Embedding(config.vocab_size,config.d_model)  #(B,T) -> (B,T,C)
        self.positional_embedding = nn.Embedding(config.n_ctx,config.d_model) #(T) -> (T,C)
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device

    def forward(self, x):
        _, T = x.shape
        tok_embed = self.token_embedding(x) #(B,T,C)
        pos_embed = self.positional_embedding(torch.arange(T,device=self.device)) #(T,C)
        x = self.dropout(tok_embed + pos_embed)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.qkv_linear_layer = nn.Linear(config.d_model,3*config.d_model,bias=False,device=config.device) #(B,T,C) -> (B,T,3C)
        self.out_linear_layer = nn.Linear(config.d_model,config.d_model) #(B,T,C) -> (B,T,C)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        if config.mask_type == 'causal':
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.n_ctx,config.n_ctx)).view(1,1,config.n_ctx,config.n_ctx)
            )
        else:
            self.register_buffer(
                "mask",
                torch.ones(config.n_ctx,config.n_ctx).view(1,1,config.n_ctx,config.n_ctx)
            )

    def forward(self, x):
        B,T,C = x.shape #B,T,C = batch size,`n_ctx`,`d_model`

        q, k, v = self.qkv_linear_layer(x).chunk(3,dim=-1)  #(B,T,3C) -> 3 lots of (B,T,C)
        q = q.view(B,T,self.n_heads,-1).transpose(1,2)      #(B,T,C) -> (B,nh,T,C/nh)
        k = k.view(B,T,self.n_heads,-1).transpose(1,2)      #(B,T,C) -> (B,nh,T,C/nh)  
        v = v.view(B,T,self.n_heads,-1).transpose(1,2)      #(B,T,C) -> (B,nh,T,C/nh)  

        attention_pattern = q @ k.transpose(-2,-1) / sqrt(C/self.n_heads)  #(B,nh,T,C/nh)*(B,nh,C/nh,T) -> (B,nh,T,T)
        attention_pattern = attention_pattern.masked_fill(self.mask[:,:,:T,:T]==0,-torch.inf)
        attention_pattern = self.attn_dropout(F.softmax(attention_pattern,dim=-1))

        v_attended_to = attention_pattern @ v #(B,nh,T,T)*(B,nh,T,C/nh)  -> (B,nh,T,C/nh)
        v_attended_to = v_attended_to.transpose(1,2).contiguous().view(B,T,C) # (B,nh,T,C/nh) -> (B,T,C)
        output = self.resid_dropout(self.out_linear_layer(v_attended_to)) #(B,T,C) -> (B,T,C)
        return  output 
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.d_model, config.d_model * config.d_mlp)
        self.layer_2 = nn.Linear(config.d_model * config.d_mlp, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.layernorm_2 = nn.LayerNorm(config.d_model)
        self.feedforward = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feedforward(self.layernorm_2(x))
        return x

def get_model(config):
    #Note: no softmax on last layer, this model outputs logits ready for crossentropy loss.
    model = nn.Sequential(
        EmbeddingLayer(config),
        *[TransformerBlock(config) for _ in range(config.n_blocks)],
        nn.Linear(config.d_model,config.vocab_size)
    )
    torch.set_float32_matmul_precision('high')
    model = torch.compile(DDP(model.to(config.device),config.device_id),mode="reduce-overhead")

    return model
