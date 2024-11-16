import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from math import sqrt

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        d_model: int,
        device: str,
        dropout: float
    ):

        super().__init__()
        self.token_embedding =  nn.Embedding(vocab_size,d_model)  #(B,T) -> (B,T,C)
        self.positional_embedding = nn.Embedding(n_ctx,d_model) #(T) -> (T,C)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        _, T = x.shape
        tok_embed = self.token_embedding(x) #(B,T,C)
        pos_embed = self.positional_embedding(torch.arange(T,device=self.device)) #(T,C)
        x = self.dropout(tok_embed + pos_embed)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        n_ctx: int,
        d_model: int,
        n_heads: int,
        device: str,
        dropout: float,
        mask_type: str
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.qkv_linear_layer = nn.Linear(d_model,3*d_model,bias=False,device=device) #(B,T,C) -> (B,T,3C)
        self.out_linear_layer = nn.Linear(d_model,d_model) #(B,T,C) -> (B,T,C)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        if mask_type == 'causal':
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(n_ctx,n_ctx)).view(1,1,n_ctx,n_ctx)
            )
        else:
            self.register_buffer(
                "mask",
                torch.ones(n_ctx,n_ctx).view(1,1,n_ctx,n_ctx)
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
    def __init__(
        self, 
        d_model: int,
        d_mlp: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_model * d_mlp)
        self.layer_2 = nn.Linear(d_model * d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        n_ctx: int,
        d_mlp: int,
        d_model: int,
        n_heads: int,
        device: str,
        dropout: float,
        mask_type: str
    ):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            n_ctx,
            d_model,
            n_heads,
            device, 
            dropout, 
            mask_type
        )

        self.layernorm_2 = nn.LayerNorm(d_model)

        self.feedforward = MLP(
            d_model,
            d_mlp,
            dropout
        )

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feedforward(self.layernorm_2(x))
        return x

def get_model(cfg):
    #Note: no softmax on last layer, this model outputs logits ready for crossentropy loss.
    model = nn.Sequential(

        EmbeddingLayer(
            cfg.vocab_size,
            cfg.n_ctx,
            cfg.d_model,
            cfg.device,
            cfg.dropout
        ),

        *[
            TransformerBlock(
                cfg.n_ctx,
                cfg.d_mlp,
                cfg.d_model,
                cfg.n_heads,
                cfg.device,
                cfg.dropout,
                cfg.mask_type
            )
            for _ in range(cfg.n_blocks)
        ],

        nn.Linear(cfg.d_model,cfg.vocab_size)

    )

    torch.set_float32_matmul_precision('high')

    model = torch.compile(
        DDP(model.to(cfg.device),cfg.device_id),
        mode="reduce-overhead"
    )

    return model
