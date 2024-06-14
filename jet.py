import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from math import sqrt

class EmbeddingLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding =  nn.Embedding(config.vocab_size,config.embed_dim)  #(B,T) -> (B,T,C)
        self.positional_embedding = nn.Embedding(config.seq_len,config.embed_dim) #(T) -> (T,C)
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device

    def forward(self, x):
        B, T = x.shape
        tok_embed = self.token_embedding(x) #(B,T,C)
        pos_embed = self.positional_embedding(torch.arange(T,device=self.device)) #(T,C)
        x = self.dropout(tok_embed + pos_embed)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.qkv_linear_layer = nn.Linear(config.embed_dim,3*config.embed_dim,bias=False,device=config.device) #(B,T,C) -> (B,T,3C)
        self.out_linear_layer = nn.Linear(config.embed_dim,config.embed_dim) #(B,T,C) -> (B,T,C)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        if config.mask_type == 'causal':
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.seq_len,config.seq_len)).view(1,1,config.seq_len,config.seq_len)
            )
        else:
            self.register_buffer(
                "mask",
                torch.ones(config.seq_len,config.seq_len).view(1,1,config.seq_len,config.seq_len)
            )

    def forward(self, x):
        B,T,C = x.shape #B,T,C = batch size,`seq_len`,`embed_dim`

        q, k, v = self.qkv_linear_layer(x).chunk(3,dim=-1) #(B,T,3C) -> 3 lots of (B,T,C)
        q = q.view(B,T,self.num_heads,-1).transpose(1,2)   #(B,T,C) -> (B,nh,T,C/nh)
        k = k.view(B,T,self.num_heads,-1).transpose(1,2)   #(B,T,C) -> (B,nh,T,C/nh)  
        v = v.view(B,T,self.num_heads,-1).transpose(1,2)   #(B,T,C) -> (B,nh,T,C/nh)  

        attention_lens = q @ k.transpose(-2,-1) / sqrt(C/self.num_heads)  #(B,nh,T,C/nh)*(B,nh,C/nh,T) -> (B,nh,T,T)
        attention_lens = attention_lens.masked_fill(self.mask[:,:,:T,:T]==0,-torch.inf)
        attention_lens = self.attn_dropout(F.softmax(attention_lens,dim=-1))

        # Now we can see v through a lens where everything attends to each other
        v_attended_to = attention_lens @ v #(B,nh,T,T)*(B,nh,T,C/nh)  -> (B,nh,T,C/nh)
        v_attended_to = v_attended_to.transpose(1,2).contiguous().view(B,T,C) # (B,nh,T,C/nh) -> (B,T,C)
        output = self.resid_dropout(self.out_linear_layer(v_attended_to)) #(B,T,C) -> (B,T,C)
        return  output 
        
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.embed_dim, config.embed_dim * config.ff_dim)
        self.layer_2 = nn.Linear(config.embed_dim * config.ff_dim, config.embed_dim)
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
        self.layernorm_1 = nn.LayerNorm(config.embed_dim)
        self.attention = MultiHeadAttention(config)
        self.layernorm_2 = nn.LayerNorm(config.embed_dim)
        self.feedforward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feedforward(self.layernorm_2(x))
        return x

def get_model(config):
    #Note: no softmax on last layer, this model outputs logits ready for crossentropy loss.
    model = nn.Sequential(
        EmbeddingLayer(config),
        *[TransformerBlock(config) for _ in range(config.num_blocks)],
        nn.Linear(config.embed_dim,config.vocab_size)
    )

    model.to(config.device)
    if config.world_size > 1:
        model = DDP(model)

    return model
