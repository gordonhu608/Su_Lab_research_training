import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, dim=512, heads=8, num_encoder=6,
                 num_decoder=6, dim_ff=2048, dropout=0.1,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(dim, heads, dim_ff, dropout)
        #encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder)
        decoder_layer = TransformerDecoderLayer(dim, heads, dim_ff, dropout)
        decoder_norm = nn.LayerNorm(dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder,decoder_norm, 
                                          return_intermediate=return_intermediate_dec)
        
        self._reset_parameters()
        self.dim = dim
        self.heads = heads
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        b, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, 
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(b, c, h, w) 


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers =_repeat(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, src, mask=None, key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, mask, key_padding_mask, pos)

        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        #Feedforward
        self.linear1 = nn.Linear(dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, mask=None, key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.attention(q, k, value=src, attn_mask=mask, 
                              key_padding_mask= key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src 
    
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _repeat(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None,
                pos = None, query_pos = None ):
        out = tgt
        intermediate = []
    
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)     
            if self.return_intermediate:
                intermediate.append(self.norm(out))  

        if self.norm is not None:
            out = self.norm(out)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(out)
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return out.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask= tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt 

def _repeat(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        dim=args.hidden_dim,
        dropout=args.dropout,
        heads=args.nheads,
        dim_ff=args.dim_feedforward,
        num_encoder=args.enc_layers,
        num_decoder=args.dec_layers,
        return_intermediate_dec=True,
    )