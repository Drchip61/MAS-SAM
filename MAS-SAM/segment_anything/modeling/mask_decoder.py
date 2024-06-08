# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d

class senet(nn.Module):
    def __init__(self,c=256,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out+max_out
        x = x*self.sigmoid(out)
        #x = x.view(b,c,h,w)
        return x+res

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, 2, 2, 0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=7, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s),g=c1)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=c_)
        #self.add = shortcut and c1 == c2

    def forward(self, x):
        return x+self.cv2(self.cv1(x))# if self.add else self.cv2(self.cv1(x))

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        #self.dw = CrossConv(c1=out_c,c2=out_c)
        self.pw2 = Conv(out_c,out_c,1,1)
    def forward(self,x):
        x = self.pw1(x)
        x = self.se(x)
        x = self.pw2(x)
        #x = self.se(x)
        return x

class conv_block1(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        #self.dw = CrossConv(c1=out_c,c2=out_c)
        self.pw2 = Conv(out_c,out_c,1,1)
    def forward(self,x):
        x = self.pw1(x)
        #x = self.se(x)
        x = self.pw2(x)
        #x = self.se(x)
        return x

class conv_block_plus(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        self.axis_dw1 = CrossConv(c1=out_c,c2=out_c,k=7)
        self.axis_dw2 = CrossConv(c1=out_c,c2=out_c,k=3)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(out_c,out_c//16,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(out_c//16,out_c,1,1,0,bias=True))
        self.pw2 = Conv(out_c,out_c,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.pw1(x)
        x1 = self.axis_dw1(x)
        x2 = self.axis_dw2(x)

        max_out = self.fc(self.max_pool(x1+x2))
        x1 = x1*self.sigmoid(max_out)
        x2 = x2*self.sigmoid(max_out)
        
        #y = torch.cat([x1,x2],dim=1)
        y = x1+x2
        res = self.pw2(y)
        
        return res

class conv_up(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c,out_c,2,2,0)
        self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x

class conv_up0(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c,out_c,1,1,0)
        self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x

class conv_up_plus(nn.Module):
    def __init__(self,in_c,out_c,k=2,s=2):
        super().__init__()
        self.conv_down = nn.ConvTranspose2d(in_c,out_c,k,s,0)
        #self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_down(x)
        #x = self.conv_fu(x)
        return x

class conv_down(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_down = nn.MaxPool2d(2,2)
        #self.conv_fu = Conv(in_c,out_c)
    def forward(self,x):
        x = self.conv_down(x)
        #x = self.conv_fu(x)
        return x

class conv_pre(nn.Module):
    def __init__(self,in_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.pre = nn.Conv2d(in_c,2,1,1,0)
        #self.conv = CrossConv(in_c,out_c)
    def forward(self,x):
        #x = self.conv(x)
        x = self.pre(x)
        return x

class conv_up_pre(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.up = Conv(in_c,out_c)
        self.pre = nn.Conv2d(out_c,2,1,1,0)
        #self.conv = CrossConv(in_c,out_c)
    def forward(self,x):
        x = self.up(x)
        x = self.pre(x)
        return x

class desam(nn.Module):
    def __init__(self,):
        super().__init__()
        self.de12_ = conv_block(768,256) 
        self.de9_ = conv_block(768,256) 
        self.de6_ = conv_block(768,256) 
        self.de3_ = conv_block(768,256) 

        
        self.de12 = conv_up0(768,256) 
        self.de9 = conv_up(768,128)
        self.de6 = nn.Sequential(conv_up(768,128),conv_up(128,64))
        self.de3 = nn.Sequential(conv_up(768,128),conv_up(128,64),conv_up(64,32))

        self.hyper = conv_block(256*4,256)
        self.hyper_up1 = conv_up(256,128)    
        self.hyper_up2 = conv_up(128,64)
        self.hyper_up3 = conv_up(64,32)

        self.up1 = conv_up(256,256)
        self.up2 = conv_up(128,128)
        self.up3 = conv_up(64,64)

        self.fu1 = conv_block(768,256)
        self.fu2 = conv_block(512,128)
        self.fu3 = conv_block(256,64)
        self.fu4 = conv_block(128,32)
     
        self.pre4 = conv_pre(32)
        self.pre3 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),conv_up_pre(64,32)) 
        self.pre2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),conv_up_pre(128,32)) 
        self.pre1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=8),conv_up_pre(256,32))   

        self.pre_final = nn.Conv2d(8,2,1,1,0) 

        
    def forward(self,mask_in,mask_embed):
        #pre_deal
        de12_ = self.de12_(mask_embed[0].permute(0,3,1,2))#256,32,32
        de9_ = self.de9_(mask_embed[1].permute(0,3,1,2))#256,32,32
        de6_ = self.de6_(mask_embed[2].permute(0,3,1,2))#256,32,32
        de3_ = self.de3_(mask_embed[2].permute(0,3,1,2))#256,32,32

        de12 = self.de12(mask_embed[0].permute(0,3,1,2))#256,32,32
        de9 = self.de9(mask_embed[1].permute(0,3,1,2))#256,64,64
        de6 = self.de6(mask_embed[2].permute(0,3,1,2))#128,128,128
        de3 = self.de3(mask_embed[2].permute(0,3,1,2))#64,256,256

        hyper = torch.cat([de12_,de9_,de6_,de3_],dim=1)# 256*4       
        hyper = self.hyper(hyper)
        hyper1 = self.hyper_up1(hyper)
        hyper2 = self.hyper_up2(hyper1)
        hyper3 = self.hyper_up3(hyper2)

        
        mask1 = torch.cat([mask_in,de12,hyper],dim=1)
        mask1_ = self.fu1(mask1)#256,32,32
        mask1 = self.up1(mask1_)
        mask2 = torch.cat([mask1,de9,hyper1],dim=1)
        mask2_ = self.fu2(mask2)#128,64,64
        mask2 = self.up2(mask2_)
        mask3 = torch.cat([mask2,de6,hyper2],dim=1)
        mask3_ = self.fu3(mask3)#64,128,128
        mask3 = self.up3(mask3_)
        mask4 = torch.cat([mask3,de3,hyper3],dim=1)
        mask4 = self.fu4(mask4)#32,256,256
   
        mask_pre4 = self.pre4(mask4)
        mask_pre3 = self.pre3(mask3_)
        mask_pre2 = self.pre2(mask2_)
        mask_pre1 = self.pre1(mask1_)

        mask_final = self.pre_final(torch.cat([mask_pre4,mask_pre3,mask_pre2,mask_pre1],dim=1))
        return mask_pre4,mask_pre3,mask_pre2,mask_pre1,mask_final

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 2#num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, 4, iou_head_depth
        )
        

        self.desam = desam()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        de_in = image_embeddings[1:]
        image_embeddings = image_embeddings[0]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        #print(src.size())
        if dense_prompt_embeddings.size(0) == 1:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        #print(src.size())
        #print(dense_prompt_embeddings.size())
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        res = self.desam(src,de_in)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return res, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
