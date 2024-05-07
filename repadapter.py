# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:48:22 2023

@author: tc922
"""


import torch
from torch import nn
import timm
# import  timm.layers.drop as DP

def forward_vit_block_adapter(self, x):
    # self.dropout=nn.Dropout(0.1)
    # self.drop_path=DP.DropPath(drop_prob=0.1)
###############################################################################
    # x = x + self.drop_path1(self.attn(self.adapter_attn(self.norm1(x))))
    # x = x + self.drop_path2(self.mlp(self.adapter_mlp(self.norm2(x))))
    # t h e v i t r e p a d a p t e r i s o w n e d b y t i a n y i c h e n
###############################################################################
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
###############################################################################
    return x

def forward_vit_attn_adapter(self, x):
    # self.dropout=nn.Dropout(0.1)
    # self.drop_path=DP.DropPath(drop_prob=0.1)
    x = x + self.drop_path1(self.attn(self.adapter_attn(self.norm1(x))))
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x


def forward_swin_block_adapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x))
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.adapter_mlp(self.norm2(x))))
    return x


def forward_swin_attn_adapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x))
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def forward_convnext_attn_adapter(self, x):
    shortcut = x
    x = self.conv_dw(self.adapter_attn(x))
    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))

    x = self.drop_path(x) + shortcut
    return x

def forward_convnext_block_adapter(self, x):
    shortcut = x
    x = self.conv_dw(self.adapter_attn(x))
    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(self.adapter_mlp(x))
    else:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(self.adapter_mlp(x))
        x = x.permute(0, 3, 1, 2)
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))

    x = self.drop_path(x) + shortcut
    return x


class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
    def forward(self, x):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))*self.scale+x
        x=x.transpose(1,2).contiguous()
        return x


class RepAdapter2D(nn.Module):
    """ Pytorch Implemention of RepAdapter for 2d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A = nn.Conv2d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv2d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        return x

def reparameterize(Wa,Wb,Ba,Bb,scale=1,do_residual=False):
    bias = 0
    id_tensor=0
    if Ba is not None:
        bias=Ba@Wb
    if Bb is not None:
        bias=bias+Bb
    if do_residual:
        id_tensor=torch.eye(Wa.shape[0],Wb.shape[1]).to(Wa.device)
    weight = Wa @ Wb*scale + id_tensor
    return weight.T,bias*scale if isinstance(bias,torch.Tensor) else None

def sparse2dense(weight,groups):
    d,cg=weight.shape
    dg=d//groups
    weight=weight.view(groups,dg,cg).transpose(1,2)
    new_weight=torch.zeros(cg*groups,d,device=weight.device,dtype=weight.dtype)
    for i in range(groups):
        new_weight[i*cg:(i+1)*cg,i*dg:(i+1)*dg]=weight[i]
    return new_weight.T


def set_RepAdapter(model, method, dim=8, s=1, args=None,set_forward=True):

    #if method == 'repblock':
    if method != None:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.ls1=_.attn
                _.ls2=_.mlp
                _.attn = RepAdapter(hidden_dim=dim,scale=s)
                _.mlp = RepAdapter(hidden_dim=dim,scale=s)
                # _.s = s
                print('set adapter!')
                bound_method = forward_vit_block_adapter.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.adapter_mlp = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_block_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_)== timm.models.convnext.ConvNeXtBlock:
                _.adapter_attn = RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.adapter_mlp = RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.s = s
                bound_method = forward_convnext_block_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s,args=args,set_forward=set_forward)
  # t h e vi t r e p ad a p t e r i s ow n e d by t i a n y i c he n
###################################################################################################
"""
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = RepAdapter(hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_vit_attn_adapter.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn =  RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_attn_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_)== timm.models.convnext.ConvNeXtBlock:
                _.adapter_attn =  RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.s = s
                bound_method = forward_convnext_attn_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s, args=args, set_forward=set_forward)
"""

def set_RepWeight(model, method, dim=8, s=1, args=None):
    if method == 'repblock':
        ################################################################################################
        # c=0   # counter for iterating on rep_model
        # for _ in model.children():
        #     print 
        #     if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
        #         if _.adapter_attn.groups>1:
        #             weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
        #             print('sparse2dense detect!')
        #             # print(weight_B)
        #         else:
        #             weight_B=_.adapter_attn.conv_B.weight.squeeze()
        #         attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
        #                                 _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
        #         qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
        #                                         attn_bias, _.attn.qkv.bias)
        #         print('Rep-PETL')
        #         with torch.no_grad():
        #             _.attn.qkv.weight.copy_(qkv_weight)
        #             _.attn.qkv.bias.copy_(qkv_bias)
        #             print('attn.qkv:')
        #             print('test_model:')
        #             print(_.attn.qkv.weight)
        #             print(_.attn.qkv.bias)
        #             for adj in rep_model.blocks[c].children():
        #                 if (type(adj)==timm.models.vision_transformer.Attention):
        #                     adj.qkv.weight.copy_(qkv_weight)
        #                     print('rep_model:')
        #                     print(adj.qkv.weight)
        #                     adj.qkv.bias.copy_(qkv_bias)
        #                     print(adj.qkv.bias)

        #         if _.adapter_mlp.groups>1:
        #             weight_B=sparse2dense(_.adapter_mlp.conv_B.weight.squeeze(),_.adapter_mlp.groups)
        #         else:
        #             weight_B=_.adapter_mlp.weight_B.squeeze()

        #         mlp_weight,mlp_bias=reparameterize(_.adapter_mlp.conv_A.weight.squeeze().T,weight_B.T,
        #                                 _.adapter_mlp.conv_A.bias,_.adapter_mlp.conv_B.bias,_.s,do_residual=True)
        #         fc_weight,fc_bias=reparameterize(mlp_weight.T,_.mlp.fc1.weight.T,
        #                                       mlp_bias, _.mlp.fc1.bias)
        #         with torch.no_grad():
        #             _.mlp.fc1.weight.copy_(fc_weight)
        #             _.mlp.fc1.bias.copy_(fc_bias)
        #             print('fc1:')
        #             print('test_model:')
        #             print(_.mlp.fc1.weight)
        #             print(_.mlp.fc1.bias)
        #             for adj in rep_model.blocks[c].children():
        #                 if (type(adj)==timm.layers.mlp.Mlp):
        #                     adj.fc1.weight.copy_(fc_weight)
        #                     print('rep_model:')
        #                     print(adj.fc1.weight)
        #                     adj.fc1.bias.copy_(fc_bias)
        #                     print(adj.fc1.bias)
        #         c=c+1
###################################################################################
        for _ in model.children():
            print 
            if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                if _.attn.groups>1:
                    weight_B=sparse2dense(_.attn.conv_B.weight.squeeze(),_.attn.groups)
                    print('sparse2dense detect!')
                    # print(weight_B)
                else:
                    weight_B=_.attn.conv_B.weight.squeeze()
                attn_weight,attn_bias=reparameterize(_.attn.conv_A.weight.squeeze().T,weight_B.T,
                                        _.attn.conv_A.bias,_.attn.conv_B.bias,s,do_residual=True)
                qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.ls1.qkv.weight.T,
                                                attn_bias, _.ls1.qkv.bias)
                # print('Rep-PETL')
                # print('Before Reparameterize:')
                # print(_.ls1.qkv.weight)
                # print(_.ls1.qkv.bias)
                with torch.no_grad():
                    _.ls1.qkv.weight.copy_(qkv_weight)
                    _.ls1.qkv.bias.copy_(qkv_bias)
                _.attn=nn.Identity()
                # print('After Reparameterize:')
                # print(_.ls1.qkv.weight)
                # print(_.ls1.qkv.bias)
                if _.mlp.groups>1:
                    weight_B=sparse2dense(_.mlp.conv_B.weight.squeeze(),_.mlp.groups)
                else:
                    weight_B=_.mlp.weight_B.squeeze()
        
                mlp_weight,mlp_bias=reparameterize(_.mlp.conv_A.weight.squeeze().T,weight_B.T,
                                        _.mlp.conv_A.bias,_.mlp.conv_B.bias,s,do_residual=True)
                fc_weight,fc_bias=reparameterize(mlp_weight.T,_.ls2.fc1.weight.T,
                                              mlp_bias, _.ls2.fc1.bias)
                with torch.no_grad():
                    _.ls2.fc1.weight.copy_(fc_weight)
                    _.ls2.fc1.bias.copy_(fc_bias)
                _.mlp=nn.Identity()
            elif len(list(_.children())) != 0:
                set_RepWeight(_, method, dim, s, args=args)
#################################################################################### waived:
        # for _ in model.children():
        #     if type(_)==timm.models.vision_transformer.Block:
        #         if _.attn.groups>1:
        #             weight_B=sparse2dense(_.attn.conv_B.weight.squeeze(),_.attn.groups)
        #         else:
        #             weight_B=_.attn.conv_B.weight.squeeze()
        #         attn_weight,attn_bias=reparameterize()
##################################################################################original :
        # for _ in model.children():
        #     print 
        #     if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
        #         if _.adapter_attn.groups>1:
        #             weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
        #             print('sparse2dense detect!')
        #             # print(weight_B)
        #         else:
        #             weight_B=_.adapter_attn.conv_B.weight.squeeze()
        #         attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
        #                                 _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
        #         qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
        #                                         attn_bias, _.attn.qkv.bias)
        #         print('Rep-PETL')
        #         with torch.no_grad():
        #             _.attn.qkv.weight.copy_(qkv_weight)
        #             _.attn.qkv.bias.copy_(qkv_bias)
        
        #         if _.adapter_mlp.groups>1:
        #             weight_B=sparse2dense(_.adapter_mlp.conv_B.weight.squeeze(),_.adapter_mlp.groups)
        #         else:
        #             weight_B=_.adapter_mlp.weight_B.squeeze()
        
        #         mlp_weight,mlp_bias=reparameterize(_.adapter_mlp.conv_A.weight.squeeze().T,weight_B.T,
        #                                 _.adapter_mlp.conv_A.bias,_.adapter_mlp.conv_B.bias,_.s,do_residual=True)
        #         fc_weight,fc_bias=reparameterize(mlp_weight.T,_.mlp.fc1.weight.T,
        #                                       mlp_bias, _.mlp.fc1.bias)
        #         with torch.no_grad():
        #             _.mlp.fc1.weight.copy_(fc_weight)
        #             _.mlp.fc1.bias.copy_(fc_bias)

        #     elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
        #         _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
        #         _.adapter_mlp = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
        #         _.s = s
        #         bound_method = forward_swin_block_adapter.__get__(_, _.__class__)
        #         setattr(_, 'forward', bound_method)
        #     elif len(list(_.children())) != 0:
        #         set_RepWeight(_, method, dim, s, args=args)
###########################################################################################################tianyi chen
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                if _.adapter_attn.groups>1:
                    weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
                else:
                    weight_B=_.adapter_attn.conv_B.weight.squeeze()
                attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
                                        _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
                qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
                                                attn_bias, _.attn.qkv.bias)
                with torch.no_grad():
                    _.attn.qkv.weight.copy_(qkv_weight)
                    _.attn.qkv.bias.copy_(qkv_bias)
            # elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            #     _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
            #     _.s = s
            #     bound_method = forward_swin_attn_adapter.__get__(_, _.__class__)
            #     setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepWeight(_, method, dim, s,args=args)
