# Based on EVA, BEIT, timm and DeiT code bases
# https://github.com/baaivision/EVA
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math #Imports the standard Python `math` module for mathematical functions.

#which allows you to fix a certain number of arguments of a function and generate a new function.
from functools import partial

import torch #for tensor operations and deep learning.
import torch.nn as nn #typically used for building model layers.

#which provides stateless functions like activation functions and loss calculations.
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint #which helps save memory by trading compute for memory

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model #custom models for use with the library.

from xraygpt.common.dist_utils import download_cached_file #that downloads a file and caches it locally

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    #DropPath implements Stochastic Depth, a regularization technique.During training, it randomly drops entire residual paths 
    # (blocks) with probability `drop_prob`.This helps prevent overfitting and improves model generalization.
   
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        ## Calls the drop_path function from timm, which applies stochastic depth.
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        # Returns a string representation showing the drop probability.
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    #Mlp (Multi-Layer Perceptron) is a feed-forward neural network block used in transformers.
    # It consists of two linear layers with an activation function and dropout in between.
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # First linear layer
        self.act = act_layer()                             # Activation function (default: GELU)
        self.fc2 = nn.Linear(hidden_features, out_features) # Second linear layer
        self.drop = nn.Dropout(drop)                        # Dropout layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    # Multi-head self-attention module with optional relative position bias.
    #Used in Vision Transformers to allow the model to focus on different parts of the input.
   
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads # Number of attention heads
        head_dim = dim // num_heads # Dimension per head
        if attn_head_dim is not None:
            head_dim = attn_head_dim # Override head dimension if specified
        all_head_dim = head_dim * self.num_heads # Total dimension for all heads
        self.scale = qk_scale or head_dim ** -0.5 # Scaling factor for queries

        # Linear layer to generate queries, keys, and values
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            # Learnable biases for queries and values
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        
        # If using windowed attention, set up relative position bias
        if window_size:
            self.window_size = window_size # Size of the attention window (height, width)
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

            # Table of learnable relative position biases
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0]) # Row coordinates
            coords_w = torch.arange(window_size[1]) # Column coordinates
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # Shift row coords to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1  # Shift col coords to start from 0
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1 # Row-major offset
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww, Fill with summed offsets
            relative_position_index[0, 0:] = self.num_relative_distance - 3 # Special index for cls-to-token
            relative_position_index[0:, 0] = self.num_relative_distance - 2 # Special index for token-to-cls
            relative_position_index[0, 0] = self.num_relative_distance - 1  # Special index for cls-to-cls

             # Register as buffer so it's saved with the model but not a parameter
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        # Dropout for attention weights and output projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        # Get batch size (B), sequence length (N), and embedding dim (C) from input
        B, N, C = x.shape
        qkv_bias = None # Initialize qkv_bias to None
        if self.q_bias is not None:
            # Concatenate query bias, zeros (for key), and value bias for custom bias in qkv projection
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Project input x to queries, keys, and values using a single linear layer
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # Reshape and permute to separate q, k, v and arrange as (3, batch, heads, seq_len, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # Split into query, key, and value tensors
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale # Scale queries for stability (usually 1/sqrt(head_dim))
        attn = (q @ k.transpose(-2, -1)) # Compute raw attention scores (dot product of q and k)

        if self.relative_position_bias_table is not None:
            # Get relative position bias for each token pair in the window
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH, Shape: (tokens, tokens, heads)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww, Shape:(heads, tokens, tokens)
            attn = attn + relative_position_bias.unsqueeze(0) # Add relative position bias to attention scores

        if rel_pos_bias is not None:
            # Add any extra relative position bias if provided
            attn = attn + rel_pos_bias 
        
        attn = attn.softmax(dim=-1) # Normalize attention scores with softmax
        attn = self.attn_drop(attn) # Apply dropout to attention weights

        # Weighted sum of values, reshape to (B, N, all_head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # Project output back to original embedding dimension
        x = self.proj(x)
        x = self.proj_drop(x) # Apply dropout to the output
        return x # Return the final output

#This is a standard transformer encoder block with LayerNorm, multi-head attention, MLP, and optional stochastic depth 
# and learnable scaling.Used as the main building block in Vision Transformers (ViT/EVA).
class Block(nn.Module):
    #Transformer block: LayerNorm -> Attention -> DropPath -> LayerNorm -> MLP -> DropPath
    #Optionally uses learnable scaling (gamma_1, gamma_2) for attention and MLP outputs.
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim) # First normalization layer
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)# Attention block
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # Second normalization layer
        mlp_hidden_dim = int(dim * mlp_ratio) # Hidden dimension for MLP
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Optional learnable scaling parameters for attention and MLP outputs
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            # Standard transformer block: add attention and MLP outputs (with drop path)
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # With learnable scaling (gamma_1, gamma_2)
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

#This module splits an image into fixed-size patches, embeds each patch into a vector, and outputs a sequence of patch embeddings—serving as 
# the input tokens for a Vision Transformer.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        #Converts img_size and patch_size to tuples 
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        #Calculates the number of patches and patch grid shape.
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        #Initializes a 2D convolution (self.proj) with kernel and stride equal to patch_size to extract non-overlapping patches 
        #and project them to the embedding dimension.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #Checks that the input image size matches the expected size.
        #Applies the convolution to extract and embed patches.
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #Flattens the spatial dimensions and transposes to shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#Generates learnable relative position biases for attention mechanisms in Vision Transformers, allowing the model to encode spatial 
# relationships between image patches.
class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        #Stores the window size and number of attention heads.
        self.window_size = window_size
        #Calculates the number of possible relative distances between patches.
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        #Creates a learnable table for all possible relative positions and heads.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        #Registers the index as a buffer (not a parameter, but saved with the model).
        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        #Looks up the bias values for each patch pair and head using the precomputed index.
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        #Reshapes and permutes the result to shape ready to be added to attention scores.
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    #This constructor sets up all the layers and parameters needed for a Vision Transformer, including patch embedding, 
    # positional/class tokens, transformer blocks, and optional relative position bias.
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_checkpoint=False):
        super().__init__()
        self.image_size = img_size # Store input image size
        self.num_classes = num_classes # Number of output classes
        self.num_features = self.embed_dim = embed_dim  # Embedding dimension (and num_features for compatibility)

        # Patch embedding: splits image into patches and projects to embedding dimension
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # Total number of patches

        # Learnable class token (for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # # Learnable absolute positional embedding (optional)
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        # Dropout applied after adding positional embedding    
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Shared relative position bias (optional, for all attention blocks)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.use_checkpoint = use_checkpoint # Whether to use gradient checkpointing for memory savings
        
        # Stochastic depth decay rule: list of drop_path rates for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.use_rel_pos_bias = use_rel_pos_bias # Whether to use per-block relative position bias
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([ 
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
#         self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
#         self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize positional embedding and class token with truncated normal distribution
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
#         if isinstance(self.head, nn.Linear):
#             trunc_normal_(self.head.weight, std=.02)
        # Apply custom weight initialization to all modules
        self.apply(self._init_weights)
        # Fix initialization scaling for attention and MLP projection weights
        self.fix_init_weight()
#         if isinstance(self.head, nn.Linear):
#             self.head.weight.data.mul_(init_scale)
#             self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):

    # Rescales the weights of attention and MLP projection layers in each transformer block.
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id)) # Divide weights by sqrt(2 * layer_id) for stability.

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1) # Rescale attention output projection weights.
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)  # Rescale MLP output projection weights.
 
    def _init_weights(self, m):
        # Initializes weights for Linear and LayerNorm layers.
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02) # Truncated normal initialization for weights.
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # Set bias to zero.
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # Set LayerNorm bias to zero.
            nn.init.constant_(m.weight, 1.0) # Set LayerNorm weight to one.

    def get_classifier(self):
        # Returns the classification head (if present).
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        # Replaces the classification head with a new one for a different number of classes.
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # Extracts features from the input image up to (but not including) the classification head.    
        x = self.patch_embed(x) # Convert image to patch embeddings.
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand class token for the batch.
        x = torch.cat((cls_tokens, x), dim=1) # Concatenate class token to patch embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed  # Add positional embeddings.
        x = self.pos_drop(x)   # Apply dropout.

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None  # Get relative position bias if used.
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias) # Use gradient checkpointing for memory efficiency.
            else:
                x = blk(x, rel_pos_bias)  # Standard forward through transformer block.
        return x # Return features (not yet classified).
#         x = self.norm(x)

#         if self.fc_norm is not None:
#             t = x[:, 1:, :]
#             return self.fc_norm(t.mean(1))
#         else:
#             return x[:, 0]

    def forward(self, x):
        # Standard forward pass: returns features (not logits, unless head is uncommented).    
        x = self.forward_features(x)
        #x = self.head(x)
        # x = self.head(x)  # (commented out: would apply classification head)
        return x
    #Extracts and returns the output (features) from each transformer block in the Vision Transformer.
    def get_intermediate_layers(self, x):
        #Converts the input image to patch embeddings.
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        #Prepends the class token and adds positional embeddings.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # Applies dropout.
        x = self.pos_drop(x)

        features = []
        #Iterates through all transformer blocks, collecting the output after each block.
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        #Returns a list of all intermediate features (one per block).
        return features
    
# Adapts (interpolates) the positional embedding weights from a checkpoint to fit a model with a different patch grid size.    
def interpolate_pos_embed(model, checkpoint_model):
    #Checks if the checkpoint contains positional embeddings.
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        #Determines the original and new grid sizes.
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        #If sizes differ, interpolates the position tokens using bicubic interpolation.
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            #Concatenates any extra tokens (like class token) unchanged.
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            #Updates the checkpoint’s positional embedding with the new, interpolated version.
            checkpoint_model['pos_embed'] = new_pos_embed
            
#Converts all applicable model weights (Conv and Linear layers) to 16-bit floating point (fp16) for 
# faster inference and reduced memory usage.            
def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        #Defines a helper that casts weights and biases to fp16 for Conv1d, Conv2d, and Linear layers.
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()
    #Applies this helper recursively to all submodules in the model.
    model.apply(_convert_weights_to_fp16)
    
#Creates and loads a large EVA Vision Transformer model with pretrained weights.    
def create_eva_vit_g(img_size=224,drop_path_rate=0.4,use_checkpoint=False,precision="fp16"):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
    )  
    #Downloads pretrained weights from a URL and loads them.
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")  
    #Interpolates positional embeddings if the patch grid size differs.  
    interpolate_pos_embed(model,state_dict)
    
    #Loads the weights into the model (non-strict to allow for shape changes).
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
#     print(incompatible_keys)
    #Optionally converts the model to fp16 precision.
    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(model)
    #Returns the ready-to-use model.    
    return model
