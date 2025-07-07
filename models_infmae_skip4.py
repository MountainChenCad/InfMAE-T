# models_infmae_skip4.py

from functools import partial
import pdb
import torch
import torch.nn as nn

from vision_transformer import PatchEmbed, Block, CBlock, PatchEmbed_F
from util.pos_embed import get_2d_sincos_pos_embed
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


class MaskedAutoencoderInfMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone (VideoMAE Adaptation) """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 clip_length=16):  # *** NEW ***
        super().__init__()
        self.clip_length = clip_length  # *** NEW ***

        # --------------------------------------------------------------------------
        # Encoder specifics
        self.patch_embed = PatchEmbed_F(img_size[0], patch_size[0] * patch_size[1] * patch_size[2], in_chans,
                                        embed_dim[2])

        self.patch_embed1 = PatchEmbed(img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans,
                                       embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0],
                                       embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1],
                                       embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.num_patches = num_patches

        # *** MODIFIED: Add temporal position embedding ***
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)  # spatial
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, clip_length, 1, embed_dim[2]))  # temporal

        self.blocks1 = nn.ModuleList([CBlock(dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],
                                             qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in
                                      range(depth[0])])
        self.blocks2 = nn.ModuleList([CBlock(dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],
                                             qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in
                                      range(depth[1])])
        self.blocks3 = nn.ModuleList([Block(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],
                                            qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in
                                      range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])
        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True,
                                                   qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      (patch_size[0] * patch_size[1] * patch_size[2]) ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** .5),
                                                    cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # *** NEW: Initialize temporal embedding ***
        torch.nn.init.normal_(self.temporal_pos_embed, std=.02)

        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # *** NEW: VideoMAE Tube Masking ***
    def random_masking_tube(self, x, mask_ratio):
        B, T, L, D = x.shape
        len_keep = int(L * (1.0 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather the kept patches across all frames (the "tube" part)
        # Expand ids_keep to gather along the T and D dimensions
        ids_keep_expanded = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, D)
        x_masked = torch.gather(x, dim=2, index=ids_keep_expanded)

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Multi-stage convolution
        x = self.patch_embed1(x)
        for blk in self.blocks1: x = blk(x)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2: x = blk(x)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)

        # *** MODIFIED: Add temporal and spatial embeddings ***
        x = x.view(B, T, self.num_patches, -1)
        x = x + self.pos_embed.unsqueeze(1) + self.temporal_pos_embed

        # *** MODIFIED: Apply tube masking ***
        x, mask, ids_restore = self.random_masking_tube(x, mask_ratio)
        x = x.view(B * T, -1, x.shape[-1])

        # Also gather the skip connections based on the spatial mask
        def gather_skip(skip_embed, B, T, L, ids_keep):
            skip_embed = skip_embed.view(B, T, L, -1)
            ids_keep_expanded = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, skip_embed.shape[-1])
            skip_masked = torch.gather(skip_embed, dim=2, index=ids_keep_expanded)
            return skip_masked.view(B * T, -1, skip_masked.shape[-1])

        stage1_embed_masked = gather_skip(stage1_embed, B, T, self.num_patches, ids_keep)
        stage2_embed_masked = gather_skip(stage2_embed, B, T, self.num_patches, ids_keep)

        # Apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)

        x = x + stage1_embed_masked + stage2_embed_masked
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B_times_T, N_keep, D = x.shape
        B = B_times_T // self.clip_length

        mask_tokens = self.mask_token.repeat(B, self.num_patches - N_keep, 1)

        # Reshape for un-shuffling
        x_ = x.view(B, self.clip_length, N_keep, D).transpose(1, 2).reshape(B * N_keep, self.clip_length, D)

        # Unshuffle logic needs to be careful with shapes
        # We restore spatial positions first
        x_restored = torch.cat(
            [x.view(B, self.clip_length, -1, D), mask_tokens.unsqueeze(1).expand(-1, self.clip_length, -1, -1)], dim=2)
        x_restored = torch.gather(x_restored, dim=2,
                                  index=ids_restore.unsqueeze(1).unsqueeze(-1).expand(-1, self.clip_length, -1, D))
        x = x_restored.view(B * T, self.num_patches, D)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        B, T, C, H, W = imgs.shape
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        # Flatten time dimension for loss calculation
        imgs_flat = imgs.view(B * T, C, H, W)
        # Mask needs to be repeated for each frame if it's not already
        if mask.shape[0] == B:
            mask = mask.repeat_interleave(T, dim=0)

        loss = self.forward_loss(imgs_flat, pred, mask)
        return loss, pred, mask


def infmae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderInfMAE(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


infmae_vit_base_patch16 = infmae_vit_base_patch16_dec512d8b