import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class VitEmbedding(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=512,
        p=0.):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x

class VitTranslationEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()

        self.vitEmbedding = VitEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.encoderLayer = nn.TransformerEncoderLayer(512, 8, ((img_size // patch_size)**2 * 3),0.1, 'gelu')
        self.encoder = nn.TransformerEncoder(self.encoderLayer, 12)

    def forward(self, x):
        out = self.vitEmbedding(x)
        out = self.encoder(out)
        return out

class VitTranslationDecoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()

        self.vitEmbedding = VitEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(512, 8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=8)

    def forward(self, memory, target):
        out = self.vitEmbedding(target)
        out = self.decoder(out, memory)
        return out

class VitTranslation(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()

        self.encoder = VitTranslationEncoder(img_size, patch_size, in_chans, embed_dim)
        self.decoder = VitTranslationDecoder(img_size, patch_size, in_chans, embed_dim)

    def forward(self, src, trg):
        src = self.encoder(src)
        output = self.decoder(src, trg)
        output = torch.sigmoid(output)

        return output 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #########################################
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(128),
            #########################################
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(256),
            #########################################
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            #########################################
            nn.Conv2d(512, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ##################################################
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ##################################################
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ##################################################
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out