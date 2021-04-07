import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            1,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.squeeze(1)
        
        return x


class VisionBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=32,
            patch_size=4,
            in_chans=3
        )

    def forward(self, x):
        a = self.patch_embed(x) 
        print("a shape: ", a)
        return a
