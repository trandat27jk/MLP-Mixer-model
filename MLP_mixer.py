import torch
import numpy as np
from einops.layers.torch import Rearrange
import torch.nn as nn

pair= lambda x: x if isinstance(x,tuple) else (x,x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patch_image(image_size,patch_size):
    image_height, image_width=pair(image_size)
    num_patchs=(image_height//patch_size)*(image_width//patch_size)

    return num_patchs
#Mixer block
class MixerBlock(nn.Module):
    def __init__(self,num_patchs,dim,expension_factor=4,dropout=0.):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.mlp2=nn.Sequential(nn.Linear(dim,expension_factor*dim),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(dim*expension_factor,dim),
                               nn.Dropout(dropout)
                               )
        self.mlp1=nn.Sequential(nn.Linear(num_patchs,expension_factor*num_patchs),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(num_patchs*expension_factor,num_patchs),
                               nn.Dropout(dropout),
                               )


    def forward(self,x):
        x1=self.norm(x)
        x1=self.mlp1(x1.permute(0,2,1))
        x1=x1.permute(0,2,1)+x
        x2=self.mlp2(self.norm(x1))+x1
        return x2

class MLPMixer(nn.Module):
    def __init__(self, image_size,channels,patch_size,depth,num_classes,dim,expansion_factor=4,dropout=0):
        super().__init__()
        self.image_size=image_size
        self.patch_size=patch_size
        self.depth=depth
        self.num_classes=num_classes
        self.dim=dim
        self.expansion_factor=expansion_factor
        self.dropout=dropout
        self.num_patchs=patch_image(image_size,patch_size)
        self.linear=nn.Linear((patch_size**2)*channels,dim)
        self.MixerBlocks=nn.ModuleList([MixerBlock(self.num_patchs,dim,expension_factor=expansion_factor,dropout=dropout) for _ in range(depth)])
        self.final_layer=nn.Linear(self.num_patchs,num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        x=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=self.patch_size,p2=self.patch_size)(x)
        x=self.linear(x)
        for block in self.MixerBlocks:
            x=block(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x=self.final_layer(x)
        return x

model=MLPMixer(28,1,7,8,10,514).to(device)

