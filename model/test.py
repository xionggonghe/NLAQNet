import torch
from ViTmodule.MobVIT import MobViTBlock

x = torch.rand([3, 4, 32, 32])
print("x.shape: ", x.shape)
vit = MobViTBlock(in_channels=4, attn_unit_dim=12)

y = vit(x)


print("y.shape: ", y.shape)


print("finish !!!")
