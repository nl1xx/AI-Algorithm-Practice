# 相对位置编码
# https://www.zhihu.com/tardis/zm/art/577855860?source_id=1005

import torch

coords_h = torch.arange(2)
coords_w = torch.arange(2)
# print(torch.meshgrid(coords_w, coords_h))
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

M = 2
relative_coords[:, :, 0] += M - 1
relative_coords[:, :, 1] += M - 1
relative_coords[:, :, 0] *= 2 * M - 1
relative_position_index = relative_coords.sum(-1)
print(relative_position_index)
