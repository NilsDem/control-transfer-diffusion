import torch.nn as nn
import torch


class SPE(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        w = 100**(2 * torch.arange(dim // 2) / dim)
        w = w.reciprocal()
        self.register_buffer("wk", w)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * 100
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).to(self.device)[None]
        wkt = self.wk * t[..., None]
        spe = torch.cat([wkt.sin(), wkt.cos()], -1)
        shape = spe.shape
        if len(spe.shape) == 2:
            spe = spe.reshape(t.shape[0], 2, -1).permute(0, 2,
                                                          1).reshape((shape[0], shape[1]))
        elif len(spe.shape) == 1:
            spe = spe.unsqueeze(0)
        else:
            spe = spe.squeeze(1)
            spe = spe.permute(0, 2, 1)
            return spe
        return spe
