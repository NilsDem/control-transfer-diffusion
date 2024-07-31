import torch.nn as nn
import torch
import torchaudio
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')


class DistanceWrap(nn.Module):

    def __init__(self, scale: float, distance: nn.Module) -> None:
        super().__init__()
        self.distance = distance
        self.scale = scale
        self.name = distance.name

    @overload
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, *args):
        return self.distance(*args)


class WaveformDistance(nn.Module):

    def __init__(self, norm: str = "L1") -> None:
        super().__init__()
        self.norm = norm
        self.name = "Waveform distance - " + norm

    def forward(self, x, y):
        return mean_difference(y, x, self.norm)


class STFTDistance(nn.Module):

    def __init__(
        self,
        n_fft: int,
        sampling_rate: int,
        norm: Union[str, Sequence[str]] = None,
        power: Union[int, None] = 1,
        normalized: bool = True,
        mel: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mel:
            self.spec = torchaudio.transforms.MelSpectrogram(
                sampling_rate,
                n_fft,
                hop_length=n_fft // 4,
                n_mels=mel,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )
        else:
            self.spec = torchaudio.transforms.Spectrogram(
                n_fft,
                hop_length=n_fft // 4,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )

        if isinstance(norm, str):
            norm = (norm, )
        self.norm = norm

    def forward(self, x, y):
        x = self.spec(x)
        y = self.spec(y)

        logx = torch.log1p(x)
        logy = torch.log1p(y)

        l_distance = mean_difference(x, y, norm='L1')
        log_distance = mean_difference(logx, logy, norm='L1')

        return l_distance + log_distance


class SpectralDistance(nn.Module):

    def __init__(self,
                 scales: List[int],
                 sr: int,
                 mel_bands: Optional[List[int]],
                 distance: nn.Module = STFTDistance) -> None:
        super().__init__()

        if mel_bands is None:
            mel_bands = [None] * len(scales)

        self.spectral_distances = nn.ModuleList([
            distance(scale, sr, mel=mel)
            for scale, mel in zip(scales, mel_bands)
        ])

        self.name = "Spectral Distance"

    def forward(self, x, y):
        spectral_distance = 0
        for dist in self.spectral_distances:
            spectral_distance = spectral_distance + dist(x, y)
        return spectral_distance


class SimpleLatentReg(nn.Module):

    def __init__(self, act: nn.Module = nn.ELU, scale: int = 3) -> None:
        super().__init__()
        self.act = act()
        self.scale = scale
        self.name = "Simple Reg"

    def forward(self, z):
        return self.act(abs(z) - self.scale).mean()


class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


from torch import nn, sin, pow
from torch.nn import functional as F
from torch.nn import Parameter


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):

    def __init__(self,
                 dim,
                 alpha=1.0,
                 alpha_trainable=True,
                 alpha_logscale=False):
        super(SnakeBeta, self).__init__()
        self.in_features = dim

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(dim) * alpha)
            self.beta = Parameter(torch.zeros(dim) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(dim) * alpha)
            self.beta = Parameter(torch.ones(dim) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1).to(
            x)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1).to(x)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


