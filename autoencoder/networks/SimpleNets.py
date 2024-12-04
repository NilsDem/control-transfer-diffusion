from math import floor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from einops_exts import rearrange_many
from torch import Tensor
from torchaudio import transforms
from functools import reduce
from torch.nn.utils import weight_norm
from ..core import SnakeBeta as Snake
from ..core import mod_sigmoid
from .pqmf import PQMF, CachedPQMF
import math
import numpy as np

import cached_conv as cc

#from .utils import closest_power_2, default, exists, groupby, prefix_dict, prod, to_list
"""
Convolutional Modules
"""


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def Conv1d(*args, **kwargs) -> nn.Module:
    return cc.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return cc.ConvTranspose1d(*args, **kwargs)


def Downsample1d(in_channels: int,
                 out_channels: int,
                 factor: int,
                 kernel_multiplier: int = 2) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return normalization(
        Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * kernel_multiplier,
            stride=factor,
            padding=cc.get_padding(2 * factor,
                                   factor),  #math.ceil(factor / 2),
        ))


def Upsample1d(in_channels: int, out_channels: int, factor: int) -> nn.Module:
    if factor == 1:
        return normalization(
            Conv1d(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=3,
                   padding=1))
    return normalization(
        ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
        ))


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        # The hook we want to remove is an instance of WeightNorm class, so
        # normally we would do `if isinstance(...)` but this class is not accessible
        # because of shadowing, so we check the module name directly.
        # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            print("Removing weight_norm from %s", self.__class__.__name__)
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class ConvBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 num_groups: int = 8,
                 use_norm: bool = True,
                 activation: nn.Module = Snake):
        super().__init__()

        self.groupnorm = (nn.GroupNorm(num_groups=min(in_channels, num_groups),
                                       num_channels=in_channels)
                          if use_norm else nn.Identity())
        self.activation = activation(dim=in_channels)
        self.project = normalization(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                #padding="same",
                padding=cc.get_padding(kernel_size, dilation=dilation),
                dilation=dilation,
            ))

    def __prepare_scriptable__(self):
        for hook in self.project._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                print("Removing weight_norm from %s", self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.project)
        return self

    def forward(self, x: Tensor) -> Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 use_norm: bool = True,
                 num_groups: int = 8,
                 use_res=True,
                 activation: nn.Module = Snake) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  use_norm=use_norm,
                                  num_groups=num_groups,
                                  activation=activation)

        self.block2 = ConvBlock1d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  use_norm=use_norm,
                                  activation=activation)

        self.to_out = (normalization(
            Conv1d(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=1))
                       if in_channels != out_channels else nn.Identity())

        self.use_res = use_res

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        if self.use_res:
            return h + self.to_out(x)
        return h


class DownsampleBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, factor: int,
                 num_groups: int, num_layers: int, dilations: Sequence[int],
                 kernel_size: int, activation: nn.Module, use_norm: bool):
        super().__init__()

        self.downsample = Downsample1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       factor=factor)
        self.act = activation(in_channels)
        self.blocks = nn.ModuleList([
            ResnetBlock1d(in_channels=in_channels,
                          out_channels=in_channels,
                          num_groups=num_groups,
                          activation=activation,
                          dilation=dilations[i],
                          kernel_size=kernel_size,
                          use_norm=use_norm) for i in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)

        x = self.act(x)
        x = self.downsample(x)
        return x


class UpsampleBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, factor: int,
                 num_layers: int, dilations: Sequence[int], kernel_size: int,
                 num_groups: int, activation: nn.Module, use_norm: bool):
        super().__init__()
        self.act = activation(dim=in_channels)
        self.blocks = nn.ModuleList([
            ResnetBlock1d(in_channels=out_channels,
                          out_channels=out_channels,
                          num_groups=num_groups,
                          activation=activation,
                          dilation=dilations[i],
                          kernel_size=kernel_size,
                          use_norm=use_norm) for i in range(num_layers)
        ])

        self.upsample = Upsample1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   factor=factor)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(x)
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x


"""
Encoders / Decoders
"""


class Bottleneck(nn.Module):

    def forward(self,
                x: Tensor,
                with_info: bool = False) -> Union[Tensor, Tuple[Tensor, Any]]:
        raise NotImplementedError()


class Encoder1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 multipliers: Sequence[int],
                 factors: Sequence[int],
                 num_blocks: Sequence[int],
                 dilations: Sequence[int],
                 kernel_size: int,
                 patch_size: int = 1,
                 resnet_groups: int = 8,
                 out_channels: Optional[int] = None,
                 recurent_layer: nn.Module = nn.Identity,
                 activation: nn.Module = Snake,
                 use_norm: bool = True):
        super().__init__()
        self.num_layers = len(multipliers) - 1
        self.downsample_factor = patch_size * prod(factors)
        self.out_channels = out_channels
        assert len(factors) == self.num_layers and len(
            num_blocks) == self.num_layers

        self.to_in = ResnetBlock1d(in_channels=in_channels,
                                   out_channels=channels * multipliers[0],
                                   use_res=True,
                                   activation=activation,
                                   use_norm=use_norm,
                                   kernel_size=kernel_size)

        self.downsamples = nn.ModuleList([
            DownsampleBlock1d(in_channels=channels * multipliers[i],
                              out_channels=channels * multipliers[i + 1],
                              factor=factors[i],
                              num_groups=resnet_groups,
                              num_layers=num_blocks[i],
                              dilations=dilations,
                              kernel_size=kernel_size,
                              activation=activation,
                              use_norm=use_norm)
            for i in range(self.num_layers)
        ])

        self.recurent_layer = recurent_layer(
            in_size=channels * multipliers[-1],
            out_size=channels * multipliers[-1])

        self.to_out = nn.Sequential(
            activation(dim=channels * multipliers[-1]),
            normalization(
                cc.Conv1d(
                    in_channels=channels * multipliers[-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=cc.get_padding(kernel_size=3, dilation=1),
                    #padding="same"
                )))

    def forward(self, x: Tensor, with_info: bool = False) -> Tensor:
        x = self.to_in(x)

        for downsample in self.downsamples:
            x = downsample(x)

        x = self.recurent_layer(x)
        x = self.to_out(x)

        return x


class Decoder1d(nn.Module):

    def __init__(
        self,
        out_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        dilations: Sequence[int],
        kernel_size: int,
        patch_size: int = 1,
        resnet_groups: int = 8,
        in_channels: Optional[int] = None,
        recurent_layer: nn.Module = nn.Identity,
        activation: nn.Module = Snake,
        use_norm: bool = True,
        use_loudness: bool = False,
        use_noise: bool = False,
    ):
        super().__init__()
        num_layers = len(multipliers) - 1

        assert len(factors) == num_layers and len(num_blocks) == num_layers

        self.use_loudness = use_loudness
        self.use_noise = use_noise

        self.to_in = normalization(
            Conv1d(in_channels=in_channels,
                   out_channels=channels * multipliers[0],
                   kernel_size=kernel_size,
                   padding=cc.get_padding(kernel_size, dilation=1)
                   #padding="same"
                   ))

        self.upsamples = nn.ModuleList([
            UpsampleBlock1d(in_channels=channels * multipliers[i],
                            out_channels=channels * multipliers[i + 1],
                            factor=factors[i],
                            num_groups=resnet_groups,
                            num_layers=num_blocks[i],
                            dilations=dilations,
                            activation=activation,
                            kernel_size=kernel_size,
                            use_norm=use_norm) for i in range(num_layers)
        ])

        self.to_out = ResnetBlock1d(in_channels=channels * multipliers[-1],
                                    out_channels=out_channels *
                                    2 if self.use_loudness else out_channels,
                                    use_res=False,
                                    activation=activation,
                                    use_norm=use_norm,
                                    kernel_size=kernel_size)

        if self.use_noise:
            self.noise_module = NoiseGeneratorV2(in_size=channels *
                                                 multipliers[-1],
                                                 hidden_size=128,
                                                 data_size=out_channels,
                                                 ratios=[2, 2, 2],
                                                 noise_bands=5)

        self.recurrent_layer = recurent_layer(in_size=in_channels,
                                              out_size=in_channels)

    def forward(self, x: Tensor, with_info: bool = False) -> Tensor:
        x = self.recurrent_layer(x)
        x = self.to_in(x)

        for upsample in self.upsamples:
            x = upsample(x)

        if self.use_noise:
            noise = self.noise_module(x)
        else:
            noise = torch.tensor(0.)

        x = self.to_out(x)

        if self.use_loudness:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        x = torch.tanh(x)

        if self.use_noise:
            x = x + noise

        return x


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequency amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


class NoiseGeneratorV2(nn.Module):

    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            data_size: int,
            ratios: int,
            noise_bands: int,
            n_channels: int = 1,
            activation=lambda dim: nn.LeakyReLU(.2),
    ):
        super().__init__()
        net = []
        self.n_channels = n_channels
        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands * n_channels)

        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    padding=(r, 0),
                    stride=r,
                ))
            if i != len(ratios) - 1:
                net.append(activation(channels[i + 1]))

        self.net = nn.Sequential(*net)
        self.data_size = data_size

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1],
                          self.n_channels * self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class GRU(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.enabled = True

        self.to_out = normalization(
            cc.Conv1d(hidden_size,
                      out_size,
                      kernel_size=3,
                      padding=cc.get_padding(3, dilation=1)))  #padding same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        x = self.to_out(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels: int,
        z_channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        dilations: Sequence[int],
        kernel_size: int,
        patch_size: int = 1,
        resnet_groups: int = 8,
        recurrent_layer: nn.Module = nn.Identity,
        activation: nn.Module = Snake,
        use_norm: bool = True,
        decoder_ratio: int = 1,
        pqmf_bands=0,
        use_loudness: bool = False,
        use_noise: bool = False,
    ):
        super().__init__()
        out_channels = in_channels

        self.pqmf_bands = pqmf_bands
        if self.pqmf_bands > 1:
            #self.pqmf = PQMF(attenuation=100, n_band=pqmf_bands)
            self.pqmf = CachedPQMF(attenuation=100, n_band=pqmf_bands)
            self.use_pqmf = True
        else:
            self.pqmf = nn.Identity()
            self.use_pqmf = False

        self.encoder = Encoder1d(in_channels=in_channels,
                                 out_channels=z_channels,
                                 channels=channels,
                                 multipliers=multipliers,
                                 factors=factors,
                                 num_blocks=num_blocks,
                                 dilations=dilations,
                                 kernel_size=kernel_size,
                                 patch_size=patch_size,
                                 resnet_groups=resnet_groups,
                                 recurent_layer=recurrent_layer,
                                 activation=activation,
                                 use_norm=use_norm)

        self.decoder = Decoder1d(
            in_channels=z_channels,
            out_channels=out_channels,
            channels=channels,
            multipliers=[int(m * decoder_ratio) for m in multipliers[::-1]],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            dilations=dilations,
            kernel_size=kernel_size,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            recurent_layer=recurrent_layer,
            activation=activation,
            use_norm=use_norm,
            use_loudness=use_loudness,
            use_noise=use_noise)

    def forward(self,
                x: Tensor,
                with_z: bool = False,
                with_multi: bool = False) -> Tensor:

        if self.pqmf_bands > 1:
            x = self.pqmf(x)

        z = self.encoder(x)

        x = self.decoder(z)

        if self.pqmf_bands > 1:
            x = self.pqmf.inverse(x)

        return x

    def encode(
        self,
        x: Tensor,
        with_multi: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if self.pqmf_bands > 1:
            x_multiband = self.pqmf(x)
        else:
            x_multiband = x

        z = self.encoder(x_multiband)

        if with_multi:
            return z, x_multiband

        return z

    def decode(self, z: Tensor, with_multi: bool = False) -> Tensor:
        x_multiband = self.decoder(z)

        if self.pqmf_bands > 1:
            x = self.pqmf.inverse(x_multiband)
        else:
            x = x_multiband

        if with_multi:
            return x, x_multiband

        return x
