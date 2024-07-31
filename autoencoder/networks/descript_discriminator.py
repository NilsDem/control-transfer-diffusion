# adapted from https://github.com/descriptinc/descript-audio-codec

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord
from autoencoder.core import mean_difference

    
    
def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen


def ls_gan(score_real, score_fake):
    loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
    loss_dis = loss_dis.mean()
    loss_gen = (score_fake - 1).pow(2).mean()
    return loss_dis, loss_gen



def kaiser_filter(wc, atten, N=None):
    """
    Computes a kaiser lowpass filter
    Parameters
    ----------
    wc: float
        Angular frequency
    
    atten: float
        Attenuation (dB, positive)
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)
    return h



def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):

    def __init__(self, period, n_channels: int = 1):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            WNConv2d(n_channels, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(128, 256, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(256, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(512, 512, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = WNConv2d(512,
                                  1,
                                  kernel_size=(3, 1),
                                  padding=(1, 0),
                                  act=False)

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):

    def __init__(self, scale: int, n_channels: int = 1):
        super().__init__()
        self.convs = nn.ModuleList([
            WNConv1d(n_channels, 16, 15, 1, padding=7),
            WNConv1d(16, 64, 41, 4, groups=4, padding=20),
            WNConv1d(64, 256, 41, 4, groups=16, padding=20),
            WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
            WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
            WNConv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)

        self.scale = scale

        if self.scale != 1:
            wc = np.pi / self.scale
            filt = kaiser_filter(wc, 140)
            if not len(filt) % 2:
                filt = np.pad(filt, (1, 0))

            self.register_buffer(
                "downsampler",
                torch.from_numpy(filt).reshape(1, 1, -1).float())

    def forward(self, x):
        if self.scale != 1:
            x = nn.functional.conv1d(
                x,
                self.downsampler,
                padding=self.downsampler.shape[-1] // 2,
                stride=self.scale,
            )

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):

    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
        n_channels: int = 1
    ):
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList([
            WNConv2d(2 * n_channels, ch, (3, 9), (1, 1), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        ])
        self.band_convs = nn.ModuleList(
            [convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch,
                                  1, (3, 3), (1, 1),
                                  padding=(1, 1),
                                  act=False)

        self.stft = Spectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=int(hop_factor * window_length),
            center=True,
            return_complex=True,
            power=None,
        )

    def spectrogram(self, x):
        x = torch.view_as_real(self.stft(x))
        x = rearrange(x, "b c f t p -> b (c p) t f")
        # Split into bands
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class DescriptDiscriminator(nn.Module):

    def __init__(
        self,
        weights: dict,
        num_skipped_features: int = 1,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
        n_channels: int = 1,
    ):
        super().__init__()
        discs = []
        discs += [MPD(p, n_channels=n_channels) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate, n_channels=n_channels) for r in rates]
        discs += [
            MRD(f, sample_rate=sample_rate, bands=bands, n_channels=n_channels) for f in fft_sizes
        ]
        self.discriminators = nn.ModuleList(discs)
        self.weights = weights
        self.num_skipped_features = num_skipped_features

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps
    
    
    def split_features(self,features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake
    
    def get_losses_names(self):
        return ["feature_matching", "pred_real", "pred_fake", "discriminator", "adversarial"]
    
    def compute_losses(self, features_real, features_fake):
    
        feature_matching_distance, pred_real, pred_fake, loss_dis, loss_adv = 0., 0., 0., 0., 0.

        for scale_real, scale_fake in zip(features_real, features_fake):

            feature_matching_func = lambda x, y : mean_difference(x, y, norm = "L1")
            current_feature_distance = sum(map(feature_matching_func, scale_real[self.num_skipped_features:],scale_fake[self.num_skipped_features:],))

            current_feature_distance /= len(scale_real[self.num_skipped_features:])

            feature_matching_distance += current_feature_distance

            _dis, _adv = hinge_gan(scale_real[-1], scale_fake[-1])

            pred_real += scale_real[-1].mean()
            pred_fake += scale_fake[-1].mean()

            loss_dis += _dis
            loss_adv += _adv


        loss_dict = {"discriminator": loss_dis.item(), "adversarial": loss_adv.item(), "feature_matching": feature_matching_distance.item(), "pred_real": pred_real.item(), "pred_fake": pred_fake.item()}
            
        loss_gen = loss_adv * self.weights["adversarial"] + feature_matching_distance * self.weights["feature_matching"]
        return loss_gen, loss_dis, loss_dict
