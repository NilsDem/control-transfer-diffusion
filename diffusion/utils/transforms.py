from torchaudio.transforms import MelSpectrogram
import torchaudio
import torch
import gin
from cqt_pytorch import CQT


@gin.configurable
class StreamableMelSpectrogram(torch.nn.Module):

    def __init__(self,
                 sr=44100,
                 n_mels=128,
                 nfft=1024,
                 hop_size=256,
                 stream=True,
                 skip_features=None):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size

        self.register_buffer('audio_buffer',
                             torch.zeros((1, 1, nfft - hop_size)))
        self.stream = stream

        transform = MelSpectrogram(sample_rate=sr,
                                   n_fft=nfft,
                                   n_mels=n_mels,
                                   win_length=nfft,
                                   hop_length=hop_size,
                                   center=not stream,
                                   normalized=True)

        self.transform = transform
        self.skip_features = skip_features

    @torch.jit.export
    def forward(self, x):
        # X : B x hop_size
        if self.stream == True:
            if self.audio_buffer.shape[0] != x.shape[0]:
                print(
                    "Resizing and resetting buffer - the batch size has changed"
                )
                self.register_buffer(
                    'audio_buffer',
                    torch.zeros(
                        (x.shape[0], 1, self.nfft - self.hop_size)).to(x))
            x = torch.cat([self.audio_buffer, x], dim=-1)

        spec = self.transform(x)[:, 0]

        if self.stream == True:
            self.audio_buffer = x[..., -(self.nfft - self.hop_size):]

        if self.skip_features is not None:
            spec = spec[:, :self.skip_features]

        spec = torch.log1p(spec)

        return spec if self.stream else spec[..., :-1]


from nnAudio.features.cqt import CQT2010v2, CQT2010, CQT1992v2, CQT1992v2


@gin.configurable
class PytorchCQT(torch.nn.Module):

    def __init__(self,
                 sr=44100,
                 num_octaves=8,
                 num_bins_per_octave=32,
                 block_length=131072):
        super().__init__()

        self.transform = CQT(num_octaves=num_octaves,
                             num_bins_per_octave=num_bins_per_octave,
                             sample_rate=sr,
                             block_length=block_length)

    def forward(self, x):
        cqt = abs(self.transform.encode(x))**2
        cqt = torch.log1p(cqt)
        cqt = cqt.squeeze(1)
        return cqt
