from autoencoder.networks.SimpleNets import AutoEncoder
from autoencoder.networks.descript_discriminator import DescriptDiscriminator
from autoencoder.core import WaveformDistance, SpectralDistance, SimpleLatentReg
from autoencoder.trainer import Trainer
import torch
import numpy as np
from dataset import SimpleDataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="test")
parser.add_argument("--bsize", type=int, default=6)
parser.add_argument("--db_path", type=str, default="")
parser.add_argument("--restart", type=int, default=0)
parser.add_argument("--gpu", type=int, default=-1)


def main(args):

    model_name = args.name
    dataset = SimpleDataset(keys=["waveform"], path=args.db_path)

    sr = 44100
    bsize = args.bsize
    max_steps = 1000000
    num_signal = 131072
    step_restart = args.restart
    warmup_steps = 200000
    freeze_encoder_step = 500000
    pqmf_bands = 8
    use_noise = True

    #SLAKGH
    autoencoder = AutoEncoder(
        in_channels=pqmf_bands,  # Number of input channels
        channels=48,
        z_channels=32,  # Number of base channels
        multipliers=[
            1, 2, 4, 4, 8, 8
        ],  # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
        factors=[2, 2, 2, 2, 4],  # Downsampling/upsampling factor per layer
        num_blocks=[3, 3, 3, 3, 3],
        dilations=[1, 3, 9],
        kernel_size=5,
        recurrent_layer=torch.nn.Identity,
        use_norm=False,
        decoder_ratio=1.5,
        pqmf_bands=pqmf_bands,
        use_loudness=True,
        use_noise=use_noise,
    )

    discriminator = DescriptDiscriminator(
        rates=[],
        periods=[2, 3, 5, 7, 11],
        fft_sizes=[2048, 1024, 512],
        sample_rate=sr,
        n_channels=1,
        num_skipped_features=1,
        weights={
            "feature_matching": 10.0,
            "adversarial": 5.0
        },
    )
    spectral_distance = SpectralDistance(
        scales=[32, 64, 128, 256, 512, 1024, 2048],
        sr=sr,
        mel_bands=[5, 10, 20, 40, 80, 160, 320])

    if pqmf_bands > 1:
        spectral_distance_multiband = SpectralDistance(
            scales=[int(1 / pqmf_bands * l) for l in [256, 512, 1024, 2048]],
            sr=sr,
            mel_bands=None)
    else:
        spectral_distance_multiband = []

    distances = [(10., spectral_distance)]  #
    multi_band_distances = [(5., spectral_distance_multiband)
                            ] if pqmf_bands > 1 else []
    reglosses = [(1.0, SimpleLatentReg())]

    x = torch.randn(1, 1, 4096)
    z = autoencoder.encode(x)
    y = autoencoder.decode(z)

    assert x.shape == y.shape, ValueError(
        f"Shape mismatch: x.shape = {x.shape}, y.shape = {y.shape}")

    ## Data
    def collate_fn(x):
        x = [l["waveform"] for l in x]
        x = [
            l[..., i0:i0 + num_signal] for l, i0 in zip(
                x, torch.randint(x[0].shape[-1] - num_signal, (len(x), )))
        ]
        x = np.stack(x)
        x = torch.from_numpy(x).reshape(x.shape[0], 1, -1).float()
        return x

    dataset, valset = torch.utils.data.random_split(
        dataset,
        (len(dataset) - int(0.99 * len(dataset)), int(0.99 * len(dataset))))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=bsize,
                                             shuffle=True,
                                             collate_fn=collate_fn,
                                             drop_last=True,
                                             num_workers=0)

    validloader = torch.utils.data.DataLoader(valset,
                                              batch_size=bsize,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              num_workers=0)

    x = next(iter(dataloader))
    print("Training size : ", x.shape)

    trainer = Trainer(autoencoder,
                      distances,
                      reglosses,
                      multi_band_distances,
                      sr=sr,
                      discriminator=discriminator,
                      max_steps=max_steps,
                      warmup_steps=warmup_steps,
                      freeze_encoder_step=freeze_encoder_step)

    if step_restart > 0:
        path = "./autoencoder/runs/" + model_name
        trainer.load_model(path, step_restart)

    trainer.fit(dataloader,
                validloader,
                tensorboard="./autoencoder/runs/" + model_name,
                device="cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
