import nn_tilde
import torch

import nn_tilde
import torch.nn as nn
from autoencoder.networks.SimpleNets import AutoEncoder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=0)
parser.add_argument("--name", type=str, default="test")

COMP_RATIO = 1024
LATENT_SIZE = 32
PQMF_BANDS = 8


class AE(nn.Module):

    def __init__(self, name, step) -> None:
        super().__init__()

        model = AutoEncoder(
            in_channels=PQMF_BANDS,  # Number of input channels
            channels=48,
            z_channels=LATENT_SIZE,  # Number of base channels
            multipliers=[1, 2, 4, 4, 8, 8],
            num_blocks=[3, 3, 3, 3, 3],
            dilations=[1, 3, 9],
            kernel_size=5,
            recurrent_layer=torch.nn.Identity,
            use_norm=False,
            decoder_ratio=1.5,
            pqmf_bands=pqmf_bands,
            use_loudness=True,
            use_noise=True,
        )

        path = "autoencoder/runs/" + name
        d = torch.load(path + "/checkpoint" + str(step) + ".pt")
        model.load_state_dict(d["model_state"])

        self.model = model
        self.latent_size = LATENT_SIZE
        self.target_channels = 1
        self.comp_ratio = COMP_RATIO

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)

        z = self.model.encoder(x)

        x = self.model.decoder(z)

        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        return x

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)
        return self.model.encoder(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model.decoder(z)
        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)
        return x


def main(args):

    ae = AE(name=args.name, step=args.step)

    test_array = torch.randn((1, 1, COMP_RATIO * 8))
    z = ae.encode(test_array)
    x = ae.decode(z)

    print(z.shape, x.shape)
    print(ae.forward(test_array).shape)

    ae = torch.jit.script(ae)
    torch.jit.save(ae, "pretrained/" + args.name + ".pt")
    print("Torch script saved to pretrained/" + args.name + ".pt")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
