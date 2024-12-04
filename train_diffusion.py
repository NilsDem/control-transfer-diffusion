import gin

gin.add_config_file_search_path('./diffusion/configs')

import torch
import os
import numpy as np

from diffusion.model import EDM_ADV
from diffusion.networks import UNET1D, Encoder1D
from diffusion.utils.general import DummyAccelerator
#from accelerate import Accelerator
from diffusion.utils.dataset import get_dataset
from dataset import SimpleDataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="test")
parser.add_argument("--bsize", type=int, default=64)
parser.add_argument("--restart", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument('--config', action="append", default=[])

parser.add_argument("--dataset_type", type=str, default="waveform")
parser.add_argument("--db_path", type=str, default=None)
parser.add_argument("--out_path", type=str, default="./diffusion/runs")
parser.add_argument("--emb_model_path", type=str)


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(args):
    gin.parse_config_files_and_bindings(
        map(add_gin_extension, args.config),
        None,
    )

    if args.restart > 0:
        config_path = "./runs/" + args.name + "/config.gin"
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    ######### BUILD MODEL #########
    model = EDM_ADV()

    emb_model = torch.jit.load(args.emb_model_path)

    model.emb_model = emb_model

    model.accelerator = DummyAccelerator(
        device="cuda:" + str(args.gpu) if args.gpu >= 0 else
        "cpu")  #if args.use_accelerator else Accelerator()
    model = model.to(model.accelerator.device)

    ######### GET THE DATASET #########
    #dataset, valset, collate_fn = get_dataset(dataset_type = args.dataset_type, db_path = args.db_path, use_cache = args.use_cache, num_recache = args.num_recache)

    if args.dataset_type == "waveform":
        dataset = SimpleDataset(path=args.db_path, keys=["waveform"])
        dataset, valset = torch.utils.data.random_split(
            dataset, (len(dataset) - 200, 200))

        x_length = gin.query_parameter("%X_LENGTH")

        def collate_fn(L):
            x = np.stack([l["waveform"] for l in L])
            x = torch.from_numpy(x).float().reshape((x.shape[0], 1, -1))

            i0 = np.random.randint(0, x.shape[-1] - x_length, x.shape[0])
            x_diff = torch.stack(
                [xc[..., i:i + x_length] for i, xc in zip(i0, x)])

            i1 = np.random.randint(0, x.shape[-1] - x_length, x.shape[0])
            x_toz = torch.stack(
                [xc[..., i:i + x_length] for i, xc in zip(i1, x)])

            return {
                "x": x_diff,
                "x_time_cond": x_diff,
                "x_toz": x_toz,
            }

    elif args.dataset_type == "midi":
        dataset = SimpleDataset(path=args.db_path, keys=["waveform", "pr"])
        dataset, valset = torch.utils.data.random_split(
            dataset, (len(dataset) - 200, 200))

        x_length = gin.query_parameter("%X_LENGTH")
        ae_ratio = gin.query_parameter("%AE_FACTOR")

        def collate_fn(L):
            x = np.stack([l["waveform"] for l in L])
            x = torch.from_numpy(x).float().reshape((x.shape[0], 1, -1))

            pr = np.stack([l["pr"] for l in L])
            pr = torch.from_numpy(pr).float()

            i0 = np.random.randint(0, pr.shape[-1] - x_length // ae_ratio,
                                   x.shape[0])
            x_diff = torch.stack([
                xc[..., i * ae_ratio:i * ae_ratio + x_length]
                for i, xc in zip(i0, x)
            ])
            pr = torch.stack(
                [xc[..., i:i + x_length // ae_ratio] for i, xc in zip(i0, pr)])

            i1 = np.random.randint(0, x.shape[-1] - x_length, x.shape[0])
            x_toz = torch.stack(
                [xc[..., i:i + x_length] for i, xc in zip(i1, x)])

            return {
                "x": x_diff,
                "x_time_cond": pr,
                "x_toz": x_toz,
            }

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.bsize,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True,
                                               collate_fn=collate_fn)

    valid_loader = torch.utils.data.DataLoader(valset,
                                               batch_size=args.bsize,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False,
                                               collate_fn=collate_fn)

    print(next(iter(train_loader))["x"].shape)
    print(next(iter(train_loader))["x_toz"].shape)

    ######### SAVE CONFIG #########
    model_dir = os.path.join(args.out_path, args.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in model.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    num_el = 0
    for p in model.encoder.parameters():
        num_el += p.numel()
    print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if model.classifier is not None:
        num_el = 0
        for p in model.classifier.parameters():
            num_el += p.numel()
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    if model.encoder_time is not None:
        num_el = 0
        for p in model.encoder_time.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    ######### TRAINING #########
    d = {
        "dataset": dataset,
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": args.restart,
        "device": model.accelerator.device,
    }

    model.fit(**d)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
