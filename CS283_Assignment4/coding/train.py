import os

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils



def infinite_dataloader(dataset, batch_size):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, num_flows, num_blocks):
    z_shapes = []

    for i in range(num_blocks - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(config, model, optimizer, dataloader, device):
    n_bins = 2.0 ** config.num_bits

    z_sample = []
    z_shapes = calc_z_shapes(config.img_num_channels, config.img_size, config.num_flows, config.num_blocks)
    
    for z in z_shapes:
        z_new = torch.randn(config.num_samples, *z) * config.temp
        z_sample.append(z_new.to(device))
        
    tqdm._instances.clear()


    with tqdm(range(config.num_iters)) as pbar:
        for i in pbar:
            image, _ = next(iter(dataloader))
            image = image.to(device)
            image = image * 255

            if config.num_bits < 8:
                image = torch.floor(image / 2 ** (8 - config.num_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
                    continue
            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, config.img_size, n_bins)
            model.zero_grad()
            loss.backward()

            warmup_lr = config.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")

            if (i + 1) % config.save_samples_freq == 0:
                os.makedirs(config.samples_dir, exist_ok=True)
                
                with torch.no_grad():
                    img = (model.reverse(z_sample).cpu().data > 0).float()
                    utils.save_image(
                        img,
                        f"{config.samples_dir}/{str(i + 1).zfill(6)}.jpeg",
                        nrow=10,
                    )

            if (i + 1) % config.save_ckpt_freq == 0:
                os.makedirs(config.checkpoints_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{config.checkpoints_dir}/model_{str(i + 1).zfill(6)}.pt")
                torch.save(optimizer.state_dict(), f"{config.checkpoints_dir}/optim_{str(i + 1).zfill(6)}.pt")
