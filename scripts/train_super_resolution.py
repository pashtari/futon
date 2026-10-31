"""Super-resolution benchmark: recover DIV2K images from their 4x bicubic downsamplings.

Every model in configs/super_resolution.yaml sees only the low-resolution (LR)
image and is scored against the high-resolution (HR) original. The benchmark's
degradation, antialiased bicubic downsampling (MATLAB's imresize), is linear
and separable: one matrix per axis describes it exactly, borders included
(inro.resize_matrix), and an LR pixel needs the field only under its kernel.
Each step renders the HR pixels under random blocks of LR pixels
(inro.SuperResolutionDataset), downsamples them with those matrices and matches
the LR pixels (inro.super_resolution_loss, L1). DIP renders the whole image from
fixed noise, and the interpolations upsample the LR image with PIL, whose
bicubic filter is MATLAB's. As is standard, PSNR and SSIM are computed on luma
(BT.601 Y) and LPIPS on RGB, all without a 4-pixel border
(inro.super_resolution_metrics).

Usage:
    python scripts/train_super_resolution.py
    python scripts/train_super_resolution.py --data data/DIV2K/DIV2K_valid_HR/0882.png --models Bicubic FUTON-sinc
"""

import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.functional import pil_to_tensor
from train_common import (
    ROOT,
    build,
    fill_rank,
    fresh,
    on_grid,
    record,
    resolve,
    run,
    warm_up,
)

import inro

DATA = sorted((ROOT / "data" / "DIV2K" / "DIV2K_valid_HR").glob("*.png"))
INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def low_resolution(path: Path, scale: int) -> Path:
    """The DIV2K release's bicubic downsampling of a high-resolution image."""
    folder = path.parent.name.replace("HR", "LR_bicubic")
    return path.parents[1] / folder / f"X{scale}" / f"{path.stem}x{scale}{path.suffix}"


def prior_loss(
    batch: dict[str, Any], model: nn.Module, kernels: Sequence[Tensor]
) -> tuple[Tensor, dict]:
    """Render the HR image from noise, downsample it, and match the LR image:
    the mean absolute error, with the fit's PSNR for the training log."""
    output = inro.downsample(model(batch["input"]).movedim(1, -1), kernels)
    target = batch["target"].movedim(1, -1)
    loss = F.l1_loss(output, target)
    mse = F.mse_loss(output.detach(), target)
    return loss, {"l1": loss.detach(), "psnr": 10 * torch.log10(1.0 / mse)}


def interpolate(
    model: dict, path: Path, scale: int, device: torch.device
) -> dict[str, Any]:
    """Upsample the LR image with PIL, scored like the trained models."""
    original = inro.ImageCoordinateDataset.load(path, mode="RGB")
    with Image.open(low_resolution(path, scale)) as image:
        start = time.perf_counter()
        upsampled = image.convert("RGB").resize(
            (original.shape[2], original.shape[1]),
            INTERPOLATIONS[model["interpolation"]],
        )
        duration = time.perf_counter() - start
    reconstruction = pil_to_tensor(upsampled)
    evaluation = {"id": path.stem, "elapsed": duration, "duration": duration}
    evaluation |= {
        name: metric(reconstruction, original)
        for name, metric in inro.super_resolution_metrics(scale, device).items()
    }
    history = [{"step": 0, "elapsed": 0.0, "duration": 0.0, "eval": evaluation}]
    return record({"config": {"num_params": 0}, "history": history}, model, evaluation)


def prepare(model: dict, path: Path) -> dict[str, Any]:
    """Fill in the HR height H and width W, and FUTON's rank from its budget."""
    width, height = Image.open(path).size
    return fill_rank(resolve(model, H=height, W=width), outputs=3)


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    scale = config["data"]["scale"]
    if "interpolation" in model:
        return interpolate(model, path, scale, device)
    model_class, kwargs = build(model)
    train = {**config["train"], **model["train"]}
    eval_dataset = inro.ImageCoordinateDataset(path)
    height, width = eval_dataset.grid_size
    kernels = [
        inro.resize_matrix(height, height // scale, device),
        inro.resize_matrix(width, width // scale, device),
    ]
    search = None
    if model_class is inro.DIPSkip:
        network = model_class(**kwargs)
        eval_dataset = inro.DIPImageDataset(
            path, noise_channels=kwargs["in_channels"], reg_noise_std=0
        )
        # The same seeded noise, at the HR size, perturbed at each step.
        train_dataset = inro.DIPImageDataset(
            low_resolution(path, scale), noise_shape=eval_dataset.input.shape
        )
        loss_fn = partial(prior_loss, kernels=kernels)
        train["chunk_size"] = None  # a CNN renders the whole image at once
    else:
        # FreSh sees what the field is fitted to: the low-resolution image.
        observed = inro.ImageCoordinateDataset(low_resolution(path, scale))
        search = fresh(
            model,
            kwargs,
            [observed.target.to(device)],
            lambda kw: on_grid(model_class(2, 3, **kw), observed, device),
            train["seed"],
        )
        network = model_class(in_features=2, out_features=3, **kwargs)
        train_dataset = inro.SuperResolutionDataset(
            low_resolution(path, scale),
            kernels,
            config["data"]["num_blocks"],
            config["data"]["block_size"],
        )
        loss_fn = inro.super_resolution_loss
    warm_up(network, train_dataset, device, config["compile"])
    res = inro.train(
        network,
        train_dataset,
        eval_dataset,
        loss_fn=loss_fn,
        metrics=inro.super_resolution_metrics(scale, device),
        # Megapixel reconstructions would crowd the logs; the report renders
        # its examples from checkpoint.pt instead.
        save_reconstruction=False,
        device=device,
        log_dir=out,
        **train,
    )
    return record(res, model, res["history"][-1]["eval"], search)


if __name__ == "__main__":
    run("super_resolution", DATA, fit, prepare)
