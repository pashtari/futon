"""Per-scene NeRF training with random ray batches and mean squared error.

Adam keeps PyTorch's defaults, as :func:`neurofield.train` does, so the
learning rate is the one setting to tune. As in torch-ngp, the learning rate
decays exponentially to ``0.1 * lr`` and each step composites onto a fresh
per-pixel random background. The dataset background is used for evaluation.
Results and logging follow the package-level :func:`neurofield.train` API.
"""

import math
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from ..utils import _format_metrics, count_parameters, setup_experiment_loggers
from .datasets import BlenderDataset
from .evaluation import evaluate
from .renderers import VolumeRenderer

__all__ = ["train"]


def train(
    field: nn.Module,
    renderer: VolumeRenderer,
    train_dataset: BlenderDataset,
    eval_dataset: BlenderDataset | None = None,
    *,
    num_steps: int = 25000,
    num_rays: int = 4096,
    max_samples_per_ray: int | None = 1024,
    lr: float = 1e-2,
    color_lr: float | None = None,
    weight_decay: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    random_background: bool = True,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    eval_interval: int | None = None,
    eval_indices: Sequence[int] | None = None,
    log_interval: int = 100,
    device: str | torch.device | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a radiance field to one scene using random ray batches.

    Each step updates occupancy, renders ``num_rays`` training rays, and
    minimizes their RGB mean squared error with Adam. Batches without occupied
    samples skip the optimizer update and count towards ``skipped_steps``.

    Args:
        field: Radiance field, moved to ``device``.
        renderer: Volume renderer, moved to ``device``.
        train_dataset: Posed images, moved to ``device`` in place for sampling.
        eval_dataset: Held-out views; ``None`` disables evaluation.
        num_steps: Number of training steps.
        num_rays: Rays sampled per step.
        max_samples_per_ray: Cap on occupied samples per ray, front to back;
            ``None`` disables it. Overrides the renderer's constructor setting.
        lr: Initial learning rate, decayed exponentially to ``0.1 * lr``.
        color_lr: Initial learning rate of the field's ``color_net``, decayed
            alike, so that a color network shared by several density networks
            can keep its own; ``None`` uses ``lr``.
        weight_decay: Adam weight decay.
        adam_betas: Adam moment decay rates.
        adam_eps: Adam epsilon.
        random_background: Composite predictions and targets onto a fresh
            random color per pixel each step. This discourages white haze in
            empty space; evaluation always uses the dataset background.
        metrics: Evaluation metrics passed to :func:`evaluate`; ``None``
            selects PSNR, SSIM, and LPIPS.
        eval_interval: Evaluate every this many steps and at the final step.
            ``None`` evaluates only at the end; non-positive values disable it.
        eval_indices: Evaluation view indices; ``None`` selects all views.
        log_interval: Steps between history entries, which update the progress
            bar and the text log. Final and evaluation steps are always
            recorded; non-positive values record only those.
        device: Compute device; ``None`` selects CUDA if available, else CPU.
        log_dir: Directory for ``log.txt``, with a line per history entry,
            ``log.json``, the field's final ``checkpoint.pt``, and
            ``renderer.pt``, the renderer's state with its occupancy grid.
            ``None`` keeps no files; the console shows a progress bar and a
            summary either way.
        seed: Seed passed to ``torch.manual_seed`` before training, after the
            supplied field and renderer have already been constructed.

    Returns:
        Dict with ``"config"``, ``"model_dict"`` (the field's final state dict),
        and ``"history"``. Configuration records model, optimizer, dataset,
        renderer, and seed settings, plus ``skipped_steps`` and ``peak_memory``,
        the most bytes a training step allocated beyond the data, field and
        occupancy grid already on the device (zero off CUDA). Each history entry
        contains ``step``, training-only ``elapsed`` seconds, ``loss``, training
        ``psnr``, and ``num_samples``; evaluation steps also contain mean held-out
        metrics under ``eval``.
    """
    logger, json_logger = setup_experiment_loggers(log_dir, __name__)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    field = field.to(device)
    renderer = renderer.to(device)
    train_dataset.to(device)
    renderer.max_samples_per_ray = (
        None if max_samples_per_ray is None else int(max_samples_per_ray)
    )
    field.train()
    renderer.train()

    num_parameters = count_parameters(field)
    config = {
        "model": type(field).__name__,
        "density_net": type(getattr(field, "density_net", None)).__name__,
        "renderer": type(renderer).__name__,
        "num_params": num_parameters,
        "num_steps": num_steps,
        "num_rays": num_rays,
        "max_samples_per_ray": max_samples_per_ray,
        "lr": lr,
        "color_lr": color_lr,
        "weight_decay": weight_decay,
        "adam_betas": tuple(adam_betas),
        "adam_eps": adam_eps,
        "random_background": random_background,
        "seed": seed,
        "device": str(device),
        "num_train_views": len(train_dataset),
        "image_size": (train_dataset.height, train_dataset.width),
        "aabb": renderer.aabb.tolist(),
        "near": renderer.near,
        "far": renderer.far,
        "render_step_size": renderer.render_step_size,
        "grid_resolution": renderer.grid_resolution,
        "density_threshold": renderer.density_threshold,
    }
    logger.debug("Config: %s", config)

    groups = [{"params": field.parameters()}]
    if color_lr is not None:
        named = field.named_parameters()
        groups = [
            {"params": [p for n, p in named if not n.startswith("color_net.")]},
            {"params": field.color_net.parameters(), "lr": color_lr},
        ]
    optimizer = torch.optim.Adam(
        groups,
        lr=lr,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 ** min(step / num_steps, 1.0)
    )

    if eval_interval is None:
        eval_interval = num_steps
    history: list[dict[str, Any]] = []
    status = ""
    elapsed = 0.0
    skipped_steps = 0
    progress = tqdm(range(1, num_steps + 1), desc="Training")
    cuda = device.type == "cuda"
    resident = torch.cuda.memory_allocated(device) if cuda else 0
    peak_memory = 0

    start_time = time.perf_counter()
    for step in progress:
        if cuda:  # each step's own peak, so that evaluations are left out
            torch.cuda.reset_peak_memory_stats(device)

        # Step zero initializes nerfacc's initially empty occupancy grid.
        renderer.update_occupancy(field, step - 1)

        batch = train_dataset.sample_rays(num_rays)
        rays_o, rays_d, rgba = batch["rays_o"], batch["rays_d"], batch["rgba"]

        background = (
            torch.rand_like(rgba[..., :3])
            if random_background
            else train_dataset.background
        )
        target = train_dataset.composite(rgba, background)

        result = renderer(field, rays_o, rays_d, background=background)
        loss = F.mse_loss(result["rgb"], target)

        optimizer.zero_grad()
        # Pure-background batches do not depend on the field's parameters.
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            skipped_steps += 1
        scheduler.step()

        if cuda:
            peak = torch.cuda.max_memory_allocated(device) - resident
            peak_memory = max(peak_memory, peak)

        should_log = (log_interval > 0 and step % log_interval == 0) or (
            step == num_steps
        )
        should_eval = (
            eval_dataset is not None
            and eval_interval > 0
            and (step % eval_interval == 0 or step == num_steps)
        )
        if not (should_log or should_eval):
            continue  # no host sync, so the GPU never waits for Python

        if cuda:
            # CUDA kernels run asynchronously; wait so timings are wall-clock.
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start_time
        mse = loss.detach().item()
        entry: dict[str, Any] = {
            "step": step,
            "elapsed": elapsed,
            "loss": mse,
            "psnr": 10 * math.log10(1 / max(mse, 1e-12)),
            "num_samples": result["num_samples"],
        }
        status = _format_metrics(entry)
        progress.set_postfix_str(status)

        if should_eval:
            entry["eval"] = evaluate(
                field,
                renderer,
                eval_dataset,
                indices=eval_indices,
                metrics=metrics,
            )["mean"]
            status += " - eval " + _format_metrics(entry["eval"])

        logger.debug("Step [%d/%d] - %s", step, num_steps, status)
        json_logger.log(entry)
        history.append(entry)
        start_time = time.perf_counter()  # evaluating and logging is not training

    logger.info("Training completed in %.1f s - %s", elapsed, status)
    if skipped_steps:
        warnings.warn(
            f"{skipped_steps} of {num_steps} steps had no occupied samples and "
            f"were skipped; the field's occupancy grid is empty or nearly so. "
            f"Treat this run's metrics as invalid rather than as a score.",
            stacklevel=2,
        )
    config["skipped_steps"] = skipped_steps
    config["peak_memory"] = peak_memory

    if log_dir:
        checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
        torch.save(field.state_dict(), checkpoint_path)
        # The grid decides which samples a render queries, so a render of the
        # saved field needs the grid the run was scored with.
        torch.save(renderer.state_dict(), os.path.join(log_dir, "renderer.pt"))
        logger.info("Saved checkpoint to %s", checkpoint_path)

    return {"config": config, "model_dict": field.state_dict(), "history": history}
