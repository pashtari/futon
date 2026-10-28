"""Training loop for neural fields on coordinate datasets."""

import copy
import itertools
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .evaluation import evaluate
from .models.futon_solvers import BCD
from .quantization import uniform_quantize
from .utils import (
    _average_dicts,
    _format_metrics,
    count_parameters,
    setup_experiment_loggers,
)

__all__ = ["train"]

LossFn: TypeAlias = Callable[
    [dict[str, Any], nn.Module], tuple[Tensor, dict[str, float | Tensor]]
]


def _mse_metrics(mse: Tensor) -> dict[str, Tensor]:
    """The MSE and the PSNR of targets in ``[-1, 1]``, left on the device."""
    return {"mse": mse, "psnr": 10 * torch.log10(4.0 / mse)}


def _mse_loss(
    batch: dict[str, Any], model: nn.Module
) -> tuple[Tensor, dict[str, Tensor]]:
    """Default reconstruction loss, with PSNR for targets in ``[-1, 1]``."""
    mse = F.mse_loss(model(batch["input"]), batch["target"])
    return mse, _mse_metrics(mse.detach())


def train(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    *,
    loss_fn: LossFn | None = None,
    batch_size: int = 1,
    num_steps: int = 1000,
    optimizer: Optimizer | None = None,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    quantize: bool = False,
    quant_max: int = 127,
    quant_interval: int = -1,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    rate_metrics: Mapping[str, Callable[[bytes, Tensor], float]] | None = None,
    log_interval: int = 100,
    eval_interval: int | None = None,
    chunk_size: int | None = None,
    save_reconstruction: bool = True,
    device: str | torch.device | None = None,
    ckpt_path: str | os.PathLike[str] | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a neural field, by default with Adam and a cosine-annealed learning rate.

    Each step draws one batch and takes one optimizer step; the loader passes
    over ``train_dataset`` again whenever it runs out. The scheduler anneals
    the learning rate over ``num_steps`` to ``lr / 100``. Moves model and
    supported datasets to ``device`` in place. Optional quantization modifies
    parameters during training, evaluates a quantized copy and quantizes the
    final model in place.

    Args:
        model: Neural field to train in place.
        train_dataset: Dataset returning ``input`` and ``target`` entries.
        eval_dataset: Evaluation dataset; ``None`` or empty uses ``train_dataset``.
        loss_fn: Called as ``loss_fn(batch, model) -> (loss, metrics)``.
            ``None`` uses MSE and logs PSNR assuming targets in ``[-1, 1]``.
            Metrics may stay scalar tensors: they are read only when logged,
            so that no step waits for the GPU.
        batch_size: Training batch size.
        num_steps: Number of training steps.
        optimizer: Steps once per batch, its learning rate under the cosine
            schedule; ``None`` builds Adam from ``lr``, ``weight_decay``,
            ``adam_betas`` and ``adam_eps``. A block coordinate descent solver
            for FUTON, :class:`~neurofield.models.futon_solvers.BCD`, fits
            each batch itself, without ``loss_fn``; its learning rate is the
            weight of the newest batch in its statistics.
        lr: Initial learning rate.
        weight_decay: Adam weight decay.
        adam_betas: Adam moment decay rates; Instant-NGP's reference code uses
            ``(0.9, 0.99)``, while the benchmarks keep PyTorch's defaults.
        adam_eps: Adam epsilon; hash grids use ``1e-15``.
        quantize: Enable quantized evaluation and final in-place quantization.
        quant_max: Largest integer quantization level.
        quant_interval: Steps between intermediate fake quantizations.
            Non-positive values disable them; requires ``quantize=True``.
        metrics: Distortion metrics passed to :func:`evaluate`.
        rate_metrics: Rate metrics passed to :func:`evaluate`.
        log_interval: Steps between history records, each also a line on the
            console and in the text log, with the evaluation's metrics when
            the step evaluates. Final and evaluation steps are always
            recorded; non-positive values record only those.
        eval_interval: Evaluate every this many steps and at the final step.
            ``None`` evaluates only at the end; non-positive values disable it.
        chunk_size: Coordinates per evaluation forward pass; ``None`` uses all.
        save_reconstruction: Save each evaluation's reconstruction under
            ``log_dir``; see :func:`evaluate`.
        device: Training device; ``None`` selects CUDA when available, else CPU.
        ckpt_path: State dict to load first, if the file exists.
        log_dir: Directory for ``log.txt``, with a line per record, ``log.json``,
            reconstructions and ``checkpoint.pt``. Text logs are truncated
            initially; JSON records append. ``None`` keeps no files; the
            console shows a progress bar, the records and a summary either way.
        seed: Seed for the global PyTorch RNG.

    Returns:
        ``config`` with the run's settings and ``peak_memory``, the most bytes
        a training step allocated beyond the data and parameters already on
        the device, drawing its batch and evaluations left out (zero off CUDA),
        ``model_dict`` with final parameters (integer codes and scales when
        quantized), and ``history`` with one record per logged step. Records
        contain ``step``, ``elapsed``, the seconds spent training so far
        (evaluating and logging left out), the training metrics of the step's
        batch, and averaged evaluation metrics under ``eval`` when evaluated.
    """
    logger, json_logger = setup_experiment_loggers(log_dir, __name__)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    eval_dataset = eval_dataset or train_dataset
    # Keep coordinates on the training device so subsampling runs there too.
    for dataset in (train_dataset, eval_dataset):
        if hasattr(dataset, "to"):
            dataset.to(device)
    on_cuda = getattr(getattr(train_dataset, "input", None), "is_cuda", False)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=not on_cuda, num_workers=0
    )

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    loss_fn = loss_fn or _mse_loss
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    lr = optimizer.defaults["lr"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr / 100
    )
    if isinstance(optimizer, BCD):
        settings = {
            key: value for key, value in optimizer.defaults.items() if key != "lr"
        }
    else:
        settings = {
            "weight_decay": optimizer.defaults.get("weight_decay", weight_decay),
            "adam_betas": tuple(optimizer.defaults.get("betas", adam_betas)),
            "adam_eps": optimizer.defaults.get("eps", adam_eps),
        }
    config: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "model": type(model).__name__,
        "num_params": count_parameters(model),
        "optimizer": type(optimizer).__name__,
        "lr": lr,
        **settings,
        "scheduler": type(scheduler).__name__,
        "chunk_size": chunk_size or -1,
        "num_steps": num_steps,
    }
    logger.debug("Config: %s", config)

    if eval_interval is None:
        eval_interval = num_steps
    # One batch per step: the loader starts over whenever it runs out.
    batches = itertools.chain.from_iterable(itertools.repeat(loader))
    history: list[dict[str, Any]] = []
    status = ""
    elapsed = 0.0
    progress = tqdm(range(1, num_steps + 1), desc="Training")
    cuda = device.type == "cuda"
    peak_memory = 0

    model.train()
    start_time = time.perf_counter()
    for step in progress:
        resident = torch.cuda.memory_allocated(device) if cuda else 0
        batch = next(batches)
        for key, value in batch.items():
            if isinstance(value, Tensor) and not key.startswith("_"):
                batch[key] = value.to(device)
        if cuda:  # the step's own peak, apart from drawing its batch
            torch.cuda.reset_peak_memory_stats(device)

        if isinstance(optimizer, BCD):
            step_metrics = _mse_metrics(optimizer.step(batch))
        else:
            loss, step_metrics = loss_fn(batch, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if cuda:
            peak = torch.cuda.max_memory_allocated(device) - resident
            peak_memory = max(peak_memory, peak)
        if quantize and quant_interval > 0 and step % quant_interval == 0:
            model, _ = uniform_quantize(model, quant_max)

        should_log = (log_interval > 0 and step % log_interval == 0) or (
            step == num_steps
        )
        should_eval = eval_interval > 0 and (
            step % eval_interval == 0 or step == num_steps
        )
        if not (should_log or should_eval):
            continue  # no host sync, so the GPU never waits for Python

        if cuda:
            # CUDA kernels run asynchronously; wait so timings are wall-clock.
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start_time
        record: dict[str, Any] = {"step": step, "elapsed": elapsed}
        for key, value in step_metrics.items():
            record[key] = value.item() if isinstance(value, Tensor) else value
        status = _format_metrics(record)
        progress.set_postfix_str(status)

        if should_eval:
            eval_model, model_dict = (
                uniform_quantize(copy.deepcopy(model), quant_max)
                if quantize
                else (model, None)
            )
            evaluation = evaluate(
                eval_model,
                eval_dataset,
                model_dict=model_dict,
                metrics=metrics,
                rate_metrics=rate_metrics,
                chunk_size=chunk_size,
                device=device,
                log_dir=log_dir,
                save_reconstruction=save_reconstruction,
            )
            record["eval"] = _average_dicts(evaluation["history"])
            if evaluated := _format_metrics(record["eval"]):  # none without metrics
                status += " - eval " + evaluated
            model.train()  # evaluation leaves the model in evaluation mode

        logger.info("Step [%d/%d] - %s", step, num_steps, status)
        json_logger.log(record)
        history.append(record)
        start_time = time.perf_counter()  # evaluating and logging is not training

    logger.info("Training completed in %.1f s - %s", elapsed, status)
    config["peak_memory"] = peak_memory

    if quantize:
        model, model_dict = uniform_quantize(model, quant_max)
    else:
        model_dict = model.state_dict()

    if log_dir:
        checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)

    return {"config": config, "model_dict": model_dict, "history": history}
