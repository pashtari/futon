"""Time every model's inference on one signal per task, under one allocation.

The benchmark's own durations come from jobs that shared their node, and
rendering is partly bound by kernel launches, so they carry whatever else ran
beside them. Timing every model inside one job makes the comparison even, the
models taking turns so that none depends on what ran before it; an exclusive
allocation makes it clean:

    sbatch --clusters=accelgor --exclusive --time=1:00:00 --gpus-per-node=1 \\
      --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out \\
      --job-name=nf-speed --wrap \\
      "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/profile_speed.py"

Usage:
    python scripts/profile_speed.py
    python scripts/profile_speed.py --tasks image occupancy --repeats 30
    python scripts/profile_speed.py --models F-INR FreSh  # add models to the files

Writes logs/speed/<task>.json: each model's median inference time over the
repeats, the points it evaluated, the rate those give, and the most memory one
inference allocated beyond the model and its input. Memory needs no exclusive
node, so a shared job can write it elsewhere with --out-dir logs/memory.
"""

import argparse
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import report_paper as report
import torch
from report_paper import field_and_renderer
from train_common import compile_model, provenance

import neurofield as nf

CHUNK = 262_144  # coordinates per forward pass, as configs/occupancy.yaml and
# configs/super_resolution.yaml evaluate them


def timed(calls: dict[str, Callable[[], object]], repeats: int) -> dict[str, float]:
    """Median seconds of each call, after two warm-ups, synchronized on CUDA.

    The calls take turns in a new random order each round, so a model's time
    depends neither on the model before it nor on when in the job it ran (two
    identical networks timed one after the other differed by 12 %).
    """
    names = list(calls)
    durations = {name: [] for name in names}
    order = random.Random(0)
    for _ in range(repeats + 2):
        order.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter()
            calls[name]()
            torch.cuda.synchronize()
            durations[name].append(time.perf_counter() - start)
    return {name: statistics.median(times[2:]) for name, times in durations.items()}


def peak_memory(call: Callable[[], object], device: torch.device) -> int:
    """Bytes a call allocates at its peak beyond what is already on the device."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    call()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) - resident


def image_inference(
    run: dict, path: Path, device, compile: bool, chunk_size: int | None = None
) -> tuple[Callable, int]:
    """Render the image from its coordinates, as the benchmark scores it: the
    Kodak images in one pass, and the DIV2K ones in chunks of CHUNK pixels."""
    dataset = nf.ImageCoordinateDataset(path)
    model = report.load_model(run, 2, 3, device)
    if compile:
        compile_model(model)
    coords = dataset.input.to(device)
    return (
        lambda: nf.chunked_inference(
            model, coords, chunk_size=chunk_size, device=device
        ),
        coords[..., 0].numel(),
    )


def occupancy_inference(
    run: dict, path: Path, device, compile: bool
) -> tuple[Callable, int]:
    """Evaluate the whole volume in chunks, as the benchmark scores it."""
    config = report.model_config(run)
    dataset = nf.OccupancyCoordinateDataset(path, **config["data"] | {"subsample": 1.0})
    model = report.load_model(run, 3, 1, device)
    if compile:
        compile_model(model)
    coords = dataset.input.to(device)
    return (
        lambda: nf.chunked_inference(model, coords, chunk_size=CHUNK, device=device),
        coords[..., 0].numel(),
    )


def nerf_inference(
    run: dict, path: Path, device, compile: bool
) -> tuple[Callable, int]:
    """Render one test view, as the benchmark scores it.

    ``field_and_renderer`` restores the run's occupancy grid, so the timing
    covers rendering alone, with the grid the run was scored with.
    """
    config = report.model_config(run)
    dataset = nf.nerf.BlenderDataset(
        path, "test", downsample=config["data"]["downsample"]
    )
    field, renderer = field_and_renderer(run, config, dataset, device)
    if compile:
        compile_model(field)
    batch = dataset[0]
    background = dataset.background.to(device)

    def render() -> object:
        with torch.no_grad(), nf.nerf.evaluation._eval_mode(field, renderer):
            return nf.nerf.render_image(
                field, renderer, batch["rays_o"], batch["rays_d"], background=background
            )

    return render, batch["rays_o"][..., 0].numel()


def super_resolution_inference(
    run: dict, path: Path, device, compile: bool
) -> tuple[Callable, int]:
    """Render the high-resolution image, as the benchmark scores it.

    Deep Image Prior renders it from its fixed noise; the interpolations are
    not timed here, their own evaluation already times PIL alone.
    """
    if run["setup"].get("class") != "DIPSkip":
        return image_inference(run, path, device, compile, CHUNK)
    model, noise = report.load_dip(run, path, device)
    if compile:
        compile_model(model)
    noise = noise.to(device)
    return (
        lambda: nf.chunked_inference(model, noise, device=device),
        noise[0, 0].numel(),
    )


INFERENCE = {
    "image": image_inference,
    "occupancy": occupancy_inference,
    "nerf": nerf_inference,
    "super_resolution": super_resolution_inference,
}
UNITS = {
    "image": "pixels",
    "occupancy": "voxels",
    "nerf": "rays",
    "super_resolution": "pixels",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", default=list(INFERENCE), choices=list(INFERENCE)
    )
    parser.add_argument(
        "--models", nargs="+", help="profile these alone, kept beside the others"
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--compile", action="store_true", help="compile each model with torch.compile"
    )
    parser.add_argument("--log-dir", type=Path, default=report.ROOT / "logs")
    parser.add_argument("--out-dir", type=Path, default=report.ROOT / "logs" / "speed")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    origin = provenance(device)

    for task in args.tasks:
        signal = report.PROFILED[task]
        path = report.ROOT / report.SIGNALS[task].format(signal)
        runs = [
            run
            for run in report.read(args.log_dir / task, task)
            if run["data"] == signal
            and "interpolation" not in run["setup"]
            and (args.models is None or run["model"] in args.models)
        ]
        print(f"\n{task} ({signal}), median of {args.repeats}")
        out = args.out_dir / f"{task}.json"
        measured = json.loads(out.read_text()) if args.models and out.exists() else {}
        calls, points, memory = {}, {}, {}
        for run in sorted(runs, key=lambda run: run["model"]):
            name = run["model"]
            calls[name], points[name] = INFERENCE[task](run, path, device, args.compile)
            calls[name]()  # once first: compiling and one-off allocations go unmeasured
            memory[name] = peak_memory(calls[name], device)
        for name, seconds in timed(calls, args.repeats).items():
            measured[name] = {
                "seconds": seconds,
                "points": points[name],
                "rate": points[name] / seconds,
                "memory": memory[name],
                "compiled": args.compile,
                "provenance": origin,  # per model, as --models adds to a file
            }
            print(
                f"  {name:14} {1000 * seconds:8.1f} ms"
                f"  {points[name] / seconds / 1e6:7.2f} M{UNITS[task]}/s"
                f"  {memory[name] / 1e6:8.1f} MB",
                flush=True,
            )
        del calls
        torch.cuda.empty_cache()
        out.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
