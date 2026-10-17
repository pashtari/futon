"""Snapshots of image training, for the paper's teaser figure.

Fits each model to one Kodak image exactly as scripts/train_image.py does,
evaluating every ten epochs, and keeps its reconstruction at the epochs by
which the model's exclusive-GPU timing run (logs/timing/image) had trained for
the given seconds, as <log-dir>/<signal>/<model>/epoch_<n>.png beside the run's
results.json. Evaluation adds no training time and draws no random numbers, so
a snapshot is the timed run's model at that epoch.

Usage:
    python scripts/snapshots.py --data data/Kodak/kodim19.png \\
        --models SIREN Instant-NGP FUTON-lanczos --seconds 0.5 1 2 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from train_common import ROOT, build, warm_up
from train_image import prepare

import neurofield as nf

EVERY = 10  # epochs between evaluations, so between snapshots


def epochs_at(history: list[dict], seconds: list[float]) -> dict[float, int]:
    """The last evaluated epoch by which a timing run had trained ``seconds``."""
    epoch = np.array([0] + [entry["epoch"] for entry in history], dtype=float)
    elapsed = np.array([0.0] + [entry["elapsed"] for entry in history])
    reached = np.interp(seconds, elapsed, epoch)  # time per epoch is steady
    return {t: int(e // EVERY * EVERY) for t, e in zip(seconds, reached)}


class Snapshots(nf.ImageCoordinateDataset):
    """The image, keeping the reconstructions of the wanted epochs."""

    def __init__(self, path: Path, wanted: set[int]) -> None:
        super().__init__(path)
        self.wanted, self.epoch = wanted, 0

    def save(self, x: torch.Tensor, path: str) -> None:
        self.epoch += EVERY  # evaluate saves once per evaluation
        if self.epoch in self.wanted:
            super().save(x, Path(path).parent / f"epoch_{self.epoch:04d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="a Kodak image")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seconds", nargs="+", type=float, required=True)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "snapshots")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / "image.yaml").read_text())
    signal = args.data.stem
    for name in args.models:
        timing = ROOT / "logs" / "timing" / "image" / signal / name / "results.json"
        history = json.loads(timing.read_text())["history"]
        wanted = epochs_at(history, args.seconds)
        model = prepare(config["models"][name], args.data)
        model_class, kwargs = build(model)
        net = model_class(in_features=2, out_features=3, **kwargs)
        train_dataset = nf.ImageCoordinateDataset(args.data, **config["data"])
        warm_up(net, train_dataset, torch.device(args.device))
        out = args.log_dir / "image" / signal / name
        settings = {
            **config["train"],
            **model["train"],
            "eval_interval": EVERY,
            "log_interval": EVERY,
        }
        result = nf.train(
            net,
            train_dataset,
            Snapshots(args.data, set(wanted.values())),
            metrics={"psnr": nf.psnr},
            device=args.device,
            log_dir=out,
            **settings,
        )
        record = {"seconds": {str(t): e for t, e in wanted.items()}, **result}
        (out / "results.json").write_text(json.dumps(record, indent=1, default=str))
        print(f"{name}: snapshots at epochs {sorted(wanted.values())}", flush=True)


if __name__ == "__main__":
    main()
