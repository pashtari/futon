"""Download the DIV2K images for 4x super-resolution.

The 100 validation images of DIV2K (Agustsson and Timofte, CVPRW 2017), 2040
pixels on the long side, and their 4x bicubic downsamplings (MATLAB's
imresize), from the official site:

    https://data.vision.ee.ethz.ch/cvl/DIV2K/

The archives unpack in the release's layout, as DIV2K_valid_HR/0801.png and
DIV2K_valid_LR_bicubic/X4/0801x4.png; notebooks/image_super_resolution.ipynb
reads image 0887 from there. ``--training`` adds the three training images
that every setting of the benchmark was tuned on, 0395, 0431 and 0777, which
configs/ablation-futon/components_width_super_resolution.yaml runs on: it
fetches the training archives, 3.8 GB, and keeps only those images. An
archive whose images all exist is skipped.

Usage:
    python scripts/download_div2k.py
    python scripts/download_div2k.py --training   # and the three tuning images
"""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/{}.zip"
VALIDATION = range(801, 901)
TRAINING = (395, 431, 777)  # the images every setting was tuned on
ARCHIVES = {  # archive: where each of its images unpacks, and the images kept
    "DIV2K_valid_HR": ("DIV2K_valid_HR/{:04d}.png", VALIDATION),
    "DIV2K_valid_LR_bicubic_X4": ("DIV2K_valid_LR_bicubic/X4/{:04d}x4.png", VALIDATION),
    "DIV2K_train_HR": ("DIV2K_train_HR/{:04d}.png", TRAINING),
    "DIV2K_train_LR_bicubic_X4": ("DIV2K_train_LR_bicubic/X4/{:04d}x4.png", TRAINING),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "DIV2K")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Also the three training images the settings were tuned on (3.8 GB)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files")
    args = parser.parse_args()

    for name, (image, images) in ARCHIVES.items():
        if images is TRAINING and not args.training:
            continue
        members = [image.format(i) for i in images]
        if all((args.output_dir / m).exists() for m in members) and not args.force:
            print(f"  {name} exists, skipping")
            continue
        print(f"  {name}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Beside the images rather than in /tmp, which is small on some clusters.
        with tempfile.TemporaryFile(dir=args.output_dir) as archive:
            with urllib.request.urlopen(URL.format(name)) as response:
                shutil.copyfileobj(response, archive)
            with zipfile.ZipFile(archive) as files:
                for member in members:
                    files.extract(member, args.output_dir)

    print(f"Done. Images in {args.output_dir}")


if __name__ == "__main__":
    main()
