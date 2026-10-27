# Reproducing the paper

Every number and figure of the paper comes from the training scripts and one that reports. This page is the full pipeline, from an empty `data/` folder to `results/`.

## 1. Data

```bash
python scripts/download_kodak.py      # data/Kodak/, 24 images, about 30 MB
python scripts/download_meshes.py     # data/occupancy/, 5 meshes, about 3 GB
python scripts/download_blender.py    # data/nerf/blender/, 8 scenes, about 2.4 GB
python scripts/download_div2k.py      # data/DIV2K/, 100 validation images and their 4x downsamplings, about 450 MB
```

The Kodak images come from Rich Franzen's mirror, the meshes from the Stanford 3D Scanning Repository (rotated into canonical pose where a release is not), the Blender scenes from a pinned Hugging Face mirror of the official Google Drive release, and the DIV2K images from the official site. All four scripts resume and keep existing files.

## 2. Training

`scripts/train_<task>.py` fits every model of `configs/<task>.yaml` to every signal of the task:

```bash
python scripts/train_image.py        # 24 images x 14 models -> logs/image/<image>/<model>/
python scripts/train_occupancy.py    # 5 shapes x 15 models  -> logs/occupancy/<shape>/<model>/
python scripts/train_nerf.py         # 8 scenes x 15 models  -> logs/nerf/<scene>/<model>/
python scripts/train_super_resolution.py  # 100 images x 18 models -> logs/super_resolution/<image>/<model>/
```

The four scripts share one command line:

| Option | Meaning |
| --- | --- |
| `--data PATH ...` | Signals to fit; default: every signal under `data/` for the task. |
| `--models NAME ...` | Model names from the config; default: all. |
| `--config FILE ...` | Configs, deep-merged in order; the last names the log directory (`configs/<name>.yaml` writes to `logs/<name>/`). |
| `--set KEY.PATH=VALUE ...` | Override one value, read as YAML, e.g. `train.num_steps=500` or `models.SIREN.train.lr=3e-4`. |
| `--log-dir DIR` | Where the runs go. |
| `--overwrite` | Rerun runs that already finished. |
| `--device DEV` | `cuda` when available, else `cpu`. |

A config holds the shared `data` and `train` settings and, per model, its NeuroField class, constructor `kwargs` and training overrides (the learning rate). Image configs may use expressions in the image height `H` and width `W`, such as `[H // 2, W // 2]`; the super-resolution config sizes its models from the high-resolution image, and FUTON's entry names a parameter `budget` whose largest CP rank the script fills in. Each config's header says how its learning rates were chosen, and `# deviation` marks a setting that departs from the authors' published ones.

Finished runs are skipped, so an interrupted sweep restarts where it stopped, and a failed run writes `error.txt` and lets the sweep continue. Runs are keyed by model name, so a variant needs its own name or its own `--log-dir`:

```bash
python scripts/train_image.py --models Gauss --overwrite                       # rerun one model in place
python scripts/train_image.py --set models.SIREN.train.lr=0.001 --log-dir logs/siren-lr
python scripts/train_image.py --config configs/ablation-futon/basis.yaml       # -> logs/ablation-futon/basis/
```

Each run directory holds `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (images) or two rendered test views (NeRF; super-resolution saves none, the report renders them), and `results.json` with the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics` (NeRF: the mean over the 200 test views, with every view under `test_views`) and the `history`. Occupancy runs write no mesh, since rendering needs a display; the report rebuilds them from the checkpoints.

Measured on A100s (1000 steps; NeRF 25,000), each signal's models in one job:

| Task | Per model | Per signal | Whole sweep |
| --- | --- | --- | --- |
| Images | 56 s | 13 min | 5.2 h |
| Occupancy | 27 s | 7 min | 34 min |
| NeRF | 5 min | 77 min | 10 h |
| Super-resolution | 1 min | 15 min | 25 h |

A run's time includes compiling its model and its 20 evaluations, with LPIPS on images and super-resolution; training alone takes from 3 s (Instant-NGP) to a minute (MFN) on an image. A super-resolution step renders 261k pixels through a 500k-parameter model, so that sweep is best spread over many short jobs, one image each. Every run uses seed 0; some GPU kernels accumulate in a nondeterministic order, so a rerun on the same hardware lands within about a tenth of a decibel of a run, and training times vary with the machine and its load.

## 3. Ablations

The FUTON ablations are configs in `configs/ablation-futon/`, run through the same scripts. All keep the benchmark's settings, so only the model changes:

| Config | Task | Models | Varies |
| --- | --- | --- | --- |
| `basis.yaml` | image | 6 | The six bases at the benchmark's size. |
| `components_rank.yaml` | image | 50 | The components per axis as a fraction $\alpha$ of the pixels, $K = (\alpha H, \alpha W)$, and the rank as a fraction $\beta$ of the smaller count, $R = \beta \min(K_1, K_2)$, both over {1/8, 1/4, 1/2, 1, 2}, sinc and cosine. |
| `tensor_net.yaml` | image | 6 | Tensor-ring and Tucker against CP, for the sinc and the cosine basis. |
| `decoder.yaml` | image | 8 | MLP decoders 64, 128 and 256 wide and the linear decoder, at equal size, for the sinc and the cosine basis. |
| `components_width_image.yaml` | image | 30 | The components per axis, $K = N/L$ for $L$ in {4, 2, 1}, against the width of the one-layer decoder, 16 to 256, each at the largest rank, a multiple of 5, within the budget, sinc and cosine. |
| `components_width_occupancy.yaml` | occupancy | 30 | The same on the five volumes. |
| `components_width_nerf.yaml` | NeRF | 30 | The same on the eight scenes. |
| `components_width_super_resolution.yaml` | super-resolution | 40 | The same with $L = 8$ added, on the three DIV2K training images the settings were tuned on. |

The image studies run on all 24 Kodak images at the benchmark's settings (1000 steps on the whole image); in `components_rank.yaml` a model's name carries its fractions as decimals, so `FUTON-sinc-a0.5-b1` has $K = (H/2, W/2)$ and $R = H/2$. The components-width studies repeat their task's data and training settings, and a model's name carries $L$ and the width: the benchmark's model is `FUTON-sinc-N2-w64` on images, volumes and radiance fields, and `FUTON-sinc-N4-w128` in super-resolution. The benchmark's FUTON runs on radiance fields are this study's runs of that configuration, copied rather than repeated; each copy's `results.json` names its `source`. Their ranks are set like the benchmark's: written out on images and radiance fields, and `rank: budget` with the budget beside it on volumes and super-resolution, whose grids change with the signal.

```bash
python scripts/train_image.py --config configs/ablation-futon/components_rank.yaml
python scripts/train_occupancy.py --config configs/ablation-futon/components_width_occupancy.yaml
python scripts/train_nerf.py --config configs/ablation-futon/components_width_nerf.yaml  # 8 scenes x 30 models
python scripts/train_super_resolution.py --config configs/ablation-futon/components_width_super_resolution.yaml \
  --data data/DIV2K/DIV2K_train_HR/0395.png data/DIV2K/DIV2K_train_HR/0431.png data/DIV2K/DIV2K_train_HR/0777.png
```

## 4. Report

`scripts/report_paper.py` turns the runs into `results/`, which holds the paper's figures and tables and nothing else:

```
results/
├── legend.{pdf,pgf}                 the models' legend, shared by every figure; super_resolution/legend adds the bicubic level
├── <task>/convergence.{pdf,pgf}     quality against training time, FUTON and the strongest model of each family
├── <task>/training_time.{pdf,pgf}   every model's final quality against its training time
├── <task>/throughput.{pdf,pgf}      the same against its inference throughput
├── <task>/{training,inference}_memory.{pdf,pgf}  and against its peak memory in either phase
├── <task>/table.{tex,md}            size, the cost of both phases and final metrics, every shape's and scene's too, best in bold, second underlined
├── <task>/qualitative_<signal>.pdf  the signal with two regions boxed and magnified for every featured model (bicubic for TensoRF in super-resolution), above each model's |error| for images and super-resolution
├── <task>/panels/<signal>/          the renders those magnifications are cut from, rebuilt from the checkpoints
├── ablation/                        a table and a convergence plot each for the bases, the combiners and the decoders, PSNR against each fraction, and quality against training time for components and width, with a table per task
└── bound/                           the error bound against trained FUTONs, from the records of scripts/futon_bound.py: figures and tables of the bound page
```

```bash
python scripts/report_paper.py                          # everything
python scripts/report_paper.py --tasks image --overwrite  # redraw one task's panels
python scripts/report_paper.py --tasks bound            # only the error bound; --tasks also takes ablation and tuning
python scripts/report_paper.py --no-panels              # tables and plots only, no GPU needed
python scripts/report_paper.py --orbit 60               # plus an orbit GIF of each NeRF model
```

The signals the paper shows are listed in `EXAMPLES` at the top of the script, and their magnified regions and the radiance-field views are set by hand in `CENTERS` and `VIEWS`, on parts of the signal where the models visibly differ. Any other signal gets them by rule: the view at the median of FUTON's gain over the strongest other model shown, and the two squares where the models shown disagree most, weighted by the detail the reference carries there, which favours no model. Images and super-resolution add each model's absolute error beneath its magnifications, on one scale per figure. Drawing the panels needs the signals in `data/` and a GPU; they are kept and reused, so recomposing a figure costs seconds. Figures come as PDF and as PGF for LaTeX, sized for a NeurIPS page: a plot is a third of the 5.5-inch text width, a strip of magnifications spans it, plot text is 7 to 8 pt, a strip labels every panel in one size, at most 7 pt, at which its widest label fits, and every tick falls on a whole or half number. The image and super-resolution tables fit the text width in 9 pt type; the occupancy and NeRF ones, with every shape's and scene's scores, are wider and take a page turned sideways. The `±` in the tables is a paired standard error over the signals: each signal's own level is removed before the error is taken, which is the uncertainty of a comparison between models rather than of an absolute level; IoU, which saturates near 100 % on every shape, takes the plain standard error over the shapes. A band in a figure spans one of them, the same error.

Costs come from the benchmark runs themselves: the training time counts the training steps alone, after a warm-up pass that compiles the model, and every evaluation records how long it took and its peak memory. Each cost in the tables is the median over a task's signals, so a run slowed by other work on its GPU does not move it.

Or skip the report and read the runs directly:

```python
import json
from pathlib import Path
import pandas as pd

records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)
```
