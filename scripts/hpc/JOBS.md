# Jobs

Every job of this work, all on one A100 of accelgor. `README.md` covers the
setup and how `train.sh` works. Each command below submits one job (the NeRF
loop, one per scene). Finished runs are skipped, so a job that stops early is
simply submitted again; `--overwrite` reruns finished runs instead.

## Benchmarks

Every model of `configs/<task>.yaml` on every signal of the task, compiled
(`--compile`), one signal per job where the signals are named by number:

| Task | Runs | Jobs | Logs |
| --- | --- | --- | --- |
| image | 24 Kodak images x 14 models = 336 | an array of 24, one image each | `logs/image/<image>/<model>/` |
| occupancy | 5 Stanford shapes x 15 models = 75 | one per shape | `logs/occupancy/<shape>/<model>/` |
| nerf | 8 Blender scenes x 15 models = 120 | two per scene, each half the models | `logs/nerf/<scene>/<model>/` |
| super_resolution | 100 DIV2K images x 18 models = 1800 | an array of 100, one image each | `logs/super_resolution/<image>/<model>/` |

```bash
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 --array=1-24 image \
  --data data/Kodak/kodim%2a.png --compile
for shape in armadillo dragon happy_buddha lucy thai_statue; do
  scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 occupancy \
    --data data/occupancy/$shape.ply --compile
done
A="RFF PE-MLP MFN SIREN FreSh Gauss WIRE FINER"
B="Instant-NGP TensoRF-CP TensoRF-VM GA-Planes F-INR FUTON-sinc FUTON-cosine"
for scene in chair drums ficus hotdog lego materials mic ship; do
  for group in A B; do
    scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 nerf \
      --data data/nerf/blender/$scene --models ${!group} --compile
  done
done
# %4a becomes the task id, zero-padded to four digits.
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 --array=801-900 super_resolution \
  --data data/DIV2K/DIV2K_valid_HR/%4a.png --compile
```

The NeRF runs first rendered their test views in passes of the renderer's
default 512 rays; `train_nerf.py` now renders each view whole, in one pass of
its 40,000 rays. The 120 runs were not trained again: four jobs, two scenes
each, rebuilt every run's field and occupancy grid from `checkpoint.pt` and
`renderer.pt`, compiled it as the benchmark does and scored the 200 test views
as `train_nerf.py` does, replacing the run's `metrics`, `test_views` and two
saved views and recording the render under `test_provenance`. Every view's
PSNR, SSIM and LPIPS agree with the first render's within 1e-5; only the times
and memory changed.

## Learning rates

Every model trains with PyTorch's default Adam at one learning rate from
{3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}, chosen on two or three
signals per task (kodim01 and kodim17; Thai Statue and Dragon; the DIV2K
training images 0395, 0431 and 0777; lego and drums on their validation
views): a neighbour on the grid replaces a model's rate only if it beats it by
more than 0.1 dB (0.02 IoU points) and loses on no signal by more, and a
winner at the edge is tried one step further. The configs hold the chosen
rates. To measure a neighbour, override the rate and send the runs elsewhere:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 image --models SIREN \
  --data data/Kodak/kodim01.png data/Kodak/kodim17.png \
  --set models.SIREN.train.lr=3.0e-3 --log-dir logs/tune-lr/image/3.0e-3 --compile
```

## FUTON ablations

Five studies on the image task, each a config in `configs/ablation-futon/`
whose runs go to the folder of the same name in `logs/ablation-futon/`. All
keep the benchmark's settings (24 Kodak images, 1000 steps on the whole
image), so only the model changes. The default model is the benchmark's
FUTON (cosine): K = (H/2, W/2) components, CP rank 280 and a one-layer MLP
decoder 64 wide (197,379 parameters), at the cosine basis's rate, 1e-1.
Every variant has the same decoder unless the decoder is what varies, and the
same rate but for three that fail at it and take 3e-2: the Tucker combiner,
which diverges at 1e-1 on 8 of the 24 images, and rank 1120 and kappa = 1/8,
where the benchmark's rule, applied to every model of the two sweeps on
kodim01 and kodim17, picks 3e-2 (1e-1 loses 4.6 and 1.4 dB there; the checks
are in `logs/ablation-futon/<study>-lr/<lr>/`). The basis study runs all six
bases, the others the cosine basis alone.

| Study | Models | What varies |
| --- | --- | --- |
| `basis` | `FUTON-<basis>`, 6 | cosine, lanczos, sinc, triangle, chebyshev and legendre, at the default size |
| `bandwidth` | `FUTON-cosine` and `-kappa<k>`, 5 | the relative bandwidth kappa = K / N over {1/8, 1/4, 1/2, 1, 2} at the budget: CP ranks 885, 515, 280, 145 and 75 |
| `rank` | `FUTON-cosine` and `-R<r>`, 5 | the CP rank over {70, 140, 280, 560, 1120} at kappa = 1/2, 49,539 to 788,739 parameters |
| `tensor_net` | `FUTON-cosine`, `-TR` and `-Tucker`, 3 | tensor-ring and Tucker combiners against the default CP one |
| `decoder` | `FUTON-cosine`, `-w128`, `-w256` and `-linear`, 4 | the MLP's width (64, 128 and 256) and the paper's linear decoder |

The variants take the largest rank that keeps FUTON smaller than every
baseline (F-INR, 198,806 parameters): tensor-ring rank 16 (180,483), Tucker
ranks `R_c = 0.163 K_c` (196,135), and CP ranks, multiples of 5, 255, 220 and
305 behind the 128- and 256-wide MLPs and the linear decoder. The default
models of `bandwidth` and `rank` are the benchmark's runs of FUTON (cosine),
`-kappa1_4` and `-kappa1` the components-width study's `FUTON-cosine-N4-w64`
and `-N1-w64`, and the kodim01 and kodim17 runs of rank 1120 and kappa = 1/8
their rate checks' (`logs/ablation-futon/<study>-lr/3.0e-2/`), all copied
rather than repeated; each copy's `results.json` names its `source`, and a
job skips the runs that exist. One image per array task:

```bash
for study in basis tensor_net decoder bandwidth rank; do
  scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 --array=1-24 image \
    --config configs/ablation-futon/$study.yaml --data data/Kodak/kodim%2a.png --compile
done
```

## Components against width, on every task

One study per task, `configs/ablation-futon/components_width_<task>.yaml`,
whose runs go to `logs/ablation-futon/components_width_<task>/`: the
components per axis, K = N/L for L in {4, 2, 1} (and 8 in super-resolution),
against the width of the one-layer MLP decoder, 16 to 256, at the rank that
fills the task's budget, for the sinc and the cosine basis. Each task's
benchmark takes one of them as its FUTON: `-N2-w64` on images, volumes and
radiance fields (its benchmark runs there are this study's, copied) and
`-N4-w128` in super-resolution. Super-resolution runs on the three DIV2K
training images its settings were tuned on, so that nothing is chosen on the
test images.

```bash
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 --array=1-24 image \
  --config configs/ablation-futon/components_width_image.yaml --data data/Kodak/kodim%2a.png --compile
for shape in armadillo dragon happy_buddha lucy thai_statue; do
  scripts/hpc/train.sh --clusters=accelgor --time=1:30:00 occupancy \
    --config configs/ablation-futon/components_width_occupancy.yaml --data data/occupancy/$shape.ply --compile
done
for scene in chair drums ficus hotdog lego materials mic ship; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf \
    --config configs/ablation-futon/components_width_nerf.yaml --data data/nerf/blender/$scene --compile
done
for image in 0395 0431 0777; do
  scripts/hpc/train.sh --clusters=accelgor --time=1:30:00 super_resolution \
    --config configs/ablation-futon/components_width_super_resolution.yaml \
    --data data/DIV2K/DIV2K_train_HR/$image.png --compile
done
```

## Error bound

`scripts/futon_bound.py` runs the error-bound study of the docs' concepts
page on the 24 Kodak images and the 5 occupancy volumes: in the cosine basis,
at every spectral resolution and at CP ranks 32, 64, 128, 256, 512 and 1024,
it bounds the best error of FUTON with a linear decoder, builds the FUTON
that attains the upper bound, and trains the same FUTON with Adam at the
benchmark's rate for FUTON (cosine), 1e-1 on images and 3e-2 on volumes;
then, on every signal at K = N/2 and the benchmark's rank, it trains for the
benchmark's steps and for five times them and runs least squares on the
whole grid. It records into `logs/bound/` and is resumable: a K with a rank
still to bound is bounded again at all its ranks, in the one call whose time
the cost reports, and a setting whose Adam run is recorded keeps it. The
Adam runs at ranks 32 to 256 and K up to N are the study's of 2026-10-26
(same script, rates, steps and A100s), and the checks' runs at the
benchmark's 1000 steps the study's of 2026-10-27, both copied rather than
repeated; each copy names its `adam_source`. One job per dataset, each on
one A100, the bound and the Adam runs it trains in the same job: 1 h 21 min
for the images, 32 min for the volumes.

```bash
for task in image occupancy; do
  sbatch --clusters=accelgor --gpus-per-node=1 --cpus-per-task=8 \
    --time=4:00:00 --chdir=$VSC_DATA/projects/inro \
    --job-name=inro-bound-$task --output=logs/slurm/%x-%j.out --wrap \
    "source $VSC_DATA/venvs/inro-env/bin/activate && python scripts/futon_bound.py --tasks $task"
done
```

Pull `logs/bound/` with `pull_logs.sh` and run `python scripts/report_paper.py
--tasks bound` locally for the figures and tables in `results/bound/`, which
take seconds and no GPU.

## Overrides

To try another value for one model, override it and send the runs elsewhere:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 nerf \
  --models SIREN --data data/nerf/blender/lego \
  --set models.SIREN.train.lr=2.0e-4 --log-dir logs/siren-lr
```

After a model's entry in a config changes, `--overwrite` reruns it in place:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 image \
  --models Gauss --overwrite
```

## Reports

Training writes each run's `results.json`, logs and `checkpoint.pt`, but no
figures. Tables, plots and examples are made afterwards from the pulled logs:

```bash
scripts/hpc/pull_logs.sh                # on the local machine
python scripts/report_paper.py          # -> results/, the paper's figures
python scripts/report_paper.py --no-panels   # without the renders, which need a GPU
```

Costs come from the benchmark runs themselves: the training time counts the
training steps alone, after a warm-up pass that compiles the model, every
evaluation records how long it took and its peak memory, and every training
step its own peak. Each cost in the tables is the median over a task's
signals, which sets aside the runs slowed by other jobs on a shared node.

`report_paper.py` writes everything the paper needs into `results/`, and
nothing else. Per task, in `results/<task>/` so that each is a LaTeX
subfigure:

- quality against training time, for FUTON and the strongest model of each
  other family, the band one standard error, as in the tables; time is
  logarithmic, where the curves span more than a decade of it, and the metric
  linear;
- every model's final quality against its training time, its inference
  throughput, and its peak training and inference memory, all axes linear;
- one table of every model's size, the cost of both phases and its mean final
  metrics, the best in bold and the second underlined. The averages carry one
  standard error over the signals, taken once each signal's own level is
  removed, but for IoU, which saturates on every shape. The occupancy table also gives each shape's IoU, and the NeRF one
  every scene's metrics under a super column, which takes a page turned
  sideways;
- for each of the task's example signals, that signal with two regions boxed,
  each in its own colour, and those regions magnified for the ground truth and
  every featured model, framed in the colour of their box. The example
  signals' regions and radiance-field views are set by hand in `CENTERS` and
  `VIEWS`; any other signal gets them by rule, as
  docs/experiments/reproduce.md describes.

The magnified panels are rebuilt from the checkpoints, which needs the signals
in `data/` and a GPU, so they are kept in `<task>/panels/<signal>/` and reused.
`--overwrite` draws them again, which a changed NeRF view calls for, and
`--no-panels` leaves them out. Meshes render offscreen, so no display is
needed; a shape file that differs from the cluster's would rebuild the wrong
mesh, which the report catches by recomputing each run's IoU.

The figures carry one TensoRF, the default variant for the dimension: CP in 2D,
where it is the only one, and VM in 3D. Four hues are as many as stay apart
under every kind of colour blindness, so a hue names a family and the line
style names the member, the stronger of a pair solid: SIREN and FINER share
the periodic-activation hue, as do the two FUTON bases. A shared `legend.pdf`
sits beside the task folders.

The FUTON ablations go to `results/ablation/`: the `basis`, `bandwidth`,
`rank`, `combiner` and `decoder` studies, each as a table with the
benchmark's columns and a figure, PSNR against training time or against the
bandwidth or rank it sweeps.

`--orbit 60` also saves an orbit GIF of each NeRF model beside its panel, for
a talk. It costs about a minute a scene and the paper takes none of them, so
it is off by default.

The whole report runs on the cluster too, where the data sits beside the runs:

```bash
sbatch --clusters=accelgor --time=1:00:00 --gpus-per-node=1 --cpus-per-task=8 \
  --chdir=$VSC_DATA/projects/inro --output=logs/slurm/%x-%j.out \
  --job-name=inro-report --wrap \
  "source $VSC_DATA/venvs/inro-env/bin/activate && python scripts/report_paper.py"
```

To read the runs directly instead:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```
