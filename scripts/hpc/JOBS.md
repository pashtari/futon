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

Four studies on the image task, each a config in `configs/ablation-futon/`
whose runs go to the folder of the same name in `logs/ablation-futon/`. All
keep the benchmark's settings (24 Kodak images, 1000 steps on the whole
image), so only the model changes. The default model is the benchmark's
FUTON: K = (H/2, W/2) components, CP rank 280 and a one-layer MLP decoder 64
wide (197,379 parameters); every variant has the same decoder unless the
decoder is what varies. The basis study runs all six bases; the others run
the benchmark's two, sinc and cosine, cosine at its own rate, 1e-1.

| Study | Models | What varies |
| --- | --- | --- |
| `basis` | `FUTON-<basis>`, 6 | cosine, lanczos, sinc, triangle, chebyshev and legendre, at the default size |
| `components_rank` | `FUTON-<basis>-a<alpha>-b<beta>`, 50 | K = (alpha H, alpha W) and R = beta min(K), both over {1/8, 1/4, 1/2, 1, 2} |
| `tensor_net` | `FUTON-<basis>`, `-TR` and `-Tucker`, 6 | tensor-ring and Tucker combiners against the default CP one |
| `decoder` | `FUTON-<basis>`, `-w128`, `-w256` and `-linear`, 8 | the MLP's width (64, 128 and 256) and the paper's linear decoder |

The variants take the largest rank that keeps FUTON smaller than every
baseline (F-INR, 198,806 parameters): tensor-ring rank 16 (180,483), Tucker
ranks `R_c = 0.163 K_c` (196,135), and CP ranks, multiples of 5, 255, 220 and
305 behind the 128- and 256-wide MLPs and the linear decoder. The CP models of `tensor_net` and `decoder` repeat those
of `basis`, so that each study stands alone. One image per array task:

```bash
for study in basis tensor_net decoder; do
  scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 --array=1-24 image \
    --config configs/ablation-futon/$study.yaml --data data/Kodak/kodim%2a.png --compile
done
# The components grid holds models up to 5.5M parameters.
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 --array=1-24 image \
  --config configs/ablation-futon/components_rank.yaml --data data/Kodak/kodim%2a.png --compile
```

## Components against width, on every task

One study per task, `configs/ablation-futon/components_width_<task>.yaml`,
whose runs go to `logs/ablation-futon/components_width_<task>/`: the
components per axis, K = N/L for L in {4, 2, 1} (and 8 in super-resolution),
against the width of the one-layer MLP decoder, 16 to 256, at the rank that
fills the task's budget, for the sinc and the cosine basis. Each study's
`-N2-w64` model (`-N4-w64` in super-resolution) is the benchmark's FUTON.
Super-resolution runs on the three DIV2K training images its settings were
tuned on, so that nothing is chosen on the test images.

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
at every spectral resolution and rank, it bounds the best error of FUTON
with a linear decoder, builds the FUTON that attains the upper bound, and
trains the same FUTON with Adam; then, on every signal at K = N/2 and one
rank, it trains for five times the steps and runs least squares on the
whole grid. It records into `logs/bound/` and is resumable. One A100 does it
all in about two hours, bound and Adam in the same job, so that the cost of
the two is compared under the same conditions.

```bash
sbatch --clusters=accelgor --gpus-per-node=1 --cpus-per-task=8 \
  --time=3:00:00 --chdir=$VSC_DATA/projects/neurofield \
  --job-name=nf-bound --output=logs/slurm/%x-%j.out --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/futon_bound.py"
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
  every featured model, framed in the colour of their box. The regions are
  found, not set: the two squares of fine detail where FUTON gains most
  squared error over the strongest baseline, which puts the magnifications
  where the models actually differ. `ZONES` restricts the search to a
  rectangle per box where a signal's telling regions are not the ones the
  search would reach, and a NeRF scene is rendered from the view where FUTON
  gains most of those that show it whole, which every run records.

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

The FUTON ablations go to `results/ablation/`: `basis`, `combiner` and
`decoder` each as a table and a convergence plot, the tensor ring against CP
and the linear decoder against the MLP for both bases, and the components
study as PSNR against the rank fraction at each component fraction and the
reverse, one figure per basis, for the fractions 1/4 to 2.

`--orbit 60` also saves an orbit GIF of each NeRF model beside its panel, for
a talk. It costs about a minute a scene and the paper takes none of them, so
it is off by default.

The whole report runs on the cluster too, where the data sits beside the runs:

```bash
sbatch --clusters=accelgor --time=1:00:00 --gpus-per-node=1 --cpus-per-task=8 \
  --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out \
  --job-name=nf-report --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/report_paper.py"
```

To read the runs directly instead:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```
