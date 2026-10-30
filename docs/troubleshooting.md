# Troubleshooting

The problems that come up first, and what to do about them.

## Coordinates and targets

**The fit is poor or the loss does not move.**
Coordinates must lie in $[-1, 1]$ on every axis. The coordinate datasets generate them that way, and FUTON's bases, the hash and feature grids, and `GridINR`'s patches are all defined on that domain. Do not pre-normalize to $[0, 1]$. Targets from the datasets are in $[-1, 1]$ too (occupancy in $\{-1, 1\}$), which is why the benchmark models end in `output_activation=torch.tanh` and why the default loss reports PSNR for that range.

**`ValueError: radius must be >= 1 and num_components >= 2 * radius`**
A local basis (`triangle`, `lanczos`) needs at least `2 * radius` components per axis: 2 for the triangle, 6 for Lanczos-3. With `num_components=[H // 2, W // 2]` this is never an issue for real images; it is for tiny test inputs.

**`grid_size` on a dataset and `subsample` together.**
`ImageCoordinateDataset(image, grid_size=...)` evaluates a model on another grid than the image's (for example a finer one) and cannot subsample; it is for evaluation.

## Memory

**CUDA out of memory during evaluation.**
Pass `chunk_size` to `nf.train`, `nf.evaluate` or `nf.chunked_inference`; the occupancy benchmark evaluates $256^3$ volumes with `chunk_size=262144`. In `nf.nerf.train`, lower `num_rays` or `max_samples_per_ray`; in `nf.nerf.evaluate` and `render_image`, pass `chunk_size` (rays per pass).

**CUDA out of memory during training.**
Draw fewer coordinates per step with `subsample` (a fraction) or `max_samples` (a count); the benchmarks train on the whole image and on 262,144 voxels per step.

## Backends

**`NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.`**
nerfacc builds its CUDA extension on first use and needs `nvcc` for that. Without it, `nf.nerf.create_renderer()` selects the PyTorch renderer, which produces the same renders more slowly; training still works. Install a CUDA toolkit matching your PyTorch build to use nerfacc.

## Data and extras

**`ModuleNotFoundError: trimesh` (or `mcubes`, `open3d`).**
Meshes as occupancy volumes, mesh export and offscreen rendering need the `3d` extra: `pip install -e ".[3d]"`. Point clouds in `.xyz` format load without it.

**No `.png` from `OccupancyCoordinateDataset.save`.**
It renders offscreen with Open3D, which needs EGL and a GPU driver but no display. Where the renderer cannot start, `save` warns and writes the mesh alone.

**LPIPS downloads weights on first use.**
`nf.lpips` loads the VGG (or AlexNet) backbone through torchmetrics, which fetches the weights once. On a machine without internet access, run one evaluation on a connected one first, with the same cache directory.

## Experiments

**A run is skipped with "holds a run of a different setup".**
Runs are keyed by model name inside the log directory. After changing a model's entry in a config, pass `--overwrite` to rerun it in place, or give the variant its own name or `--log-dir`.

**A model diverges.**
Learning rates in the configs come from the grid {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}, each the best of the grid on two or three signals of the task, as each config's header says. Sine networks (SIREN, FINER) are the most sensitive; FUTON trains at 1e-2 to 1e-1 across tasks.

**Training times differ from the tables.**
Times vary with the machine and its load; the tables take every cost from the benchmark runs themselves, each the median over a task's signals, as the [reproduction page](experiments/reproduce.md) explains.

## Models

**`hidden_layers` counts hidden layers.**
A network with `hidden_layers=h` has `h + 1` weight layers (MFN: `h` Gabor filters and `h` linear layers). The configs state the counts as the constructors expect them.

**Loading an `MLP` checkpoint.**
`MLP.layers` contains only weight layers, so a network with `hidden_layers=h` has keys `layers.0` to `layers.h`; a shared activation with parameters (for example `nn.PReLU`) lives under `activation.*`.

**Quantized checkpoints.**
With `nf.train(..., quantize=True)`, `checkpoint.pt` holds the quantized weights as an ordinary state dict, while the returned `model_dict` holds integer codes with per-tensor scales; restore those with `nf.uniform_dequantize(model, model_dict)`.
