# Occupancy volumes

A shape is represented as an occupancy field $[-1, 1]^3 \to \{-1, 1\}$, fitted to a voxelized mesh and scored by intersection over union. This follows `configs/occupancy.yaml` and `scripts/train_occupancy.py`.

## Data

`OccupancyCoordinateDataset` loads a mesh (any format trimesh reads, with the `3d` extra) or an `.xyz` point cloud with normals, normalizes it into $[-0.45, 0.45]^3$ as FINER and BACON do, voxelizes it at `resolution` samples per unit length, crops the volume to its occupied bounds, and caches the result beside the source as `.cache/<name>_<resolution>.pt`.

```python
import torch
import neurofield as nf

path = "data/occupancy/thai_statue.ply"
train_set = nf.OccupancyCoordinateDataset(path, resolution=256, max_samples=2**18)
eval_set = nf.OccupancyCoordinateDataset(path, resolution=256)

eval_set.input.shape, eval_set.target.shape   # (D, H, W, 3) coordinates, (D, H, W, 1) targets in {-1, 1}
eval_set.original.shape                        # (1, D, H, W) float occupancy in {0, 1}
```

At 256 samples per unit length a volume, cropped to the shape's bounds, holds 2 to 8 million voxels; `max_samples=2**18` trains on 262,144 of them at random per step, as many for every shape. The five meshes of the benchmark are the Stanford armadillo, dragon, happy buddha, lucy and thai statue, fetched by `scripts/download_meshes.py`.

## Model

The benchmark FUTON uses 128 components per axis, CP rank 328 and a one-layer MLP decoder 16 wide, 131,233 parameters:

```python
model = nf.FUTON(
    3, 1,
    basis=("sinc", {"num_components": 128, "grid_size": eval_set.grid_size}),
    combiner=("cp", {"rank": 300}),
    decoder=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
    output_activation=torch.tanh,
)
```

The cosine variant replaces the basis with `("cosine", {"num_components": 128, "grid_size": eval_set.grid_size})`. On the voxel grid, which `grid_size` declares and which the crop sizes differently for every shape, either basis is evaluated once, when it is built, and each point reads its rows.

## Training and evaluation

```python
result = nf.train(
    model,
    train_set,
    eval_set,
    num_steps=1000,
    lr=3e-2,
    metrics={"iou": nf.iou},
    chunk_size=262144,
    log_interval=50,
    eval_interval=50,
    save_reconstruction=False,
    log_dir="logs/tutorial/thai_statue/FUTON-sinc",
)
print(result["history"][-1]["eval"]["iou"])
```

`chunk_size=262144` bounds the memory of evaluating the whole volume. `nf.iou` thresholds the model's output at zero and compares it with the `{0, 1}` original. The benchmark models reach 99.5 to 99.9 % IoU in 2000 steps; FUTON's convergence is shown on the [benchmarks page](../experiments/benchmarks.md#occupancy).

## Exporting a mesh

With the `3d` extra, `eval_set.save(volume, path)` smooths the occupancy, runs marching cubes, writes a `.dae` mesh and renders it offscreen to a `.png` with Open3D:

```python
volume = nf.chunked_inference(model, eval_set.input, chunk_size=262144)
occupancy = eval_set.postprocess(volume.cpu())       # (1, D, H, W) in {0, 1}
eval_set.save(occupancy, "thai_statue_futon")         # thai_statue_futon.dae and .png
```

The render needs a display, which headless machines lack; that is why the call above and the benchmark script pass `save_reconstruction=False` and rebuild meshes from `checkpoint.pt` afterwards (`nf.evaluate(model, eval_set, ckpt_path=..., log_dir=...)` on a workstation).

## The benchmark script

```bash
python scripts/train_occupancy.py                                       # all shapes, all models
python scripts/train_occupancy.py --data data/occupancy/lucy.ply --models FUTON-cosine
python scripts/train_occupancy.py --config configs/ablation-futon/basis.yaml   # -> logs/ablation-futon/basis/
```

The four FUTON ablation configs in `configs/ablation-futon/` run on this task; the [ablations page](../experiments/ablations.md) reports them.
