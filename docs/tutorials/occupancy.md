# Occupancy volumes

A shape is represented as an occupancy field $[-1, 1]^3 \to \{-1, 1\}$, fitted to a voxelized mesh and scored by intersection over union. This follows `configs/occupancy.yaml` and `scripts/train_occupancy.py`.

## Data

`OccupancyCoordinateDataset` loads a mesh (any format trimesh reads, with the `3d` extra) or an `.xyz` point cloud with normals, normalizes it into $[-0.45, 0.45]^3$ as FINER and BACON do, voxelizes it at `resolution` samples per unit length, crops the volume to its occupied bounds, and caches the result beside the source as `.cache/<name>_<resolution>.pt`.

```python
import torch
import neurofield as nf

path = "data/occupancy/lucy.ply"
train_set = nf.OccupancyCoordinateDataset(path, resolution=256, max_samples=2**18)
eval_set = nf.OccupancyCoordinateDataset(path, resolution=256)

eval_set.input.shape, eval_set.target.shape   # (D, H, W, 3) coordinates, (D, H, W, 1) targets in {-1, 1}
eval_set.original.shape                        # (1, D, H, W) float occupancy in {0, 1}
```

At 256 samples per unit length a volume, cropped to the shape's bounds, holds 2 to 8 million voxels; `max_samples=2**18` trains on 262,144 of them at random per step, as many for every shape. The five meshes of the benchmark are the Stanford armadillo, dragon, happy buddha, lucy and thai statue, fetched by `scripts/download_meshes.py`.

## Model

The benchmark FUTON uses half the voxels along each axis of the cropped grid, a one-layer MLP decoder 64 wide, and the largest CP rank, a multiple of 5, that keeps it smaller than every baseline, the smallest of which has 131,457 parameters. Rank $R$ costs $R$ times the components plus the width, and the decoder adds $2 \cdot 64 + 1$, so Lucy's $135 \times 78 \times 231$ grid gets 460, for 131,229 parameters:

```python
components = [size // 2 for size in eval_set.grid_size]
rank = (131_456 - 129) // (sum(components) + 64) // 5 * 5
model = nf.FUTON(
    3, 1,
    basis=("sinc", {"num_components": components, "grid_size": eval_set.grid_size}),
    combiner=("cp", {"rank": rank}),
    decoder=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
    output_activation=torch.tanh,
)
```

The cosine variant replaces the basis with `("cosine", {"num_components": components, "grid_size": eval_set.grid_size})`. On the voxel grid, which `grid_size` declares and which the crop sizes differently for every shape, either basis is evaluated once, when it is built, and each point reads its rows.

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
    log_dir="logs/tutorial/lucy/FUTON-sinc",
)
print(result["history"][-1]["eval"]["iou"])
```

`chunk_size=262144` bounds the memory of evaluating the whole volume. `nf.iou` thresholds the model's output at zero and compares it with the `{0, 1}` original. Every model's IoU after the 1000 steps, and FUTON's convergence, are on the [benchmarks page](../experiments/benchmarks.md#occupancy).

## Exporting a mesh

With the `3d` extra, `eval_set.save(volume, path)` smooths the occupancy, runs marching cubes, writes a `.dae` mesh and renders it offscreen to a `.png` with Open3D, drawing the grid's axis `up` upwards: Lucy stands along the last, the other four shapes along the second, the default:

```python
volume = nf.chunked_inference(model, eval_set.input, chunk_size=262144)
occupancy = eval_set.postprocess(volume.cpu())  # (1, D, H, W) in {0, 1}
eval_set.save(occupancy, "lucy_futon", up=2)    # lucy_futon.dae and .png
```

The render runs offscreen with Open3D and needs EGL and a GPU driver, which not every machine has; that is why the call above and the benchmark script pass `save_reconstruction=False` and rebuild meshes from `checkpoint.pt` afterwards (`nf.evaluate(model, eval_set, ckpt_path=..., log_dir=...)` on a workstation).

## The benchmark script

```bash
python scripts/train_occupancy.py                                       # all shapes, all models
python scripts/train_occupancy.py --data data/occupancy/lucy.ply --models FUTON-cosine
```

FUTON's ablations run on images; the [ablations page](../experiments/ablations.md) reports them.
