# Your first neural field

This page fits FUTON to one image, reads what the training loop returns, and swaps in another model. It needs an image; the Kodak images from the [installation page](installation.md#benchmark-data) are used below, but any RGB image works.

## Fit an image

```python
import torch
import neurofield as nf

image = "data/Kodak/kodim19.png"
dataset = nf.ImageCoordinateDataset(image)  # every pixel, at every step
H, W, _ = dataset.target.shape

model = nf.FUTON(
    in_features=2,
    out_features=3,
    basis=("sinc", {"num_components": [H // 2, W // 2]}),
    combiner=("cp", {"rank": 280}),
    decoder=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
    output_activation=torch.tanh,
)

result = nf.train(
    model,
    dataset,
    dataset,
    num_steps=1000,
    lr=3e-2,
    metrics={"psnr": nf.psnr, "ssim": nf.ssim},
    log_interval=250,
    eval_interval=250,
)
print(result["history"][-1]["eval"])
```

On a GPU this takes about half a minute (34 s on an RTX 4060 Ti) and ends near 37.3 dB PSNR. Step by step:

1. **The dataset turns the image into coordinate/value pairs.** `dataset.input` holds the pixel-centre coordinates, shape `(H, W, 2)` in $[-1, 1]$; `dataset.target` the RGB values, shape `(H, W, 3)`, also in $[-1, 1]$. Each access returns the whole image; `subsample=0.1` would instead draw a fresh random 10 % of the pixels per access, flattened to `(N, 2)` and `(N, 3)`.
2. **The model is a function of coordinates.** `FUTON` is assembled from three specs, each a registry key, a `(key, kwargs)` pair, a class, or a module instance. The sinc basis here has half as many functions per axis as the image has pixels, 384 × 256 on this 768 × 512 image, evaluated once per image row and column; the CP combiner has rank 280; the decoder is an MLP with one hidden layer of width 64. `output_activation=torch.tanh` matches the target range. This is the benchmark's FUTON (sinc), 197,379 parameters; the benchmark also tabulates the basis on the pixel grid (`grid_size`) and compiles the model, which trains it in about 5 s on an A100.
3. **`nf.train` runs Adam with cosine annealing** from `lr` down to `lr / 100`, one optimizer step per batch; this dataset has one item, the whole image, so every step sees every pixel. A progress bar shows the training metrics. Every `eval_interval` steps, and at the last one, it evaluates on `dataset`, mapping predictions back to `uint8` with `dataset.postprocess` before computing the metrics.

The result is a plain dictionary:

```python
result["config"]       # seed, device, model, num_params, optimizer settings, ...
result["model_dict"]   # the final state dict
result["history"][-1]  # {"step": 1000, "elapsed": ..., "mse": ..., "psnr": ...,
                       #  "eval": {"psnr": ..., "ssim": ...}}
```

## Reconstruct and save

The trained model can be evaluated anywhere. To rebuild the image, evaluate it on the dataset's coordinate grid in chunks and map the output back to pixels:

```python
output = nf.chunked_inference(model, dataset.input, chunk_size=65536)  # (H, W, 3) in [-1, 1]
reconstruction = dataset.postprocess(output.cpu())                     # uint8 (3, H, W)

print(nf.psnr(reconstruction, dataset.original))
dataset.save(reconstruction, "kodim19_futon.png")
```

Coordinates need not lie on the pixel grid. `nf.create_coordinates((2 * H, 2 * W))` gives a twice-finer grid on the same domain, and `model(coords)` evaluates the continuous representation there; how well it interpolates depends on the basis and on `num_components`, see [FUTON](../concepts/futon.md).

## Log to disk

Pass `log_dir` to keep a record of the run:

```python
result = nf.train(model, dataset, dataset, num_steps=1000, lr=3e-2, log_dir="logs/quickstart")
```

This writes `log.txt` (a line per logged step, which the console shows as a progress bar), `log.json` (the same records as JSON, read back with `nf.read_json_records`), the final `checkpoint.pt`, and the evaluation reconstruction `kodim19_reconstructed.png`.

## Swap the model

Every model in the [zoo](../concepts/models.md) takes `in_features` and `out_features` first and maps `(..., in_features)` coordinates in $[-1, 1]$ to `(..., out_features)` values, so the training call does not change:

```python
model = nf.SIREN(2, 3, hidden_features=256, hidden_layers=4, output_activation=torch.tanh)
result = nf.train(model, dataset, dataset, num_steps=1000, lr=1e-3, metrics={"psnr": nf.psnr})
```

The learning rates that the benchmark uses for each model are in `configs/image.yaml`; the [image tutorial](../tutorials/image.md) walks through them. The same loop fits occupancy volumes and, through `nf.nerf`, radiance fields; the [tutorials](../tutorials/index.md) cover each.
