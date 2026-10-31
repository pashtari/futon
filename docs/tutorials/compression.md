# Image compression

A neural field is a compressed representation of its signal: the parameters are the code. `inro.compression` makes this concrete for FUTON, with a codec that fits a quantized model and serializes it, PIL codecs as baselines, and one function that measures any codec's rate-distortion point.

## A codec in two calls

```python
from torchvision.io import ImageReadMode, read_image
import inro

image = read_image("data/Kodak/kodim19.png", mode=ImageReadMode.RGB)   # uint8 (3, H, W)

encoded = inro.compression.encode_futon(
    image,
    basis=("cosine", {"num_components": 128}),
    combiner=("cp", {"rank": 64}),
    decoder="linear",
    num_steps=500,
    lr=0.1,
    alpha=0.005,
    quant_max=127,
)
reconstruction = inro.compression.decode_futon(encoded)   # uint8 (3, H, W)
```

`encode_futon` builds a `FUTON` from the keyword arguments (`in_features` and `out_features` default to 2 and the channel count), trains it with `inro.rate_distortion_loss`, quantizes every parameter tensor to `quant_max` levels, and packs the integer codes, their scales, the model configuration, the image ID and the spatial shape into gzip-compressed bytes. `decode_futon` rebuilds the model from those bytes and evaluates it on the pixel grid. The specs must be registry keys or `(key, dict)` pairs so that they survive the round trip.

The rate-distortion loss is $\text{MSE} + \alpha \cdot H$, where $H$ is the size-weighted mean of a Laplace entropy estimate of each parameter tensor, in bits; `alpha=0` disables the rate term. `quant_interval` rounds the parameters to the quantization grid every so many steps during training, so that the model adapts to it (quantization-aware training); the final quantization always runs.

## Measuring a codec

```python
futon = inro.compression.evaluate_compression(
    image,
    inro.compression.encode_futon,
    inro.compression.decode_futon,
    basis=("cosine", {"num_components": 128}),
    combiner=("cp", {"rank": 64}),
    decoder="linear",
    num_steps=500,
)
jpeg = inro.compression.evaluate_compression(
    image, inro.compression.encode_pil, inro.compression.decode_pil, format="JPEG", quality=50
)
webp = inro.compression.evaluate_compression(
    image, inro.compression.encode_pil, inro.compression.decode_pil, format="WEBP", quality=50
)
```

Each call returns `{"PSNR (dB)", "bit rate (bpp)", "encoding time (ms)", "decoding time (ms)"}`, with the decoded image under `"reconstructed"` if `return_reconstruction=True`. PSNR uses `data_range=255` and is infinite for a lossless codec. The keyword arguments go to the encoder, so a rate-distortion curve is a loop over `rank` (or `num_components`, `quant_max`, `alpha`) for FUTON and over `quality` for the PIL codecs, and `inro.LinePlot` draws the curves:

```python
records = [
    {"codec": "FUTON", "rank": rank, **inro.compression.evaluate_compression(
        image, inro.compression.encode_futon, inro.compression.decode_futon,
        basis=("cosine", {"num_components": 128}), combiner=("cp", {"rank": rank}), decoder="linear",
    )}
    for rank in (16, 32, 64, 128)
]
plot = inro.LinePlot(records)
fig, ax = plot.plot(x="bit rate (bpp)", y="PSNR (dB)", hue="codec")
```

`encode_pil` accepts a path, a file object, or a `(C, H, W)` tensor and forwards its keyword arguments to `PIL.Image.save`; `format` is required because the image is written to memory.

## Quantization in the training loop

The same machinery is available in `inro.train`:

```python
result = inro.train(
    model,
    train_set,
    eval_set,
    quantize=True,
    quant_max=127,
    quant_interval=100,
    metrics={"psnr": inro.psnr},
    rate_metrics={"bpp": inro.bits_per_pixel},
)
```

Every evaluation then scores a quantized copy of the model, the rate metrics see its parameters serialized with `inro.encode_dict` (`torch.save` plus gzip), and `result["model_dict"]` holds the integer codes and scales, which `inro.uniform_dequantize(model, model_dict)` loads back (`checkpoint.pt` keeps the quantized weights as an ordinary state dict). `inro.uniform_quantize(model, quant_max)` and `inro.uniform_dequantize` are also usable on their own.

## Entropy coding

`inro.compression.encode_arithmetic(symbols, pmfs)` and `decode_arithmetic` implement integer arithmetic coding with a separate probability mass function per position, for codecs that model the parameter distribution explicitly. `inro.soft_entropy` gives a differentiable entropy estimate from soft histogram assignments, and `inro.laplace_entropy` the closed-form one that `rate_distortion_loss` uses.
