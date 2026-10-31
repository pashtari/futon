# API reference

The public API lives in three namespaces. Everything for fitting coordinate/value pairs is exported at the top level; image compression and radiance fields have their own.

| Namespace | Contents |
| --- | --- |
| `inro` | Models, datasets, `train`, `evaluate`, `chunked_inference`, losses, metrics, quantization, plotting and utilities |
| `inro.compression` | `encode_futon`, `decode_futon`, `encode_pil`, `decode_pil`, `evaluate_compression`, arithmetic coding |
| `inro.nerf` | `BlenderDataset`, `RadianceField`, renderers, `train`, `evaluate`, `render_image`, `render_views`, camera helpers |

The pages of this section are organized by module. Each documents the public classes and functions with their signatures, tensor shapes and defaults, as written in the source.

- [inro.models](models.md): every architecture except FUTON, with its layers and encodings.
- [FUTON, bases and combiners](futon.md): `FUTON`, the six bases and the features they return, the combiners, the block coordinate descent solvers, and the error bounds.
- [Datasets](datasets.md): coordinate datasets for images, volumes and MRI, the Deep Image Prior dataset, and super-resolution's blocks with the bicubic operator they are seen through.
- [Training and evaluation](training.md): `train`, `evaluate`, `chunked_inference`.
- [Metrics and losses](metrics.md): PSNR, SSIM, MS-SSIM, LPIPS, NMSE, IoU, bits per pixel, and super-resolution's luma metrics; rate-distortion, SDF, super-resolution and entropy losses.
- [Quantization](quantization.md): uniform quantization of a model's parameters.
- [Compression](compression.md): the codecs, their evaluation, and arithmetic coding.
- [Radiance fields](nerf.md): the `inro.nerf` package.
- [Utilities](utils.md): module specs, parameter counts, logging, serialization, and the seaborn line plot.

!!! tip "Docstring conventions"
    Shapes are written as `(N, C)` for a flat batch and `(*, C)` or `(..., C)` when any leading dimensions are accepted. `C` is the number of coordinate axes, `K` or `K_c` the number of basis functions per axis, `R` a rank, and `D` the number of output channels.
