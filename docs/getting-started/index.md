# Getting started

Two pages take you from an empty environment to a trained neural field.

- [Installation](installation.md): requirements, the optional extras, and the benchmark data.
- [Your first neural field](quickstart.md): fit FUTON to an image, read the result, and swap in another model.

Everything is importable from the top-level package:

```python
import inro

inro.FUTON, inro.SIREN, inro.InstantNGP     # models
inro.ImageCoordinateDataset                 # datasets
inro.train, inro.evaluate, inro.chunked_inference
inro.psnr, inro.ssim, inro.lpips, inro.iou  # metrics
inro.compression, inro.nerf                 # image compression and radiance fields
```
