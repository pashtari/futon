# Notebooks

Five notebooks in `notebooks/` run one signal end to end. Four compare FUTON, with its sinc and cosine bases, against five representative baselines of a task at equal size: RFF, SIREN, FINER, Instant-NGP and TensoRF, or in super-resolution the last four and bicubic interpolation. The fifth compares the ways to fit FUTON. They are the interactive version of the training scripts and the fastest way to see how the models compare on your own data.

| Notebook | Signal | Models | Runtime on an A100 |
| --- | --- | --- | --- |
| `image_representation.ipynb` | `data/Kodak/kodim17.png` | 7, about 197k parameters each, 1000 steps on the whole image | about 7 minutes |
| `occupancy_volume.ipynb` | `data/occupancy/thai_statue.ply` at $256^3$ | 7, about 131k parameters each, 1000 steps of $2^{18}$ voxels | about 4 minutes |
| `nerf.ipynb` | `data/nerf/blender/lego` under FINER's protocol, in a quick pass | 7, about 75k parameters each, 5000 steps | about 11 minutes (set `NUM_STEPS = 25_000` for the benchmark's full schedule, about 35 minutes) |
| `image_super_resolution.ipynb` | `data/DIV2K/DIV2K_valid_HR/0887.png` from its 4× bicubic downsampling | 6 fields, about 500k parameters each, 1000 steps of 135 blocks, and bicubic interpolation | about 6 minutes |
| `futon_solvers.ipynb` | `data/Kodak/kodim19.png` | 4 FUTONs, about 197k parameters each: an MLP or a linear decoder, fitted by Adam (1000 steps on the whole image) or by alternating least squares or multiplicative updates (500 sweeps) | about 4 minutes |

Each notebook is self-contained and has the same sections: *Setup* imports the packages, sets the seaborn theme and each model's colour and marker, and defines the plotting helpers; *Data* shows the signal and the region the comparison magnifies; *Models* gives every model's constructor arguments and learning rate, with `# deviation` comments where a setting departs from the authors' code; *Training* runs `nf.train` or `nf.nerf.train`; *Convergence* plots the metric against steps and against training time; *Summary* tabulates every model's size, costs and scores, the best of each column in bold and the second underlined, and plots the scores against the costs; and *Qualitative Comparison* sets the reconstructions side by side, magnified, above their errors. FUTON is drawn in black, its cosine basis dashed, and the baselines in seaborn's colour-blind palette. The model settings are those of `configs/<task>.yaml`, which holds every model of the benchmark, the notebooks' and the rest. The super-resolution notebook also walks through its forward model, which `nf.resize_matrix`, `nf.SuperResolutionDataset` and `nf.super_resolution_loss` provide: the models never see the high-resolution image, and each step renders the pixels under random blocks of low-resolution pixels, downsamples them with the benchmark's bicubic kernel, and matches the low-resolution pixels.

```bash
pip install -e ".[3d,nerf]" jupyterlab
python scripts/download_kodak.py --images 17 19
python scripts/download_meshes.py --meshes thai_statue
python scripts/download_blender.py --scenes lego
python scripts/download_div2k.py
jupyter lab notebooks/
```

The notebooks read data from `../data/`, relative to their folder, and are committed with the outputs of the runs timed above, so they read without being run; rerun them top to bottom. The NeRF notebook also renders an orbit of its best field as a GIF with `imageio`, which the `nerf` extra provides.
