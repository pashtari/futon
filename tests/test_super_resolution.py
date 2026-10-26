"""The super-resolution benchmark's forward model, minibatches and FUTON's rank."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import neurofield as nf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def task():
    """The benchmark script as a module; it imports train_common from its folder."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "train_super_resolution", SCRIPTS / "train_super_resolution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def image_pair(task, tmp_path):
    """A random 32 x 48 image and its 4x downsampling, laid out as DIV2K is."""
    torch.manual_seed(0)
    image = torch.randint(256, (3, 32, 48), dtype=torch.uint8)
    device = torch.device("cpu")
    kernels = [task.resize_matrix(32, 8, device), task.resize_matrix(48, 12, device)]
    low = task.downsample(image.movedim(0, -1).float(), kernels)
    low = low.round().clamp(0, 255).to(torch.uint8).movedim(-1, 0)
    (tmp_path / "DIV2K_valid_HR").mkdir()
    (tmp_path / "DIV2K_valid_LR_bicubic" / "X4").mkdir(parents=True)
    path = tmp_path / "DIV2K_valid_HR" / "0801.png"
    to_pil_image(image).save(path)
    to_pil_image(low).save(task.low_resolution(path, 4))
    return path, image, kernels


def test_resize_matrix_is_the_antialiased_bicubic_operator(task):
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 48)
    expected = F.interpolate(image, size=(8, 12), mode="bicubic", antialias=True)
    device = torch.device("cpu")
    kernels = [task.resize_matrix(32, 8, device), task.resize_matrix(48, 12, device)]
    downsampled = task.downsample(image[0].movedim(0, -1), kernels).movedim(-1, 0)
    torch.testing.assert_close(downsampled, expected[0], atol=1e-5, rtol=1e-5)
    assert int((kernels[0] != 0).sum(1).max()) == 16  # taps per LR pixel at 4x


def test_blocks_downsample_to_the_low_resolution_pixels(task, image_pair):
    path, image, kernels = image_pair
    dataset = task.SuperResolutionDataset(task.low_resolution(path, 4), kernels, 12, 2)
    torch.manual_seed(0)
    batch = dataset[0]
    # A field that returns the HR pixels under the coordinates it is given.
    rows = ((batch["input"][..., 0] + 1) / 2 * 31).round().long()
    columns = ((batch["input"][..., 1] + 1) / 2 * 47).round().long()
    rendered = dataset.preprocess(image)[rows, columns]  # (B, L, L, C) in [-1, 1]
    fitted = task.downsample(rendered, (batch["weight_y"], batch["weight_x"]))
    assert batch["input"].shape[1:] == (dataset.footprint, dataset.footprint, 2)
    assert batch["target"].shape[1:] == (2, 2, 3)
    torch.testing.assert_close(fitted, batch["target"], atol=2 / 255, rtol=0)


@pytest.mark.parametrize("outputs", [1, 3])
def test_futon_rank_is_the_largest_multiple_of_5_within_the_budget(task, outputs):
    from train_common import futon_rank  # the fixture put the scripts on the path

    for components, budget in (((339, 510), 500_006), ((69, 115, 59), 131_456)):
        rank = futon_rank(components, budget, 64, outputs)

        def count(rank: int) -> int:
            model = nf.FUTON(
                in_features=len(components),
                out_features=outputs,
                basis=("lanczos", {"num_components": components, "radius": 3}),
                combiner=("cp", {"rank": rank}),
                decoder=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
            )
            return nf.count_parameters(model)

        assert rank % 5 == 0
        assert count(rank) <= budget < count(rank + 5)


def test_prepare_fills_futon_rank_from_the_budget(task, image_pair):
    path, _, _ = image_pair  # 32 x 48 pixels
    entry = {
        "class": "FUTON",
        "budget": 20_000,
        "kwargs": {
            "basis": ["sinc", {"num_components": ["H // 4", "W // 4"]}],
            "combiner": ["cp", {"rank": "budget"}],
            "decoder": ["mlp", {"hidden_layers": 1, "hidden_features": 64}],
        },
    }
    model = task.prepare(entry, path)
    assert model["kwargs"]["basis"][1]["num_components"] == [8, 12]
    # (20,000 - 64 - 64 * 3 - 3) // (8 + 12 + 64) = 235, a multiple of 5
    assert model["kwargs"]["combiner"][1]["rank"] == 235
    # a model without a budget, a baseline, passes through as it is
    siren = {"class": "SIREN", "kwargs": {"hidden_features": "H // 2"}}
    assert task.prepare(siren, path) == {
        "class": "SIREN",
        "kwargs": {"hidden_features": 16},
    }
