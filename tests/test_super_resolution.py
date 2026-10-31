"""Super-resolution: the forward model, its minibatches, loss and metrics, and
the benchmark script's FUTON rank."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import to_pil_image

import inro

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
def images():
    """A random 32 x 48 image, its 4x downsampling rounded to uint8, and the
    downsampling's kernels."""
    torch.manual_seed(0)
    image = torch.randint(256, (3, 32, 48), dtype=torch.uint8)
    kernels = [inro.resize_matrix(32, 8), inro.resize_matrix(48, 12)]
    low = inro.downsample(image.movedim(0, -1).float(), kernels)
    low = low.round().clamp(0, 255).to(torch.uint8).movedim(-1, 0)
    return image, low, kernels


@pytest.fixture
def image_pair(task, images, tmp_path):
    """The image and its downsampling as files, laid out as DIV2K is."""
    image, low, kernels = images
    (tmp_path / "DIV2K_valid_HR").mkdir()
    (tmp_path / "DIV2K_valid_LR_bicubic" / "X4").mkdir(parents=True)
    path = tmp_path / "DIV2K_valid_HR" / "0801.png"
    to_pil_image(image).save(path)
    to_pil_image(low).save(task.low_resolution(path, 4))
    return path, image, kernels


class Lookup(nn.Module):
    """A field that returns the pixels of an image under the coordinates it is
    given, plus a learnable offset, so that a loss has a gradient."""

    def __init__(self, image: torch.Tensor) -> None:
        super().__init__()
        self.pixels = inro.ImageCoordinateDataset.preprocess(image)  # (H, W, C)
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.pixels.shape[:2]
        rows = ((x[..., 0] + 1) / 2 * (height - 1)).round().long()
        columns = ((x[..., 1] + 1) / 2 * (width - 1)).round().long()
        return self.pixels[rows, columns] + self.offset


def test_resize_matrix_is_the_antialiased_bicubic_operator():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 48)
    expected = F.interpolate(image, size=(8, 12), mode="bicubic", antialias=True)
    kernels = [inro.resize_matrix(32, 8), inro.resize_matrix(48, 12)]
    downsampled = inro.downsample(image[0].movedim(0, -1), kernels).movedim(-1, 0)
    torch.testing.assert_close(downsampled, expected[0], atol=1e-5, rtol=1e-5)
    assert int((kernels[0] != 0).sum(1).max()) == 16  # taps per LR pixel at 4x


def test_blocks_downsample_to_the_low_resolution_pixels(images):
    image, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, num_blocks=12, block_size=2)
    torch.manual_seed(0)
    batch = dataset[0]
    rendered = Lookup(image)(batch["input"])  # (B, L, L, C) in [-1, 1]
    fitted = inro.downsample(rendered, (batch["weight_y"], batch["weight_x"]))
    assert batch["input"].shape == (12, dataset.footprint, dataset.footprint, 2)
    assert batch["weight_y"].shape == batch["weight_x"].shape == (12, 2, 16 + 4)
    assert batch["target"].shape == (12, 2, 2, 3)
    torch.testing.assert_close(fitted, batch["target"], atol=2 / 255, rtol=0)
    assert torch.equal(batch["_original"], low)


def test_loss_matches_the_low_resolution_pixels_through_the_kernels(images):
    """The true image leaves only the rounding of the LR pixels, and the loss
    pulls an offset back towards it."""
    image, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, num_blocks=12, block_size=4)
    batch = torch.utils.data.default_collate([dataset[0]])  # as inro.train draws it
    field = Lookup(image)
    loss, metrics = inro.super_resolution_loss(batch, field)
    assert loss.item() <= 1 / 255 and set(metrics) == {"l1", "psnr"}
    assert float(metrics["psnr"]) > 50
    with torch.no_grad():
        field.offset.fill_(0.1)
    loss, _ = inro.super_resolution_loss(batch, field)
    loss.backward()
    assert float(field.offset.grad) > 0


def test_dataset_rejects_kernels_and_blocks_that_do_not_fit(images):
    _, low, kernels = images
    with pytest.raises(ValueError, match="do not map"):
        inro.SuperResolutionDataset(low, kernels[::-1], num_blocks=4)
    with pytest.raises(ValueError, match="block_size"):
        inro.SuperResolutionDataset(low, kernels, num_blocks=4, block_size=9)
    with pytest.raises(ValueError, match="footprint"):  # 4 * 5 + 16 = 36 > 32
        inro.SuperResolutionDataset(low, kernels, num_blocks=4, block_size=6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dataset_moves_its_kernels_with_it(images):
    _, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, 4, block_size=4).to("cuda")
    batch = dataset[0]
    trained = ("input", "weight_y", "weight_x", "target")
    assert all(batch[key].is_cuda for key in trained)
    assert batch["_original"].device.type == "cpu"


def test_luma_is_bt601_in_16_to_235():
    image = torch.zeros(3, 2, 2, dtype=torch.uint8)
    image[:, 0] = 255
    torch.testing.assert_close(
        inro.luma(image), torch.tensor([[[235.0, 235.0], [16.0, 16.0]]])
    )


@pytest.mark.parametrize("border", [0, 4])
def test_metrics_score_luma_and_rgb_without_the_border(border):
    torch.manual_seed(0)
    target = torch.randint(256, (3, 40, 48), dtype=torch.uint8)
    pred = (target.int() + torch.randint(-8, 9, target.shape)).clamp(0, 255).byte()
    metrics = inro.super_resolution_metrics(border)
    inner = (slice(None), slice(border, 40 - border), slice(border, 48 - border))
    assert metrics["psnr"](pred, target) == inro.psnr(
        inro.luma(pred[inner]), inro.luma(target[inner]), data_range=255
    )
    assert metrics["ssim"](pred, target) == inro.ssim(
        inro.luma(pred[inner]), inro.luma(target[inner]), data_range=255
    )


@pytest.mark.parametrize("outputs", [1, 3])
def test_futon_rank_is_the_largest_multiple_of_5_within_the_budget(task, outputs):
    from train_common import futon_rank  # the fixture put the scripts on the path

    for components, budget in (((339, 510), 500_006), ((69, 115, 59), 131_456)):
        rank = futon_rank(components, budget, 64, outputs)

        def count(rank: int) -> int:
            model = inro.FUTON(
                in_features=len(components),
                out_features=outputs,
                basis=("lanczos", {"num_components": components, "radius": 3}),
                combiner=("cp", {"rank": rank}),
                decoder=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
            )
            return inro.count_parameters(model)

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
