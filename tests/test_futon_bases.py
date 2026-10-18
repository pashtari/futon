"""Tests for the FUTON compactly supported bases and their RCS fast path."""

import pytest
import torch
from torch import nn

from neurofield.models import RCSMatrix
from neurofield.models.futon import (
    FUTON,
    ChebyshevBasis,
    CPCombiner,
    HadamardCombiner,
    LanczosBasis,
    LegendreBasis,
    SincBasis,
    TRCombiner,
    TuckerCombiner,
    TriangleBasis,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def reference_lanczos(x, num_components, radius):
    """Direct evaluation from the definition, over every center."""
    K, a = num_components, radius
    centers = torch.linspace(-1.0, 1.0, K, dtype=x.dtype, device=x.device)
    w = 2.0 / (K - 1)
    t = (x.unsqueeze(-1) - centers) / w
    kernel = torch.sinc(t) * torch.sinc(t / a)
    return torch.where(t.abs() < a, kernel, torch.zeros_like(kernel))


class TestKernel:
    @pytest.mark.parametrize("radius", [1, 2, 3, 4])
    @pytest.mark.parametrize("K", [16, 33])
    def test_matches_definition(self, radius, K):
        basis = LanczosBasis(1, K, radius=radius, normalize=False, sparse=False)
        x = torch.rand(500) * 2 - 1
        got = basis(x.unsqueeze(-1))[0]
        assert torch.allclose(got, reference_lanczos(x, K, radius), atol=1e-6)

    @pytest.mark.parametrize(
        "basis, taps",
        [
            (TriangleBasis(1, 32), 2),
            (LanczosBasis(1, 32, radius=1), 2),
            (LanczosBasis(1, 32), 6),  # default radius=3 -> Lanczos-3
            (LanczosBasis(1, 32, radius=3), 6),
        ],
    )
    def test_stores_two_taps_per_radius(self, basis, taps):
        """The RCS segment length is 2 * radius."""
        assert basis.radius * 2 == taps
        feats = basis((torch.rand(16) * 2 - 1).unsqueeze(-1))[0]
        assert feats.values.shape[1] == taps

    def test_support_is_l_contiguous(self):
        K, radius = 32, 2
        basis = LanczosBasis(1, K, radius=radius, normalize=False, sparse=False)
        dense = basis((torch.rand(200) * 2 - 1).unsqueeze(-1))[0]
        nz = dense != 0
        assert int(nz.sum(-1).max()) <= 2 * radius
        # every nonzero run is contiguous
        first = torch.argmax(nz.int(), dim=-1)
        last = nz.shape[-1] - 1 - torch.argmax(nz.int().flip(-1), dim=-1)
        assert bool((last - first < 2 * radius).all())

    def test_cardinal_at_grid_points(self):
        """Un-normalized kernels are 1 at their own center, 0 at the others."""
        K = 16
        basis = LanczosBasis(1, K, radius=2, normalize=False, sparse=False)
        dense = basis(torch.linspace(-1, 1, K).unsqueeze(-1))[0]
        assert torch.allclose(dense, torch.eye(K), atol=1e-6)

    def test_approaches_sinc_for_large_radius(self):
        """SincBasis is the radius -> infinity limit."""
        K = 64
        x = (torch.rand(100) * 2 - 1).unsqueeze(-1)
        sinc = SincBasis(1, K, normalize=False)(x)[0]
        errors = [
            (LanczosBasis(1, K, radius=a, normalize=False, sparse=False)(x)[0] - sinc)
            .abs()
            .max()
            .item()
            for a in (2, 8, 24)
        ]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.05

    def test_partition_of_unity_of_interpolant(self):
        """Lanczos reconstruction of a constant signal is near-constant."""
        basis = LanczosBasis(1, 64, radius=3, normalize=False, sparse=False)
        x = (torch.rand(500) * 1.8 - 0.9).unsqueeze(-1)
        assert (basis(x)[0].sum(-1) - 1.0).abs().max() < 0.02


def reference_polynomials(x, count, legendre):
    """The three-term recurrences, degree by degree."""
    x = x.unsqueeze(-1)
    polynomials = [torch.ones_like(x), x]
    for k in range(2, count):
        previous, before = polynomials[-1], polynomials[-2]
        if legendre:
            polynomials.append(((2 * k - 1) * x * previous - (k - 1) * before) / k)
        else:
            polynomials.append(2 * x * previous - before)
    return torch.cat(polynomials[:count], -1)


class TestPolynomials:
    @pytest.mark.parametrize("count", [2, 8, 64, 256])
    def test_chebyshev_matches_the_recurrence(self, count):
        """The closed form agrees with the recurrence to double precision."""
        basis = ChebyshevBasis(2, [count, max(count // 2, 2)], normalize=False).double()
        x = torch.rand(500, 2, dtype=torch.float64) * 2 - 1
        for axis, feature in enumerate(basis(x)):
            expected = reference_polynomials(
                x[:, axis], basis.num_components[axis], False
            )
            assert torch.allclose(feature, expected, atol=1e-10)

    @pytest.mark.parametrize("count", [2, 8, 64, 256])
    def test_legendre_matches_the_recurrence(self, count):
        """The Chebyshev map agrees with Bonnet's recurrence to double precision."""
        basis = LegendreBasis(2, [count, max(count // 2, 2)], normalize=False).double()
        x = torch.rand(500, 2, dtype=torch.float64) * 2 - 1
        for axis, feature in enumerate(basis(x)):
            expected = reference_polynomials(
                x[:, axis], basis.num_components[axis], True
            )
            assert torch.allclose(feature, expected, atol=1e-6)

    def test_differentiable_inside_the_interval(self):
        x = (torch.rand(20, 1) * 1.8 - 0.9).requires_grad_(True)
        LegendreBasis(1, 16)(x)[0].sum().backward()
        assert bool(x.grad.isfinite().all())


def reference_tent(x, num_components):
    """The tent formula as originally written, in coordinate units."""
    K = num_components
    centers = torch.linspace(-1.0, 1.0, K, dtype=x.dtype, device=x.device)
    width = 2.0 / (K - 1)
    return (1.0 - (x.unsqueeze(-1) - centers).abs() / width).clamp(min=0.0)


class TestTriangle:
    @pytest.mark.parametrize("K", [2, 8, 33, 64, 512])
    def test_matches_original_formula(self, K):
        """Refactoring to grid-index units must not change the values.

        Checked in float64: in float32 both this form and the original sit
        ~1e-5 from the exact value once K is large (cancellation in the
        ``x - mu_c`` subtraction), so a float32 comparison of the two would
        only be measuring rounding noise. See
        :meth:`test_float32_accuracy_matches_original`.
        """
        basis = TriangleBasis(1, K, normalize=False, sparse=False)
        x = torch.rand(500, dtype=torch.float64) * 2 - 1
        got = basis(x.unsqueeze(-1))[0]
        assert torch.allclose(got, reference_tent(x, K), atol=1e-12)

    @pytest.mark.parametrize("K", [64, 512])
    def test_float32_accuracy_matches_original(self, K):
        """In float32 the new form is no less accurate than the original."""
        x = torch.rand(2000) * 2 - 1
        truth = reference_tent(x.double(), K)
        new = TriangleBasis(1, K, normalize=False, sparse=False)(x.unsqueeze(-1))[0]
        old = reference_tent(x, K)
        err_new = (new.double() - truth).abs().max()
        err_old = (old.double() - truth).abs().max()
        assert err_new < 1e-4
        assert err_new < 4 * err_old

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("normalize", [True, False])
    def test_sparse_equals_dense(self, device, normalize):
        K = 64
        x = torch.rand(1000, 2, device=device) * 2 - 1
        dense = TriangleBasis(2, K, normalize=normalize, sparse=False).to(device)(x)
        sparse = TriangleBasis(2, K, normalize=normalize, sparse=True).to(device)(x)
        for d, s in zip(dense, sparse):
            assert isinstance(s, RCSMatrix)
            assert torch.allclose(s.to_dense(), d, atol=1e-5)

    def test_two_taps_only(self):
        feats = TriangleBasis(1, 32, normalize=False)(
            (torch.rand(500) * 2 - 1).unsqueeze(-1)
        )[0]
        assert feats.values.shape[1] == 2
        assert int((feats.to_dense() != 0).sum(-1).max()) <= 2

    def test_partition_of_unity(self):
        basis = TriangleBasis(1, 32, normalize=False, sparse=False)
        x = (torch.rand(500) * 2 - 1).unsqueeze(-1)
        assert torch.allclose(basis(x)[0].sum(-1), torch.ones(500), atol=1e-5)

    def test_cardinal_at_grid_points(self):
        K = 16
        basis = TriangleBasis(1, K, normalize=False, sparse=False)
        got = basis(torch.linspace(-1, 1, K).unsqueeze(-1))[0]
        assert torch.allclose(got, torch.eye(K), atol=1e-6)

    def test_grid_sample_equivalence(self):
        """Un-normalized tents + a linear map == linear interpolation on a grid."""
        K, R = 16, 3
        grid = torch.randn(K, R)
        x = torch.rand(200) * 2 - 1
        feats = TriangleBasis(1, K, normalize=False, sparse=True)(x.unsqueeze(-1))[0]
        pos = (x + 1) / 2 * (K - 1)
        i = pos.floor().clamp(0, K - 2).long()
        frac = (pos - i).unsqueeze(-1)
        expected = grid[i] * (1 - frac) + grid[i + 1] * frac
        assert torch.allclose(feats @ grid, expected, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_futon_equivalence(self, device):
        torch.manual_seed(0)
        x = torch.rand(4, 9, 2, device=device) * 2 - 1
        outs = []
        for sparse in (False, True):
            torch.manual_seed(1)
            model = FUTON(
                2,
                3,
                ("triangle", {"num_components": 32, "sparse": sparse}),
                ("cp", {"rank": 8}),
                "linear",
            ).to(device)
            outs.append(model(x))
        assert outs[0].shape == (4, 9, 3)
        assert torch.allclose(outs[1], outs[0], atol=1e-5)


class TestSparseEqualsDense:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("radius", [1, 2, 4])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_features_match(self, device, radius, normalize):
        K = 48
        x = torch.rand(1000, 2, device=device) * 2 - 1
        common = dict(radius=radius, normalize=normalize)
        dense = LanczosBasis(2, K, sparse=False, **common).to(device)(x)
        sparse = LanczosBasis(2, K, sparse=True, **common).to(device)(x)
        for d, s in zip(dense, sparse):
            assert isinstance(s, RCSMatrix)
            assert s.shape == d.shape
            assert torch.allclose(s.to_dense(), d, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_boundaries_and_grid_points(self, device):
        """Clamped windows at the domain edges stay exact."""
        K = 32
        x = torch.cat(
            [
                torch.linspace(-1, 1, K, device=device),  # exactly on grid
                torch.tensor([-1.0, 1.0, -0.999, 0.999], device=device),
                torch.rand(200, device=device) * 2 - 1,
            ]
        ).unsqueeze(-1)
        common = dict(radius=3, normalize=False)
        dense = LanczosBasis(1, K, sparse=False, **common).to(device)(x)[0]
        sparse = LanczosBasis(1, K, sparse=True, **common).to(device)(x)[0]
        assert torch.allclose(sparse.to_dense(), dense, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_per_axis_num_components(self, device):
        x = torch.rand(64, 3, device=device) * 2 - 1
        common = dict(num_components=[16, 24, 32], radius=2)
        dense = LanczosBasis(3, sparse=False, **common).to(device)(x)
        sparse = LanczosBasis(3, sparse=True, **common).to(device)(x)
        for d, s in zip(dense, sparse):
            assert torch.allclose(s.to_dense(), d, atol=1e-5)


class TestCombiners:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("combiner_cls", [CPCombiner, TRCombiner, TuckerCombiner])
    def test_matches_dense_path(self, device, combiner_cls):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(128, C, device=device) * 2 - 1
        combiner = combiner_cls([K] * C, rank=8).to(device)
        dense = LanczosBasis(C, K, radius=2, sparse=False).to(device)(x)
        sparse = LanczosBasis(C, K, radius=2, sparse=True).to(device)(x)
        out, ref = combiner(sparse), combiner(dense)
        assert torch.allclose(out, ref, atol=1e-5)
        grad = torch.randn_like(out)
        params = list(combiner.parameters())
        grads = torch.autograd.grad(out, params, grad)
        for got, expected in zip(grads, torch.autograd.grad(ref, params, grad)):
            assert torch.allclose(got, expected, atol=1e-4)

    def test_tucker_contracts_the_core(self):
        torch.manual_seed(0)
        combiner = TuckerCombiner([7, 6, 5], rank=[3, 4, 2])
        assert combiner.core.shape == (3, 4, 2, 4)  # out_features: the largest rank
        features = [torch.randn(10, width) for width in (7, 6, 5)]
        a, b, c = (f @ factor for factor, f in zip(combiner.factors, features))
        expected = torch.einsum("na,nb,nc,abco->no", a, b, c, combiner.core)
        assert torch.allclose(combiner(features), expected, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_hadamard_densifies(self, device):
        K, C = 32, 2
        x = torch.rand(64, C, device=device) * 2 - 1
        combiner = HadamardCombiner([K] * C)
        dense = LanczosBasis(C, K, radius=2, sparse=False).to(device)(x)
        sparse = LanczosBasis(C, K, radius=2, sparse=True).to(device)(x)
        assert torch.allclose(combiner(sparse), combiner(dense), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradients_match(self, device):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(128, C, device=device) * 2 - 1
        grads = {}
        for sparse in (False, True):
            torch.manual_seed(1)
            combiner = CPCombiner([K] * C, rank=8).to(device)
            feats = LanczosBasis(C, K, radius=2, sparse=sparse).to(device)(x)
            combiner(feats).square().sum().backward()
            grads[sparse] = [p.grad.clone() for p in combiner.parameters()]
        for g_dense, g_sparse in zip(grads[False], grads[True]):
            assert torch.allclose(g_sparse, g_dense, atol=1e-4, rtol=1e-4)


class TestFuton:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_shape", [(), (7,), (4, 5), (2, 8, 8)])
    def test_shapes_and_equivalence(self, device, batch_shape):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(*batch_shape, C, device=device) * 2 - 1
        outs = []
        for sparse in (False, True):
            torch.manual_seed(1)
            model = FUTON(
                C,
                3,
                ("lanczos", {"num_components": K, "radius": 3, "sparse": sparse}),
                ("cp", {"rank": 8}),
                "linear",
            ).to(device)
            outs.append(model(x))
        assert outs[0].shape == (*batch_shape, 3)
        assert torch.allclose(outs[1], outs[0], atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_other_bases_unaffected(self, device):
        """Flattening in FUTON.forward must not change existing bases."""
        torch.manual_seed(0)
        x = torch.rand(3, 6, 2, device=device) * 2 - 1
        for name in ("cosine", "sinc", "triangle", "chebyshev"):
            torch.manual_seed(1)
            model = FUTON(
                2, 3, (name, {"num_components": 16}), ("cp", {"rank": 8}), "linear"
            ).to(device)
            out = model(x)
            flat = model(x.reshape(-1, 2))
            assert out.shape == (3, 6, 3)
            assert torch.allclose(out.reshape(-1, 3), flat, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_training_step(self, device):
        torch.manual_seed(0)
        model = FUTON(
            2,
            1,
            ("lanczos", {"num_components": 64, "radius": 3}),
            ("cp", {"rank": 16}),
            "linear",
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.rand(512, 2, device=device) * 2 - 1
        y = torch.sin(3 * x[:, :1]) * torch.cos(3 * x[:, 1:])
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.6 * losses[0]


def grid_points(sizes, count=None, dtype=torch.float64):
    """The points of ``linspace(-1, 1, size)`` per axis, as a coordinate dataset
    has them, or ``count`` of them at random, as a training batch draws them."""
    axes = [torch.linspace(-1.0, 1.0, size).to(dtype) for size in sizes]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    if count is None:
        return points
    points = points.reshape(-1, len(sizes))
    return points[torch.randperm(len(points))[:count]]


def per_point(model, x):
    """FUTON with each point's dense features of its own, shared with no other."""
    points = x.reshape(-1, model.in_features)
    basis = model.basis
    features = [basis.axis_features(xc, c) for c, xc in enumerate(points.unbind(-1))]
    out = model.output_activation(model.decoder(model.combiner(features)))
    return out.reshape(*x.shape[:-1], model.out_features)


def shares_rows(model, x):
    """Whether the basis gives the points of ``x`` shared rows of a table."""
    features = model.basis(x.reshape(-1, model.in_features))
    return all(isinstance(f, RCSMatrix) and f.table is not None for f in features)


class TestGridEvaluation:
    """On a grid, a dense basis evaluates each distinct coordinate once."""

    @staticmethod
    def model(sizes, basis, combiner, device, **basis_kwargs):
        torch.manual_seed(0)
        model = FUTON(
            len(sizes),
            3,
            (basis, {"num_components": 8, **basis_kwargs}),
            combiner,
            ("mlp", {"hidden_layers": 1}),
            torch.tanh,
        )
        for parameter in model.parameters():  # the biases too, which start at zero
            nn.init.normal_(parameter, std=0.3)
        return model.double().to(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sizes", [(9, 12), (5, 6, 7)])
    @pytest.mark.parametrize(
        "combiner",
        [
            ("cp", {"rank": 6}),
            ("tr", {"rank": 3}),
            ("tucker", {"rank": 4}),
            "hadamard",
        ],
    )
    @pytest.mark.parametrize("basis", ["cosine", "sinc", "chebyshev", "legendre"])
    def test_matches_per_point(self, device, basis, combiner, sizes):
        model = self.model(sizes, basis, combiner, device)
        x = grid_points(sizes, count=60).to(device)
        assert shares_rows(model, x)
        out, ref = model(x), per_point(model, x)
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
        params = list(model.parameters())
        grads = torch.autograd.grad(out.square().sum(), params)
        expected = torch.autograd.grad(ref.square().sum(), params)
        for got, want in zip(grads, expected):
            torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "basis, basis_kwargs",
        [("sinc", {}), ("lanczos", {"radius": 2, "sparse": False})],
        ids=["sinc", "dense-lanczos"],
    )
    def test_whole_grid(self, device, basis, basis_kwargs):
        model = self.model((9, 12), basis, ("cp", {"rank": 6}), device, **basis_kwargs)
        x = grid_points((9, 12)).to(device)
        out = model(x)
        assert shares_rows(model, x) and out.shape == (9, 12, 3)
        torch.testing.assert_close(out, per_point(model, x), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scattered_points_are_evaluated_one_by_one(self, device):
        model = self.model((9, 12), "sinc", ("cp", {"rank": 6}), device)
        x = (torch.rand(60, 2, dtype=torch.float64) * 2 - 1).to(device)
        assert not any(isinstance(f, RCSMatrix) for f in model.basis(x))
        out, ref = model(x), per_point(model, x)
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("basis", ["triangle", "lanczos"])
    def test_sparse_bases_keep_their_taps(self, device, basis):
        model = self.model((9, 12), basis, ("cp", {"rank": 6}), device)
        x = grid_points((9, 12), count=60).to(device)
        assert all(f.table is None for f in model.basis(x))
        out, ref = model(x), per_point(model, x)
        torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-12)

    def test_batches_with_a_shape_are_evaluated_one_by_one(self):
        basis = SincBasis(2, 8)
        x = grid_points((9, 12))  # (9, 12, 2)
        assert not any(isinstance(f, RCSMatrix) for f in basis(x))
        assert all(isinstance(f, RCSMatrix) for f in basis(x.reshape(-1, 2)))

    @pytest.mark.parametrize("device", DEVICES)
    def test_coordinate_gradients(self, device):
        model = self.model((9, 12), "sinc", ("cp", {"rank": 6}), device)
        x = grid_points((9, 12), count=60).to(device).requires_grad_()
        assert not shares_rows(model, x)
        (grad,) = torch.autograd.grad(model(x).sum(), x)
        (expected,) = torch.autograd.grad(per_point(model, x).sum(), x)
        torch.testing.assert_close(grad, expected)

    @CUDA
    def test_gradients_reproduce_bitwise(self):
        torch.manual_seed(0)
        basis = SincBasis(2, [16, 24]).cuda()
        combiner = CPCombiner([16, 24], rank=32).cuda()
        features = basis(grid_points((40, 50), count=4096, dtype=torch.float32).cuda())
        grads = []
        for _ in range(2):
            combiner.zero_grad()
            combiner(features).square().sum().backward()
            grads.append([p.grad.clone() for p in combiner.parameters()])
        assert all(map(torch.equal, *grads))


@pytest.mark.parametrize(
    "name", ["cosine", "sinc", "legendre", "chebyshev", "triangle", "lanczos"]
)
class TestNumComponents:
    def test_int_and_per_axis(self, name):
        from neurofield.models.futon import BASES

        def dense(feature):
            return feature.to_dense() if isinstance(feature, RCSMatrix) else feature

        x = torch.rand(20, 3) * 2 - 1
        shared = BASES[name](3, 8, normalize=False)
        # At least 2 * radius = 6 components per axis for the default Lanczos-3.
        per_axis = BASES[name](3, [6, 8, 10], normalize=False)
        assert shared.num_components == [8, 8, 8]
        assert per_axis.num_components == [6, 8, 10]
        features = [dense(f) for f in per_axis(x)]
        assert [f.shape for f in features] == [(20, 6), (20, 8), (20, 10)]
        # Axis 1 has the same count in both bases, so its features must match.
        assert torch.allclose(dense(shared(x)[1]), features[1])

    def test_wrong_length(self, name):
        from neurofield.models.futon import BASES

        with pytest.raises(ValueError):
            BASES[name](3, [8, 8])


class TestValidation:
    @pytest.mark.parametrize("radius", [0, -2])
    def test_radius_must_be_positive(self, radius):
        with pytest.raises(ValueError):
            LanczosBasis(1, 32, radius=radius)

    def test_num_components_at_least_twice_radius(self):
        with pytest.raises(ValueError):
            LanczosBasis(1, 4, radius=3)  # needs K >= 6
        with pytest.raises(ValueError):
            TriangleBasis(1, 1)  # needs K >= 2

    def test_wrong_in_features(self):
        basis = LanczosBasis(3, 16, radius=2)
        with pytest.raises(ValueError):
            basis(torch.rand(8, 2))

    def test_registry(self):
        from neurofield.models.futon import BASES

        assert BASES["lanczos"] is LanczosBasis
        assert BASES["triangle"] is TriangleBasis


@CUDA
class TestSparseIsFaster:
    def test_forward_beats_dense(self):
        """The compact path should win where K >> L (its whole purpose)."""
        torch.manual_seed(0)
        K, C, M = 512, 2, 200_000
        x = torch.rand(M, C, device="cuda") * 2 - 1
        combiner = CPCombiner([K] * C, rank=32).cuda()

        def run(sparse):
            basis = LanczosBasis(C, K, radius=4, sparse=sparse).cuda()
            return lambda: combiner(basis(x))

        import time

        times = {}
        for sparse in (False, True):
            fn = run(sparse)
            with torch.no_grad():
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                times[sparse] = time.perf_counter() - t0
        assert times[True] < times[False], times
