"""FUTON: coordinate bases, tensor-network combiners, and the FUTON model.

FUTON evaluates ``decoder(combiner(basis(x)))`` in three stages:

1. A **basis** maps coordinates ``(N, C)`` in ``[-1, 1]`` to one feature tensor
   ``(N, K_c)`` per axis, where ``K_c = num_components[c]``.
2. A **combiner** projects the features of each axis and combines the
   projections into one ``(N, out_features)`` tensor.
3. A **decoder**, ``nn.Linear`` or :class:`MLP`, maps the fused features to the
   output.

The first two stages make up :class:`FUTONEncoding`, which :class:`FUTON`
decodes.

Every basis takes ``num_components``, an int shared by all axes or one count
per axis; ``normalize``, which L2-normalizes each feature vector (not each
basis function); and ``grid_size``, the sizes of the grid its inputs lie on,
if they do. The local bases :class:`TriangleBasis` and
:class:`LanczosBasis` default to sparse mode: each axis holds :class:`Taps`,
only the few nonzero features of each coordinate.

On a grid, where every axis of a flat batch repeats its coordinates, a basis
evaluates each distinct coordinate once, or reads the table of the grid that
its ``grid_size`` declares, evaluated once when it is built, and returns each
axis as :class:`SharedRows` of that table, dense or the taps of a local basis:
the combiners then project the table rather than every point, the same function
at a fraction of the cost. Both kinds multiply like ``(N, K_c)`` matrices, so
every combiner treats every axis alike; compiled, the gathers fuse with the
product and the decoder.
"""

import math
from collections.abc import Callable, Sequence
from functools import reduce
from typing import NamedTuple, Self

import opt_einsum as oe
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import ModuleSpec, build_module
from .mlp import MLP

__all__ = [
    "SharedRows",
    "Taps",
    "CosineBasis",
    "SincBasis",
    "LegendreBasis",
    "ChebyshevBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "TuckerCombiner",
    "FUTONEncoding",
    "FUTON",
]


def _broadcast(value: int | Sequence[int], length: int, name: str) -> list[int]:
    """Expand a shared int to ``length`` values, or check a per-axis sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _grid_position(x: Tensor, size: int) -> Tensor:
    """Map coordinates in ``[-1, 1]`` to continuous indices in ``[0, size - 1]``."""
    return (x + 1) / 2 * (size - 1)


def _offsets(x: Tensor, size: int) -> Tensor:
    """Grid-step offsets ``(*, size)`` from the centers ``linspace(-1, 1, size)``."""
    centers = torch.arange(size, device=x.device, dtype=x.dtype)
    return _grid_position(x, size).unsqueeze(-1) - centers


@torch.compiler.disable()  # sizes that depend on the data; compiled, a graph per axis
def _shared(x: Tensor) -> list[tuple[Tensor, Tensor]] | None:
    """Each axis's distinct coordinates and the row of every point among them.

    For a flat batch ``(N, C)`` whose axes repeat their coordinates; ``None``
    for other shapes, for coordinates that need gradients, which shared rows do
    not carry, and when an axis has more distinct coordinates than half the
    points, the rule of :class:`~neurofield.models.finr.FINR`.
    """
    if x.ndim != 2 or x.requires_grad:
        return None
    shared = []
    for coordinate in x.unbind(-1):
        values, rows = torch.unique(coordinate, return_inverse=True)
        if 2 * len(values) > len(coordinate):
            return None
        shared.append((values, rows))
    return shared


class SharedRows(NamedTuple):
    """An axis's features on a grid: ``(N, K)``, row ``n`` being ``table[rows[n]]``.

    ``table`` holds the features of ``S`` coordinates of the axis, its distinct
    ones or those of its grid, dense or as a local basis's :class:`Taps`, and
    ``rows`` the row of each point, so a product with a factor projects ``S``
    rows rather than ``N``.
    """

    table: "Tensor | Taps"
    rows: Tensor

    def __matmul__(self, other: Tensor) -> Tensor:
        return (self.table @ other)[self.rows]

    def dense(self) -> Tensor:
        """The ``(N, K)`` features."""
        table = self.table if isinstance(self.table, Tensor) else self.table.dense()
        return table[self.rows]


class Taps(NamedTuple):
    """An axis's local features: ``(N, size)``, row ``n`` holding the ``L``
    values ``values[n]`` in columns ``start[n]`` to ``start[n] + L - 1``.

    ``compiled`` records whether a compiled graph made the taps, whose product
    with a factor then gathers windows of its rows. It is set where the taps are
    made, not read in :meth:`__matmul__`: taps that cross a graph break have
    their product traced later, when some PyTorch versions no longer report
    ``torch.compiler.is_compiling()``.
    """

    values: Tensor
    start: Tensor
    size: int
    compiled: bool = False

    def __matmul__(self, other: Tensor) -> Tensor:
        if self.compiled:
            # Each point gathers its window of L consecutive rows, which the
            # compiler fuses with the weighted sum into one kernel.
            windows = other.unfold(0, self.values.shape[1], 1)
            return (windows[self.start] * self.values.unsqueeze(1)).sum(-1)
        # Eager, the same sum in one kernel, without gathering the windows.
        return F.embedding_bag(
            self._columns(), other, per_sample_weights=self.values, mode="sum"
        )

    def dense(self) -> Tensor:
        """The ``(N, size)`` features, zero outside the taps."""
        zeros = self.values.new_zeros(len(self.values), self.size)
        return zeros.scatter(1, self._columns(), self.values)

    def _columns(self) -> Tensor:
        taps = torch.arange(self.values.shape[1], device=self.start.device)
        return self.start.unsqueeze(1) + taps


# An axis's features: dense, the rows a grid shares, or a local basis's taps.
Features = Tensor | SharedRows | Taps


# Bases --------------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class for bases; subclasses implement :meth:`_features` for one axis."""

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_components = _broadcast(num_components, in_features, "num_components")
        self.normalize = normalize
        self._tabulate(grid_size)

    def axis_features(self, x: Tensor, axis: int) -> Tensor:
        """The dense features ``(*, K_axis)`` of one axis at coordinates ``(*,)``."""
        return self._normalize(self._features(x, axis))

    def forward(self, x: Tensor) -> tuple[Features, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        if self.grid_size is not None and not x.requires_grad:
            # Each point's row of its axis's table is a rounding: no sort, no
            # synchronization and no evaluation of the basis.
            return tuple(
                SharedRows(self._grid_table(c), _grid_position(xc, n).round().long())
                for c, (xc, n) in enumerate(zip(x.unbind(-1), self.grid_size))
            )
        shared = _shared(x)
        if shared is None:
            return tuple(self._evaluate(xc, c) for c, xc in enumerate(x.unbind(-1)))
        return tuple(
            SharedRows(self._evaluate(values, c), rows)
            for c, (values, rows) in enumerate(shared)
        )

    def _features(self, x: Tensor, axis: int) -> Tensor:
        """The features of one axis before normalization."""
        raise NotImplementedError

    def _evaluate(self, x: Tensor, axis: int) -> Tensor | Taps:
        """The features of one axis at coordinates ``(N,)``, dense or as taps."""
        return self.axis_features(x, axis)

    def _normalize(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def _tabulate(
        self,
        grid_size: int | Sequence[int] | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Evaluate each axis once on its grid ``linspace(-1, 1, size)``, as the
        buffers that :meth:`_grid_table` reads: ``_table{axis}``, dense or the
        taps' values, and ``_start{axis}``, the taps' first columns or ``None``.
        Subclasses whose features need state of their own call it again once
        that is set."""
        self.grid_size = None
        if grid_size is not None:
            self.grid_size = _broadcast(grid_size, self.in_features, "grid_size")
        for c, size in enumerate(self.grid_size or ()):
            grid = torch.linspace(-1, 1, size, device=device, dtype=dtype)
            table, start = self._evaluate(grid, c), None
            if isinstance(table, Taps):
                table, start = table.values, table.start
            self.register_buffer(f"_table{c}", table, persistent=False)
            self.register_buffer(f"_start{c}", start, persistent=False)

    def _grid_table(self, axis: int) -> Tensor | Taps:
        """The table of one axis on its declared grid."""
        table, start = getattr(self, f"_table{axis}"), getattr(self, f"_start{axis}")
        if start is None:
            return table
        size = self.num_components[axis]
        return Taps(table, start, size, torch.compiler.is_compiling())

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # A move or a cast evaluates the tables again where it takes them, as
        # nn.RNNBase rebuilds its flat weights: a cast would keep the rounding
        # of the dtype they were built in.
        super()._apply(fn, recurse)
        if self.grid_size is not None:
            table = getattr(self, "_table0")
            self._tabulate(self.grid_size, table.device, table.dtype)
        return self


class CosineBasis(_Basis):
    r"""Cosine basis :math:`\cos(k \pi u)` with :math:`u = (x + 1) / 2`.

    Axis ``c`` uses frequencies :math:`k = 0, \dots, K_c - 1`. This is the
    FUTON paper's basis without its :math:`\sqrt{2}` factor; see
    :class:`FUTON`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of frequencies ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        u = ((x + 1) / 2).unsqueeze(-1)
        k = torch.arange(self.num_components[axis], device=x.device, dtype=x.dtype)
        return torch.cos(math.pi * k * u)


class SincBasis(_Basis):
    r"""Cardinal sine basis centered on a uniform grid over ``[-1, 1]``.

    Component ``k`` is :math:`\operatorname{sinc}((x - \mu_k) / w)`, with
    centers :math:`\mu_k = -1 + k w` and spacing :math:`w = 2 / (K_c - 1)`.
    Each component equals one at its own center and zero at all other
    centers, so ``normalize=False`` gives cardinal (Whittaker) interpolation.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        return torch.sinc(_offsets(x, self.num_components[axis]))


def _chebyshev_to_legendre(count: int) -> Tensor:
    """The ``(count, count)`` matrix ``M`` with ``P_n = sum_k M[n, k] T_k``.

    Bonnet's recurrence, :math:`k P_k = (2k - 1) x P_{k-1} - (k - 1) P_{k-2}`,
    run on Chebyshev coefficients with :math:`x T_j = (T_{j+1} + T_{j-1}) / 2`.
    The entries are nonnegative and every row sums to one, so the map is as
    well conditioned as the polynomials themselves.
    """
    matrix = torch.zeros(count, count, dtype=torch.float64)
    matrix[0, 0] = 1.0
    if count > 1:
        matrix[1, 1] = 1.0
    for n in range(2, count):
        previous = matrix[n - 1]
        times_x = torch.zeros_like(previous)
        times_x[1:] += previous[:-1] / 2
        times_x[:-1] += previous[1:] / 2
        times_x[1] += previous[0] / 2  # x T_0 = T_1
        matrix[n] = ((2 * n - 1) * times_x - (n - 1) * matrix[n - 2]) / n
    return matrix.to(torch.get_default_dtype())


class ChebyshevBasis(_Basis):
    r"""Chebyshev polynomials of the first kind on ``[-1, 1]``.

    Degree ``k`` is :math:`T_k(x) = \cos(k \arccos x)`, so every degree
    comes from one cosine over a ``(*, K_c)`` grid. The unscaled polynomials
    are orthogonal under the weight :math:`(1 - x^2)^{-1/2}`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        degrees = torch.arange(count, dtype=x.dtype, device=x.device)
        return torch.cos(degrees * torch.arccos(x.unsqueeze(-1)))


class LegendreBasis(ChebyshevBasis):
    r"""Legendre polynomials on ``[-1, 1]``.

    Each is a fixed combination of Chebyshev polynomials, with nonnegative
    coefficients that sum to one, so the basis is the Chebyshev one followed
    by a matrix product; see :func:`_chebyshev_to_legendre`. The polynomials
    are the standard, unscaled orthogonal ones.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    _from_chebyshev: Tensor

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        matrix = _chebyshev_to_legendre(max(self.num_components))
        self.register_buffer("_from_chebyshev", matrix, persistent=False)
        self._tabulate(grid_size)

    def _features(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        matrix = self._from_chebyshev[:count, :count]
        return super()._features(x, axis) @ matrix.T


class _LocalBasis(_Basis):
    """Compact kernels centered on ``linspace(-1, 1, K_c)``.

    Subclasses implement :meth:`_kernel` on offsets ``t`` measured in grid
    steps. The kernel must vanish for ``|t| >= radius``, so each coordinate
    touches at most ``2 * radius`` consecutive components; sparse mode
    evaluates only those taps, as :class:`Taps`, at scattered points and on the
    coordinates of a grid alike.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int,
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        if radius < 1 or min(self.num_components) < 2 * radius:
            raise ValueError(
                "radius must be >= 1 and num_components >= 2 * radius, got "
                f"radius {radius} and num_components {self.num_components}"
            )
        self.radius = int(radius)
        self.sparse = bool(sparse)
        self._tabulate(grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def _features(self, x: Tensor, axis: int) -> Tensor:
        return self._kernel(_offsets(x, self.num_components[axis]))

    def _evaluate(self, x: Tensor, axis: int) -> Tensor | Taps:
        if not self.sparse:
            return self.axis_features(x, axis)
        # The 2 * radius taps around each coordinate. Clamping keeps each window
        # inside the basis; taps shifted by the clamp lie outside the kernel's
        # support and evaluate to zero.
        size = self.num_components[axis]
        position = _grid_position(x, size)
        start = (position.floor() - (self.radius - 1)).clamp(0, size - 2 * self.radius)
        taps = torch.arange(2 * self.radius, device=x.device, dtype=x.dtype)
        values = self._normalize(self._kernel((position - start).unsqueeze(-1) - taps))
        return Taps(values, start.long(), size, torch.compiler.is_compiling())


class TriangleBasis(_LocalBasis):
    r"""Triangle (hat) basis for piecewise-linear interpolation on ``[-1, 1]``.

    Component ``k`` is :math:`\max(0, 1 - |x - \mu_k| / w)` with centers spaced
    by :math:`w = 2 / (K_c - 1)`, so at most two components are nonzero. With
    ``normalize=False``, a linear map of these features equals sampling a 1D
    feature grid with ``align_corners=True``; a CP combiner and MLP decoder
    then give the TensoRF-CP parametrization.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: L2-normalize each feature vector.
        sparse: Evaluate only the two taps of each coordinate, as :class:`Taps`,
            per point off a grid and per grid coordinate on one.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or :class:`Taps`
          in sparse mode, off a grid and as the tables of :class:`SharedRows`.

    References:
        Chen et al., "TensoRF: Tensorial Radiance Fields", ECCV 2022.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, 1, normalize, sparse, grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_LocalBasis):
    r"""Windowed-sinc (Lanczos) basis centered on a uniform grid over ``[-1, 1]``.

    With grid offset :math:`t = (x - \mu_k) / w` and radius :math:`a`, the
    kernel is :math:`\operatorname{sinc}(t) \operatorname{sinc}(t / a)` for
    :math:`|t| < a` and zero elsewhere. Each component equals one at its own
    center and zero at all other centers; at most ``2 * radius`` are nonzero.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2 * radius``, shared or per
            axis.
        radius: Kernel radius in grid steps: ``2`` gives Lanczos-2 (four taps),
            ``3`` (the default) gives Lanczos-3 (six taps).
        normalize: L2-normalize each feature vector.
        sparse: Evaluate only the ``2 * radius`` taps of each coordinate, as
            :class:`Taps`, per point off a grid and per grid coordinate on one.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` that every input
            lies on, shared or per axis, where the features are evaluated once;
            see :class:`FUTONEncoding`.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or :class:`Taps`
          in sparse mode, off a grid and as the tables of :class:`SharedRows`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int = 3,
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            in_features, num_components, radius, normalize, sparse, grid_size
        )

    def _kernel(self, t: Tensor) -> Tensor:
        window = torch.sinc(t) * torch.sinc(t / self.radius)
        return torch.where(t.abs() < self.radius, window, 0.0)


# Combiners ----------------------------------------------------------------------------


class _Combiner(nn.Module):
    """Base class for combiners; ``in_features[c]`` is the width ``K_c`` of axis ``c``.

    The features of an axis are dense, :class:`SharedRows` of a grid or
    :class:`Taps` of a local basis, and all three multiply a matrix with ``@``.
    """

    out_features: int

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__()
        self.in_features = list(in_features)


class HadamardCombiner(_Combiner):
    """Elementwise product of per-axis features that share one width ``K``.

    Shared rows and taps are expanded to their dense features first.

    Args:
        in_features: Feature width of each axis; all entries must be equal.

    Shape:
        - Input: ``C`` tensors of shape :math:`(*, K)`.
        - Output: :math:`(*, K)`.
    """

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__(in_features)
        if len(set(self.in_features)) != 1:
            raise ValueError(
                f"Expected all in_features entries to be equal, got {self.in_features}"
            )
        self.out_features = self.in_features[0]

    def forward(self, features: Sequence[Features]) -> Tensor:
        dense = (f if isinstance(f, Tensor) else f.dense() for f in features)
        return reduce(torch.mul, dense)


class CPCombiner(_Combiner):
    r"""Canonical polyadic (CP) combiner.

    Projects each axis to ``rank`` channels with its CP factor
    :math:`\mathbf{U}^{(c)}`, ``factors[c]`` of shape ``(K_c, rank)`` (FUTON
    paper, Eqs. 10–11), and multiplies the projections elementwise.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: CP rank, which is also the output width.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank})`.

    References:
        Kolda and Bader, "Tensor Decompositions and Applications", SIAM Review 2009.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank
        self.factors = nn.ParameterList(
            nn.Parameter(torch.empty(width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factors Kaiming-uniform, with fan-in ``K_c``."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)

    def forward(self, features: Sequence[Features]) -> Tensor:
        return reduce(torch.mul, (f @ u for f, u in zip(features, self.factors)))


class TRCombiner(_Combiner):
    r"""Tensor-ring (TR) combiner.

    Axis ``c`` contracts its features with a core of shape
    ``(rank, K_c, rank)`` into a ``rank x rank`` matrix. The ordered product of
    these matrices is flattened to ``rank**2`` features, leaving the ring
    closure (e.g. a trace) to the decoder. This extends the paper's CP model.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Tensor-ring rank; the output width is ``rank**2``.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank}^2)`.

    References:
        Zhao et al., "Tensor Ring Decomposition", arXiv 2016.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank**2
        self.cores = nn.ParameterList(
            nn.Parameter(torch.empty(rank, width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the cores with Kaiming-uniform weights."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    def forward(self, features: Sequence[Features]) -> Tensor:
        rank = self.rank
        # Each axis contracts its features with its core, folded to (K, R * R).
        matrices = (
            (f @ core.transpose(0, 1).flatten(1)).unflatten(-1, (rank, rank))
            for f, core in zip(features, self.cores)
        )
        return reduce(torch.matmul, matrices).flatten(start_dim=-2)


class TuckerCombiner(_Combiner):
    r"""Tucker combiner.

    Projects axis ``c`` to ``rank[c]`` channels with its factor
    ``factors[c]`` of shape ``(K_c, rank[c])``, as the CP combiner does, and
    contracts the projections with a core of shape
    ``(rank[0], ..., rank[C - 1], out_features)`` instead of multiplying them
    elementwise: with a linear decoder the weight tensor is in Tucker format,
    the core with one factor matrix per axis and the decoder as the last.
    CP is the special case of a superdiagonal core. This extends the paper's
    CP model. The core absorbs one projection at a time, largest rank first.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Multilinear rank, one per axis or one for all.
        out_features: Width of the output, the core's last mode; ``None``
            takes the largest rank.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{out\_features})`.

    References:
        Tucker, "Some mathematical notes on three-mode factor analysis",
        Psychometrika 1966.
    """

    def __init__(
        self,
        in_features: Sequence[int],
        rank: int | Sequence[int],
        out_features: int | None = None,
    ) -> None:
        super().__init__(in_features)
        modes = len(self.in_features)
        self.rank = _broadcast(rank, modes, "rank")
        self.out_features = max(self.rank) if out_features is None else out_features
        self.factors = nn.ParameterList(
            nn.Parameter(torch.empty(width, r))
            for width, r in zip(self.in_features, self.rank)
        )
        self.core = nn.Parameter(torch.empty(*self.rank, self.out_features))
        # The core absorbs one projection at a time, largest rank first, so
        # the intermediates stay smallest and lead with the points (the
        # ellipsis); opt_einsum's own optimum starts with an outer product of
        # the projections, slower and larger. Each result goes to the end of
        # the operands, hence the path; given the path, the points' count of 1
        # is a placeholder.
        self.order = sorted(range(modes), key=self.rank.__getitem__, reverse=True)
        self.contract = oe.contract_expression(
            tuple(self.core.shape),
            [*range(modes + 1)],
            *(arg for c in self.order for arg in ((1, self.rank[c]), [..., c])),
            [..., modes],
            optimize=[(0, 1)] + [(0, n) for n in range(modes - 1, 0, -1)],
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factors (fan-in ``K_c``) and the core Kaiming-uniform."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)
        nn.init.kaiming_uniform_(self.core.view(-1, self.out_features))

    def forward(self, features: Sequence[Features]) -> Tensor:
        projections = [f @ factor for f, factor in zip(features, self.factors)]
        return self.contract(self.core, *(projections[c] for c in self.order))


# FUTON --------------------------------------------------------------------------------


class FUTONEncoding(nn.Module):
    r"""FUTON's encoding: the features of a basis, fused by a combiner.

    Computes ``combiner(basis(x))``, which :class:`FUTON` decodes. Each stage is
    given as for :class:`FUTON`.

    Args:
        in_features: Number of coordinate axes ``C``.
        basis: Basis spec, built as ``basis(in_features, **kwargs)``. Custom
            bases must expose ``num_components``, one count per axis.
        combiner: Combiner spec, built as
            ``combiner(basis.num_components, **kwargs)``. Custom combiners must
            expose ``out_features``, the encoding's width.

    Shape:
        - Input: :math:`(N, C)` coordinates in ``[-1, 1]``, a flat batch,
          whose points the basis shares or sparsifies.
        - Output: :math:`(N, \text{out\_features})`.

    Evaluation on a grid:
        A projection :math:`\mathbf{U}^{(c)\top} \boldsymbol{\varphi}(x_c)`
        depends on one coordinate alone, so points on a grid, such as the
        pixels of an image or a batch drawn from them, share it. When every
        axis repeats its coordinates, the basis evaluates each distinct
        coordinate once and the combiner projects that table, each point
        gathering its rows, as :class:`~neurofield.models.finr.FINR` does with
        its networks. This computes the same function and gradients in
        :math:`O(R \sum_c S_c K_c + NCR)` for :math:`N` points with
        :math:`S_c` distinct coordinates on axis :math:`c`, instead of
        :math:`O(NR \sum_c K_c)`; a local basis's tables hold only its
        :math:`L` taps per coordinate, :math:`O(RL \sum_c S_c + NCR)` instead
        of :math:`O(NCRL)`. A basis given ``grid_size``, the grid that every input
        lies on, as a coordinate dataset's points do, evaluates its table there
        once, when it is built, and rounds each point to its row, which spares
        searching every batch for its distinct coordinates, a sort and a
        synchronization per axis. A point off the grid would read its nearest
        grid point, so a radiance field, whose samples are scattered, declares
        none. Scattered points and coordinates that need gradients are
        evaluated point by point.

        The tables are buffers outside the model's ``state_dict``. Moving or
        casting the model, as ``to`` and ``double`` do, evaluates them again on
        its new device and in its new dtype, so they are exact in any precision.
    """

    def __init__(self, in_features: int, basis: ModuleSpec, combiner: ModuleSpec) -> None:
        super().__init__()
        self.in_features = in_features
        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.num_components)
        self.out_features = self.combiner.out_features

    def forward(self, x: Tensor) -> Tensor:
        features = self.basis(x)
        return self.combiner(features)


class FUTON(nn.Module):
    r"""Fourier Tensor Network for implicit neural representations.

    Computes ``output_activation(decoder(encoding(x)))``, where ``encoding`` is
    the :class:`FUTONEncoding` ``combiner(basis(x))``. Each stage is a module
    instance, a registry key, a module class, or a ``(key_or_class, kwargs)``
    pair. Registry keys:

    - Bases: ``"cosine"``, ``"sinc"``, ``"legendre"``, ``"chebyshev"``,
      ``"triangle"``, ``"lanczos"``.
    - Combiners: ``"hadamard"``, ``"cp"``, ``"tr"``, ``"tucker"``.
    - Decoders: ``"linear"``, ``"mlp"``.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels.
        basis: Basis spec of the encoding; see :class:`FUTONEncoding`.
        combiner: Combiner spec of the encoding; see :class:`FUTONEncoding`.
        decoder: Decoder spec, built as
            ``decoder(encoding.out_features, out_features, **kwargs)``.
        output_activation: Callable applied to the output; ``None`` uses the
            identity.

    Shape:
        - Input: :math:`(*, C)` coordinates in ``[-1, 1]``.
        - Output: :math:`(*, \text{out\_features})`.

    Relation to the paper:
        The cosine basis, CP combiner, and linear decoder implement Eq. (9):
        ``encoding.combiner.factors[c]`` stores :math:`\mathbf{U}^{(c)}` and
        ``decoder.weight`` stores :math:`\mathbf{V}`; the decoder bias is an
        addition. Coordinates are mapped internally from ``[-1, 1]`` to
        ``[0, 1]``. The basis omits the paper's :math:`\sqrt{2}` factor and
        L2-normalizes features by default; ``normalize=False`` recovers Eq. (1)
        up to per-frequency constants absorbed by the factors. Experiments use
        ``torch.tanh`` at the output. The other bases, the Hadamard and
        tensor-ring combiners, and the MLP decoder are extensions.

    Example::

        model = FUTON(
            in_features=2,
            out_features=3,
            basis=("cosine", {"num_components": 256}),
            combiner=("cp", {"rank": 256}),
            decoder="linear",
            output_activation=torch.tanh,
        )
        rgb = model(torch.rand(64, 64, 2) * 2 - 1)  # (64, 64, 3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: ModuleSpec,
        combiner: ModuleSpec,
        decoder: ModuleSpec,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoding = FUTONEncoding(in_features, basis, combiner)
        self.decoder = build_module(
            decoder, DECODERS, self.encoding.out_features, out_features
        )
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        # The encoding takes a flat batch, whose points its basis shares or sparsifies.
        points = x.reshape(-1, self.in_features)
        fused = self.encoding(points)
        out = self.output_activation(self.decoder(fused))
        return out.reshape(*x.shape[:-1], self.out_features)


# Registries ---------------------------------------------------------------------------

BASES: dict[str, type[nn.Module]] = {
    "cosine": CosineBasis,
    "sinc": SincBasis,
    "legendre": LegendreBasis,
    "chebyshev": ChebyshevBasis,
    "triangle": TriangleBasis,
    "lanczos": LanczosBasis,
}

COMBINERS: dict[str, type[nn.Module]] = {
    "hadamard": HadamardCombiner,
    "cp": CPCombiner,
    "tr": TRCombiner,
    "tucker": TuckerCombiner,
}

DECODERS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
}
