"""Block coordinate descent for FUTON: solvers in place of a gradient optimizer.

A FUTON with a CP combiner and a linear decoder, both without bias, is the
paper's model, and it is multilinear: with the other blocks held, its output
is linear in each factor matrix and in the decoder, so fitting one block is a
least-squares problem. :class:`BCD` sweeps the blocks once per batch and
leaves each block's update to a subclass: :class:`LeastSquares` solves the
block's problem, for any basis, and :class:`MultiplicativeUpdate` takes Lee
and Seung's step, which keeps a nonnegative model nonnegative. Both are
optimizers that :func:`neurofield.train` takes in place of Adam, on the same
datasets.

References:
    Lee and Seung, "Algorithms for non-negative matrix factorization",
    NeurIPS 2001.
    Gillis and Glineur, "Accelerated multiplicative updates and hierarchical
    ALS algorithms for nonnegative matrix factorization", Neural Computation
    2012.
    Mairal, Bach, Ponce and Sapiro, "Online learning for matrix factorization
    and sparse coding", JMLR 2010.
"""

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .futon import FUTON, CPCombiner

__all__ = ["BCD", "LeastSquares", "MultiplicativeUpdate"]

MODES = "ijklmn"  # einsum letters of a grid's axes and, last, its channels


def _cp(factors: Sequence[Tensor]) -> Tensor:
    """The CP tensor ``(M_1, ..., M_n)`` of factor matrices ``(M_k, R)``."""
    modes = MODES[: len(factors)]
    return torch.einsum(",".join(m + "r" for m in modes) + "->" + modes, *factors)


def _correlation(grid: Tensor, factors: Sequence[Tensor], mode: int) -> Tensor:
    """``grid`` contracted with every factor matrix but ``mode``'s: ``(M_mode, R)``."""
    modes = MODES[: grid.dim()]
    rest = [k for k in range(grid.dim()) if k != mode]
    operands = ",".join([modes, *(modes[k] + "r" for k in rest)])
    return torch.einsum(
        f"{operands}->{modes[mode]}r", grid, *(factors[k] for k in rest)
    )


@torch.compiler.disable()  # sizes that depend on the data
def _distinct(x: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Each axis's distinct coordinates and the row of every point among them."""
    return [torch.unique(u, return_inverse=True) for u in x.unbind(-1)]


def ridged(gram: Tensor, ridge: float) -> Tensor:
    """A Gram matrix with ``ridge`` times its mean diagonal added to the diagonal."""
    eye = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    return gram + ridge * gram.diagonal().mean() * eye


class BCD(Optimizer):
    """Block coordinate descent over a FUTON's factor matrices and decoder.

    On a grid with ``M_c`` coordinates along axis ``c`` the model is a CP
    tensor, with one factor per axis, the response ``T_c A_c`` of the basis's
    table ``T_c`` ``(M_c, K_c)`` and the factor matrix ``A_c`` ``(K_c, R)``,
    and one for the channels, the decoder ``W`` ``(D, R)``. With the other
    blocks held, the mean squared error's normal equations for ``A_c`` are
    ``P A_c Q = B``, small and of the size of ``A_c``: ``P = T_c^T T_c / M_c``
    is the basis's Gram matrix, ``Q`` the elementwise product of ``W^T W`` and
    of the other axes' Gram matrices ``(T_j A_j)^T (T_j A_j) / M_j``, and
    ``B`` the targets' correlation with the rest of the model. The decoder's
    are ``W G = C``, with ``G`` the product of every axis's Gram matrix.

    A batch is taken as points of the grid the basis declares, or else of the
    grid of its distinct coordinates. The Gram matrices are the grid's, and a
    right-hand side is what the model explains on the grid plus what it misses
    at the batch's points, ``B = P A_c Q + T_c^T E_c / N`` for the residual's
    correlation ``E_c`` over the ``N`` points: exact on a whole grid and, on a
    random share of it, the whole grid's in expectation, with the noise of the
    residual rather than of the signal. A sweep thus costs a few products of
    matrices over the grid whatever share of it the batch holds, and a whole
    grid is where the solvers need no momentum and the fewest sweeps. There,
    without momentum, no update raises the batch's error.

    :meth:`step` updates the factors in order and then the decoder, each from
    the others' latest values, through a subclass's :meth:`solve`, and holds
    every axis's response at unit mean square, the decoder's columns taking
    the scales, which fixes the freedom of a CP model to trade scale between
    its blocks. :meth:`compile` compiles the sweep with ``torch.compile``.

    ``momentum`` averages each block's statistics over the batches, as online
    matrix factorization does, with weights that sum to one from the first
    batch on. The newest batch's weight, ``1 - momentum``, is the solver's
    ``lr``, which a scheduler anneals as it would a gradient step's:
    :func:`neurofield.train` takes it down to a hundredth, so that the
    average reaches further back as the model settles.

    Args:
        model: A FUTON with a CP combiner and a linear decoder, both without
            bias, whose basis evaluates one axis at a time through
            ``axis_features``, as the built-in ones do. Its parameters are
            updated in place.
        momentum: Weight of the past batches in a block's statistics; ``0``
            takes each batch on its own, which suits whole grids.
        initialize: Draw the parameters anew with :meth:`reset_parameters`;
            ``False`` keeps the model's.
        **defaults: A subclass's settings, kept in ``self.defaults`` with the
            others.
    """

    def __init__(
        self,
        model: FUTON,
        momentum: float = 0.9,
        initialize: bool = True,
        **defaults: Any,
    ) -> None:
        if not (
            isinstance(model, FUTON)
            and isinstance(combiner := model.encoding.combiner, CPCombiner)
            and isinstance(decoder := model.decoder, nn.Linear)
            and decoder.bias is None
        ):
            raise ValueError(
                "block coordinate descent fits a FUTON with a CP combiner and a"
                " linear decoder without bias"
            )
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        # The blocks, which the solver reads here rather than from param_groups,
        # whose lr changes at every step and would make a compiled sweep recompile.
        self._factors, self._weight = combiner.factors, decoder.weight
        settings = {"lr": 1 - momentum, "initialize": initialize}
        super().__init__([*self._factors, self._weight], settings | defaults)
        self.model = model
        self._sweep = self._sweep_eagerly
        if initialize:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Draw the parameters as the solver wants them.

        A subclass draws the entries; each axis's response is then scaled to
        about one in the middle of the domain and the decoder sums the ``R``
        components to about one half, so that the first sweep starts from an
        output of the signal's order.
        """
        basis: Any = self.model.encoding.basis  # any basis with axis_features
        for c, factor in enumerate(self._factors):
            middle = basis.axis_features(factor.new_zeros(1), c)
            factor.copy_(self._draw(factor) / middle.sum())
        self._weight.copy_(self._draw(self._weight) / (2 * self._weight.shape[1]))

    def _draw(self, param: Tensor) -> Tensor:
        """A parameter's entries drawn anew, before scaling."""
        raise NotImplementedError

    def _left(self, table: Tensor) -> Tensor:
        """An axis's basis Gram matrix ``P``, as :meth:`solve` takes it."""
        return table.T @ table / len(table)

    def _check(self, x: Tensor, y: Tensor) -> None:
        """Check the first batch, its points ``x`` and values ``y``, before its
        sweep; the base class has no checks."""

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        """A block's new value, from its problem ``left @ value @ gram = right``.

        Args:
            value: The block, ``(K_c, R)`` for a factor and ``(D, R)`` for the
                decoder.
            right: The right-hand side, of the block's shape.
            gram: The other blocks' Gram matrix ``Q`` or ``G``, ``(R, R)``.
            left: The basis's Gram matrix as :meth:`_left` gives it, or
                ``None``, the identity, for the decoder.
        """
        raise NotImplementedError

    def compile(
        self, batch: Mapping[str, Tensor] | None = None, **options: Any
    ) -> None:
        """Compile the sweep with ``torch.compile``, given ``options``.

        With a ``batch``, compile it now, on that batch's shapes, and leave the
        model and the solver as they were, so that no timed step pays for it.
        """
        self._sweep = torch.compile(self._sweep_eagerly, **options)
        if batch is not None:
            params = [*self._factors, self._weight]
            saved = [p.detach().clone() for p in params]
            self.step(batch)
            with torch.no_grad():
                for param, value in zip(params, saved):
                    param.copy_(value)
            self.state.clear()

    @torch.no_grad()
    def step(self, batch: Mapping[str, Tensor]) -> Tensor:  # type: ignore[override]
        """One sweep over the factor matrices and the decoder on a batch.

        Args:
            batch: ``input``, coordinates ``(*, C)`` in [-1, 1], and
                ``target``, their values ``(*, D)``, as a dataset yields them.
                The points must lie on a grid, as a coordinate dataset's do.

        Returns:
            The batch's mean squared error before the sweep.
        """
        x = batch["input"].reshape(-1, self.model.in_features)
        y = batch["target"].reshape(-1, self.model.out_features)
        if not self.state:
            self._start(x)
            self._check(x, y)
        self._lr.fill_(self.param_groups[0]["lr"])
        return self._sweep(x, y)

    def _start(self, x: Tensor) -> None:
        """Every block's averages, which start at zero, before the first batch.

        On a declared grid the tables are fixed, so each axis's left-hand side
        is too, and is made once.
        """
        self._lr = x.new_zeros(())
        square = (self._weight.shape[1],) * 2
        for param in [*self._factors, self._weight]:  # a block's (right, gram)
            statistics = (torch.zeros_like(param), x.new_zeros(square))
            self.state[param] = {"total": x.new_zeros(()), "statistics": statistics}
        if getattr(self.model.encoding.basis, "grid_size", None) is not None:
            for factor, table in zip(self._factors, self._grid(x)[0]):
                self.state[factor]["left"] = self._left(table)

    def _average(self, param: Tensor, *statistics: Tensor) -> tuple[Tensor, ...]:
        """A block's statistics, averaged over the batches seen."""
        state = self.state[param]
        # The batches' weights sum to one, whatever the step size was at each.
        state["total"].mul_(1 - self._lr).add_(self._lr)
        weight = self._lr / state["total"]
        for average, new in zip(state["statistics"], statistics):
            average.lerp_(new, weight)
        return state["statistics"]

    def _grid(self, x: Tensor) -> tuple[list[Tensor], tuple[Tensor, ...]]:
        """Each axis's table, over the grid the basis declares or else over the
        batch's distinct coordinates, and every point's row in it."""
        basis: Any = self.model.encoding.basis  # any basis with axis_features
        if getattr(basis, "grid_size", None) is not None:
            features = basis(x)
            tables = [
                f.table if isinstance(f.table, Tensor) else f.table.dense()
                for f in features
            ]
            return tables, tuple(f.rows for f in features)
        distinct = _distinct(x)
        tables = [basis.axis_features(u, c) for c, (u, _) in enumerate(distinct)]
        return tables, tuple(rows for _, rows in distinct)

    def _targets(self, y: Tensor, output: Tensor) -> Tensor:
        """What the decoder's output should fit for the model's to approach ``y``.

        Without an output activation the values themselves. With one, the
        Gauss-Newton target from the current ``output``: the output plus the
        residual over the activation's slope, with the squared slopes bounded
        by their largest so that every block stays an unweighted
        least-squares problem. For an affine activation it is exact. The
        activation acts elementwise, so its slopes are one forward derivative.
        """
        activation = self.model.output_activation
        if isinstance(activation, nn.Identity):
            return y
        value, slope = torch.func.jvp(
            activation, (output,), (torch.ones_like(output),)
        )[:2]
        return output + slope * (y - value) / slope.square().max()

    def _sweep_eagerly(self, x: Tensor, y: Tensor) -> Tensor:
        count = len(x)
        tables, rows = self._grid(x)
        factors, weight = self._factors, self._weight
        # On the grid the output is a CP tensor, whose factors are the axes'
        # responses and the decoder; the points pick their entries.
        modes = [table @ factor for table, factor in zip(tables, factors)] + [weight]
        grams = [m.T @ m / len(m) for m in modes[:-1]]
        output = _cp(modes)[rows]
        loss = (self.model.output_activation(output) - y).square().mean()
        targets = self._targets(y, output)

        def missed(output: Tensor) -> Tensor:
            """What the model misses at the points, summed into the grid."""
            grid = y.new_zeros(*(len(table) for table in tables), y.shape[1])
            return grid.index_put_(rows, targets - output, accumulate=True)

        error = missed(output)
        for c, (table, factor) in enumerate(zip(tables, factors)):
            gram = reduce(torch.mul, grams[:c] + grams[c + 1 :], weight.T @ weight)
            # What the model explains, from the Gram matrices, and what it
            # misses, from the batch; see the class's notes.
            explained = modes[c] @ gram / len(table)
            right = table.T @ (explained + _correlation(error, modes, c) / count)
            right, gram = self._average(factor, right, gram)
            left = self.state[factor].get("left")  # made once on a declared grid
            if left is None:
                left = self._left(table)
            value = self.solve(factor, right, gram, left)
            # A component scaled in a factor and back in the decoder leaves
            # the model as it was: unit mean square responses fix the scales.
            response = table @ value
            scale = response.square().mean(0).sqrt()
            scale = torch.where(scale > 0, scale, 1.0)
            factor.copy_(value / scale)
            weight.mul_(scale)
            modes[c] = response / scale
            grams[c] = modes[c].T @ modes[c] / len(table)
            error = missed(_cp(modes)[rows])
        gram = reduce(torch.mul, grams)
        right = weight @ gram + _correlation(error, modes, len(tables)) / count
        right, gram = self._average(weight, right, gram)
        weight.copy_(self.solve(weight, right, gram))
        return loss


class LeastSquares(BCD):
    """Alternating least squares: every block takes its problem's solution.

    The problem ``P A Q = B`` is solved one side at a time, each with a ridge
    against collinear functions or components: from the left by the inverse
    of ``P``, which depends on the basis and the grid alone and is made once
    on a declared grid, and from the right by the Cholesky factorization of
    ``Q``. Any basis will do, signed or not. With an activation at the output
    the targets are Gauss-Newton ones, refitted every sweep, so the sweeps
    descend the activated model's error rather than minimize a fixed one. The
    parameters start from normal draws.

    Args:
        model: See :class:`BCD`.
        ridge: Added to the diagonal of each Gram matrix, relative to its
            mean diagonal.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.
    """

    def __init__(
        self,
        model: FUTON,
        ridge: float = 1e-3,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        super().__init__(model, momentum, initialize, ridge=ridge)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.randn_like(param)

    def _left(self, table: Tensor) -> Tensor:
        gram = ridged(super()._left(table), self.defaults["ridge"])
        return torch.cholesky_inverse(torch.linalg.cholesky_ex(gram).L)

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        if left is not None:
            right = left @ right
        # The Gram matrix is symmetric, so the right solve is a left one transposed.
        factor = torch.linalg.cholesky_ex(ridged(gram, self.defaults["ridge"])).L
        return torch.cholesky_solve(right.T, factor).T


class MultiplicativeUpdate(BCD):
    """Lee and Seung's multiplicative updates, for a nonnegative model.

    With everything nonnegative, the two parts of a block's gradient are too,
    and the update multiplies the block elementwise by their ratio,
    ``A <- A * B / (P A Q)``, or sets an entry to zero where a minibatch's
    estimate of ``B`` is negative. That is the minimizer of the auxiliary
    function Lee and Seung build for a quadratic with a nonnegative Hessian:
    the block's problem is never worse after the update, the block stays
    nonnegative, and one at a fixed point stays there. A block's problem is
    fixed for the sweep and an update costs two small products, so each block
    takes ``inner`` of them in a row, as Gillis and Glineur propose.

    The model needs a nonnegative basis, the triangle with
    ``normalize=False`` so that its functions sum to one, nonnegative targets
    and nonnegative parameters, which are drawn uniformly in [0.5, 1.5]
    before scaling. A signal in [-1, 1] is fitted through the affine output
    activation ``lambda z: 2 * z - 1``, under which the targets are its
    values rescaled to [0, 1].

    Args:
        model: See :class:`BCD`.
        inner: Updates of each block per sweep.
        floor: Lower bound on a denominator, against division by zero where a
            basis function sees no sample.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.
    """

    def __init__(
        self,
        model: FUTON,
        inner: int = 16,
        floor: float = 1e-12,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        if inner < 1:
            raise ValueError("inner must be at least 1")
        super().__init__(model, momentum, initialize, inner=inner, floor=floor)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.rand_like(param) + 0.5

    def _check(self, x: Tensor, y: Tensor) -> None:
        tables, rows = self._grid(x)
        params = [*self._factors, self._weight]
        output = _cp([t @ a for t, a in zip(tables, params)] + params[-1:])[rows]
        # The targets of an affine activation are exact up to a few roundings,
        # which may leave a zero slightly negative.
        lowest = [*tables, *params, self._targets(y, output) + 1e-5]
        if torch.stack([t.amin() for t in lowest]).amin() < 0:
            raise ValueError(
                "multiplicative updates need a nonnegative basis, nonnegative"
                " targets (an affine output activation, 2 z - 1, for a signal in"
                " [-1, 1]) and nonnegative parameters"
            )

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        floor = self.defaults["floor"]
        for _ in range(self.defaults["inner"]):
            denominator = value @ gram if left is None else left @ value @ gram
            # A coefficient no sample of the batch reaches has both parts at
            # zero and keeps its value; one whose right-hand side is negative,
            # which a minibatch's estimate can be, goes to zero, the
            # constrained minimizer of its auxiliary function.
            ratio = (right / denominator.clamp_min(floor)).clamp_min(0)
            value = value * torch.where(denominator > floor, ratio, 1.0)
        return value
