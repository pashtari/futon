"""F-INR: a functional tensor decomposition, one SIREN per input axis."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .siren import SineLayer

__all__ = ["FINR"]


class FINR(nn.Module):
    """Functional tensor decomposition with SIREN axes (Vemuri et al., WACV 2026).

    One network per input coordinate maps that coordinate alone to factors,
    and a tensor format combines the factors of all coordinates, per output
    channel, with no decoder after it:

    - ``"cp"``: ``y_c = sum_r prod_i a_i[c, r]``, for any number of inputs;
    - ``"tt"``: for three inputs, the tensor train ``y_c = b[c]^T G[c] d[c]``,
      whose ``R x R`` core comes from the first coordinate's network, as in
      the authors' code.

    Each axis network is a SIREN as in the authors' image code:
    ``hidden_layers`` sine layers ``sin(omega (W x + b))``, with ``omega``
    ``first_omega`` in the first and ``hidden_omega`` after, and a linear
    head. Weights are drawn from ``U(-1/n, 1/n)`` in the first layer and from
    ``U(-sqrt(6/n) / hidden_omega, sqrt(6/n) / hidden_omega)`` in the later
    layers and the head, ``n`` being the fan-in, and every bias is zero.

    ``positional_encoding`` puts the authors' positional encoding before each
    network, as their image experiments do: ``sin(2^k x)`` and then
    ``cos(2^k x)`` for ``k`` from ``int(-(n - 1) / 2)`` up to, but not
    including, ``int((n + 1) / 2)``, computed as their code does, so that
    ``n = 10`` gives the nine bands ``2^-4, ..., 2^4``, and no raw coordinate.

    On a grid, each network runs once per distinct value of its coordinate in
    a batch, and its factors are gathered per point: the output is the same
    as running it per point, and it is the separable evaluation the
    decomposition is made for.

    Args:
        in_features: Number of input coordinates, one network each.
        out_features: Number of output channels.
        rank: Rank ``R`` of the decomposition.
        hidden_features: Width of each axis network.
        hidden_layers: Number of sine layers of each axis network; at least one.
        mode: ``"cp"``, or ``"tt"`` for three inputs.
        first_omega: Frequency of the first sine layer.
        hidden_omega: Frequency of the later sine layers.
        positional_encoding: The authors' ``pos_enc``, the size of the
            positional encoding before each network; 0 feeds it the raw
            coordinate.
        grid: Whether inputs lie on a grid, so that a coordinate takes few
            distinct values in a batch; off for scattered points, such as a
            radiance field's samples, where finding them would only cost time.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        mode: str = "cp",
        first_omega: float = 100.0,
        hidden_omega: float = 30.0,
        positional_encoding: int = 0,
        grid: bool = True,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("cp", "tt"):
            raise ValueError(f"mode must be 'cp' or 'tt', got {mode!r}")
        if mode == "tt" and in_features != 3:
            raise ValueError("the tensor train combines exactly three inputs")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.mode = mode
        self.hidden_omega = hidden_omega
        self.grid = grid
        self.output_activation = output_activation

        bands = torch.empty(0)
        if positional_encoding > 0:
            low = int(-(positional_encoding - 1) / 2)  # int rounds toward zero
            bands = 2.0 ** torch.arange(low, int((positional_encoding + 1) / 2))
        self.register_buffer("bands", bands, persistent=False)
        encoded = 2 * len(bands) if len(bands) else 1

        sizes = [out_features * rank] * in_features
        if mode == "tt":
            sizes[0] = out_features * rank * rank
        self.axes = nn.ModuleList(
            nn.Sequential(
                SineLayer(encoded, hidden_features, first_omega),
                *(
                    SineLayer(hidden_features, hidden_features, hidden_omega)
                    for _ in range(hidden_layers - 1)
                ),
                nn.Linear(hidden_features, size),
            )
            for size in sizes
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize as the authors' image code does, biases at zero."""
        for network in self.axes:
            for index, layer in enumerate(network):
                linear = layer.linear if isinstance(layer, SineLayer) else layer
                fan_in = linear.in_features
                bound = (
                    1 / fan_in
                    if index == 0
                    else math.sqrt(6 / fan_in) / self.hidden_omega
                )
                linear.weight.uniform_(-bound, bound)
                linear.bias.zero_()

    def _encode(self, coordinate: Tensor) -> Tensor:
        """A ``(n,)`` coordinate as a network's ``(n, features)`` input."""
        if not len(self.bands):
            return coordinate[:, None]
        angles = coordinate[:, None] * self.bands
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def _factors(self, network: nn.Module, coordinate: Tensor) -> Tensor:
        """``network`` on each point's coordinate, once per distinct value on a grid."""
        if self.grid and not coordinate.requires_grad:
            values, inverse = torch.unique(coordinate, return_inverse=True)
            if 2 * len(values) <= len(coordinate):
                return network(self._encode(values))[inverse]
        return network(self._encode(coordinate))

    def forward(self, x: Tensor) -> Tensor:
        points = x.reshape(-1, self.in_features)
        count, channels, rank = len(points), self.out_features, self.rank
        factors = [
            self._factors(network, points[:, axis])
            for axis, network in enumerate(self.axes)
        ]
        if self.mode == "cp":
            out = factors[0].view(count, channels, rank)
            for factor in factors[1:]:
                out = out * factor.view(count, channels, rank)
            out = out.sum(-1)
        else:
            core = factors[0].view(count, channels, rank, rank)
            left = factors[1].view(count, channels, rank)
            right = factors[2].view(count, channels, rank)
            out = torch.einsum("ncpq,ncp,ncq->nc", core, left, right)
        out = out.reshape(*x.shape[:-1], channels)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out
