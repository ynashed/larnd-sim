#Reproduced from https://github.com/ott-jax/ott/blob/4bba69fbc039026c9e96173873b133202587b9a3/src/ott/geometry/costs.py


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import abc
from typing import Optional, Callable, Union, Tuple, Any
import numpy as np

@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """Base class for all costs.

  Cost functions evaluate a function on a pair of inputs. For convenience,
  that function is split into two norms -- evaluated on each input separately --
  followed by a pairwise cost that involves both inputs, as in:

  .. math::
    c(x, y) = norm(x) + norm(y) + pairwise(x, y)

  If the :attr:`norm` function is not implemented, that value is handled as
  :math:`0`, and only :func:`pairwise` is used.
  """

  # no norm function created by default.
  norm: Optional[Callable[[jnp.ndarray], Union[float, jnp.ndarray]]] = None

  @abc.abstractmethod
  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost.
    """

  def barycenter(self, weights: jnp.ndarray,
                 xs: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
    """Barycentric operator.

    Args:
      weights: Convex set of weights.
      xs: Points.

    Returns:
      A list, whose first element is the barycenter of `xs` using `weights`
      coefficients, followed by auxiliary information on the convergence of
      the algorithm.
    """
    raise NotImplementedError("Barycenter is not implemented.")

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    """Create a padding vector of adequate dimension, well-suited to a cost.

    Args:
      dim: Dimensionality of the data.

    Returns:
      The padding vector.
    """
    return jnp.zeros((1, dim))

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost, optionally including the :attr:`norms <norm>` of
      :math:`x`/:math:`y`.
    """
    cost = self.pairwise(x, y)
    if self.norm is None:
      return cost
    return cost + self.norm(x) + self.norm(y)

  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs, including the :attr:`norms <norm>`.

    Args:
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def all_pairs_pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs, excluding the :attr:`norms <norm>`.

    Args:
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self.pairwise(x_, y_))(y))(x)

  def twist_operator(
      self, vec: jnp.ndarray, dual_vec: jnp.ndarray, variable: bool
  ) -> jnp.ndarray:
    r"""Twist inverse operator of the cost function.

    Given a cost function :math:`c`, the twist operator returns
    :math:`\nabla_{1}c(x, \cdot)^{-1}(z)` if ``variable`` is ``0``,
    and :math:`\nabla_{2}c(\cdot, y)^{-1}(z)` if ``variable`` is ``1``, for
    :math:`x=y` equal to ``vec`` and :math:`z` equal to ``dual_vec``.

    Args:
      vec: ``[p,]`` point at which the twist inverse operator is evaluated.
      dual_vec: ``[q,]`` point to invert by the operator.
      variable: apply twist inverse operator on first (i.e. value set to ``0``
        or equivalently ``False``) or second (``1`` or ``True``) variable.

    Returns:
      A vector.
    """
    raise NotImplementedError("Twist operator is not implemented.")

  def tree_flatten(self):  # noqa: D102
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)
  
@jax.tree_util.register_pytree_node_class
class TICost(CostFn):
  """Base class for translation invariant (TI) costs.

  Such costs are defined using a function :math:`h`, mapping vectors to
  real-values, to be used as:

  .. math::
    c(x, y) = h(z), z := x - y.

  If that cost function is used to form an Entropic map using the
  :cite:`brenier:91` theorem, then the user should ensure :math:`h` is
  strictly convex, as well as provide the Legendre transform of :math:`h`,
  whose gradient is necessarily the inverse of the gradient of :math:`h`.
  """

  @abc.abstractmethod
  def h(self, z: jnp.ndarray) -> float:
    """TI function acting on difference of :math:`x-y` to output cost.

    Args:
      z: Array of shape ``[d,]``.

    Returns:
      The cost.
    """

  def h_legendre(self, z: jnp.ndarray) -> float:
    """Legendre transform of :func:`h` when it is convex."""
    raise NotImplementedError("Legendre transform of `h` is not implemented.")

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost as evaluation of :func:`h` on :math:`x-y`."""
    return self.h(x - y)

  def twist_operator(
      self, vec: jnp.ndarray, dual_vec: jnp.ndarray, variable: bool
  ) -> jnp.ndarray:
    # Note: when `h` is pair, i.e. h(z) = h(-z), the expressions below coincide
    if variable:
      return vec + jax.grad(self.h_legendre)(-dual_vec)
    return vec - jax.grad(self.h_legendre)(dual_vec)
  
@jax.tree_util.register_pytree_node_class
class SqEuclidean(TICost):
  r"""Squared Euclidean distance.

  Implemented as a translation invariant cost, :math:`h(z) = \|z\|^2`.
  """

  def norm(self, x: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """Compute squared Euclidean norm for vector."""
    return jnp.sum(x ** 2, axis=-1)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    return -2.0 * jnp.vdot(x, y)

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.sum(z ** 2)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.25 * jnp.sum(z ** 2)

  def barycenter(self, weights: jnp.ndarray,
                 xs: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
    """Output barycenter of vectors when using squared-Euclidean distance."""
    return jnp.average(xs, weights=weights, axis=0), None

@jax.tree_util.register_pytree_node_class
class SoftDTW(CostFn):
  """Soft dynamic time warping (DTW) cost :cite:`cuturi:17`.

  Args:
    gamma: Smoothing parameter :math:`> 0` for the soft-min operator.
    ground_cost: Ground cost function. If ``None``,
      use :class:`~ott.geometry.costs.SqEuclidean`.
    debiased: Whether to compute the debiased soft-DTW :cite:`blondel:21`.
  """

  def __init__(
      self,
      gamma: float,
      ground_cost: Optional[CostFn] = None,
      debiased: bool = False
  ):
    self.gamma = gamma
    self.ground_cost = SqEuclidean() if ground_cost is None else ground_cost
    self.debiased = debiased

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:  # noqa: D102
    c_xy = self._soft_dtw(x, y)
    if self.debiased:
      return c_xy - 0.5 * (self._soft_dtw(x, x) + self._soft_dtw(y, y))
    return c_xy

  def _soft_dtw(self, t1: jnp.ndarray, t2: jnp.ndarray) -> float:

    def body(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        current_antidiagonal: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
      # modified from: https://github.com/khdlr/softdtw_jax
      two_ago, one_ago = carry

      diagonal, right, down = two_ago[:-1], one_ago[:-1], one_ago[1:]
      # jax.debug.print("{diagonal}, {right}, {down}", diagonal=diagonal, right=right, down=down)
      best = -self.gamma * jsp.special.logsumexp(jnp.stack([diagonal, right, down], axis=-1) / -self.gamma, axis=-1)

      next_row = best + current_antidiagonal
      next_row = jnp.pad(next_row, (1, 0), constant_values=1e10)

      return (one_ago, next_row), next_row

    t1 = t1[:, None] if t1.ndim == 1 else t1
    t2 = t2[:, None] if t2.ndim == 1 else t2
    dist = self.ground_cost.all_pairs(t1, t2)

    n, m = dist.shape
    if n < m:
      dist = dist.T
      n, m = m, n

    model_matrix = jnp.full((n + m - 1, n), fill_value=1e10)
    mask = np.tri(n + m - 1, n, k=0, dtype=bool)
    mask = mask & mask[::-1, ::-1]
    model_matrix = model_matrix.T.at[mask.T].set(dist.ravel()).T

    init = (
        jnp.pad(model_matrix[0], (1, 0), constant_values=1e10),
        jnp.pad(
            model_matrix[1] + model_matrix[0, 0], (1, 0),
            constant_values=1e10
        )
    )

    (_, carry), _ = jax.lax.scan(body, init, model_matrix[2:])
    # (_, carry), _ = body(init, model_matrix[0])
    return carry[-1]
    # return carry

  def tree_flatten(self):  # noqa: D102
    return (self.gamma, self.ground_cost), {"debiased": self.debiased}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)