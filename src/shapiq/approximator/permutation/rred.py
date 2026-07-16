"""This module implements the Permutation Sampling approximator for the Rred index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game

ValidPermutationRredIndices = Literal["Rred"]


class PermutationSamplingRred(Approximator[ValidPermutationRredIndices]):
    """Permutation Sampling approximator for the Rred index.
    """

    #: override the valid indices for this approximator
    valid_indices: tuple[ValidPermutationRredIndices, ...] = tuple(
        get_args(ValidPermutationRredIndices)
    )
    """The valid indices for this permutation sampling approximator."""

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: ValidPermutationRredIndices = "Rred",
        *,
        top_order: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the Permutation Sampling approximator for Rred.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The interaction index to compute. Must be Rrred.

            top_order: Whether to approximate only the top order interactions (``True``) or all
                orders up to the specified order (``False``, default).

            random_state: The random state to use for the permutation sampling. Defaults to
                ``None``.

        """
        if index not in ["Rred"]:
            msg = f"Invalid index {index}. Must be Rred"
            raise ValueError(msg)
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
        )
        self.iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Compute the cost of a single iteration of the permutation sampling.

        Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the Rred index.

        Returns:
            int: The cost of a single iteration.

        """
        iteration_cost: int = 0
        min_order = 1 if not self.top_order else self.max_order
        for s in range(min_order, self.max_order + 1):
            iteration_cost += (self.n - s + 1) * 2**s
        return iteration_cost

    def _compute_order_iterator(self) -> np.ndarray:
        """Computes the order iterator for the Rred index.

        Returns:
            np.ndarray: The order iterator.

        """
        min_order = 1 if not self.top_order else self.max_order
        return np.arange(min_order, self.max_order + 1)

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        batch_size: int | None = 5,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the interaction values.

        Args:
            budget: The budget for the approximation.
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If ``None``, the batch size is set to ``1``. Defaults to ``5``.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            InteractionValues: The estimated interaction values.

        """
        batch_size = 1 if batch_size is None else batch_size
        used_budget = 0

        result = self._init_result()
        counts = self._init_result(dtype=int)

        empty_value = game(np.zeros(self.n, dtype=bool))[0]
        used_budget += 1

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - used_budget,
            batch_size,
            self.iteration_cost,
        )

        # main permutation sampling loop
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size

            # create the permutations: a 2d matrix of shape (batch_size, n) where each row is a
            # permutation of the players
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            n_permutations = permutations.shape[0]
            n_subsets = n_permutations * self.iteration_cost

            # get all subsets to evaluate per iteration
            subsets = np.zeros(shape=(n_subsets, self.n), dtype=bool)
            subset_index = 0
            for permutation_id in range(n_permutations):
                for order in self._compute_order_iterator():
                    for k in range(self.n - order + 1):
                        subset = permutations[permutation_id, k : k + order]
                        previous_subset = permutations[permutation_id, :k]
                        for subset_ in powerset(subset, min_size=0):
                            subset_eval = np.concatenate((previous_subset, subset_)).astype(int)
                            subsets[subset_index, subset_eval] = True
                            subset_index += 1

            # evaluate all subsets on the game
            game_values = game(subsets)

            # update the interaction scores by iterating over the permutations again
            subset_index = 0
            for permutation_id in range(n_permutations):
                for order in self._compute_order_iterator():
                    for k in range(self.n - order + 1):
                        interaction = permutations[permutation_id, k : k + order]
                        interaction = tuple(sorted(interaction))
                        interaction_index = self._interaction_lookup[interaction]
                        counts[interaction_index] += 1
                        # update the discrete derivative given the subset
                        for subset_ in powerset(interaction, min_size=0):
                            game_value = game_values[subset_index]
                            update = game_value * (-1) ** (order - len(subset_))
                            result[interaction_index] += update
                            subset_index += 1

            used_budget += self.iteration_cost * batch_size

        # compute mean of interactions
        result = np.divide(result, counts, out=result, where=counts != 0)

        return InteractionValues(
            n_players=self.n,
            values=result,
            index=self.approximation_index,
            interaction_lookup=self._interaction_lookup,
            baseline_value=empty_value,
            min_order=self.min_order,
            max_order=self.max_order,
            estimated=True,
            estimation_budget=used_budget,
            target_index=self.index,
        )
