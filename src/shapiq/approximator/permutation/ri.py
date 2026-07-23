"""This module implements the Permutation Sampling approximator for the RI index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import FloatVector, IntVector, Model

ValidPermutationRIIndices = Literal["RI"]


class PermutationSamplingRI(Approximator[ValidPermutationRIIndices]):
    """Permutation Sampling approximator for the RI index.
    """

    #: override the valid indices for this approximator
    valid_indices: tuple[ValidPermutationRIIndices, ...] = tuple(
        get_args(ValidPermutationRIIndices)
    )
    """The valid indices for this permutation sampling approximator."""

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: ValidPermutationRIIndices = "RI",
        
        *,
        approximator: Approximator | None = None,
        top_order: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the Permutation Sampling approximator for RI.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The interaction index to compute. Must be RI.

            top_order: Whether to approximate only the top order interactions (``True``) or all
                orders up to the specified order (``False``, default).

            random_state: The random state to use for the permutation sampling. Defaults to
                ``None``.

        """
        if index not in ["RI"]:
            msg = f"Invalid index {index}. Must be RI"
            raise ValueError(msg)
        self.approximator=approximator
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            approximator=approximator
        )
        self.iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Compute the cost of a single iteration of the permutation sampling.

        Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the RI index.

        Returns:
            int: The cost of a single iteration.

        """
        iteration_cost: int = 0
        min_order = 1 if not self.top_order else self.max_order
        for s in range(min_order, self.max_order + 1):
            iteration_cost += (self.n - s + 1) * 2**s
        return iteration_cost

    def _compute_order_iterator(self) -> np.ndarray:
        """Computes the order iterator for the RI index.

        Returns:
            np.ndarray: The order iterator.

        """
        min_order = 1 if not self.top_order else self.max_order
        return np.arange(min_order, self.max_order + 1)
    def approximate(
        self,
        budget: int,
        x: np.ndarray,
        data: np.ndarray,
        game: Game | Callable[[np.ndarray], np.ndarray],
        batch_size: int | None = 5,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation

            game: The game function as a callable that takes a set of players and returns the value.

            batch_size: The size of the batch. If ``None``, the batch size is set to ``1``.
                Defaults to ``5``.

            *args: Additional positional arguments (not used, only for compatibility).

            **kwargs: Additional keyword arguments (not used, only for compatibility).

        Returns:
            The estimated interaction values.

        """
        ksii_interaction_values = np.asarray(self.approximator.approximate(x=x, budget=budget, game=game))
        print(self.approximator)
        result: FloatVector = self._init_result()
        counts: IntVector = self._init_result(dtype=int)

        batch_size = 1 if batch_size is None else batch_size

        # store the values of the empty and full coalition
        # this saves 2 evaluations per permutation
        empty_val = float(game(np.zeros(self.n, dtype=bool))[0])
        full_val = float(game(np.ones(self.n, dtype=bool))[0])


        used_budget = 2

        # catch special case of single player game, otherwise iteration through permutations fails
        if self.n == 1:
            interaction_index = self._interaction_lookup[self._grand_coalition_tuple]
            result[interaction_index] = full_val - empty_val
            counts[interaction_index] = 1

            return InteractionValues(
                values=result,
                interaction_lookup=self._interaction_lookup,
                baseline_value=empty_val,
                min_order=self.min_order,
                max_order=self.max_order,
                n_players=self.n,
                index=self.approximation_index,
                estimated=True,
                estimation_budget=used_budget,
                target_index=self.index,
            )

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - 2,
            batch_size,
            self.iteration_cost,
        )
        coalitions = data

        result: FloatVector = self._init_result()
        span_mean: FloatVector = self._init_result()
        counts: IntVector = self._init_result(dtype=int)
        
        interaction_index = self.n+1
        for u in range(1,self.n+1):
            for v in range(1,self.n+1):
                if v<=u:
                    continue
                else:
                    #print(u,v,interaction_index)
                    #print(ksii_interaction_values[u],ksii_interaction_values[v],ksii_interaction_values[interaction_index])
                    result[interaction_index]=ksii_interaction_values[interaction_index]/(ksii_interaction_values[interaction_index]+ksii_interaction_values[u]+ksii_interaction_values[v])
                                
                    interaction_index+=1
        result = np.divide(result, counts, out=result, where=counts != 0)
        span_mean = np.divide(span_mean, counts, out=span_mean, where=counts != 0)
        for i in range(len(span_mean)):
            span_mean[i]+=1
        result = np.divide(result, span_mean, out=result, where=span_mean != 0)

        return InteractionValues(
            values=result,
            interaction_lookup=self._interaction_lookup,
            baseline_value=empty_val,
            min_order=2,
            max_order=2,
            n_players=self.n,
            index=self.approximation_index,
            estimated=True,
            estimation_budget=used_budget,
            target_index=self.index,
        )