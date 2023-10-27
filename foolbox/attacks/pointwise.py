from typing import Union, Any, Optional, Tuple, Callable, List
import eagerpy as ep
import numpy as np
import logging

from ..criteria import Criterion

from .base import FlexibleDistanceMinimizationAttack
from .saltandpepper import SaltAndPepperNoiseAttack

from ..devutils import flatten
from .base import Model
from .base import MinimizationAttack
from .base import get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds


class PointwiseAttack(FlexibleDistanceMinimizationAttack):
    """Starts with an adversarial and performs a binary search between
    the adversarial and the original for each dimension of the input
    individually. [#Sch18]_

    References:
        .. [#Sch18] Lukas Schott, Jonas Rauber, Matthias Bethge, Wieland Brendel,
               "Towards the first adversarially robust neural network model on MNIST",
               https://arxiv.org/abs/1805.09190
    """

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        l2_binary_search: bool = True,
    ):
        self.init_attack = init_attack
        self.l2_binary_search = l2_binary_search

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        starting_points: Optional[ep.Tensor] = None,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        del kwargs

        x, restore_type = ep.astensor_(inputs)
        del inputs

        verify_input_bounds(x, model)

        criterion_ = get_criterion(criterion)
        del criterion
        is_adversarial = get_is_adversarial(criterion_, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = SaltAndPepperNoiseAttack()
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)
            starting_points = init_attack.run(model, x, criterion_)

        x_adv = ep.astensor(starting_points)
        assert is_adversarial(x_adv).all()

        original_shape = x.shape
        N = len(x)

        x_flat = flatten(x)
        x_adv_flat = flatten(x_adv)

        # was there a pixel left in the samples to manipulate,
        # i.e. reset to the clean version?
        found_index_to_manipulate = ep.from_numpy(x, np.ones(N, dtype=bool))

        while ep.any(found_index_to_manipulate):
            diff_mask = (ep.abs(x_flat - x_adv_flat) > 1e-8).numpy()
            diff_idxs = [z.nonzero()[0] for z in diff_mask]
            untouched_indices = [z.tolist() for z in diff_idxs]
            untouched_indices = [
                np.random.permutation(it).tolist() for it in untouched_indices
            ]

            found_index_to_manipulate = ep.from_numpy(x, np.zeros(N, dtype=bool))

            # since the number of pixels still left to manipulate might differ
            # across different samples we track each of them separately and
            # and manipulate the images until there is no pixel left for
            # any of the samples. to not update already finished samples, we mask
            # the updates such that only samples that still have pixels left to manipulate
            # will be updated
            i = 0
            while i < max([len(it) for it in untouched_indices]):
                # mask all samples that still have pixels to manipulate left
                relevant_mask_lst = [len(it) > i for it in untouched_indices]
                relevant_mask: np.ndarray[Any, np.dtype[np.bool_]] = np.array(
                    relevant_mask_lst, dtype=bool
                )
                relevant_mask_index: np.ndarray[
                    Any, np.dtype[np.int_]
                ] = np.flatnonzero(relevant_mask)

                # for each image get the index of the next pixel we try out
                relevant_indices = [it[i] for it in untouched_indices if len(it) > i]

                old_values = x_adv_flat[relevant_mask_index, relevant_indices]
                new_values = x_flat[relevant_mask_index, relevant_indices]
                x_adv_flat = ep.index_update(
                    x_adv_flat, (relevant_mask_index, relevant_indices), new_values
                )

                # check if still adversarial
                is_adv = is_adversarial(x_adv_flat.reshape(original_shape))
                found_index_to_manipulate = ep.index_update(
                    found_index_to_manipulate,
                    relevant_mask_index,
                    ep.logical_or(found_index_to_manipulate, is_adv)[relevant_mask],
                )

                # if not, undo change
                new_or_old_values = ep.where(
                    is_adv[relevant_mask], new_values, old_values
                )
                x_adv_flat = ep.index_update(
                    x_adv_flat,
                    (relevant_mask_index, relevant_indices),
                    new_or_old_values,
                )

                i += 1

            if not ep.any(found_index_to_manipulate):
                break

        if self.l2_binary_search:
            while True:
                diff_mask = (ep.abs(x_flat - x_adv_flat) > 1e-12).numpy()
                diff_idxs = [z.nonzero()[0] for z in diff_mask]
                untouched_indices = [z.tolist() for z in diff_idxs]
                # draw random shuffling of all indices for all samples
                untouched_indices = [
                    np.random.permutation(it).tolist() for it in untouched_indices
                ]

                # whether that run through all values made any improvement
                improved = ep.from_numpy(x, np.zeros(N, dtype=bool)).astype(bool)

                logging.info("Starting new loop through all values")

                # use the same logic as above
                i = 0
                while i < max([len(it) for it in untouched_indices]):
                    # mask all samples that still have pixels to manipulate left
                    relevant_mask_lst = [len(it) > i for it in untouched_indices]
                    relevant_mask = np.array(relevant_mask_lst, dtype=bool)
                    relevant_mask_index = np.flatnonzero(relevant_mask)

                    # for each image get the index of the next pixel we try out
                    relevant_indices = [
                        it[i] for it in untouched_indices if len(it) > i
                    ]

                    old_values = x_adv_flat[relevant_mask_index, relevant_indices]
                    new_values = x_flat[relevant_mask_index, relevant_indices]

                    x_adv_flat = ep.index_update(
                        x_adv_flat, (relevant_mask_index, relevant_indices), new_values
                    )

                    # check if still adversarial
                    is_adv = is_adversarial(x_adv_flat.reshape(original_shape))

                    improved = ep.index_update(
                        improved,
                        relevant_mask_index,
                        ep.logical_or(improved, is_adv)[relevant_mask],
                    )

                    if not ep.all(is_adv):
                        # run binary search for examples that became non-adversarial
                        updated_new_values = self._binary_search(
                            x_adv_flat,
                            relevant_mask,
                            relevant_mask_index,
                            relevant_indices,
                            old_values,
                            new_values,
                            (-1, *original_shape[1:]),
                            is_adversarial,
                        )
                        x_adv_flat = ep.index_update(
                            x_adv_flat,
                            (relevant_mask_index, relevant_indices),
                            ep.where(
                                is_adv[relevant_mask], new_values, updated_new_values
                            ),
                        )

                        improved = ep.index_update(
                            improved,
                            relevant_mask_index,
                            ep.logical_or(
                                old_values != updated_new_values,
                                improved[relevant_mask],
                            ),
                        )

                    i += 1

                if not ep.any(improved):
                    # no improvement for any of the indices
                    break

        x_adv = x_adv_flat.reshape(original_shape)

        return restore_type(x_adv)

    def _binary_search(
        self,
        x_adv_flat: ep.Tensor,
        mask: Union[ep.Tensor, List[bool], np.ndarray[Any, np.dtype[np.bool_]]],
        mask_indices: Union[ep.Tensor, np.ndarray[Any, np.dtype[np.int_]]],
        indices: Union[ep.Tensor, List[int]],
        adv_values: ep.Tensor,
        non_adv_values: ep.Tensor,
        original_shape: Tuple,
        is_adversarial: Callable,
    ) -> ep.Tensor:
        for i in range(10):
            next_values = (adv_values + non_adv_values) / 2
            x_adv_flat = ep.index_update(
                x_adv_flat, (mask_indices, indices), next_values
            )
            is_adv = is_adversarial(x_adv_flat.reshape(original_shape))[mask]

            adv_values = ep.where(is_adv, next_values, adv_values)
            non_adv_values = ep.where(is_adv, non_adv_values, next_values)

        return adv_values
