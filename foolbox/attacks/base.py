from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict
from typing_extensions import final, overload
from abc import ABC, abstractmethod
from collections.abc import Iterable
import eagerpy as ep

from ..models import Model

from ..criteria import Criterion
from ..criteria import Misclassification

from ..devutils import atleast_kd

from ..distances import Distance


T = TypeVar("T")
CriterionType = TypeVar("CriterionType", bound=Criterion)


# TODO: support manually specifying early_stop in __call__


class Attack(ABC):
    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T]:
        ...

    @abstractmethod  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], T], Tuple[T, T]]:
        # in principle, the type of criterion is Union[Criterion, T]
        # but we want to give subclasses the option to specify the supported
        # criteria explicitly (i.e. specifying a stricter type constraint)
        ...

    @abstractmethod
    def repeat(self, times: int) -> "Attack":
        ...

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"


class AttackWithDistance(Attack):
    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> Attack:
        return Repeated(self, times)


class Repeated(AttackWithDistance):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: AttackWithDistance, times: int):
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")

        self._attack = attack
        self._times = times

    @property
    def distance(self) -> Distance:
        return self._attack.distance

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T]:
        ...

    def __call__(  # noqa: F811
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], T], Tuple[T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        xps, success = self._attack(model, x, criterion, epsilons=epsilons, **kwargs)
        assert len(xps) == K
        for xp in xps:
            assert xp.shape == x.shape
        assert success.shape == (K, N)

        best_xps = xps
        best_success = success

        for _ in range(1, self._times):
            xps, success = self._attack(
                model, x, criterion, epsilons=epsilons, **kwargs
            )
            assert len(xps) == K
            for xp in xps:
                assert xp.shape == x.shape
            assert success.shape == (K, N)

            # TODO: test if stacking the list to a single tensor and
            # getting rid of the loop is faster

            for k, epsilon in enumerate(epsilons):
                first = best_success[k].logical_not()
                assert first.shape == (N,)
                if epsilon is None:
                    # TODO: maybe cache some of these distances
                    # and then remove the else part
                    closer = self.distance(x, xps[k]) < self.distance(x, best_xps[k])
                    assert closer.shape == (N,)
                    new_best = ep.logical_and(success[k], ep.logical_or(closer, first))
                else:
                    new_best = ep.logical_and(success[k], first)
                best_xps[k] = ep.where(
                    atleast_kd(new_best, x.ndim), xps[k], best_xps[k]
                )

            best_success = ep.logical_or(success, best_success)

        best_xps_ = [restore_type(xp) for xp in best_xps]
        if was_iterable:
            return best_xps_, restore_type(best_success)
        else:
            assert len(best_xps_) == 1
            return best_xps_[0], restore_type(best_success.squeeze(axis=0))

    def repeat(self, times: int) -> "Repeated":
        return Repeated(self._attack, self._times * times)


class FixedEpsilonAttack(AttackWithDistance):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    @abstractmethod
    def run(
        self, model: Model, inputs: T, criterion: Any, *, epsilon: float, **kwargs: Any
    ) -> T:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], T], Tuple[T, T]]:

        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            # TODO: implement a binary search
            raise NotImplementedError(
                "FixedEpsilonAttack subclasses do not yet support None in epsilons"
            )
        real_epsilons = [eps for eps in epsilons if eps is not None]
        del epsilons

        xps = [
            self.run(model, x, criterion, epsilon=epsilon, **kwargs)
            for epsilon in real_epsilons
        ]

        is_adv = ep.stack([is_adversarial(xp) for xp in xps], axis=0)
        assert is_adv.shape == (K, N)

        in_limits = ep.stack(
            [
                self.distance(x, xp) <= epsilon
                for xp, epsilon in zip(xps, real_epsilons)
            ],
            axis=0,
        )
        assert in_limits.shape == (K, N)

        if not in_limits.all():
            # TODO handle (numerical) violations
            # warn user if run() violated the epsilon constraint
            import pdb

            pdb.set_trace()

        success = ep.logical_and(in_limits, is_adv)
        assert success.shape == (K, N)

        xps_ = [restore_type(xp) for xp in xps]

        if was_iterable:
            return xps_, restore_type(success)
        else:
            assert len(xps) == 1
            (xp_,) = xps_
            return xp_, restore_type(success.squeeze(axis=0))


class MinimizationAttack(AttackWithDistance):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    @abstractmethod
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], T], Tuple[T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            early_stop = None
        else:
            early_stop = min(epsilons)
        limit_epsilons = [eps if eps is not None else ep.inf for eps in epsilons]
        del epsilons

        # run the actual attack
        xp = self.run(model, x, criterion, early_stop=early_stop, **kwargs)
        # TODO: optionally improve using a binary search?
        # TODO: optionally reduce size to the different epsilons and recompute is_adv

        is_adv = is_adversarial(xp)
        assert is_adv.shape == (N,)

        distances = self.distance(x, xp)
        assert distances.shape == (N,)

        in_limits = ep.stack(
            [distances <= epsilon for epsilon in limit_epsilons], axis=0
        )
        assert in_limits.shape == (K, N)

        success = ep.logical_and(in_limits, is_adv)
        assert success.shape == (K, N)

        xp_ = restore_type(xp)

        if was_iterable:
            return [xp_] * K, restore_type(success)
        else:
            return xp_, restore_type(success.squeeze(axis=0))


class FlexibleDistanceMinimizationAttack(MinimizationAttack):
    def __init__(self, *, distance: Optional[Distance] = None):
        self._distance = distance

    @property
    def distance(self) -> Distance:
        if self._distance is None:
            # we delay the error until the distance is needed,
            # e.g. when __call__ is executed (that way, run
            # can be used without specifying a distance)
            raise ValueError(
                "unknown distance, please pass `distance` to the attack initializer"
            )
        return self._distance


def get_is_adversarial(
    criterion: Criterion, model: Model
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)

    return is_adversarial


def get_criterion(criterion: Union[Criterion, Any]) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)


def get_channel_axis(model: Model, ndim: int) -> Optional[int]:
    data_format = getattr(model, "data_format", None)
    if data_format is None:
        return None
    if data_format == "channels_first":
        return 1
    if data_format == "channels_last":
        return ndim - 1
    raise ValueError(
        f"unknown data_format, expected 'channels_first' or 'channels_last', got {data_format}"
    )


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(
            f"attack got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )
