# Adding Adversarial Attacks

::: tip NOTE
The [development guidelines](./development) explain how to get started with
with developing features and adversarial attacks for Foolbox.
:::

## The `Attack` base class

Adversarial attacks in Foolbox should either directly or indirectly subclass
the `Attack` base class in `foolbox/attacks/base.py`.

An attack in Foolbox needs to implement two methods, `__call__` and `repeat`.

Both methods need to be implemented with the same signature as the base class.
The type annotation for the `criterion` argument of `__call__` can be made
more precise, see `foolbox/attacks/carlini_wagner.py` for an example.

The `__call__` method should return three values, a list of raw tensors (one
for each epsilon) with the internal raw attack results, a list of tensors
corresponding to the raw tensors but with perturbation sizes guaranteed to
be clipped to the given epsilons, and a boolean tensor with `len(epsilons)`
rows and `len(inputs)` columns indicating for each returned sample whether
it is a successful adversarial example given the respective epsilon and
criterion. If `epsilons` is a single scalar epsilon (and not a list with
one element), then the first and second return value should be a tensor
rather than a list and the third return value should be 1-D tensor.

All returned tensors must have the same type as the input tensors. In
particular, native tensors should be returned as native tensors and
EagerPy-wrapped tensors should be returned as EagerPy-wrapped tensors.
Use `astensor_` or `astensors_` and `restore_type`.

The `repeat` method should return a version of the attack that repeats itself
n times and returns the best result.

::: warning NOTE
In practice, it is usually not necessary to subclass `Attack` directly.
Instead, for most attacks it is easiest to subclass either `FixedEpsilonAttack`
or `MinimizatonAttack`.
:::

## The `FixedEpsilonAttack` base class

Attacks that require a fixed epsilon and try to find an adversarial
perturbation given this perturbation budget (e.g. `FGSM` and `PGD`) should
be implemented by subclassing `FixedEpsilonAttack`. It already provides
implementations of `__call__` and `repeat`. The attack just needs
to specify the `distance` property (simply assign a class variable) and
implement the `run` method that gets a single `epsilon` and returns a batch
of perturbed inputs, ideally adversarial and ideally with a perturbation
size smaller or equal to `epsilon`.
The `distance` is used by `__call__` to determine which perturbed inputs
are actually adversarials given `epsilon` and by `repeat` to determine the
run.

## The `MinimizatonAttack` base class

Attacks that try to find adversarial examples with minimal perturbation size
(e.g. the `Carlini & Wagner` attack or the `Boundary Attack`) should
be implemented by subclassing `MinimizatonAttack`. It already provides
implementations of `__call__` and `repeat`. The attack just needs
to specify the `distance` property (simply assign a class variable) and
implement the `run` method that returns a batch of minimally perturbed
adversarials. For `MinimizatonAttack` subclasses, `run` gets called only once
by `__call__` independent of how many `epsilons` are given. The `__call__`
method then compares the minimal adversarial perturbation to the different
epsilons.

::: tip
You should have a look at the implementation of existing attacks
to get an impression of the best practices and conventions used in Foolbox.
:::
