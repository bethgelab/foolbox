from __future__ import print_function
from __future__ import division

import warnings
import threading
import queue
import time
import sys
import collections

# requires Python 3.2 or newer, or a backport for Python 2
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Executor
from concurrent.futures import Future

from .base import Attack
from .blended_noise import BlendedUniformNoiseAttack
from ..criteria import Misclassification

import numpy as np
from numpy.linalg import norm


class BoundaryAttack(Attack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    This is the reference implementation for the attack introduced in [1]_.

    Notes
    -----
    This implementation provides several advanced features:

    * ability to continue previous attacks by passing an instance of the
      Adversarial class
    * ability to pass an explicit starting point; especially to initialize
      a targeted attack
    * ability to pass an alternative attack used for initialization
    * fine-grained control over logging
    * ability to specify the batch size
    * optional automatic batch size tuning
    * optional multithreading for random number generation
    * optional multithreading for candidate point generation

    References
    ----------
    .. [1] Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge,
           "Decision-Based Adversarial Attacks: Reliable Attacks
           Against Black-Box Machine Learning Models",
           https://arxiv.org/abs/1712.04248

    """

    def __init__(self, model=None, criterion=Misclassification()):
        super(BoundaryAttack, self).__init__(model=model, criterion=criterion)

    def __call__(
            self,
            image,
            label=None,
            unpack=True,
            iterations=5000,
            max_directions=25,
            starting_point=None,
            initialization_attack=None,
            log_every_n_steps=1,
            spherical_step=1e-2,
            source_step=1e-2,
            step_adaptation=1.5,
            batch_size=1,
            tune_batch_size=True,
            threaded_rnd=True,
            threaded_gen=True,
            alternative_generator=False,
            internal_dtype=np.float64,
            verbose=False):

        """Applies the Boundary Attack.

        Parameters
        ----------
        image : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        iterations : int
            Maximum number of iterations to run. Might converge and stop
            before that.
        max_directions : int
            Maximum number of trials per ieration.
        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point, in particular
            for targeted attacks.
        initialization_attack : :class:`Attack`
            Attack to use to find a starting point. Defaults to
            BlendedUniformNoiseAttack.
        log_every_n_steps : int
            Determines verbositity of the logging.
        spherical_step : float
            Initial step size for the orthogonal (spherical) step.
        source_step : float
            Initial step size for the step towards the target.
        step_adaptation : float
            Factor by which the step sizes are multiplied or divided.
        batch_size : int
            Batch size or initial batch size if tune_batch_size is True
        tune_batch_size : bool
            Whether or not the batch size should be automatically chosen
            between 1 and max_directions.
        threaded_rnd : bool
            Whether the random number generation should be multithreaded.
        threaded_gen : bool
            Whether the candidate point generation should be multithreaded.
        alternative_generator: bool
            Whether an alternative implemenation of the candidate generator
            should be used.
        internal_dtype : np.float32 or np.float64
            Higher precision might be slower but is numerically more stable.
        verbose : bool
            Controls verbosity of the attack.

        """

        # overwriting __call__ to make the list of parameters and default
        # values as well as the documentation easily accessible to the user

        # make some of the parameters available to other methods without
        # the need to explicitly pass them
        self.log_every_n_steps = log_every_n_steps
        self._starting_point = starting_point
        self._initialization_attack = initialization_attack
        self.batch_size = batch_size
        self.max_directions = max_directions
        self.step_adaptation = step_adaptation
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.internal_dtype = internal_dtype
        self.verbose = verbose

        if not verbose:
            print('run with verbose=True to see details')

        if alternative_generator:
            self.generate_candidate = self.generate_candidate_alternative
        else:
            self.generate_candidate = self.generate_candidate_default

        return super(BoundaryAttack, self).__call__(
            image=image,
            label=label,
            unpack=unpack,
            iterations=iterations,
            tune_batch_size=tune_batch_size,
            threaded_rnd=threaded_rnd,
            threaded_gen=threaded_gen)

    def _apply(
            self,
            *args,
            **kwargs):

        # ===========================================================
        # Start optional threads for parallel candidate generation
        # ===========================================================

        if kwargs['threaded_gen'] is True:
            # default value if True, but allow users to pass a number instead
            kwargs['threaded_gen'] = 13

        if kwargs['threaded_gen']:
            n = kwargs['threaded_gen']
            with ThreadPoolExecutor(max_workers=n) as pool:
                return self._apply_inner(pool, *args, **kwargs)
        else:
            with DummyExecutor() as pool:
                return self._apply_inner(pool, *args, **kwargs)

    def _apply_inner(
            self,
            pool,
            a,
            iterations,
            tune_batch_size,
            threaded_rnd,
            threaded_gen):

        self.t_initial = time.time()

        # ===========================================================
        # Increase floating point precision
        # ===========================================================

        external_dtype = a.original_image.dtype

        assert self.internal_dtype in [np.float32, np.float64]
        assert external_dtype in [np.float32, np.float64]

        assert not (external_dtype == np.float64 and
                    self.internal_dtype == np.float32)

        a.set_distance_dtype(self.internal_dtype)

        # ===========================================================
        # Find starting point
        # ===========================================================

        self.initialize_starting_point(a)

        if a.image is None:
            warnings.warn(
                'Initialization failed. If the criterion is targeted,'
                ' it might be necessary to pass an explicit starting'
                ' point or targeted initialization attack.')
            return

        assert a.image.dtype == external_dtype

        # ===========================================================
        # Initialize variables, constants, hyperparameters, etc.
        # ===========================================================

        # make sure repeated warnings are shown
        warnings.simplefilter('always', UserWarning)

        # get bounds
        bounds = a.bounds()
        min_, max_ = bounds

        # get original and starting point in the right format
        original = a.original_image.astype(self.internal_dtype)
        perturbed = a.image.astype(self.internal_dtype)
        distance = a.distance

        # determine next step for batch size tuning
        self.init_batch_size_tuning(tune_batch_size)

        # make sure step size is valid
        self.printv(
            'Initial spherical_step = {:.2f}, source_step = {:.2f}'.format(
                self.spherical_step, self.source_step))

        # ===========================================================
        # intialize stats
        # ===========================================================

        stats_initialized = False

        # time measurements
        self.stats_success = np.zeros((self.max_directions,), dtype=np.int)
        self.stats_fail = 0
        self.stats_generator_duration = np.zeros((self.max_directions,))
        self.stats_prediction_duration = np.zeros((self.max_directions,))
        self.stats_spherical_prediction_duration \
            = np.zeros((self.max_directions,))
        self.stats_hyperparameter_update_duration = 0

        # counters
        self.stats_generator_calls \
            = np.zeros((self.max_directions,), dtype=np.int)
        self.stats_prediction_calls \
            = np.zeros((self.max_directions,), dtype=np.int)
        self.stats_spherical_prediction_calls \
            = np.zeros((self.max_directions,), dtype=np.int)
        self.stats_numerical_problems = 0

        # recent successes
        self.stats_spherical_adversarial = collections.deque(maxlen=100)
        self.stats_step_adversarial = collections.deque(maxlen=30)

        # ===========================================================
        # Start optional threads for parallel sampling from std normal
        # ===========================================================

        if threaded_rnd is True:
            # default value if True, but allow users to pass a number instead
            threaded_rnd = 4

        if threaded_rnd:
            # create a queue to cache samples
            queue_size = 2 * self.max_directions + threaded_rnd + threaded_gen
            rnd_normal_queue = queue.Queue(queue_size)

            try:
                import randomstate
            except ImportError:  # pragma: no cover
                raise ImportError('To use the BoundaryAttack,'
                                  ' please install the randomstate'
                                  ' module (e.g. pip install randomstate)')

            def sample_std_normal(thread_id, shape, dtype):
                # create a thread-specifc RNG
                rng = randomstate.RandomState(seed=20 + thread_id)

                t = threading.currentThread()
                while getattr(t, 'do_run', True):
                    rnd_normal = rng.standard_normal(
                        size=shape, dtype=dtype, method='zig')
                    rnd_normal_queue.put(rnd_normal)

            self.printv('Using {} threads to create random numbers'.format(
                threaded_rnd))

            # start threads that sample from std normal distribution
            rnd_normal_threads = []
            for thread_id in range(threaded_rnd):
                rnd_normal_thread = threading.Thread(
                    target=sample_std_normal,
                    args=(thread_id, original.shape, original.dtype))
                rnd_normal_thread.start()
                rnd_normal_threads.append(rnd_normal_thread)
        else:
            rnd_normal_queue = None

        # ===========================================================
        # Iteratively refine adversarial by following the boundary
        # between adversarial and non-adversarial images
        # ===========================================================

        generation_args = None

        # log starting point
        self.log_step(0, distance)

        initial_convergence_steps = 100
        convergence_steps = initial_convergence_steps
        resetted = False

        for step in range(1, iterations + 1):
            t_step = time.time()

            # ===========================================================
            # Check converges
            # ===========================================================

            check_strict = convergence_steps == initial_convergence_steps
            if self.has_converged(check_strict):
                self.log_step(step - 1, distance, always=True)
                if resetted:
                    self.printv(
                        'Looks like attack has converged after {} steps,'
                        ' {} remaining'.format(step, convergence_steps))
                    convergence_steps -= 1
                    if convergence_steps == 0:
                        break
                else:
                    resetted = True
                    self.printv(
                        'Looks like attack has converged after' +
                        ' {} steps'.format(step) +
                        ' for the first time. Resetting steps to be sure.')
                    self.spherical_step = 1e-2
                    self.source_step = 1e-2
            elif (convergence_steps <
                    initial_convergence_steps):  # pragma: no cover
                self.log_step(step - 1, distance, always=True)
                warnings.warn('Attack has not converged!')
                convergence_steps = initial_convergence_steps
                resetted = False

            # ===========================================================
            # Determine optimal batch size
            # ===========================================================

            if tune_batch_size and step == self.next_tuning_step:
                if not stats_initialized:
                    self.initialize_stats(
                        a, pool, external_dtype, generation_args)
                    stats_initialized = True

                    # during initialization, predictions are performed
                    # and thus better adversarials might have been found
                    # if a.distance.value != distance.value:
                    #     assert a.distance.value < distance.value
                    if a.distance.value < distance.value:
                        self.printv(
                            'During initialization, a better adversarial'
                            ' has been found. Continuing from there.')
                        perturbed = a.image.astype(self.internal_dtype)
                        distance = a.distance
                        # becaue we are resetting perturbed, it's important
                        # that the new generator is created afterwards

                self.tune_batch_size(a)

            # ===========================================================
            # Create a generator for new candidates
            # ===========================================================

            unnormalized_source_direction, source_direction, source_norm \
                = self.prepare_generate_candidates(original, perturbed)

            generation_args = (
                rnd_normal_queue,
                bounds,
                original,
                perturbed,
                unnormalized_source_direction,
                source_direction,
                source_norm,
                self.spherical_step,
                self.source_step,
                self.internal_dtype)

            # ===========================================================
            # Try to find a better adversarial
            # ===========================================================

            # only check spherical every 10th step
            # or in every step when we are in convergence confirmation mode,
            # i.e. after resetting
            do_spherical = (step % 10 == 0) or resetted

            n_batches = (self.max_directions - 1) // self.batch_size + 1

            # for the first batch
            t = time.time()
            futures = [
                pool.submit(self.generate_candidate, *generation_args)
                for _ in range(self.batch_size)]
            t = time.time() - t
            self.stats_generator_duration[self.batch_size - 1] += t

            for i in range(n_batches):
                # for the last batch, reduce the batch size if necessary
                if i == n_batches - 1:
                    # last batch
                    remaining = self.max_directions - i * self.batch_size
                    current_batch_size = remaining
                    next_batch_size = 0
                elif i == n_batches - 2:
                    # second to last batch
                    current_batch_size = self.batch_size
                    remaining = self.max_directions - (i + 1) * self.batch_size
                    next_batch_size = remaining
                else:
                    # other batches
                    current_batch_size = self.batch_size
                    next_batch_size = self.batch_size

                assert len(futures) == current_batch_size

                batch_shape = (current_batch_size,) + original.shape

                # sample a batch of candidates
                candidates = np.empty(batch_shape, dtype=original.dtype)
                if do_spherical:
                    spherical_candidates = np.empty(
                        batch_shape, dtype=original.dtype)

                for j in range(current_batch_size):
                    t = time.time()
                    candidate, spherical_candidate \
                        = futures[j].result()
                    if do_spherical:
                        spherical_candidates[j] = spherical_candidate
                    candidates[j] = candidate
                    t = time.time() - t
                    self.stats_generator_duration[
                        current_batch_size - 1] += t
                    self.stats_generator_calls[current_batch_size - 1] += 1

                # for the next batch
                if next_batch_size > 0:
                    t = time.time()
                    futures = [
                        pool.submit(self.generate_candidate, *generation_args)
                        for _ in range(next_batch_size)]
                    t = time.time() - t
                    self.stats_generator_duration[next_batch_size - 1] += t
                else:
                    futures = None

                # check spherical ones
                if do_spherical:
                    t = time.time()
                    _, batch_is_adversarial = a.batch_predictions(
                        spherical_candidates.astype(external_dtype),
                        strict=False)
                    t = time.time() - t

                    assert batch_is_adversarial.shape == (current_batch_size,)

                    self.stats_spherical_prediction_duration[
                        current_batch_size - 1] += t
                    self.stats_spherical_prediction_calls[
                        current_batch_size - 1] += 1

                    indices = []
                    for j in range(current_batch_size):
                        spherical_is_adversarial \
                            = batch_is_adversarial[j]
                        self.stats_spherical_adversarial.appendleft(
                            spherical_is_adversarial)
                        if spherical_is_adversarial:
                            indices.append(j)

                    if len(indices) == 0:
                        continue  # next batch

                    # if at least one of the spherical candidates was
                    # adversarial, get real candidates

                    candidates = np.take(candidates, indices, axis=0)
                    reduced_shape = (len(indices),) + batch_shape[1:]
                    assert candidates.shape == reduced_shape

                    t = time.time()
                    _, batch_is_adversarial = a.batch_predictions(
                        candidates.astype(external_dtype),
                        strict=False)
                    t = time.time() - t
                    # TODO: use t

                    assert batch_is_adversarial.shape == (len(indices),)

                    self.stats_step_adversarial.extendleft(
                        batch_is_adversarial)

                    for j in range(len(indices)):
                        is_adversarial = batch_is_adversarial[j]

                        if is_adversarial:
                            new_perturbed = candidates[j]
                            new_distance = a.normalized_distance(new_perturbed)
                            # rough correction factor
                            f = current_batch_size / len(indices)
                            candidate_index = i * self.batch_size + int(j * f)
                            self.stats_success[candidate_index] += 1
                            break
                    else:
                        continue  # next batch
                    break  # found advesarial candidate
                else:
                    # check if one of the candidates is adversarial
                    t = time.time()
                    _, is_adversarial, adv_index, is_best, candidate_distance \
                        = a.batch_predictions(
                            candidates.astype(external_dtype), greedy=True,
                            strict=False, return_details=True)
                    t = time.time() - t
                    self.stats_prediction_duration[self.batch_size - 1] += t
                    self.stats_prediction_calls[
                        self.batch_size - 1] += 1

                    if is_adversarial:
                        new_perturbed = candidates[adv_index]
                        new_distance = candidate_distance
                        candidate_index = i * self.batch_size + adv_index
                        self.stats_success[candidate_index] += 1
                        break

            else:  # if the for loop doesn't break
                new_perturbed = None
                self.stats_fail += 1

            # ===========================================================
            # Handle the new adversarial
            # ===========================================================

            if new_perturbed is not None:
                if not new_distance < distance:
                    # assert not is_best  # consistency with adversarial object
                    self.stats_numerical_problems += 1
                    warnings.warn('Internal inconsistency, probably caused by '
                                  'numerical errors')
                else:
                    # assert is_best  # consistency with adversarial object
                    # Jonas 24.10.2017: this can be violated because spherical
                    # step can be better and adv (numerical issues)
                    abs_improvement = distance.value - new_distance.value
                    rel_improvement = abs_improvement / distance.value
                    message = 'd. reduced by {:.2f}% ({:.4e})'.format(
                        rel_improvement * 100, abs_improvement)

                    # update the variables
                    perturbed = new_perturbed
                    distance = new_distance
            else:
                message = ''

            # ===========================================================
            # Update step sizes
            # ===========================================================

            t = time.time()
            self.update_step_sizes()
            t = time.time() - t
            self.stats_hyperparameter_update_duration += t

            # ===========================================================
            # Log the step
            # ===========================================================

            t_step = time.time() - t_step
            message += ' (took {:.5f} seconds)'.format(t_step)
            self.log_step(step, distance, message)
            sys.stdout.flush()

        # ===========================================================
        # Stop threads that generate random numbers
        # ===========================================================

        if threaded_rnd:
            for rnd_normal_thread in rnd_normal_threads:
                rnd_normal_thread.do_run = False
            for rnd_normal_thread in rnd_normal_threads:
                try:
                    rnd_normal_queue.get(block=False)
                except queue.Empty:  # pragma: no cover
                    pass
            for rnd_normal_thread in rnd_normal_threads:
                rnd_normal_thread.join()

        # ===========================================================
        # Log overall runtime
        # ===========================================================

        self.log_time()

    # ===============================================================
    #
    # Other methods
    #
    # ===============================================================

    def initialize_starting_point(self, a):
        starting_point = self._starting_point
        init_attack = self._initialization_attack

        if a.image is not None:
            print(
                'Attack is applied to a previously found adversarial.'
                ' Continuing search for better adversarials.')
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring starting_point parameter because the attack'
                    ' is applied to a previously found adversarial.')
            if init_attack is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring initialization_attack parameter because the'
                    ' attack is applied to a previously found adversarial.')
            return

        if starting_point is not None:
            a.predictions(starting_point)
            assert a.image is not None, ('Invalid starting point provided.'
                                         ' Please provide a starting point'
                                         ' that is adversarial.')
            return

        if init_attack is None:
            init_attack = BlendedUniformNoiseAttack
            self.printv(
                'Neither starting_point nor initialization_attack given.'
                ' Falling back to {} for initialization.'.format(
                    init_attack.__name__))

        if issubclass(init_attack, Attack):
            # instantiate if necessary
            init_attack = init_attack()

        init_attack(a)

    def log_step(self, step, distance, message='', always=False):
        if not always and step % self.log_every_n_steps != 0:
            return
        print('Step {}: {:.5e}, stepsizes = {:.1e}/{:.1e}: {}'.format(
            step,
            distance.value,
            self.spherical_step,
            self.source_step,
            message))

    @staticmethod
    def prepare_generate_candidates(original, perturbed):
        unnormalized_source_direction = original - perturbed
        source_norm = norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm
        return unnormalized_source_direction, source_direction, source_norm

    @staticmethod
    def generate_candidate_default(
            rnd_normal_queue,
            bounds,
            original,
            perturbed,
            unnormalized_source_direction,
            source_direction,
            source_norm,
            spherical_step,
            source_step,
            internal_dtype,
            rng=None):

        if rng is None:
            try:
                import randomstate
            except ImportError:  # pragma: no cover
                raise ImportError('To use the BoundaryAttack,'
                                  ' please install the randomstate'
                                  ' module (e.g. pip install randomstate)')
            rng = randomstate

        # ===========================================================
        # perform initial work
        # ===========================================================

        assert original.dtype == internal_dtype
        assert perturbed.dtype == internal_dtype

        shape = original.shape

        min_, max_ = bounds

        # ===========================================================
        # draw a random direction
        # ===========================================================

        # randomstate's rnd is faster and more flexible than numpy's if
        # has a dtype argument and supports the much faster Ziggurat method
        if rnd_normal_queue is None:
            perturbation = rng.standard_normal(
                size=shape, dtype=original.dtype, method='zig')
        else:
            perturbation = rnd_normal_queue.get()

        assert perturbation.dtype == internal_dtype

        # ===========================================================
        # calculate candidate on sphere
        # ===========================================================

        dot = np.vdot(perturbation, source_direction)
        perturbation -= dot * source_direction
        perturbation *= spherical_step * source_norm / norm(perturbation)

        D = 1 / np.sqrt(spherical_step**2 + 1)
        direction = perturbation - unnormalized_source_direction
        spherical_candidate = original + D * direction

        np.clip(spherical_candidate, min_, max_, out=spherical_candidate)

        # ===========================================================
        # add perturbation in direction of source
        # ===========================================================

        new_source_direction = original - spherical_candidate
        new_source_direction_norm = norm(new_source_direction)

        assert perturbed.dtype == internal_dtype
        assert original.dtype == internal_dtype
        assert spherical_candidate.dtype == internal_dtype

        # length if spherical_candidate would be exactly on the sphere
        length = source_step * source_norm

        # length including correction for deviation from sphere
        deviation = new_source_direction_norm - source_norm
        length += deviation

        # make sure the step size is positive
        length = max(0, length)

        # normalize the length
        length = length / new_source_direction_norm

        candidate = spherical_candidate + length * new_source_direction
        np.clip(candidate, min_, max_, out=candidate)

        assert spherical_candidate.dtype == internal_dtype
        assert candidate.dtype == internal_dtype

        data = (candidate, spherical_candidate)

        return data

    @staticmethod
    def generate_candidate_alternative(
            rnd_normal_queue,
            bounds,
            original,
            perturbed,
            unnormalized_source_direction,
            source_direction,
            source_norm,
            spherical_step,
            source_step,
            internal_dtype,
            rng=None):

        if rng is None:
            try:
                import randomstate
            except ImportError:  # pragma: no cover
                raise ImportError('To use the BoundaryAttack,'
                                  ' please install the randomstate'
                                  ' module (e.g. pip install randomstate)')
            rng = randomstate

        # ===========================================================
        # perform initial work
        # ===========================================================

        assert original.dtype == internal_dtype
        assert perturbed.dtype == internal_dtype

        shape = original.shape

        min_, max_ = bounds

        # ===========================================================
        # draw a random direction
        # ===========================================================

        # randomstate's rnd is faster and more flexible than numpy's if
        # has a dtype argument and supports the much faster Ziggurat method
        if rnd_normal_queue is None:
            perturbation = rng.standard_normal(
                size=shape, dtype=original.dtype, method='zig')
        else:
            perturbation = rnd_normal_queue.get()

        assert perturbation.dtype == internal_dtype

        # ===========================================================
        # normalize perturbation and subtract source direction
        # (to stay on sphere)
        # ===========================================================

        perturbation *= spherical_step * source_norm / norm(perturbation)
        perturbation -= np.vdot(perturbation, source_direction) \
            * source_direction

        spherical_perturbation = perturbed + perturbation
        np.clip(spherical_perturbation, min_, max_, out=spherical_perturbation)

        # refine spherical perturbation
        refinement_threshold = min(1e-5, source_step / 10)
        for refinements in range(30):
            spherical_source_direction = spherical_perturbation - original
            spherical_norm = norm(spherical_source_direction)
            diff_norm = spherical_norm - source_norm
            if np.abs(diff_norm) / source_norm <= refinement_threshold:
                break
            spherical_perturbation -= diff_norm / spherical_norm \
                * spherical_source_direction
            np.clip(
                spherical_perturbation,
                min_,
                max_,
                out=spherical_perturbation)
        else:  # pragma: no cover
            refinements += 1

        # ===========================================================
        # add perturbation in direction of source
        # ===========================================================

        new_source_direction = original - spherical_perturbation
        new_source_direction_norm = norm(new_source_direction)
        assert perturbed.dtype == internal_dtype
        assert original.dtype == internal_dtype
        assert spherical_perturbation.dtype == internal_dtype

        perturbation = spherical_perturbation.copy()
        length = source_step * source_norm / new_source_direction_norm
        perturbation += length * new_source_direction
        np.clip(perturbation, min_, max_, out=perturbation)

        assert spherical_perturbation.dtype == internal_dtype
        assert perturbation.dtype == internal_dtype

        data = (perturbation, spherical_perturbation)
        return data

    def initialize_stats(self, a, pool, external_dtype, generation_args):
        self.printv('Initializing generation and prediction'
                    ' time measurements. This can take a few'
                    ' seconds.')

        _next = self.generate_candidate(*generation_args)
        candidate, spherical_candidate = _next
        # batch_shape = (self.max_directions,) + candidate.shape
        # samples = np.empty(batch_shape, candidate.dtype)

        # after initialization, we should have 1000 data points
        # and at least `max_directions` new ones to fill the array
        # n = max(1000 - self.stats_generator_calls, self.max_directions)

        for batch_size in range(1, self.max_directions + 1):
            t = time.time()
            futures = [
                pool.submit(self.generate_candidate, *generation_args)
                for _ in range(batch_size)]
            t = time.time() - t
            self.stats_generator_duration[batch_size - 1] += t

            batch_shape = (batch_size,) + candidate.shape
            samples = np.empty(batch_shape, candidate.dtype)

            for i in range(batch_size):
                t = time.time()
                candidate, _ = futures[i].result()
                samples[i] = candidate
                t = time.time() - t
                self.stats_generator_duration[batch_size - 1] += t
                self.stats_generator_calls[batch_size - 1] += 1

            batch = samples

            current = self.stats_prediction_calls[batch_size - 1]
            # more data points for small batch sizes, fewer
            # for large batch sizes
            target = 2 + (2 * self.max_directions) // batch_size
            n = max(target - current, 0)

            for i in range(n):
                t = time.time()
                _, is_adversarial, adv_index, is_best, candidate_distance \
                    = a.batch_predictions(
                        batch.astype(external_dtype), greedy=True,
                        strict=False, return_details=True)
                t = time.time() - t

                self.stats_prediction_duration[batch_size - 1] += t
                self.stats_prediction_calls[batch_size - 1] += 1

                t = time.time()
                _, _ = a.batch_predictions(
                    batch.astype(external_dtype), strict=False)
                t = time.time() - t

                self.stats_spherical_prediction_duration[batch_size - 1] \
                    += t
                self.stats_spherical_prediction_calls[batch_size - 1] += 1

    def log_time(self):
        t_total = time.time() - self.t_initial

        rel_generate = self.stats_generator_duration.sum() / t_total
        rel_prediction = self.stats_prediction_duration.sum() / t_total
        rel_spherical \
            = self.stats_spherical_prediction_duration.sum() / t_total
        rel_hyper = self.stats_hyperparameter_update_duration / t_total
        rel_remaining = 1 - rel_generate - rel_prediction \
            - rel_spherical - rel_hyper

        self.printv('Time since beginning: {:.5f}'.format(t_total))
        self.printv('   {:2.1f}% for generation ({:.5f})'.format(
            rel_generate * 100, self.stats_generator_duration.sum()))
        self.printv('   {:2.1f}% for spherical prediction ({:.5f})'.format(
            rel_spherical * 100,
            self.stats_spherical_prediction_duration.sum()))
        self.printv('   {:2.1f}% for prediction ({:.5f})'.format(
            rel_prediction * 100, self.stats_prediction_duration.sum()))
        self.printv('   {:2.1f}% for hyperparameter update ({:.5f})'.format(
            rel_hyper * 100, self.stats_hyperparameter_update_duration))
        self.printv('   {:2.1f}% for the rest ({:.5f})'.format(
            rel_remaining * 100, rel_remaining * t_total))

    def init_batch_size_tuning(self, tune_batch_size):
        if not tune_batch_size:
            return

        if tune_batch_size is True:
            # user provided a boolean
            self.steps_to_next_tuning = 100
        else:
            # user provded a concrete number
            self.steps_to_next_tuning = tune_batch_size
            tune_batch_size = True

        self.next_tuning_step = 1 + self.steps_to_next_tuning
        assert self.next_tuning_step > 1, (
            'Estimating the optimal batch size cannot be done'
            ' before the first step.')

        if self.steps_to_next_tuning < 50:
            warnings.warn('Batch size tuning after so few steps'
                          ' is not very reliable.')

    def tune_batch_size(self, a):
        self.printv('Estimating optimal batch size')

        max_directions = self.max_directions

        self.log_time()

        # ===========================================================
        # for each batch size, we estimate the time per step given the
        # distribution over the number of candidates needed per step
        # ===========================================================

        step_duration = np.zeros((max_directions,))

        # how long does it take to generate a candidate
        T_generate = self.stats_generator_duration / self.stats_generator_calls

        # how long does it take to get predictions of a batch
        T_prediction = self.stats_prediction_duration \
            / self.stats_prediction_calls

        self.printv('current estimate of the time to generate a candidate'
                    ' depending on the batch size:')
        self.printv(T_generate / np.arange(1, max_directions + 1))

        self.printv('current estimate of the time to get predictions for a'
                    ' candidate depending on the batch size:')
        self.printv(T_prediction / np.arange(1, max_directions + 1))

        # how often did we need to use the corresponding
        # number of candidates
        frequencies = [self.stats_fail] + list(self.stats_success)
        candidates = [max_directions] + list(range(1, max_directions + 1))

        s = sum(frequencies)

        self.printv('Relative frequencies for failing and success after k')
        self.printv(np.asarray(frequencies) / s)

        for batch_size in range(1, max_directions + 1):
            t_generate = 0
            t_prediction = 0

            for frequency, samples in zip(frequencies, candidates):
                # number of full batches
                max_full = max_directions // batch_size

                # same as round_up(samples / batch_size)
                full = (samples - 1) // batch_size + 1

                if full > max_full:
                    # the last batch will be smaller
                    full -= 1
                    remaining = max_directions - full * batch_size

                    t_generate += frequency * T_generate[remaining - 1]
                    t_prediction += frequency * T_prediction[remaining - 1]

                t_generate += frequency * full * T_generate[batch_size - 1]
                t_prediction += frequency * full * T_prediction[batch_size - 1]

            t_total = t_generate + t_prediction
            step_duration[batch_size - 1] = t_total

            self.printv(
                'Using batch size {:3d}, an average step would have taken'
                ' {:.5f} = {:.5f} + {:.5f} seconds'.format(
                    batch_size, t_total / s, t_generate / s, t_prediction / s))

        # ===========================================================
        # determine the best batch size and print comparisons
        # ===========================================================

        best_batch_size = np.argmin(step_duration) + 1
        worst_batch_size = np.argmax(step_duration) + 1

        self.printv('batch size was {}, optimal batch size would have'
                    ' been {}'.format(self.batch_size, best_batch_size))

        best_step_duration = step_duration[best_batch_size - 1]
        self.printv('setting batch size to {}: expected step duration:'
                    ' {:.5f}'.format(best_batch_size, best_step_duration / s))

        for name, value in (
                ('old', self.batch_size),
                ('worst', worst_batch_size),
                ('smallest', 1),
                ('largest', max_directions)):

            improvement = step_duration[value - 1] / best_step_duration

            self.printv('improvement compared to {} batch size'
                        ' ({}): {:.1f}x'.format(name, value, improvement))

        change = best_batch_size - self.batch_size

        if change == 0:
            self.steps_to_next_tuning *= 2
        elif change in [-1, 1]:
            pass
        else:
            if self.steps_to_next_tuning > 100:
                self.steps_to_next_tuning //= 2

        self.next_tuning_step += self.steps_to_next_tuning
        self.printv('next batch size tuning in {} steps, after step {}'.format(
            self.steps_to_next_tuning, self.next_tuning_step - 1))

        # finally, set the new batch size
        self.batch_size = best_batch_size

        # and reset the distribution over number of candidates needed
        # in a step, as it changes over time
        self.stats_fail = 0
        self.stats_success *= 0

    def update_step_sizes(self):
        def is_full(deque):
            return len(deque) == deque.maxlen

        if not (is_full(self.stats_spherical_adversarial) or
                is_full(self.stats_step_adversarial)):
            # updated step size recently, not doing anything now
            return

        def estimate_probability(deque):
            if len(deque) == 0:
                return None
            return np.mean(deque)

        p_spherical = estimate_probability(self.stats_spherical_adversarial)
        p_step = estimate_probability(self.stats_step_adversarial)

        n_spherical = len(self.stats_spherical_adversarial)
        n_step = len(self.stats_step_adversarial)

        def log(message):
            _p_spherical = p_spherical
            if _p_spherical is None:  # pragma: no cover
                _p_spherical = -1.

            _p_step = p_step
            if _p_step is None:
                _p_step = -1.

            self.printv('  {} {:.2f} ({:3d}), {:.2f} ({:2d})'.format(
                message,
                _p_spherical,
                n_spherical,
                _p_step,
                n_step))

        if is_full(self.stats_spherical_adversarial):
            if p_spherical > 0.5:
                message = 'Boundary too linear, increasing steps:    '
                self.spherical_step *= self.step_adaptation
                self.source_step *= self.step_adaptation
            elif p_spherical < 0.2:
                message = 'Boundary too non-linear, decreasing steps:'
                self.spherical_step /= self.step_adaptation
                self.source_step /= self.step_adaptation
            else:
                message = None

            if message is not None:
                self.stats_spherical_adversarial.clear()
                log(message)

        if is_full(self.stats_step_adversarial):
            if p_step > 0.5:
                message = 'Success rate too high, increasing source step:'
                self.source_step *= self.step_adaptation
            elif p_step < 0.2:
                message = 'Success rate too low, decreasing source step: '
                self.source_step /= self.step_adaptation
            else:
                message = None

            if message is not None:
                self.stats_step_adversarial.clear()
                log(message)

    def has_converged(self, strict):
        if strict:
            return self.source_step < 1e-7
        return self.source_step < 2e-7

    def printv(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class DummyExecutor(Executor):

    def __init__(self):
        self._shutdown = False
        self._shutdownLock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:  # pragma: no cover
                raise RuntimeError(
                    'cannot schedule new futures after shutdown')

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:  # pragma: no cover
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
