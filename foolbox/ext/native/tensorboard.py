"""Internal module for attacks that support logging to TensorBoard"""
import eagerpy as ep
from functools import wraps


def maybenoop(f):
    @wraps(f)
    def wrapper(self, *args, **kwds):
        if self.writer is None:
            return
        return f(self, *args, **kwds)

    return wrapper


class TensorBoard:
    """A custom TensorBoard class that accepts EagerPy tensors and that
    can be disabled by turned into a noop by passing logdir=False.

    This makes it possible to add tensorboard logging without any if
    statements and without any computational overhead if it's disabled.
    """

    def __init__(self, logdir):
        if logdir or (logdir is None):
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    @maybenoop
    def close(self):
        self.writer.close()

    @maybenoop
    def scalar(self, tag, x, step):
        self.writer.add_scalar(tag, x, step)

    @maybenoop
    def mean(self, tag, x: ep.Tensor, step):
        self.writer.add_scalar(tag, x.mean(axis=0).item(), step)

    @maybenoop
    def probability(self, tag, x: ep.Tensor, step):
        self.writer.add_scalar(tag, x.float32().mean(axis=0).item(), step)

    @maybenoop
    def conditional_mean(self, tag, x: ep.Tensor, cond: ep.Tensor, step):
        cond_ = cond.numpy()
        if ~cond_.any():
            return
        x_ = x.numpy()
        x_ = x_[cond_]
        self.writer.add_scalar(tag, x_.mean(axis=0).item(), step)

    @maybenoop
    def probability_ratio(self, tag, x: ep.Tensor, y: ep.Tensor, step):
        x_ = x.float32().mean(axis=0).item()
        y_ = y.float32().mean(axis=0).item()
        if y_ == 0:
            return
        self.writer.add_scalar(tag, x_ / y_, step)

    @maybenoop
    def histogram(self, tag, x: ep.Tensor, step, *, first=True):
        x = x.numpy()
        self.writer.add_histogram(tag, x, step)
        if first:
            self.writer.add_scalar(tag + "/0", x[0].item(), step)
