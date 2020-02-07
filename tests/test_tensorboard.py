from typing import Union, Any
from typing_extensions import Literal
import pytest
import eagerpy as ep
import foolbox as fbn


@pytest.mark.parametrize("logdir", [False, "temp"])
def test_tensorboard(
    logdir: Union[Literal[False], None, str], tmp_path: Any, dummy: ep.Tensor
) -> None:
    if logdir == "temp":
        logdir = tmp_path

    if logdir:
        before = len(list(tmp_path.iterdir()))

    tb = fbn.tensorboard.TensorBoard(logdir)

    tb.scalar("a_scalar", 5, step=1)

    x = ep.ones(dummy, 10)
    tb.mean("a_mean", x, step=2)

    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    tb.probability("a_probability", x, step=2)

    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.conditional_mean("a_conditional_mean", x, cond, step=2)

    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.conditional_mean("a_conditional_mean_false", x, cond, step=2)

    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    y = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.probability_ratio("a_probability_ratio", x, y, step=5)

    x = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    y = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.probability_ratio("a_probability_ratio_y_zero", x, y, step=5)

    x = ep.arange(dummy, 10).float32()
    tb.histogram("a_histogram", x, step=9, first=False)
    tb.histogram("a_histogram", x, step=10, first=True)

    tb.close()

    if logdir:
        after = len(list(tmp_path.iterdir()))
        assert after > before  # make sure something has been written
