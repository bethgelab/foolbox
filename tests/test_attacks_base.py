import pytest
import eagerpy as ep
import foolbox as fbn

from conftest import ModeAndDataAndDescription


attacks = [
    fbn.attacks.InversionAttack(distance=fbn.distances.l2),
    fbn.attacks.InversionAttack(distance=fbn.distances.l2).repeat(3),
    fbn.attacks.L2ContrastReductionAttack(),
    fbn.attacks.L2ContrastReductionAttack().repeat(3),
]


@pytest.mark.parametrize("attack", attacks)
def test_call_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription, attack: fbn.Attack,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks

    assert ep.istensor(x)
    assert ep.istensor(y)

    raw, clipped, success = attack(fmodel, x, y, epsilons=1.0)
    assert ep.istensor(raw)
    assert ep.istensor(clipped)
    assert ep.istensor(success)
    assert raw.shape == x.shape
    assert clipped.shape == x.shape
    assert success.shape == (len(x),)


def test_get_channel_axis() -> None:
    class Model:
        data_format = None

    model = Model()
    model.data_format = "channels_first"  # type: ignore
    assert fbn.attacks.base.get_channel_axis(model, 3) == 1  # type: ignore
    model.data_format = "channels_last"  # type: ignore
    assert fbn.attacks.base.get_channel_axis(model, 3) == 2  # type: ignore
    model.data_format = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        assert fbn.attacks.base.get_channel_axis(model, 3)  # type: ignore
