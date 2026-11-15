import pytest

from vigia.cli import DEMOS, main


@pytest.mark.parametrize("demo", sorted(DEMOS))
def test_every_demo_has_working_help(demo, capsys):
    with pytest.raises(SystemExit) as e:
        main([demo, "--help"])
    assert e.value.code == 0
    assert "--source" in capsys.readouterr().out


def test_usage_and_unknown_demo(capsys):
    assert main([]) == 0
    assert main(["nope"]) == 2


def test_every_zero_is_rejected(capsys):
    with pytest.raises(SystemExit) as e:
        main(["tables", "--source", "x.mp4", "--every", "0"])
    assert e.value.code == 2
    assert "--every" in capsys.readouterr().err
