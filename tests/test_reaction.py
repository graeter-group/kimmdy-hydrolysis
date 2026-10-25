from pathlib import Path
from types import SimpleNamespace

import pytest

from kimmdy_hydrolysis.reaction import HydrolysisReaction


def _runmng(external_force: float, **root_options):
    return SimpleNamespace(
        config=SimpleNamespace(
            reactions=SimpleNamespace(
                hydrolysis=SimpleNamespace(external_force=external_force)
            ),
            **root_options,
        )
    )


def test_hydrolysis_requires_edissoc_kimmdy_with_plumed():
    with pytest.raises(ValueError, match="edissoc_kimmdy"):
        HydrolysisReaction("hydrolysis", _runmng(external_force=-1))

    assert HydrolysisReaction("hydrolysis", _runmng(external_force=1.0))
    assert HydrolysisReaction(
        "hydrolysis",
        _runmng(external_force=-1, edissoc_kimmdy=Path("edissoc_kimmdy.dat")),
    )
