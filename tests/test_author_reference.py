"""Regression against external FEAP data, not against our own solver output."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('dolfin')
from harandi.fem import solve
from harandi.physics import Problem,grid


def test_rectangular_fem_agrees_with_author_temperature_and_displacement_shape():
    p=Path(__file__).parents[1]
    a=np.genfromtxt(p/'data/authors_geometry2_fe.csv',delimiter=',',names=True)
    names=a.dtype.names
    r=solve(Problem('rectangles'),50,grid(51))['fields']
    T=a[names[2]]
    assert np.linalg.norm(r[:,5]-T)/np.linalg.norm(T)<1e-5
    ref=np.column_stack((a[names[0]],a[names[1]]))
    # Independent data contains an almost exact 0.98 amplitude factor. Report
    # the raw discrepancy; do not silently change alpha to fit the labels.
    assert np.linalg.norm(r[:,:2]-ref)/np.linalg.norm(ref)<.025
    factor=np.sum(r[:,:2]*ref)/np.sum(r[:,:2]**2)
    assert abs(factor-.98)<1e-5
