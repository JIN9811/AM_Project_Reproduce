"""Catch wrong thermal factors, shear conventions and misplaced inclusions."""
import numpy as np
import pytest
from harandi.physics import Problem, material, constitutive, homogeneous_exact, grid


def test_plane_strain_includes_out_of_plane_thermal_strain():
    s = constitutive(np.zeros((1, 2, 2)), np.ones(1), np.ones(1), .3, 1.)
    np.testing.assert_allclose(s, [[-2.5, -2.5, 0]], atol=1e-13)


def test_engineering_shear_has_no_factor_of_two_error():
    g = np.array([[[0., .02], [0., 0.]]])
    s = constitutive(g, np.zeros(1), np.ones(1), .3, 1.)
    np.testing.assert_allclose(s, [[0, 0, .02/2.6]], atol=1e-13)


def test_rectangles_match_independent_author_phase_data():
    from pathlib import Path
    a = np.loadtxt(Path(__file__).parents[1]/'data/authors_geometry2_nodes.txt')
    e, k = material(a[:, 2:4], Problem('rectangles'), mode='nodal')
    np.testing.assert_allclose(e, a[:, 4], atol=2e-12)
    np.testing.assert_allclose(k, a[:, 4], atol=2e-12)


def test_circle_location_and_material_contrast():
    p=Problem('circle')
    e,k=material(np.array([[.6,.35],[0.,0.],[.6,.51]]),p,'sharp')
    np.testing.assert_allclose(e,[1.,.3,.3]); np.testing.assert_allclose(k,e)


def test_homogeneous_exact_solution_and_boundary_values():
    xy=np.array([[0.,.4],[.5,.5],[1.,.6]])
    f=homogeneous_exact(xy,Problem('homogeneous'))
    np.testing.assert_allclose(f[:,0],[0,13/56,0],atol=1e-13)
    np.testing.assert_allclose(f[:,1],0,atol=1e-13)
    np.testing.assert_allclose(f[:,2],-1.25,atol=1e-13)
    np.testing.assert_allclose(f[:,5],[1,.5,0],atol=1e-13)
    np.testing.assert_allclose(f[:,6],1,atol=1e-13)
    np.testing.assert_allclose(f[:,7],0,atol=1e-13)


def test_nonphysical_materials_are_rejected():
    with pytest.raises(ValueError): Problem('circle',nu=.5)
    with pytest.raises(ValueError): Problem('rectangles',ratio=-1)
