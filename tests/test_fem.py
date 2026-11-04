"""Integration checks run with Ubuntu /usr/bin/python3 and real FEniCS."""
import numpy as np
import pytest
pytest.importorskip('dolfin')
from harandi.fem import solve
from harandi.physics import Problem, grid, homogeneous_exact


@pytest.mark.parametrize('coefficient_mode',['sharp','nodal'])
def test_fem_recovers_analytic_thermoelastic_solution(coefficient_mode):
    xy=grid(17)
    result=solve(Problem('homogeneous',material_grid=4),16,xy,coefficient_mode=coefficient_mode)
    exact=homogeneous_exact(xy,Problem('homogeneous'))
    np.testing.assert_allclose(result['fields'][:,[0,1,5]],exact[:,[0,1,5]],atol=2e-10)
    np.testing.assert_allclose(result['fields'][:,2],-1.25,atol=2e-10)
    np.testing.assert_allclose(result['fields'][:,6],1,atol=2e-10)
    assert result['diagnostics']['thermal_relative_residual'] < 1e-10
    assert result['diagnostics']['mechanical_relative_residual'] < 1e-10
    assert abs(result['diagnostics']['heat_balance']) < 1e-10


def test_raw_xdmf_preserves_cell_fields_and_material(tmp_path):
    import xml.etree.ElementTree as ET
    import h5py
    solve(Problem('homogeneous'),4,grid(5),tmp_path)
    tree=ET.parse(tmp_path/'raw_cell_fields.xdmf')
    attrs=tree.findall('.//Attribute')
    assert {a.attrib['Name'] for a in attrs}=={'E','sxx','syy','sxy','qx','qy'}
    assert all(a.attrib['Center']=='Cell' for a in attrs)
    with h5py.File(tmp_path/'raw_cell_fields.h5','r') as f:
        assert f['cells'].shape==(16,4)
        np.testing.assert_allclose(f['qx'][:],1,atol=1e-12)
