"""Actual FEniCS/DOLFIN quadrilateral thermoelastic solver; run under WSL.

No numerical substitute is used. The thermal problem is solved first because
steady thermoelasticity has one-way coupling for the constitutive assumptions.
"""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import dolfin as df
from .physics import Problem, material, grid, FIELDS


def _sample_q1(f,xy,n):
    # DOLFIN's point-search is simplex-only. Q1 interpolation on this known
    # Cartesian quad mesh avoids that unsupported search, including on edges.
    coords=f.function_space().tabulate_dof_coordinates().reshape(-1,2)
    ij=np.rint(coords*n).astype(int)
    nodal=np.empty((n+1,n+1)); nodal[ij[:,1],ij[:,0]]=f.vector().get_local()
    a=np.clip(xy*n,0,n); base=np.minimum(a.astype(int),n-1); frac=a-base
    ix,iy=base.T; fx,fy=frac.T
    return ((1-fx)*(1-fy)*nodal[iy,ix]+fx*(1-fy)*nodal[iy,ix+1]
            +(1-fx)*fy*nodal[iy+1,ix]+fx*fy*nodal[iy+1,ix+1])


def _relative_residual(A,b,x):
    r=b.copy(); A.mult(x,r); r.axpy(-1.,b)
    return float(r.norm('l2')/max(b.norm('l2'),1e-30))


def _write_cell_fields(out,mesh,E,values):
    """DOLFIN's ordinary Function export recovers DG values at vertices.

    Write explicit Cell attributes instead, preserving discontinuities and all
    six scalar fields in a single ParaView-compatible XDMF/HDF5 dataset.
    """
    import h5py
    vertices=mesh.coordinates();cells=mesh.cells().astype(np.int64)
    corners=vertices[cells];relative=corners-corners.mean(1,keepdims=True)
    order=np.argsort(np.arctan2(relative[:,:,1],relative[:,:,0]),axis=1)
    cells=np.take_along_axis(cells,order,axis=1)
    arrays={'E':E,**{name:values[:,i] for i,name in enumerate(('sxx','syy','sxy','qx','qy'))}}
    with h5py.File(out/'raw_cell_fields.h5','w') as f:
        f.create_dataset('vertices',data=vertices);f.create_dataset('cells',data=cells)
        for name,a in arrays.items():f.create_dataset(name,data=a)
    root=ET.Element('Xdmf',Version='3.0');domain=ET.SubElement(root,'Domain')
    g=ET.SubElement(domain,'Grid',Name='Raw cell averages',GridType='Uniform')
    topology=ET.SubElement(g,'Topology',TopologyType='Quadrilateral',NumberOfElements=str(len(cells)))
    def data(parent,name,dimensions,datatype='Float'):
        item=ET.SubElement(parent,'DataItem',Dimensions=dimensions,DataType=datatype,Precision='8',Format='HDF')
        item.text=f'raw_cell_fields.h5:/{name}'
    data(topology,'cells',f'{len(cells)} 4','Int')
    geometry=ET.SubElement(g,'Geometry',GeometryType='XY');data(geometry,'vertices',f'{len(vertices)} 2')
    for name in arrays:
        attr=ET.SubElement(g,'Attribute',Name=name,AttributeType='Scalar',Center='Cell')
        data(attr,name,str(len(cells)))
    ET.indent(root)
    ET.ElementTree(root).write(out/'raw_cell_fields.xdmf',encoding='utf-8',xml_declaration=True)


def solve(p, n=100, xy=None, output=None, coefficient_mode='sharp'):
    if n < 2: raise ValueError('need at least two elements per direction')
    if df.MPI.size(df.MPI.comm_world)!=1:
        raise RuntimeError('This nodal-recovery implementation requires one MPI rank.')
    df.set_log_level(40)
    t0=time.perf_counter()
    mesh=df.UnitSquareMesh.create([n,n],df.CellType.Type.quadrilateral)
    V=df.FunctionSpace(mesh,'Q',1)
    U=df.VectorFunctionSpace(mesh,'Q',1)
    W=df.FunctionSpace(mesh,'DQ',0)
    mid=np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    ev,kv=material(mid,p,coefficient_mode)
    if coefficient_mode=='sharp':
        E=df.Function(W,name='Young_modulus'); k=df.Function(W,name='conductivity')
        ec=E.vector().get_local(); kc=k.vector().get_local()
        for c in df.cells(mesh):
            dof=W.dofmap().cell_dofs(c.index())[0]
            ec[dof]=ev[c.index()]; kc[dof]=kv[c.index()]
        E.vector().set_local(ec); k.vector().set_local(kc)
    elif coefficient_mode=='nodal':
        if n%p.material_grid:
            raise ValueError('matched nodal-material FEM needs n divisible by material_grid')
        E=df.Function(V,name='Young_modulus'); k=df.Function(V,name='conductivity')
        coords=V.tabulate_dof_coordinates().reshape(-1,2)
        ec,kc=material(coords,p,'nodal')
        E.vector().set_local(ec);k.vector().set_local(kc)
    else:raise ValueError('unknown coefficient mode')
    dx=df.Measure('dx',domain=mesh,metadata={'quadrature_degree':4})
    left='near(x[0],0.0) && on_boundary'
    right='near(x[0],1.0) && on_boundary'
    vertical='(near(x[0],0.0) || near(x[0],1.0)) && on_boundary'
    horizontal='(near(x[1],0.0) || near(x[1],1.0)) && on_boundary'
    T=df.Function(V,name='temperature')
    t,v=df.TrialFunction(V),df.TestFunction(V)
    aT=k*df.dot(df.grad(t),df.grad(v))*dx
    lT=df.Constant(0.)*v*dx
    tbcs=[df.DirichletBC(V,df.Constant(1.),left),
          df.DirichletBC(V,df.Constant(0.),right)]
    AT,bT=df.assemble_system(aT,lT,tbcs)
    df.solve(AT,T.vector(),bT,'mumps')
    mu=E/(2*(1+p.nu)); lam=E*p.nu/((1+p.nu)*(1-2*p.nu))
    I=df.Identity(2)
    eps=lambda a: df.sym(df.grad(a))
    elastic=lambda a: 2*mu*eps(a)+lam*df.tr(eps(a))*I
    thermal=E*p.alpha*(T-p.T0)/(1-2*p.nu)
    u=df.Function(U,name='displacement')
    du,w=df.TrialFunction(U),df.TestFunction(U)
    aU=df.inner(elastic(du),eps(w))*dx
    lU=thermal*df.tr(eps(w))*dx
    ubcs=[df.DirichletBC(U.sub(0),df.Constant(0.),vertical),
          df.DirichletBC(U.sub(1),df.Constant(0.),horizontal)]
    AU,bU=df.assemble_system(aU,lU,ubcs)
    df.solve(AU,u.vector(),bU,'mumps')
    sigma=elastic(u)-thermal*I
    q=-k*df.grad(T)
    # Cell averages are retained. Continuous plots use explicit patch recovery,
    # like the nodally averaged FE data supplied by the authors.
    Z=df.VectorFunctionSpace(mesh,'DQ',0,dim=5)
    cellfield=df.project(df.as_vector((sigma[0,0],sigma[1,1],sigma[0,1],q[0],q[1])),
                         Z,solver_type='mumps',form_compiler_parameters={'quadrature_degree':4})
    cv=np.empty((len(mid),5))
    for j,f in enumerate(cellfield.split(deepcopy=True)):
        values=f.vector().get_local()
        for c in df.cells(mesh):
            cv[c.index(),j]=values[f.function_space().dofmap().cell_dofs(c.index())[0]]
    nv=mesh.num_vertices(); patch=np.zeros((nv,5)); weight=np.zeros(nv)
    for c in df.cells(mesh):
        ids=c.entities(0); vol=c.volume()
        patch[ids]+=cv[c.index()]*vol; weight[ids]+=vol
    patch/=weight[:,None]
    recovered=[]; vd=df.vertex_to_dof_map(V)
    for j in range(5):
        f=df.Function(V,name=('sxx','syy','sxy','qx','qy')[j])
        values=f.vector().get_local(); values[vd]=patch[:,j]
        f.vector().set_local(values); recovered.append(f)
    if xy is None: xy=grid(101)
    xy=np.asarray(xy)
    uv=np.column_stack([_sample_q1(f,xy,n) for f in u.split(deepcopy=True)])
    tv=_sample_q1(T,xy,n)
    rv=np.column_stack([_sample_q1(f,xy,n) for f in recovered])
    fields=np.column_stack((uv,rv[:,:3],tv,rv[:,3:]))
    # Reactions from the unmodified heat system are conservative, unlike the
    # pointwise boundary gradient of a finite-resolution primal solution.
    rawAT=df.assemble(aT); rawbT=df.assemble(lT)
    reactions=rawbT.copy(); rawAT.mult(T.vector(),reactions); reactions.axpy(-1,rawbT)
    lr=list(tbcs[0].get_boundary_values()); rr=list(tbcs[1].get_boundary_values())
    heat_left=float(np.sum(reactions.get_local()[lr]))
    heat_right=float(np.sum(reactions.get_local()[rr]))
    diagnostics={
        'thermal_relative_residual':_relative_residual(AT,bT,T.vector()),
        'mechanical_relative_residual':_relative_residual(AU,bU,u.vector()),
        'heat_left_reaction':heat_left,'heat_right_reaction':heat_right,
        'heat_balance':heat_left+heat_right,
        'conductive_energy':float(df.assemble(k*df.dot(df.grad(T),df.grad(T))*dx)),
        'mechanical_virtual_work':float(df.assemble(df.inner(sigma,eps(u))*dx)),
        'elements':mesh.num_cells(),'vertices':mesh.num_vertices(),
        'elapsed_seconds':time.perf_counter()-t0,'fenics_version':df.__version__,
    }
    result={'xy':xy,'fields':fields,'diagnostics':diagnostics,
            'cell_xy':mid,'cell_fields':cv,'cell_E':ev,
            'vertices':mesh.coordinates(),'cells':mesh.cells()}
    if output:
        out=Path(output); out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'solution.npz',**{k:v for k,v in result.items() if k!='diagnostics'})
        (out/'metadata.json').write_text(json.dumps({'problem':p.to_dict(),'n':n,'coefficient_mode':coefficient_mode,
            'field_names':FIELDS,'diagnostics':diagnostics},indent=2))
        with df.XDMFFile(str(out/'solution.xdmf')) as f:
            f.parameters['functions_share_mesh']=True
            f.parameters['rewrite_function_mesh']=False
            f.write(T,0.); f.write(u,0.)
            for g in recovered: f.write(g,0.)
        _write_cell_fields(out,mesh,ev,cv)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',choices=['circle','rectangles','homogeneous'],default='rectangles')
    parser.add_argument('--n',type=int,default=100)
    parser.add_argument('--eval-grid',type=int,default=101)
    parser.add_argument('--ratio',type=float)
    parser.add_argument('--coefficient-mode',choices=['sharp','nodal'],default='sharp')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    r=solve(Problem(args.geometry,ratio=args.ratio),args.n,grid(args.eval_grid),args.output,args.coefficient_mode)
    print(json.dumps(r['diagnostics'],indent=2),flush=True)


if __name__=='__main__': main()
