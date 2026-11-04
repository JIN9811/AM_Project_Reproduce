"""Geometry, coefficients and physical conventions shared by FEM and PINNs.

Field order throughout: ux, uy, sxx, syy, sxy, T, qx, qy.
Coordinates are the paper's numerical unit square. E is expressed in GPa;
alpha=1 and delta T=1 are benchmark values, not small-strain design loads.
"""
from dataclasses import dataclass, asdict
import numpy as np

FIELDS = ('ux','uy','sxx','syy','sxy','T','qx','qy')
RECTANGLES = ((.10,.34,.12,.44),(.50,.92,.18,.42),
              (.42,.94,.52,.94),(.06,.34,.60,.82))


@dataclass(frozen=True)
class Problem:
    geometry: str = 'rectangles'
    nu: float = .3
    alpha: float = 1.
    T0: float = 0.
    ratio: float | None = None
    circle_x: float = .6
    circle_y: float = .35
    circle_radius: float = .15
    material_grid: int = 50

    def __post_init__(self):
        if self.geometry not in ('circle','rectangles','homogeneous'):
            raise ValueError('geometry must be circle, rectangles or homogeneous')
        if not -1 < self.nu < .5:
            raise ValueError('isotropic stability requires -1 < nu < 0.5')
        if self.ratio is not None and self.ratio <= 0:
            raise ValueError('material ratio must be positive')
        if self.circle_radius <= 0 or self.material_grid < 2:
            raise ValueError('radius and material grid must be positive')

    @property
    def phases(self):
        # Section 4.4 convention here is explicitly E_matrix/E_inclusion.
        if self.ratio is not None:
            return 1., 1./self.ratio
        return (.3,1.) if self.geometry=='circle' else (1.,.5)

    def to_dict(self):
        return asdict(self)


def grid(n=101):
    a=np.linspace(0.,1.,n)
    x,y=np.meshgrid(a,a,indexing='xy')
    return np.column_stack((x.ravel(),y.ravel()))


def inclusion(xy, p):
    x,y=np.asarray(xy).T
    if p.geometry=='homogeneous': return np.zeros_like(x,dtype=bool)
    if p.geometry=='circle':
        return (x-p.circle_x)**2+(y-p.circle_y)**2 < p.circle_radius**2
    inside=np.zeros_like(x,dtype=bool)
    for x0,x1,y0,y1 in RECTANGLES:
        inside |= (x>x0)&(x<x1)&(y>y0)&(y<y1)
    return inside


def nodal_phase(p):
    """Average adjacent element phases on the 50x50 material grid (authors)."""
    n=p.material_grid
    a=(np.arange(n)+.5)/n
    x,y=np.meshgrid(a,a,indexing='xy')
    c=inclusion(np.c_[x.ravel(),y.ravel()],p).reshape(n,n).astype(float)
    v=np.zeros((n+1,n+1)); weights=np.zeros_like(v)
    for i in range(2):
        for j in range(2):
            v[i:i+n,j:j+n]+=c
            weights[i:i+n,j:j+n]+=1
    return v/weights


def material(xy,p,mode='sharp'):
    xy=np.asarray(xy)
    if mode=='sharp': phase=inclusion(xy,p).astype(float)
    elif mode=='nodal':
        n=p.material_grid
        a=np.clip(xy*n,0,n)
        ij=np.minimum(a.astype(int),n-1); f=a-ij
        ix,iy=ij.T; fx,fy=f.T; v=nodal_phase(p)
        phase=((1-fx)*(1-fy)*v[iy,ix]+fx*(1-fy)*v[iy,ix+1]
               +(1-fx)*fy*v[iy+1,ix]+fx*fy*v[iy+1,ix+1])
    else: raise ValueError('material mode must be sharp or nodal')
    em,ei=p.phases
    e=em+(ei-em)*phase
    if p.geometry=='homogeneous': e=np.ones_like(e)
    return e,e.copy()


def constitutive(grad_u,T,E,nu=.3,alpha=1.,T0=0.):
    """3D isotropic plane strain, sigma_zz is eliminated after constitutive law.

    grad_u[...,i,j] = du_i/dx_j. Includes eps^e_zz=-alpha*(T-T0).
    sigma_xy=mu*(du_x/dy+du_y/dx), avoiding the printed Eq. (49) typo.
    """
    E=np.asarray(E); T=np.asarray(T)
    mu=E/(2*(1+nu)); lam=E*nu/((1+nu)*(1-2*nu))
    ex=grad_u[...,0,0]; ey=grad_u[...,1,1]
    b=E*alpha*(T-T0)/(1-2*nu)
    return np.stack(((lam+2*mu)*ex+lam*ey-b,
                     lam*ex+(lam+2*mu)*ey-b,
                     mu*(grad_u[...,0,1]+grad_u[...,1,0])),axis=-1)


def homogeneous_exact(xy,p):
    """Exact roller-boundary solution, derived independently of FEM/PINN code."""
    x,y=np.asarray(xy).T
    e=1.; mu=e/(2*(1+p.nu)); lam=e*p.nu/((1+p.nu)*(1-2*p.nu))
    b=e*p.alpha/(1-2*p.nu); a=b/(2*(lam+2*mu))
    ux=a*x*(1-x); uy=np.zeros_like(x); T=1-x
    sxx=np.full_like(x,-b/2+b*p.T0)
    syy=lam*a*(1-2*x)-b*(1-x-p.T0)
    return np.column_stack((ux,uy,sxx,syy,0*x,T,np.ones_like(x),0*x))
