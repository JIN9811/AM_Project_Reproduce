"""First-order mixed PINNs with eight independent tanh subnetworks.

The head dimension batches independent matrix products. Weights are never
shared between fields. Differentiating each head's own copy of x,y computes
all first derivatives in one autograd call without mixing field derivatives.
"""
import numpy as np
import torch
from torch import nn
from .physics import Problem, grid, material


class ParallelMLP(nn.Module):
    def __init__(self,heads,width=40,layers=5,input_dim=2):
        super().__init__()
        dims=[input_dim]+[width]*layers+[1]
        self.weights=nn.ParameterList()
        self.biases=nn.ParameterList()
        for a,b in zip(dims[:-1],dims[1:]):
            w=nn.Parameter(torch.empty(heads,a,b))
            for h in range(heads): nn.init.xavier_normal_(w[h],gain=1.)
            self.weights.append(w);self.biases.append(nn.Parameter(torch.zeros(heads,1,b)))

    def forward(self,z):
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=torch.bmm(z,w)+b
            if i<len(self.weights)-1:z=torch.tanh(z)
        return z[:,:,0]


class MixedPINN(nn.Module):
    def __init__(self,width=40,layers=5,hard=True,parametric=False,jacobian_backend='autograd'):
        super().__init__()
        if jacobian_backend not in ('autograd','jets'):raise ValueError('unknown Jacobian backend')
        self.jacobian_backend=jacobian_backend
        self.hard=hard;self.parametric=parametric
        self.width=width;self.layers=layers
        self.mechanical=ParallelMLP(5,width,layers,4 if parametric else 2)
        self.thermal=ParallelMLP(3,width,layers,4 if parametric else 2)

    def _group(self,xy,kind,derivatives=False,create_graph=True,parameters=None):
        if derivatives and self.jacobian_backend=='jets':
            return self._jet_group(xy,kind,create_graph,parameters)
        model=self.mechanical if kind=='mechanical' else self.thermal
        nh=5 if kind=='mechanical' else 3
        z=xy[None].expand(nh,-1,-1).clone()
        if derivatives:z.requires_grad_(True)
        inputs=z
        if self.parametric:
            if parameters is None:raise ValueError('parametric model needs global E and k ratios')
            # Global ratios identify the entire BVP. Material properties are
            # detached from spatial differentiation, exactly as prescribed data.
            params=parameters[None].expand(nh,-1,-1)
            inputs=torch.cat((z,params),-1)
        raw=model(2*inputs-1) if not self.parametric else model(torch.cat((2*z-1,params),-1))
        if self.hard:
            x=z[:,:,0];y=z[:,:,1]
            if kind=='mechanical':
                f=torch.stack((x[0]*(1-x[0])*raw[0],y[1]*(1-y[1])*raw[1],
                               raw[2],raw[3],raw[4]))
            else:f=torch.stack((1-x[0]+x[0]*(1-x[0])*raw[0],raw[1],raw[2]))
        else:f=raw
        if not derivatives:return f.T
        g=torch.autograd.grad(f.sum(),z,create_graph=create_graph)[0]
        return f.T,g.permute(1,0,2)

    def _jet_group(self,xy,kind,create_graph=True,parameters=None):
        from .jets import forward_jet
        model=getattr(self,kind);nh=5 if kind=='mechanical' else 3
        z=xy[None].expand(nh,-1,-1)
        inputs=2*z-1
        if self.parametric:
            if parameters is None:raise ValueError('parametric model needs global E and k ratios')
            inputs=torch.cat((inputs,parameters[None].expand(nh,-1,-1)),-1)
        raw,jac=forward_jet(model,inputs);jac=2*jac
        if self.hard:
            x=z[:,:,0];y=z[:,:,1];zero=torch.zeros_like(raw[0])
            if kind=='mechanical':
                f=torch.stack((x[0]*(1-x[0])*raw[0],y[1]*(1-y[1])*raw[1],*raw[2:]))
                gx=x[0,:,None]*(1-x[0,:,None])*jac[0]+torch.stack(((1-2*x[0])*raw[0],zero),1)
                gy=y[1,:,None]*(1-y[1,:,None])*jac[1]+torch.stack((zero,(1-2*y[1])*raw[1]),1)
                jac=torch.stack((gx,gy,*jac[2:]))
            else:
                f=torch.stack((1-x[0]+x[0]*(1-x[0])*raw[0],*raw[1:]))
                gt=x[0,:,None]*(1-x[0,:,None])*jac[0]+torch.stack(((1-2*x[0])*raw[0]-1,zero),1)
                jac=torch.stack((gt,*jac[1:]))
        else:f=raw
        if not create_graph:jac=jac.detach()
        return f.T,jac.permute(1,0,2)

    def predict(self,xy,parameters=None):
        return torch.cat((self._group(xy,'mechanical',parameters=parameters),
                          self._group(xy,'thermal',parameters=parameters)),1)

    def state(self,xy,create_graph=True,parameters=None):
        fm,gm=self._group(xy,'mechanical',True,create_graph,parameters)
        ft,gt=self._group(xy,'thermal',True,create_graph,parameters)
        return torch.cat((fm,ft),1),torch.cat((gm,gt),1)


def make_batch(p,n=51,device='cpu',dtype=torch.float64,seed=42,circle_points=5000,left_refine=0):
    if left_refine<0:raise ValueError('left_refine must be nonnegative')
    xy=grid(n)
    if p.geometry=='circle' and n>=51 and circle_points>len(xy):
        xy=np.r_[xy,np.random.default_rng(seed).random((circle_points-len(xy),2))]
    if left_refine:
        xy=np.r_[xy,np.column_stack((np.zeros(left_refine),np.random.default_rng(seed+91).uniform(0,1,left_refine)))]
    E,k=material(xy,p,'nodal')
    x,y=xy.T
    masks={'left':x==0,'right':x==1,'bottom':y==0,'top':y==1,
           'body':(x>0)&(x<1)&(y>0)&(y<1)}
    if not masks['body'].any():raise ValueError('need interior collocation points')
    to=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    b={'xy':to(xy),'E':to(E),'k':to(k)}
    b.update({name:torch.as_tensor(a,device=device) for name,a in masks.items()})
    if p.ratio is not None:
        b['parameters']=to(np.full((len(xy),2),np.log(p.ratio)/np.log(10.)))
    return b


def physical_state(f,g,E,k,p):
    ex=g[:,0,0];ey=g[:,1,1];gamma=g[:,0,1]+g[:,1,0]
    mu=E/(2*(1+p.nu));lam=E*p.nu/((1+p.nu)*(1-2*p.nu))
    beta=E*p.alpha*(f[:,5]-p.T0)/(1-2*p.nu)
    stress=torch.stack(((lam+2*mu)*ex+lam*ey-beta,
                        lam*ex+(lam+2*mu)*ey-beta,mu*gamma),1)
    flux=-k[:,None]*g[:,5,:]
    eth=p.alpha*(f[:,5]-p.T0)
    # Complete 3D plane-strain elastic energy includes sigma_zz*eps^e_zz.
    szz=lam*(ex+ey)-beta
    energy=.5*(stress[:,0]*(ex-eth)+stress[:,1]*(ey-eth)
               +stress[:,2]*gamma-szz*eth)
    work=stress[:,0]*ex+stress[:,1]*ey+stress[:,2]*gamma
    return {'stress':stress,'flux':flux,'elastic_energy':energy,
            'mechanical_work':work,'gamma':gamma}


def mixed_loss(model,b,p,energy='balance',energy_weight=1.,detach_thermal=False,active='coupled'):
    pars=b.get('parameters')
    # In a thermal block, compute only thermal subnetworks. In a mechanical
    # block, temperature is a prescribed, detached field, not a trainable target.
    if active=='thermal':
        ft,gt=model._group(b['xy'],'thermal',True,True,pars)
        f=torch.cat((torch.zeros((len(ft),5),device=ft.device,dtype=ft.dtype),ft),1)
        g=torch.cat((torch.zeros((len(ft),5,2),device=ft.device,dtype=ft.dtype),gt),1)
    elif active=='mechanical':
        fm,gm=model._group(b['xy'],'mechanical',True,True,pars)
        ft=model._group(b['xy'],'thermal',parameters=pars).detach()
        f=torch.cat((fm,ft),1)
        g=torch.cat((gm,torch.zeros((len(ft),3,2),device=fm.device,dtype=fm.dtype)),1)
    else:f,g=model.state(b['xy'],parameters=pars)
    if detach_thermal:
        f=torch.cat((f[:,:5],f[:,5:].detach()),1)
        g=torch.cat((g[:,:5],g[:,5:].detach()),1)
    s=physical_state(f,g,b['E'],b['k'],p)
    body=b['body'];L=b['left'];R=b['right'];B=b['bottom'];U=b['top']
    zero=f.sum()*0
    mse=lambda a: a.square().mean()
    terms={}
    if active!='thermal':
        terms['M_connection']=((f[:,2:5]-s['stress']).square().sum(1)).mean()
        div=torch.stack((g[:,2,0]+g[:,4,1],g[:,4,0]+g[:,3,1]),1)
        terms['M_equilibrium']=div[body].square().sum(1).mean()
        terms['M_dirichlet']=mse(f[L,0])+mse(f[R,0])+mse(f[B,1])+mse(f[U,1])
        # Roller supports prescribe only normal displacement. Tangential
        # traction is zero: sigma_xy=0 on all four edges (missing in Eq.56).
        terms['M_neumann']=sum(mse(f[mask,4]) for mask in (L,R,B,U))
        if energy=='balance':em=s['mechanical_work'][body].mean().square()
        elif energy=='paper':
            eth=p.alpha*(f[:,5]-p.T0)
            # Expanded Eq.52, including its printed shear contraction and no
            # zz term. Stress itself always uses the corrected plane-strain law.
            em=(.5*(s['stress'][:,0]*(g[:,0,0]-eth)
                    +s['stress'][:,1]*(g[:,1,1]-eth)
                    +s['stress'][:,2]*s['gamma']/2)[body].mean()).abs()
        elif energy=='authors':
            sig=s['stress']
            ext=(-(sig[L,0]*f[L,0]+sig[L,2]*f[L,1]).mean()
                 +(sig[R,0]*f[R,0]+sig[R,2]*f[R,1]).mean()
                 -(sig[B,2]*f[B,0]+sig[B,1]*f[B,1]).mean()
                 +(sig[U,2]*f[U,0]+sig[U,1]*f[U,1]).mean())
            em=(.5*s['mechanical_work'][body].mean()-ext).abs()
        elif energy=='variational':
            # Mechanics varies u at prescribed T. Letting elastic energy vary
            # T would invent reverse coupling absent from div(k grad T)=0.
            fixed=torch.cat((f[:,:5],f[:,5:].detach()),1)
            em=physical_state(fixed,g,b['E'],b['k'],p)['elastic_energy'][body].mean()
        elif energy=='none':em=zero
        else:raise ValueError(f'unknown energy formulation: {energy}')
        terms['M_energy']=energy_weight*em
    if active!='mechanical':
        terms['T_connection']=(f[:,6:8]-s['flux']).square().sum(1).mean()
        terms['T_equilibrium']=mse((g[:,6,0]+g[:,7,1])[body])
        terms['T_dirichlet']=mse(f[L,5]-1)+mse(f[R,5])
        terms['T_neumann']=mse(f[B,7])+mse(f[U,7])
        conduction=(b['k']*g[:,5].square().sum(1))[body].mean()
        # Outward normal surface integral of q*T on all four boundaries.
        surf=(-(f[L,6]*f[L,5]).mean()+(f[R,6]*f[R,5]).mean()
              -(f[B,7]*f[B,5]).mean()+(f[U,7]*f[U,5]).mean())
        if energy=='balance':et=(-conduction-surf).square()
        elif energy=='paper':
            # Expanded Eq.57: 1/2 q.grad(T) + left qx*T. Kept as a
            # literal reproduction option; this is not a vanishing energy law.
            et=(-.5*conduction+(s['flux'][L,0]*f[L,5]).mean()).abs()
        elif energy=='authors':
            qc=s['flux']
            ext=(-(qc[L,0]*f[L,5]).mean()+(qc[R,0]*f[R,5]).mean()
                 -(qc[B,1]*f[B,5]).mean()+(qc[U,1]*f[U,5]).mean())
            et=(-.5*conduction-ext).abs()
        elif energy=='variational':et=.5*conduction
        elif energy=='none':et=zero
        else:raise ValueError(f'unknown energy formulation: {energy}')
        terms['T_energy']=energy_weight*et
    terms['mechanical']=sum((v for k,v in terms.items() if k.startswith('M_')),zero)
    terms['thermal']=sum((v for k,v in terms.items() if k.startswith('T_')),zero)
    terms['total']=terms['mechanical']+terms['thermal']
    return terms
