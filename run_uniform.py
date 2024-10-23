#!/usr/bin/python
from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys


# SS added
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters["ghost_mode"] = "shared_facet"
set_log_active(False)
set_log_level(LogLevel.ERROR) 

# SS added
iMin=2; iMax = 7

# Define parameters
gamma0 = 20.0
gamma1 = 1.0
k=1
c_frac=3.0*sqrt(pi)/4.0 # Gamma(5/2)


def usage():
  print("-h   or --help")
  print("-g g or --gamma g       to specify gamma_0")
  print("-G G or --Gamma G       to specify gamma_1")
  print("-k       to specify k")
  print("-i i or --iMin  i       to specify iMin")
  print("-j j or --jMin  j       to specify jMin")
  print("-I i or --iMax  i       to specify iMax")
  print("-J j or --jMax  j       to specify jMax")
  print(" ")
  os.system('date +%Y_%m_%d_%H-%M-%S')
  print (time.strftime("%d/%m/%Y at %H:%M:%S"))

# parse the command line
try:
  opts, args = getopt.getopt(sys.argv[1:], "hg:G:k:i:I:j:J:",
                   [
                    "help",           # obvious
                    "gamma0=",        # gamma0
                    "gamma1=",        # gamma1
                    "k=",             # degree of polynomials
                    "iMin=",          # iMin
                    "iMax=",          # iMax
                    ])

except getopt.GetoptError as err:
  # print help information and exit:
  print(err) # will print something like "option -a not recognized"
  usage()
  sys.exit(2)

for o, a in opts:
  if o in ("-h", "--help"):
    usage()
    sys.exit()
  elif o in ("-g", "--gamma"):
    gamma0 = float(a)
    print('setting:  gamma0 = %f;' % gamma0),
  elif o in ("-G", "--Gamma"):
    gamma1 = float(a)
    print('setting:  gamma1 = %f;' % gamma1),    
  elif o in ("-k"):
    k = int(a)
    print('setting:  k = %d;' % k),
  elif o in ("-i", "--iMin"):
    iMin = int(a)
    print('setting:  iMin = %f;' % iMin),
  elif o in ("-I", "--iMax"):
    iMax = int(a)
    print('setting:  iMax = %f;' % iMax),
  else:
    assert False, "unhandled option"



# save data for error
L2u_error=np.zeros((iMax-iMin,2), dtype=np.float64)
H1u_error=np.zeros((iMax-iMin,2), dtype=np.float64)
post_error=np.zeros((iMax-iMin,2), dtype=np.float64)

order_convergence=np.zeros((iMax-iMin,3), dtype=object)

# problem data
T = 0.01     # total simulation time


# define coefficients B_{n,i} on the quadrature rule
def Qw(n):
    Bn=[]
    if n==0:
        Bn.append(0.0)
    else:
        Bn.append(n**0.5*(1.5-n)+(n-1.0)**1.5)
        for i in range (1,n):
            Bn.append((n-i-1.0)**1.5+(n-i+1.0)**1.5-2.0*(n-i)**1.5)
        Bn.append(1.0)
    return Bn



#-----------------------------------------------------------------------------------------------------------------

ux=Expression(("utn*sin(pi*x[0])*sin(pi*x[1])","utn*x[0]*x[1]*(1.0-x[0])*(1.0-x[1])"),utn=1.0,degree=5)
wx=Expression(("wtn*sin(pi*x[0])*sin(pi*x[1])","wtn*x[0]*x[1]*(1.0-x[0])*(1.0-x[1])"),wtn=0.0,degree=5)


# define time function values of u w.r.t. time derivative
def uTime(tn):
    return (1.0+tn**4.0)

def wTime(tn):
    return (4.0*tn**3.0)

def fracwTime(tn):
    return (4.0*math.gamma(4)/math.gamma(4.5)*tn**3.5)

f=Expression(("(utn+fwtn)*(1.5*pi*pi*sin(pi*x[0])*sin(pi*x[1])+0.5*(2.0*x[0]-1.0)*(1.0-2.0*x[1]))",\
              "(utn+fwtn)*(-0.5*pi*pi*cos(pi*x[0])*cos(pi*x[1])+2.0*x[0]*(1-x[0])+x[1]*(1.0-x[1]))"),\
              utn=1.0,fwtn=0,degree=5)

# Cauchy-infinitestimal strain tensor
def strain(v):
    Dv=grad(v)
    return 0.5*(Dv+Dv.T)
    
#===================================================================================================================
tol=1E-15
for i in range(iMin,iMax):
    Nxy=pow(2,i)
    mesh = UnitSquareMesh(Nxy, Nxy)
    V = VectorFunctionSpace(mesh, 'DG', k)

    # Define the boundary partition
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_parts.set_all(0)

    # Mark subdomain 0 for \Gamma_N etc
    # GammaNeumann Neumann BC(left edge)
    class GammaNeumann(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0.0,tol)
    Gamma_Neumann = GammaNeumann()
    Gamma_Neumann.mark(boundary_parts, 1)


    class GammaDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near((1.0-x[0]),0.0,tol) \
                    or near((1.0-x[1]),0.0,tol)\
                    or near(x[1],0.0,tol))
    Gamma_Dirichlet = GammaDirichlet()
    Gamma_Dirichlet.mark(boundary_parts, 2)
    
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

    # Define normal vector and mesh size
    n = FacetNormal(mesh)
    h = FacetArea(mesh)
    h_avg = (h('+') + h('-'))/2
    
    # Initial condition
    ux.utn=1.0; wx.wtn=0.0;
    u, v = TrialFunction(V), TestFunction(V)

    # bilinear form for the solver  
    uh = Function(V)    
    wh = Function(V)
    oldu=Function(V)
    oldw=Function(V)
    
    
    # compute Q(wh)+Q(oldw)-wh for r.h.s.
    numDof=len(wh.vector().get_local())   #the number of degree of freedoms
  
    def Quad(oldB,newB,n):
        Sq=(newB[0]+oldB[0])*np.array(W[0:numDof])
        for i in range(1,n+1):
            Sq=np.add(Sq,(newB[i]+oldB[i])*np.array(W[numDof*(i):numDof*(i+1)]))
        return Sq
    
    # define the sum of Bn*Wn
    def fracIntegral(newB,n):
        Sq=(newB[0]+oldB[0])*np.array(W[0:numDof])
        for i in range(0,n+1):            
            Sq=np.add(Sq,(newB[i])*np.array(W[numDof*(i):numDof*(i+1)]))
        return Sq

    # approximate the exact solution to define surface traction
    U = VectorFunctionSpace(mesh, 'Lagrange', 5)
    ux.utn=1.0
    UX = interpolate(ux, U)

    # to define linear systems
    stiffness = inner(strain(u),strain(v))*dx- inner(avg(strain(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
                - inner(avg(strain(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
                + gamma0/(h_avg**gamma1)*dot(jump(u), jump(v))*dS \
                - inner(strain(u), outer(v,n))*ds(2) \
                - inner(outer(u,n), strain(v))*ds(2) \
                + gamma0/(h**gamma1)*dot(u,v)*ds(2)
    
    
    # assemble the system matrix once and for all
    A = assemble(stiffness)
  
    Nt = Nxy    # number of time steps
    dt = T/Nt      # time step

    # Initializing
    ux.utn=1.0; wx.wtn=0.0;

    oldu = project(ux,V)    #initial displacement
    oldw = project(wx,V)    #initial velocity
    
    W=[]
    W.extend(oldw.vector().get_local())
    
    P = dt/4.0*A+0.5*sqrt(dt)/c_frac*A

    
    # assemble only once, before the time stepping
    b = None 
    b2= None   
    oldB = Qw(0)
    for nt in range(0,Nt):
        # update data and solve for tn+k
        tn=dt*(nt+1);th=dt*nt;
        newB = Qw(nt+1)                          
        utn=0.5*(uTime(tn)+uTime(th));
        fwtn=0.5*(fracwTime(tn)+fracwTime(th));
        f.utn=utn; f.fwtn=fwtn;
        
        # assemble the right hand side
        L=inner(f,v)*dx+(utn+fwtn)*inner(strain(UX),outer(v,n))*ds(1)
        b = assemble(L, tensor=b)
        b2=(-dt/4.0*A)*oldw.vector().get_local()\
            -A*oldu.vector().get_local()\
            -0.5*sqrt(dt)/c_frac*A*Quad(oldB,newB,nt)
        b.add_local(b2)

        # solve the linear system to get new velocity and new displacement
        solve(P, wh.vector(), b, 'lu')   
        uh.vector()[:]=oldu.vector().get_local()\
                        +dt/2.0*(wh.vector().get_local()+oldw.vector().get_local())
        

        # update old terms
        oldw.assign(wh);oldu.assign(uh);W.extend(wh.vector().get_local())
        oldB = newB
        
    # compute error at last time step
    ux.utn = uTime(T); wx.wtn = wTime(T); utn=uTime(T); fwtn=fracwTime(T);

    err1 = errornorm(ux,uh,'L2')
    err2 = errornorm(ux,uh,'H1')
    
    L2u_error[i-iMin,0]=Nxy; L2u_error[i-iMin,1]=err1;
    H1u_error[i-iMin,0]=Nxy; H1u_error[i-iMin,1]=err2;

    # compute the posteriori error estimator
    Qwh=fracIntegral(newB,Nt)
    varpi=uh
    varpi.vector()[:]=varpi.vector().get_local()+sqrt(dt)/c_frac*Qwh
    stress = strain(varpi)
    div_stress = div(stress)

    f.utn=utn; f.fwtn=fwtn;

    local_eta1 = f+div_stress
    local_eta2 = jump(varpi)
    local_eta3 = jump(stress-(utn+fwtn)*strain(UX))
    local_eta4 = dot(stress,n)-(utn+fwtn)*dot(strain(UX),n)
    total_eta =  h_avg*h_avg*dot(local_eta1,local_eta1)*dx \
                +dot(local_eta2,local_eta2)/h_avg*dS + dot(varpi,varpi)/h*ds(2)\
                + h_avg*inner(local_eta3,local_eta3)*dS\
                + h*dot(local_eta4,local_eta4)*ds(1)

    posterror = None
    posterror = assemble(total_eta, tensor=posterror)
    posterror = sqrt(posterror)
    post_error[i-iMin,0]=Nxy; post_error[i-iMin,1]=posterror;
         
np.savetxt(f"L2_error_{k}.txt",L2u_error,fmt="%2.3e")
np.savetxt(f"H1_error_{k}.txt",H1u_error,fmt="%2.3e")
np.savetxt(f"post_error_{k}.txt",post_error,fmt="%2.3e")
        

# post-processcing to compute the convergence order

datasets = [L2u_error, H1u_error, post_error]
datanames = ["L2u_error","H1u_error","post_error"]
for i in range(0,len(datasets)):

    order_convergence[0,i] = datanames[i]
    error = datasets[i][:,1]
    for j in range(0,len(error)-1):
        order = np.log(error[j]/error[j+1])/np.log(2)
        order_convergence[j+1,i]=order

np.savetxt(f"convergence_Order_{k}.txt",order_convergence,fmt="%s")

