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

tol=1E-16

# Define parameters
gamma0 = 20.0   # penalty parameter
gamma1 = 1.0    # penalty parameter
k = 2           # degree of polynomials

lame1 = Constant(0.456)     # first Lame's parameter
lame2 = Constant(0.228)     # second Lame's parameter

varphi0 = Constant(0.685)    # Young's modulus coefficient


# problem data
Nx = 60 
Ny = 30  
Nt = 50
dt = 0.001      # time step
T  = Nt*dt     # total simulation time

mesh = RectangleMesh(Point(0.0, 0.0), Point(2, 1), Nx, Ny,"left")


#-----------------------------------------------------------------------------------------------------------------

ux = Expression(("0.0","0.0"), tn=0, degree=5)

# define the body force f
amp = 1 # amplitude of body force
f = Expression(("amp","0.0"),amp=amp, degree=5) 

def strain(v):
    Dv=grad(v)
    return 0.5*(Dv+Dv.T)

# Stress tensor (elasticity part)
def sigma(v):
    return  lame1*tr(strain(v))*Identity(2) +2*lame2*strain(v)
    
V = VectorFunctionSpace(mesh, 'DG', k)

# Define the boundary partition
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_parts.set_all(0)

# Mark subdomain 0 for \Gamma_N etc
# homogeneous GammaNeumann Neumann BC(top and bottom edges)
class GammaNeumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near((1.0-x[1]),0.0,tol)\
                    or near(x[1],0.0,tol))
Gamma_Neumann = GammaNeumann()
Gamma_Neumann.mark(boundary_parts, 1)

# homogeneous Dirichlet BC (left and right edges) 
class GammaDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0],2.0,tol) \
                or near(x[0],0.0,tol))

Gamma_Dirichlet = GammaDirichlet()
Gamma_Dirichlet.mark(boundary_parts, 2)

dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

# Define normal vector and mesh size
n = FacetNormal(mesh)
h = FacetArea(mesh)
h_avg = (h('+') + h('-'))/2

u, v = TrialFunction(V), TestFunction(V)

# bilinear form for the solver  
uh = Function(V,name="Displacement")
oldu=Function(V)

# to define linear systems
stiffness = inner(sigma(u),strain(v))*dx- inner(avg(sigma(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
            - inner(avg(sigma(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
            + gamma0/(h_avg**gamma1)*dot(jump(u), jump(v))*dS \
            - inner(sigma(u), outer(v,n))*ds(2) \
            - inner(outer(u,n), sigma(v))*ds(2) \
            + gamma0/(h**gamma1)*dot(u,v)*ds(2)


# assemble the system matrix once and for all
A = assemble(stiffness)


oldu = project(ux,V)    # zero initial displacement

P = varphi0/2.0*A

# assemble only once, before the time stepping
b = None 
b2= None
fileResult_u = XDMFFile("elastic_vecfield_norm/output_displacement.xdmf")
fileResult_u.parameters["flush_output"] = True
fileResult_u.parameters["functions_share_mesh"] = True

tn = 0.0

TIME = [0.0]
DISP = [0.0]
for nt in range(0,Nt):
    tn +=dt

    # assemble the right hand side
    L=dot(f,v)*dx

    b = assemble(L, tensor=b)
    b2= -varphi0/2.0*A*oldu.vector().get_local()
    b.add_local(b2)

    # solve the linear system to get new velocity and new displacement
    solve(P, uh.vector(), b, 'lu')   
    
    # update old terms
    oldu.assign(uh);
    
    xval, yval = uh(0.5,1)

    TIME.append(tn)
    DISP.append(xval)
    fileResult_u.write(uh, tn)

stress = sigma(uh)
div_stress = div(stress)


local_eta1 = f+div_stress
local_eta2 = jump(uh)
local_eta3 = jump(stress)
local_eta4 = stress
total_eta = h_avg*h_avg*dot(local_eta1,local_eta1)*dx\
            +dot(local_eta2,local_eta2)/h_avg*dS + dot(uh,uh)/h*ds(2)\
            +h_avg*inner(local_eta3,local_eta3)*dS\
            +h*inner(local_eta4,local_eta4)*ds(1)

posterror = None
posterror = assemble(total_eta, tensor=posterror)
posterror = sqrt(posterror)
print(posterror)
np.savetxt("Time_elasticity.txt",TIME , fmt="%.5e")
np.savetxt("DISP(elasticity).txt",DISP , fmt="%.5e")