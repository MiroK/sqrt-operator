from dolfin import *

from block.object_pool import vec_pool
from block.block_base import block_base

from math import ceil
import numpy as np


class BPOperator(block_base):
    '''
    Solver for equations of the type L ^ s u = b. Solution u is computed as 
    u = L^{-s} b where action of L^{-s} is computed from using Bochner integral 
    representation. The numerical integration requires solutions of shifted 
    problems (L + p*I).

    '''
    def __init__(self, V, s, solve_shifted_problem=None, compute_k=0.5):
        '''
        Solve L^s x = b, where L is operator over V. 

        INPUT:
          s = fractionality
          V = function space
          solve_shifted_problem = (A, x, b) -> (niters, solution of A*x = b)
          compute_k = (fract, dim(V), hmin(mesh(V))) -> control for number of integration points
                      small k is greater accuracy
        '''
        assert between(s, (0, 1))
        self.V = V
        # This are really only for computing k
        self.mesh_hmin = MPI.min(V.mesh().mpi_comm(), V.mesh().hmin())
        # Monitors
        self.nsolves, self.niters = 0, 0
        # Fractionality for deciding action
        self.s = s
        # Specialization for L
        
        if solve_shifted_problem is None:
            # Fall back to direct solve
            self.solve_shifted_problem = lambda A, x, b: (solve(A, x, b), (1, x))[1]
        else:
            self.solve_shifted_problem = solve_shifted_problem

        if isinstance(compute_k, (int, float)):
            # Fall back
            self.compute_k = lambda a, b, c, k=compute_k: k
        else:
            self.compute_k = compute_k

    def matvec(self, b):
        x, nsolves, niters = self.apply_negative_power(b, self.s)
        self.nsolves += nsolves
        self.niters += niters

        return x

    @vec_pool
    def create_vec(self, dim=1):
        return Function(self.V).vector()
        
    def apply_negative_power(self, b, beta):
        '''
        Using exponentially convergenging quadrature from Bonito&Pesciak

          Numerical approximation of fractional powers of elliptic operators
        '''
        assert between(beta, (0, 1))
        
        x = self.create_vec(1); x.zero()
        xk = x.copy()
        # NOTE: k controls the number of quadrature points, we let user
        # set that based on the fractianality, # of unknowns, mesh size
        k = self.compute_k(beta, b.size(), self.mesh_hmin)

        M = int(ceil(pi**2/(4.*beta*k**2)))  # cf Remark 3.1
        N = int(ceil(pi**2/(4*(1 - beta)*k**2)))

        nsolves = 0
        iter_count = 0
        for l in range(-M, N+1):
            nsolves += 1
            
            yl = l*k
            shift = exp(-2.*yl)  # equation (47) in section 3.3

            A = self.shifted_operator(shift)
            # Keep track of number of iteration in inner solves
            count, xk = self.solve_shifted_problem(A, xk, b)
            iter_count += count
            
            x.axpy(exp(2*yl*(beta - 1.)), xk)
        x *= 2*k*sin(pi*beta)/pi

        return x, nsolves, iter_count

    def shifted_problem(self, shift):
        raise NotImplementedError

    
class InvFHelmholtz(BPOperator):
    '''
    BPOperator with L = -Delta + I; fractional Helmholtz, i.e. compute 
    the solution of L u = f for f given.
    '''
    def __init__(self, V, s, bcs, solve_shifted_problem=None, compute_k=0.5):
        
        BPOperator.__init__(self, V, s, solve_shifted_problem, compute_k=compute_k)

        # For operator build
        u, v = TrialFunction(V), TestFunction(V)
        h = CellDiameter(V.mesh())

        if V.ufl_element().family() == 'Discontinuous Lagrange':
            assert V.ufl_element().value_shape() == ()
            assert V.ufl_element().degree() == 0

            h_avg = avg(h)
            a = h_avg**(-1)*dot(jump(v), jump(u))*dS + inner(u, v)*ds + inner(u, v)*dx
        else:
            a = inner(grad(u), grad(v))*dx + inner(u, v)*dx            

        self.L = a
        self.I = inner(u, v)*dx
        self.f = inner(Constant(np.zeros(V.ufl_element().value_shape())), v)*dx

        self.bcs = bcs

    def shifted_operator(self, shift):
        return assemble_system(self.L + Constant(shift)*self.I, self.f, self.bcs)[0]

        
class InvFLaplace(BPOperator):
    '''
    BPOperator with L = -Delta, fractional Laplacian, i.e. compute the 
    solution of L u = f for f given.
    '''
    def __init__(self, V, s, bcs, solve_shifted_problem=None, compute_k=0.5):
        assert bcs is not None
        BPOperator.__init__(self, V, s, solve_shifted_problem, compute_k)

        # For operator build
        u, v = TrialFunction(V), TestFunction(V)
        h = CellDiameter(V.mesh())

        if V.ufl_element().family() == 'Discontinuous Lagrange':
            assert V.ufl_element().value_shape() == ()
            assert V.ufl_element().degree() == 0

            h_avg = avg(h)
            a = h_avg**(-1)*dot(jump(v), jump(u))*dS + inner(u, v)*ds 
        else:
            a = inner(grad(u), grad(v))*dx

        self.L = a
        self.I = inner(u, v)*dx
        self.f = inner(Constant(np.zeros(V.ufl_element().value_shape())), v)*dx
        self.bcs = bcs

    def shifted_operator(self, shift):
        return assemble_system(self.L + Constant(shift)*self.I, self.f, self.bcs)[0]
    
# -------------------------------------------------------------------

# NOTE: this is for debugging
if __name__ == '__main__':
    from dolfin import UnitIntervalMesh, FunctionSpace, Expression, assemble
    from dolfin import Function, interpolate, errornorm, ln
    from dolfin import UnitSquareMesh

    from block.iterative import ConjGrad
    from block.algebraic.petsc import AMG
    
    s = 0.5
    k = 2.
    dim = 2

    # -------------------------------------
    
    f = Expression('cos(k*pi*x[0])', k=k, degree=4)
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)

    def amg_solve(A, x, b):
        Ainv = ConjGrad(A, precond=AMG(A), tolerance=1E-8, show=2)
        x[:] = Ainv*b
        return (len(Ainv.residuals), x)
    
    get_bcs = lambda V: None
    # NOTE: `compute_k` here controls the number of quadrature points
    # For optimal converrgence the value should be adapted on mesh resultion etc
    get_B = lambda V, bcs, s: InvFHelmholtz(V, s, bcs, compute_k=0.3,
                                            solve_shifted_problem=amg_solve)

    # NOTE: `solve_shifted_problem=None`  falls back to LU
    # FIXME: preconditioned based on AMG of unshifted problem
    
    e0 = None
    h0 = None

    n_range = {1: [2**i for i in range(5, 13)],
               2: [2**i for i in range(2, 7)]}
    
    for n in n_range[dim]:
        mesh = {1: UnitIntervalMesh, 2: UnitSquareMesh}[dim](*(n, )*dim)
        V = FunctionSpace(mesh, 'CG', 1)

        bcs = get_bcs(V)
        B = get_B(V, bcs, s)

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        b = assemble(inner(f, v)*dx)

        if bcs is not None: bcs.apply(b)

        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = np.nan
        else:
            rate = ln(e/e0)/ln(h/h0)

        e0 = e
        h0 = float(h)
        nsolves, niters_per_solve = B.nsolves, float(B.niters)/B.nsolves 
        
        print('%d | %.2E %.4E %.2f [%d(%.4f)]' % (V.dim(), mesh.hmin(), e0, rate, nsolves, niters_per_solve))

    u_exact = interpolate(u_exact, V)
    u_exact.vector().axpy(-1, df.vector())
