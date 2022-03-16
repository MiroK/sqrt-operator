# Action of inv sqrt for a block operator
from block import block_mat, block_vec
from block.object_pool import vec_pool
from bonito_pesciak import BPOperator
import numpy as np


def block_diagonal(diagonal):
    '''Diagonal block_mat'''
    n = len(diagonal)
    blocks = [[0]*n for _ in range(n)]  # Alloc
    for i in range(n):  # Fill
        blocks[i][i] = diagonal[i]

    return block_mat(blocks)


class BlockInvFHelmholtz(BPOperator):
    '''
    BPOperator with L = diag([-Delta + I]*nblocks); fractional Helmholtz, i.e. compute 
    the solution of L u = f for f given.
    '''
    def __init__(self, nblocks, V, s, bcs, solve_shifted_problem=None, compute_k=0.5):
        assert nblocks >= 1
        
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

        self.nblocks = nblocks

    def shifted_operator(self, shift):
        Apiece = assemble_system(self.L + Constant(shift)*self.I, self.f)[0]
        # The actual operator is
        A = block_diagonal([Apiece]*self.nblocks)
        # NOTE: for the iterative solver the A above would be sufficient
        
        return A

    @vec_pool
    def create_vec(self, dim=1):
        '''Inform about the shape'''
        return block_vec([Function(self.V).vector() for _ in range(self.nblocks)])
    

# -------------------------------------------------------------------

# NOTE: this is for debugging
if __name__ == '__main__':
    from block.iterative import ConjGrad
    from block.algebraic.petsc import AMG
    from dolfin import *
    
    s = 0.5
    k = 2.
    dim = 1

    # -------------------------------------
    # For one block
    f = Expression('cos(k*pi*x[0])', k=k, degree=4)
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)

    nblocks = 3    

    def amg_solve(A, x, b, n=nblocks):
        # Here A is block OP
        # We make the preconditioner the same way
        Ainv = ConjGrad(A, precond=block_diagonal([AMG(A[0][0])]*n), tolerance=1E-8, show=2)
        x[:] = Ainv*b
        return (len(Ainv.residuals), x)
    
    # NOTE: `compute_k` here controls the number of quadrature points
    # For optimal converrgence the value should be adapted on mesh resultion etc
    get_B = lambda V, s: BlockInvFHelmholtz(nblocks=nblocks,
                                            V=V, s=s, bcs=None, compute_k=0.3,
                                            solve_shifted_problem=amg_solve)

    e0 = None
    h0 = None

    n_range = {1: [2**i for i in range(5, 13)],
               2: [2**i for i in range(2, 7)]}

    for n in n_range[dim]:
        mesh = {1: UnitIntervalMesh, 2: UnitSquareMesh}[dim](*(n, )*dim)
        V = FunctionSpace(mesh, 'CG', 1)

        B = get_B(V, s)

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        b = assemble(inner(f, v)*dx)
        b = block_vec([b]*nblocks)

        x = B*b  # A block function
        h = mesh.hmin()
        e = 0
        for i in range(nblocks):
            df = Function(V, x[i])
            e += errornorm(u_exact, df, 'L2', degree_rise=3)**2
        e = sqrt(e)
        
        if e0 is None:
            rate = np.nan
        else:
            rate = ln(e/e0)/ln(h/h0)

        e0 = e
        h0 = float(h)
        nsolves, niters_per_solve = B.nsolves, float(B.niters)/B.nsolves 
        
        print('%d | %.2E %.4E %.2f [%d(%.4f)]' % (V.dim(), mesh.hmin(), e0, rate, nsolves, niters_per_solve))
