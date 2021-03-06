from xii.linalg.matrix_utils import (is_petsc_vec, is_petsc_mat, diagonal_matrix,
                                     is_number, as_petsc, petsc_serial_matrix,
                                     zero_matrix)

from block.block_compose import block_mul, block_add, block_sub, block_transpose
from block import block_mat, block_vec
from dolfin import PETScVector, PETScMatrix
from dolfin import Vector, GenericVector, Matrix, MPI
from scipy.sparse import bmat as numpy_block_mat
from scipy.sparse import csr_matrix, vstack as sp_vstack
from petsc4py import PETSc
import numpy as np
import itertools
import operator
from functools import reduce

COMM = PETSc.COMM_WORLD


def convert(bmat, algorithm='numpy'):
    '''
    Attempt to convert bmat to a PETSc(Matrix/Vector) object.
    If succed this is at worst a number.
    '''
    # Block vec conversion
    if isinstance(bmat, block_vec):
        array = block_vec_to_numpy(bmat)
        vec = PETSc.Vec().createWithArray(array)
        vec.assemble()
        return PETScVector(vec)
    
    # Conversion of bmat is bit more involved because of the possibility
    # that some of the blocks are numbers or composition of matrix operations
    if isinstance(bmat, block_mat):
        # Create collpsed bmat
        row_sizes, col_sizes = bmat_sizes(bmat)
        nrows, ncols = len(row_sizes), len(col_sizes)
        indices = itertools.product(list(range(nrows)), list(range(ncols)))

        blocks = np.zeros((nrows, ncols), dtype='object')
        for block, (i, j) in zip(bmat.blocks.flatten(), indices):
            # This might is guaranteed to be matrix or number
            A = collapse(block)

            if is_number(A):
                # Diagonal matrices can be anything provided square
                if i == j and row_sizes[i] == col_sizes[j]:
                    A = diagonal_matrix(row_sizes[i], A)
                else:
                    # Offdiagonal can only be zero
                    A = zero_matrix(row_sizes[i], col_sizes[j])
                #else:
                #    A = 0
            # The converted block
            blocks[i, j] = A
        # Now every block is a matrix/number and we can make a monolithic thing
        bmat = block_mat(blocks)

        assert all(is_petsc_mat(block) or is_number(block)
                   for block in bmat.blocks.flatten())
        
        # Opt out of monolithic
        if not algorithm:
            set_lg_map(bmat)
            return bmat
        
        # Monolithic via numpy (fast)
        # Convert to numpy
        array = block_mat_to_numpy(bmat)
        # Constuct from numpy
        bmat = numpy_to_petsc(array)
        set_lg_map(bmat)

        return bmat

    # Try with a composite
    return collapse(bmat)


def collapse(bmat):
    '''Collapse what are blocks of bmat'''
    # Single block cases
    # Do nothing
    if is_petsc_mat(bmat) or is_number(bmat) or is_petsc_vec(bmat):
        return bmat

    if isinstance(bmat, (Vector, Matrix, GenericVector)):
        return bmat

    # Multiplication
    if isinstance(bmat, block_mul):
        return collapse_mul(bmat)
    # +
    elif isinstance(bmat, block_add):
        return collapse_add(bmat)
    # -
    elif isinstance(bmat, block_sub):
        return collapse_sub(bmat)
    # T
    elif isinstance(bmat, block_transpose):
        return collapse_tr(bmat)

    # Some things in cbc.block know their matrix representation
    # This is typically diagonals like InvLumpDiag etc
    elif hasattr(bmat, 'v'):
        # So now we make that diagonal matrix
        diagonal = bmat.v

        n = diagonal.size
        mat = PETSc.Mat().createAIJ(comm=COMM, size=[[n, n], [n, n]], nnz=1)
        mat.assemblyBegin()
        mat.setDiagonal(diagonal)
        mat.assemblyEnd()

        return PETScMatrix(mat)
    # Some operators actually have matrix repre (HsMG)
    elif hasattr(bmat, 'matrix'):
        return bmat.matrix

    # Try:
    elif hasattr(bmat, 'collapse'):
        return bmat.collapse()
    
    elif hasattr(bmat, 'create_vec'):
        x = bmat.create_vec()
        columns = []
        for ei in Rn_basis(x):
            y = bmat*ei
            columns.append(csr_matrix(convert(y).get_local()))
        bmat = (sp_vstack(columns).T).tocsr()

        return numpy_to_petsc(bmat)
    
    raise ValueError('Do not know how to collapse %r' % type(bmat))


def collapse_tr(bmat):
    '''to Transpose'''
    # Base
    # FIXME:!!!
    A = bmat.A
    if is_petsc_mat(A):
        A_ = as_petsc(A)
        C_ = PETSc.Mat()
        A_.transpose(C_)
        return PETScMatrix(C_)
    # Recurse
    return collapse_tr(collapse(bmat))


def collapse_add(bmat):
    '''A + B to single matrix'''
    A, B = bmat.A, bmat.B
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A + B
        C_.axpy(1., B_, PETSc.Mat.Structure.DIFFERENT)
        return PETScMatrix(C_)
    # Recurse
    return collapse_add(collapse(A) + collapse(B))


def collapse_sub(bmat):
    '''A - B to single matrix'''
    A, B = bmat.A, bmat.B
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A - B
        C_.axpy(-1., B_, PETSc.Mat.Structure.DIFFERENT)
        return PETScMatrix(C_)
    # Recurse
    return collapse_sub(collapse(A) - collapse(B))


def collapse_mul(bmat):
    '''A*B*C to single matrix'''
    # A0 * A1 * ...
    A, B = bmat.chain[0], bmat.chain[1:]

    if len(B) == 1:
        B = B[0]
        # Two matrices
        if is_petsc_mat(A) and is_petsc_mat(B):
            A_ = as_petsc(A)
            B_ = as_petsc(B)
            assert A_.size[1] == B_.size[0]
            C_ = PETSc.Mat()
            A_.matMult(B_, C_)

            return PETScMatrix(C_)
        # One of them is a number
        elif is_petsc_mat(A) and is_number(B):
            A_ = as_petsc(A)
            C_ = A_.copy()
            C_.scale(B)
            return PETScMatrix(C_)

        elif is_petsc_mat(B) and is_number(A):
            B_ = as_petsc(B)
            C_ = B_.copy()
            C_.scale(A)
            return PETScMatrix(C_)
        # Some compositions
        else:
            return collapse(collapse(A)*collapse(B))
    # Recurse
    else:
        return collapse_mul(collapse(A)*collapse(reduce(operator.mul, B)))                                    

    
# Conversion via numpy
def block_vec_to_numpy(bvec):
    '''Collapsing block bector to numpy array'''
    return np.hstack([v.get_local() for v in bvec])


def block_mat_to_numpy(bmat):
    '''Collapsing block mat of matrices to scipy's bmat'''
    # A single matrix
    if is_petsc_mat(bmat):
        bmat = as_petsc(bmat)
        return csr_matrix(bmat.getValuesCSR()[::-1], shape=bmat.size)
    # 0
    if is_number(bmat):
        return None  # What bmat accepts
    # Recurse on blocks
    blocks = np.array(list(map(block_mat_to_numpy, bmat.blocks.flatten())))
    blocks = blocks.reshape(bmat.blocks.shape)
    # The bmat
    return numpy_block_mat(blocks).tocsr()


def numpy_to_petsc(mat):
    '''Build PETScMatrix with array structure'''
    # Dense array to matrix
    if isinstance(mat, np.ndarray):
        if mat.ndim == 1:
            vec = PETSc.Vec().createWithArray(mat)
            vec.assemble()
            return PETScVector(vec)

        return numpy_to_petsc(csr_matrix(mat))
    # Sparse
    A = PETSc.Mat().createAIJ(comm=COMM,
                              size=mat.shape,
                              csr=(mat.indptr, mat.indices, mat.data))
    A.assemble()
    return PETScMatrix(A)


def block_mat_to_petsc(bmat):
    '''Block mat to PETScMatrix via assembly'''
    # This is beautiful but slow as hell :)
    def iter_rows(matrix):
        for i in range(matrix.size(0)):
            yield matrix.getrow(i)

    row_sizes, col_sizes = get_sizes(bmat)
    row_offsets = np.cumsum([0] + list(row_sizes))
    col_offsets = np.cumsum([0] + list(col_sizes))

    with petsc_serial_matrix(row_offsets[-1], col_offsets[-1]) as mat:
        row = 0
        for row_blocks in bmat.blocks:
            # Zip the row iterators of the matrices together
            for indices_values in zip(*list(map(iter_rows, row_blocks))):
                indices, values = list(zip(*indices_values))

                indices = [list(index+offset) for index, offset in zip(indices, col_offsets)]
                indices = sum(indices, [])
            
                row_values = np.hstack(values)

                mat.setValues([row], indices, row_values, PETSc.InsertMode.INSERT_VALUES)

                row += 1
    return PETScMatrix(mat)


def get_dims(thing):
    '''
    Size of Rn vector or operator Rn to Rm. We return None for scalars
    and raise when such an operator cannot be established, i.e. there 
    are consistency checks going on 
    '''
    if is_petsc_vec(thing): return thing.size()

    if is_petsc_mat(thing): return (thing.size(0), thing.size(1))
    
    if is_number(thing): return None
    
    # Now let's handdle block stuff
    # Multiplication
    if isinstance(thing, block_mul):
        A, B = thing.chain[0], thing.chain[1:]

        dims_A, dims_B = get_dims(A), get_dims(B[0])
        # A number does not change
        if dims_A is None:
            return dims_B
        if dims_B is None:
            return dims_A
        # Otherwise, consistency
        if len(B) == 1:
            assert len(dims_A) == len(dims_B) 
            assert dims_A[1] == dims_B[0], (dims_A, dims_B) 
            return (dims_A[0], dims_B[1])
        else:
            dims_B = get_dims(reduce(operator.mul, B))
            
            assert len(dims_A) == len(dims_B) 
            assert dims_A[1] == dims_B[0], (dims_A, dims_B)
            return (dims_A[0], dims_B[1])
    # +, -
    if isinstance(thing, (block_add, block_sub)):
        A, B = thing.A, thing.B
        if is_number(A):
            return get_dims(B)

        if is_number(B):
            return get_dims(A)

        dims = get_dims(A)
        assert dims == get_dims(B), (dims, get_dims(B))
        return dims
    # T
    if isinstance(thing, block_transpose):
        dims = get_dims(thing.A)
        return (dims[1], dims[0])
    
    # Some things in cbc.block know their maE.g. InvLumpDiag...Almost last resort
    if hasattr(thing, 'A'):
        assert is_petsc_mat(thing.A)
        return get_dims(thing.A)

    if hasattr(thing, '__sizes__'):
        return thing.__sizes__
    
    if hasattr(thing, 'create_vec'):
        return (thing.create_vec(0).size(), thing.create_vec(1).size())

    raise ValueError('Cannot get_dims of %r, %s' % (type(thing), thing))

    
def bmat_sizes(bmat):
    '''Return a tuple which represents sizes of (blocks of) bmat'''
    if isinstance(bmat, block_vec):
        return tuple(map(get_dims, block_vec.blocks))

    if isinstance(bmat, block_mat):
        vec = bmat.create_vec(0)
        vecs = (vec, ) if not hasattr(vec, 'blocks') else vec.blocks
        row_sizes = tuple(vec.size() for vec in vecs)

        vec = bmat.create_vec(1)
        vecs = (vec, ) if not hasattr(vec, 'blocks') else vec.blocks
        col_sizes = tuple(vec.size() for vec in vecs)
        
        return row_sizes, col_sizes
    
    raise ValueError('Cannot bmat_sizes of %r, %s' % (type(bmat), bmat))
