import numpy
import itertools
from functools import reduce
import torch
from .utils import (
    ipow, pauli_tokenize, 
    clifford_rotate, pauli_transform,
    batch_dot, aggregate)

class Pauli(object):
    '''Represents a Pauli operator.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).'''
    def __init__(self, g, p = None):
        self.g = g
        self.p = 0 if p is None else p

    def __repr__(self):
        # interprete phase factor
        if self.N > 0:
            if self.p == 0:
                txt = ' +'
            elif self.p == 1:
                txt = '+i'
            elif self.p == 2:
                txt = ' -'
            elif self.p == 3:
                txt = '-i'
        else:
            txt = 'null'
        # interprete Pauli string
        for i in range(self.N):
            x = self.g[2*i  ]
            z = self.g[2*i+1]
            if x == 0:
                if z == 0:
                    txt += 'I'
                elif z == 1:
                    txt += 'Z'
            elif x == 1:
                if z == 0:
                    txt += 'X'
                elif z == 1:
                    txt += 'Y'
        return txt

    def __getattr__(self, item):
        if item is 'N': # number of qubits
            return self.g.shape[0]//2
        else:
            return super().__getattribute__(item)

    def __neg__(self):
        return type(self)(self.g, (self.p + 2) % 4)

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.g, (self.p + 1) % 4)
        elif c == -1:
            return type(self)(self.g, (self.p + 2) % 4)
        elif c == -1j:
            return type(self)(self.g, (self.p + 3) % 4)
        else: # upgrade to PauliMonomial
            return c * self.as_monomial()

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, Pauli):
            p = (self.p + other.p + ipow(self.g, other.g)) % 4
            g = (self.g + other.g) % 2
            return Pauli(g, p)
        elif isinstance(other, (PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else: 
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def copy(self):
        return Pauli(self.g.copy(), self.p)

    def as_monomial(self):
        '''cast a Pauli operator to a Pauli monomial assuming coefficient = 1'''
        return PauliMonomial(self.g, self.p)

    def as_polynomial(self):
        '''cast a Pauli operator to a Pauli polynomial'''
        return self.as_monomial().as_polynomial()

class PauliList(object):
    '''Represents a list of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).'''
    def __init__(self, gs, ps = None):
        self.gs = gs
        self.ps = numpy.zeros(self.L, dtype=numpy.int_) if ps is None else ps

    def __repr__(self):
        return '\n'.join([repr(pauli) for pauli in self])

    def __len__(self):
        return self.L

    def __getattr__(self, item):
        if item is 'L':
            return self.gs.shape[0]
        if item is 'N':
            return self.gs.shape[1]//2
        else:
            return super().__getattribute__(item)

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return Pauli(self.gs[item], self.ps[item])
        return PauliList(self.gs[item], self.ps[item])

    def __neg__(self):
        return type(self)(self.gs, (self.ps + 2) % 4)

    def __truediv__(self, other):
        return (1/other) * self

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.gs, (self.ps + 1) % 4)
        elif c == -1:
            return type(self)(self.gs, (self.ps + 2) % 4)
        elif c == -1j:
            return type(self)(self.gs, (self.ps + 3) % 4)
        else: # upgrade to PauliPolynomial
            raise NotImplementedError('multiplication is not defined for {} when factor is not 1, -1, 1j, -1j.'.format(type(self).__name__))

    def copy(self):
        return PauliList(self.gs.copy(), self.ps.copy())

    def as_polynomial(self):
        return PauliPolynomial(self.gs, self.ps)

    def rotate_by(self, generator, mask=None):
        # perform Clifford rotation by Pauli generator (in-place)
        if mask is None:
            clifford_rotate(generator.g, generator.p, self.gs, self.ps)
        else:
            mask2 = numpy.repeat(mask,  2)
            self.gs[:,mask2], self.ps = clifford_rotate(
                generator.g, generator.p, self.gs[:,mask2], self.ps)
        return self

    def transform_by(self, clifford_map, mask=None):
        # perform Clifford transformation by Clifford map (in-place)
        if mask is None:
            self.gs, self.ps = pauli_transform(self.gs, self.ps, 
                clifford_map.gs, clifford_map.ps)
        else:
            mask2 = numpy.repeat(mask, 2)
            self.gs[:,mask2], self.ps = pauli_transform(
                self.gs[:,mask2], self.ps, clifford_map.gs, clifford_map.ps)
        return self

    def tokenize(self):
        return pauli_tokenize(self.gs, self.ps)

class PauliMonomial(Pauli):
    '''Represent a Pauli operator with a coefficient.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).
    c: comlex - coefficient.'''
    def __init__(self, *args, **kwargs):
        super(PauliMonomial, self).__init__(*args, **kwargs)
        self.c = 1.+0.j # default coefficient

    def __repr__(self):
        # interprete coefficient
        c = self.c * 1j**self.p
        if c.imag == 0.:
            c = c.real
            if c.is_integer():
                txt = '{:d} '.format(int(c))
            else: 
                txt = '{:.2f} '.format(c)
        else:
            txt = '({:.2f}) '.format(c)
        # interprete Pauli string
        for i in range(self.N):
            x = self.g[2*i  ]
            z = self.g[2*i+1]
            if x == 0:
                if z == 0:
                    txt += 'I'
                elif z == 1:
                    txt += 'Z'
            elif x == 1:
                if z == 0:
                    txt += 'X'
                elif z == 1:
                    txt += 'Y'
        return txt

    def __neg__(self):
        return PauliMonomial(self.g, self.p).set_c(-self.c)

    def __rmul__(self, c):
        return PauliMonomial(self.g, self.p).set_c(c * self.c)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def set_c(self, c):
        self.c = c
        return self

    def copy(self):
        return PauliMonomial(self.g.copy(), self.p).set_c(self.c)
        
    def as_polynomial(self):
        '''cast the Pauli monomial to a single-term Pauli polynomial'''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=numpy.int_)
        cs = numpy.array([self.c], dtype=numpy.complex_)
        return PauliPolynomial(gs, ps).set_cs(cs)

    def inverse(self):
        return Pauli(self.g)/(self.c * 1j**self.p)

class PauliPolynomial(PauliList):
    '''Represent a linear combination of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).
    cs: comlex - coefficients.'''
    def __init__(self, *args, **kwargs):
        super(PauliPolynomial, self).__init__(*args, **kwargs)
        self.cs = numpy.ones(self.ps.shape, dtype=numpy.complex_) # default coefficient

    def __repr__(self):
        txt = ''
        for k, term in enumerate(self):
            txt_term = repr(term)
            if k != 0:
                if txt_term[0] == '-':
                    txt_term = ' ' + txt_term
                else:
                    txt_term = ' +' + txt_term
            txt  = txt + txt_term
        return txt

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return PauliMonomial(self.gs[item], self.ps[item]).set_c(self.cs[item])
        return PauliPolynomial(self.gs[item], self.ps[item]).set_cs(self.cs[item])

    def __neg__(self):
        return PauliPolynomial(self.gs, self.ps).set_cs(-self.cs)

    def __rmul__(self, c):
        return PauliPolynomial(self.gs, self.ps).set_cs(c * self.cs)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        if not isinstance(other, PauliPolynomial):
            if isinstance(other, (PauliMonomial, Pauli, PauliList)):
                other = other.as_polynomial()
            else: # otherwise assuming other is a number
                other = other * pauli_identity(self.N)
        gs = numpy.concatenate([self.gs, other.gs])
        ps = numpy.concatenate([self.ps, other.ps])
        cs = numpy.concatenate([self.cs, other.cs])
        return PauliPolynomial(gs, ps).set_cs(cs).reduce()

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            other = other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))
        gs, ps, cs = batch_dot(self.gs, self.ps, self.cs, other.gs, other.ps, other.cs)
        return PauliPolynomial(gs, ps).set_cs(cs)

    def set_cs(self, cs):
        '''set coefficients'''
        self.cs = cs
        return self

    def copy(self):
        return PauliPolynomial(self.gs.copy(), self.ps.copy()).set_cs(self.cs.copy())

    def as_polynomial(self):
        return self

    def reduce(self, tol=1.e-10):
        '''Reduce the Pauli polynomial by 
            1. combine simiilar terms,
            2. move phase factors to coefficients,
            3. drop terms that are too small (coefficient < tol).'''
        gs, inds = numpy.unique(self.gs, return_inverse=True, axis=0)
        cs = aggregate(self.cs * 1j**self.ps, inds, gs.shape[0])
        mask = (numpy.abs(cs) > tol)
        return PauliPolynomial(gs[mask]).set_cs(cs[mask])

# ---- constructors ----
def pauli(obj, N = None):
    if isinstance(obj, Pauli):
        return obj
    elif isinstance(obj, (tuple, list, numpy.ndarray)):
        N = len(obj)
        inds = enumerate(obj)
    elif isinstance(obj, dict):
        if N is None:
            raise ValueError('pauli(inds, N) must specify qubit number N when inds is dict.')
        inds = obj.items()
    elif isinstance(obj, str):
        return pauli(list(obj))
    else:
        raise TypeError('pauli(obj) recieves obj of type {}, which is not implemented.'.format(type(obj).__name__))
    g = numpy.zeros(2*N, dtype=numpy.int_)
    h = 0
    p = 0
    for i, mu in inds:
        assert i-h < N, 'qubit {} is out of bounds for system size {}.'.format(i, N)
        if mu == 0 or mu == 'I':
            continue
        elif mu == 1 or mu == 'X':
            g[2*(i-h)] = 1
        elif mu == 2 or mu == 'Y':
            g[2*(i-h)] = 1
            g[2*(i-h)+1] = 1
        elif mu == 3 or mu == 'Z':
            g[2*(i-h)+1] = 1
        elif mu == '+':
            p = 0
            h += 1
        elif mu == '-':
            p = 2
            h += 1
        elif mu == 'i':
            p += 1
            h += 1
        else:
            h += 1
    if h == 0:
        return Pauli(g, p)
    else:
        return Pauli(g[:-2*h], p)

import types
def paulis(*objs):
    # short cut if PauliList is passed in
    if len(objs) == 1 :
        if isinstance(objs[0], PauliList):
            return objs[0]
        if isinstance(objs[0], (tuple, list, set, types.GeneratorType)):
            objs = objs[0]
    # otherwise construct data for Pauli operators
    objs = [pauli(obj) for obj in objs]
    gs = numpy.stack([obj.g for obj in objs])
    ps = numpy.array([obj.p for obj in objs])
    return PauliList(gs, ps)

def pauli_identity(N):
    '''Pauli polynomial of an idenity operator of N qubits.'''
    return PauliPolynomial(numpy.zeros((1,2*N), dtype=numpy.int_))

def pauli_zero(N):
    '''Pauli polynomial of zero operator of N qubit'''
    return 0 * pauli_identity(N)


######## return support of pauli string ########
def pauli_support(g):
    return numpy.ndarray.flatten(numpy.argwhere(g[::2]+g[1::2]))

######## return range of sites on which pauli string acts ########
def pauli_range(g):
    supp,n=pauli_support(g),len(g)//2
    if len(supp)%n==0:
        return supp
    #rotate so that the first site is nonzero
    col=numpy.roll(g[::2]+g[1::2],-supp[0])
    #find the contiguous strings of zeros
    ids=numpy.ndarray.flatten(numpy.argwhere(col==0))
    components,curr=[],[ids[0]]
    for i in range(1,len(ids)):
        if ids[i]==(curr[-1]+1):
            curr.append(ids[i])
        else :
            components.append(curr)
            curr=[ids[i]]
    components.append(curr)
    #the longest zero string's complement is the range
    lens=[len(com) for com in components]
    biggest=lens.index(max(lens))
    return numpy.array(list(range(supp[0]+components[biggest][-1]+1, \
                                  supp[0]+components[biggest][0]+n))) % n


######## return mps tensors for stabilizer state given generators ########
def stabilizer_mps(gs, ps):
    n=len(gs[0])//2
    #define range for each generator
    ranges=[pauli_range(gen) for gen in gs]
    
    #determine which generators overlap at a site
    overlaps=[[i for i in range(n) if j in ranges[i]] for j in range(n)]
    
    #determine which of the generators extend over a site to determine virt. dim.
    tensors,gens,D=[],[],1
    for i in range(n):
        lg=list(set(overlaps[i-1]) & set(overlaps[i]))
        rg=list(set(overlaps[(i+1)%n]) & set(overlaps[i]))
        ig=list(set(overlaps[i]) - set(overlaps[i-1]) - set(overlaps[(i+1)%n]))
        mg=list(set(lg) & set(rg))
        #sort for consistency
        lg.sort()
        rg.sort()        
        ig.sort()
        mg.sort()
        gens.append([ig,lg,rg,mg])
        tensors.append(numpy.full(tuple([2]*len(lg+rg)), pauli_zero(1)))
        D=max(D,2**len(lg),2**len(rg))
    
    #determine the matrix elements of the onsite tensors
    for i in range(n):
        ig,lg,rg,mg=gens[i]
        lrg=numpy.array(lg+rg)
        for alphas in itertools.product(range(2),repeat=len(lg+rg)):
            if numpy.prod([alphas[lg.index(j)]==alphas[rg.index(j)+len(lg)] for j in mg]):
                nzg=list(set(lrg[numpy.flatnonzero(alphas)]))
                me=pauli('I')
                for j in nzg :
                    x=pauli([gs[j][::2][i]+3*gs[j][1::2][i]-2*gs[j][::2][i]*gs[j][1::2][i]])
                    me=1j**(ps[j]) * (me @ x) if i == ranges[j][0] else (me @ x)
                for j in ig :
                    x=pauli('I') + 1j**(ps[j])*pauli([gs[j][::2][i]+3*gs[j][1::2][i]-2*gs[j][::2][i]*gs[j][1::2][i]])
                    me=(me @ x).reduce()
                tensors[i][alphas]=me
        tensors[i] = tensors[i].reshape(2**len(lg), 2**len(rg))
    return tensors,D

######## convert tensor list to mps form ########
def mps_canonical_form(tensors,D):
    n=len(tensors)
    mats=torch.zeros((n,D,D,4),dtype=torch.complex64)
    for site in range(n):
        t=tensors[site]
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                s=t[i,j].as_polynomial()
                for k in range(len(s.gs)):
                    g,p,c=s.gs[k],s.ps[k],s.cs[k]
                    mats[site,i,j,g[0]+3*g[1]-2*g[0]*g[1]]+=((1j)**p)*c
    return mats

######## return stabilizer bond dimension ########
def stabilizer_bond_dimension(gs,ps):
    n=len(gs[0])//2
    #define range for each generator
    ranges=[pauli_range(gen) for gen in gs]
    
    #determine which generators overlap at a site
    overlaps=[[i for i in range(n) if j in ranges[i]] for j in range(n)]
    
    #determine which of the generators extend over a site to determine virt. dim.
    gens,D=[],1
    for i in range(n):
        lg=list(set(overlaps[i-1]) & set(overlaps[i]))
        rg=list(set(overlaps[(i+1)%n]) & set(overlaps[i]))
        ig=list(set(overlaps[i]) - set(overlaps[i-1]) - set(overlaps[(i+1)%n]))
        mg=list(set(lg) & set(rg))
        #sort for consistency
        lg.sort()
        rg.sort()        
        ig.sort()
        mg.sort()
        gens.append([ig,lg,rg,mg])
        D=max(D,2**len(lg),2**len(rg))
    return D

######## return stabilizer mps given bond dimension ########
def stabilizer_local_tensor(gs,ps,D,i):
    n=len(gs[0])//2
    #define range for each generator
    ranges=[pauli_range(gen) for gen in gs]
    
    #determine which generators overlap at a site
    overlaps=[[i for i in range(n) if j in ranges[i]] for j in range(n)]
    
    #determine which of the generators extend over a site to determine virt. dim.    
    lg=list(set(overlaps[i-1]) & set(overlaps[i]))
    rg=list(set(overlaps[(i+1)%n]) & set(overlaps[i]))
    ig=list(set(overlaps[i]) - set(overlaps[i-1]) - set(overlaps[(i+1)%n]))
    mg=list(set(lg) & set(rg))
    #sort for consistency
    lg.sort()
    rg.sort()        
    ig.sort()
    mg.sort()
    
    tensor=numpy.full(tuple([2]*len(lg+rg)), pauli_zero(1))

    #determine the matrix elements of the onsite tensors
    lrg=numpy.array(lg+rg)
    for alphas in itertools.product(range(2),repeat=len(lg+rg)):
        if numpy.prod([alphas[lg.index(j)]==alphas[rg.index(j)+len(lg)] for j in mg]):
            nzg=list(set(lrg[numpy.flatnonzero(alphas)]))
            me=pauli('I')
            for j in nzg :
                x=pauli([gs[j][::2][i]+3*gs[j][1::2][i]-2*gs[j][::2][i]*gs[j][1::2][i]])
                me=1j**(ps[j]) * (me @ x) if i == ranges[j][0] else (me @ x)
            for j in ig :
                x=pauli('I') + 1j**(ps[j])*pauli([gs[j][::2][i]+3*gs[j][1::2][i]-2*gs[j][::2][i]*gs[j][1::2][i]])
                me=(me @ x).reduce()
            tensor[alphas]=me
    tensor = tensor.reshape(2**len(lg), 2**len(rg))
    
    #convert tensor list to mps form
    mats=torch.zeros((D,D,4),dtype=torch.complex64)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            s=tensor[i,j].as_polynomial()
            for k in range(len(s.gs)):
                g,p,c=s.gs[k],s.ps[k],s.cs[k]
                mats[i,j,g[0]+3*g[1]-2*g[0]*g[1]]+=((1j)**p)*c
    return mats
    

######## return stabilizer mps given bond dimension ########
#def stabilizer_local_tensor(gs,ps,i):
#    print(i)
#    n=len(gs[0])//2
#    #define range for each generator
#    ranges=[pauli_range(gen) for gen in gs]
#    #determine which generators overlap at a site
#    overlaps=[k for k in range(n) if i in ranges[k]]
#    print(overlaps)
#    if len(overlaps)>0 :
#        #construct tensors
#        tensors=[projtensor(gs[j],ps[j],i) for j in overlaps]
#        print([tensor.shape for tensor in tensors])
#        fv=fusion_vertex()
#        result=reduce(lambda x,y:pfuse(x,y,fv), tensors)
#        print(result.shape)
#        return result
#    else :
#        return torch.tensor([[[1,0,0,0]]],dtype=torch.complex64)
#
#def fusion_vertex():
#    fv=torch.zeros((4,4,4),dtype=torch.complex64)
#    for i in range(4):
#        for j in range(4):
#            for k in range(4):
#                a=sum((pauli([i]) @ pauli([j]) @ pauli([k])).g)==0
#                b=1j**(pauli([i]) @ pauli([j]) @ pauli([k])).p
#                fv[i,j,k]= a*b
#    return fv
#
#def projtensor(g,p,i):
#    rng=pauli_range(g)
#    shape=[1 if i==rng[0] else 2,1 if i==rng[-1] else 2]
#    gi=g[::2][i]+3*g[1::2][i]-2*g[::2][i]*g[1::2][i]
#    tensor=torch.zeros(shape+[4])
#    tensor[0,0,0]=1.0
#    tensor[-1,-1,gi]=1j**p if i==rng[0] else 1.0
#    return tensor

#def pfuse(t1,t2,vx):
#    Da1,Da2,Db1,Db2=t1.shape[0],t1.shape[1],t2.shape[0],t2.shape[1]
#    t3=kron(t1.reshape(-1,4),t2.reshape(-1,4)).type(torch.complex64).mm(vx.reshape(-1,4))
#    return t3.reshape(Da1,Da2,Db1,Db2,4).transpose(1,2).reshape(Da1*Db1,Da2*Db2,4)
#
#def kron(A, B):
#    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0), \
#                                                  A.size(1)*B.size(1))