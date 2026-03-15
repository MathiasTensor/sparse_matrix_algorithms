# -----------------------------------------------------------------------------
#  Sparse FEAST (projector iteration) for the EVP
#
#      A x = λ x
#
#  where A is large, sparse, real symmetric (stored as SparseMatrixCSC{Float64}).
#
#  Goal: compute the eigenpairs whose eigenvalues lie in a real interval
#
#      λ ∈ [emin, emax]
#
#  without computing the full spectrum.
#
# -----------------------------------------------------------------------------
#  Spectral projector:
#
#  Let Γ be a positively oriented (counterclockwise) closed contour in ℂ that
#  encloses exactly the part of the spectrum we want, i.e. all eigenvalues
#  λ_j ∈ (emin, emax) and no other eigenvalues.
#
#  The spectral projector onto the invariant subspace S spanned by those
#  eigenvectors is
#
#      P = (1 / (2π i)) ∮_Γ (z I - A)^{-1} dz .
#
#  For an eigenvector x_j:
#      - if λ_j is inside Γ, then P x_j = x_j
#      - if λ_j is outside Γ, then P x_j = 0
#
#  Hence, applying P to any matrix of vectors Q keeps only the components in
#  the desired eigenspace:
#
#      Y = P Q  ≈  Σ_k w_k (z_k I - A)^{-1} Q .
#
#  FEAST uses this as a "rational filter" subspace iteration:
#      Q ← orth(P Q) , then Rayleigh–Ritz in span(Q).
#
# -----------------------------------------------------------------------------
#  Numerical approximation:
#
#  We approximate the contour integral by the trapezoidal rule on an ellipse:
#
#      z(θ) = c + a cos θ + i b sin θ ,  θ ∈ [0, 2π)
#      c = (emin+emax)/2,  a = (emax-emin)/2,  b = eta*a.
#
#  The quadrature weights include the factor (1/(2π i)) dz:
#
#      P ≈ Σ_{k=1..nodes} w_k (z_k I - A)^{-1}
#
#  where:
#      z_k = z(θ_k),  θ_k = (k-1) * 2π/nodes,
#      w_k = (1/(2π i)) z'(θ_k) Δθ.
#
# -----------------------------------------------------------------------------
#  Chebyshev polynomial filtered subspace iteration for the symmetric EVP
#
#      A x = λ x
#
#  where A is large, sparse, real symmetric (SparseMatrixCSC{Float64,Int}).
#
#  Goal: compute eigenpairs with λ ∈ [emin,emax].
#
#  The filter is a polynomial approximation to the spectral window 1_[emin,emax],
#  built from Chebyshev polynomials on a rescaled spectral interval [smin,smax].
#
# CENTRAL LITARETURE:
#   FEAST Eigensolver for Nonlinear Eigenvalue Problems, Gavin B., Międlar, A, Pollizi E. https://arxiv.org/abs/1801.09794
#   Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, Güttel S., Tak Peter Tang P., Viaud G., https://arxiv.org/abs/1407.8078
#   Polynomially Filtered Exact Diagonalization Approach to Many-Body Localization" by Sierant, P., Lewenstein, M., Zakrzewski, J.
#
# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# EDITING: Orel 10/3/2026
# * If editing file/want to add things, give your surname and date in the comment above, and briefly describe what you changed.
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# HIGH LEVEL API
#
# - FEAST
# general_feast_sparse_interval(...) <-> GENERAL SPARSE MATRIX
# general_feast_sparse_interval_distributed(...) <-> GENERAL SPARSE MATRIX, DISTRIBUTED (low preallocations, slightly slower but good on RAM)
# sym_feast_sparse_interval_half(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR
# sym_feast_sparse_interval_half_distributed(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR, DISTRIBUTED (low preallocations, slightly slower but good on RAM)
#
# - CHEBYSHEV FILTER (WINDOW [a,b])
# chebyshev_poly_real_symm_sparse(...) <-> REAL SYMMETRIC SPARSE MATRIX
#
# - ZOLOTAREV RATIONAL FILTER ! Very fast, should be the GOTO
# ! REQUIRES THE CONSTRUCTION OF ZolotarevParams (constructor below) TO PRECOMPUTE THE POLES !
# general_feast_sparse_interval_zolotarev(...) <-> GENERAL SPARSE MATRIX
# sym_feast_sparse_interval_half_zolotarev(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR
# general_feast_sparse_interval_zolotarev_distributed(...) <-> GENERAL SPARSE MATRIX, DISTRIBUTED
# sym_feast_sparse_interval_half_zolotarev_distributed(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR, DISTRIBUTED
#
# - CHEBYSHEV FILTER (POLFED) (DELTA APPROX near target eigenvalue)
# ! USEFUL FOR VERY LARGE MATRIX SIZES WHERE SHIFT INVERTS ARE EXPENSIVE, BUT CHEBYSHEV POLYVALS ARE CHEAP !
# chebyshev_poly_real_symm_sparse(...) <-> REAL SYMMETRIC SPARSE MATRIX
# chebyshev_poly_real_symm_sparse_distributed(...) <-> REAL SYMMETRIC SPARSE MATRIX, DISTRIBUTED
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# EXAMPLES (see tests below)
# -----------------------------------------------------------------------------



using LinearAlgebra
using SparseArrays
using Random
using Printf
using BenchmarkTools
using LinearSolve
using ProgressMeter
using QuadGK
using Krylov
using Statistics

try
    @eval using MKL
catch
    @warn "MKL.jl not installed/available"
end
try
    @eval using MKLSparse
catch
    @warn "MKLSparse.jl not installed/available"
end

#= #TODO NOT YET IMPLEMENTED FULLY - ignore for now. For Sparse we need to becnhmark againts KLUFactorization since it seems the fastets in that regard
const HAVE_PARDISO=Ref(false)
try
    @eval using Pardiso
    HAVE_PARDISO[]=true
catch
    @warn "Pardiso.jl not installed/available — falling back to other solvers"
end
try
    @eval using MKL_jll,Libdl
    MKLRT=MKL_jll.libmkl_rt
    push!(Libdl.DL_LOAD_PATH,dirname(MKLRT))
catch
    @warn "MKL_jll and/or Libdl not installed/available; MKL Pardiso may fail to load libmkl_rt"
end
=#

##################################################
#################### MACROS ######################
##################################################

"""
    use_threads(args...)
    
The macro expects either:
(a) A keyword argument "multithreading" followed by a loop expression, or
b) A lone loop expression (in which case multithreading defaults to true).
NOTE: Already @inbounds 
"""
macro use_threads(args...)
    if length(args)>=2 && args[1] isa Expr && args[1].head==:(=) && args[1].args[1]==:multithreading
        if length(args)!=2
            error("Usage: @use_threads multithreading=[true|false] for ...")
        end
        # Extract the provided multithreading value and the loop expression.
        multithreading_val=args[1].args[2]
        loop_expr=args[2]
        # If the provided value is literally true or false, we branch accordingly at compile time.
        if multithreading_val===true
            return esc(:(@inbounds Threads.@threads $loop_expr))
        elseif multithreading_val===false
            return esc(:(@inbounds $loop_expr))
        else
            # If not a literal (i.e. it's some expression that will be evaluated at runtime),
            # we generate code that conditionally selects the threaded version.
            return esc(quote
                @inbounds begin
                    if $multithreading_val
                        Threads.@threads $loop_expr
                    else
                        $loop_expr
                    end
                end
            end)
        end
    elseif length(args)==1
        # No keyword argument provided. Default behavior is multithreading=true.
        loop_expr=args[1]
        return esc(:(@inbounds Threads.@threads $loop_expr))
    else
        error("Usage: @use_threads [multithreading=[true|false]] for ...")
    end
end

"""
    struct DiagPtr{Ti<:Integer}
        p::Vector{Ti}
    end

A small helper that stores, for each column/row j, the index `t` into `A.nzval`
where the diagonal entry A[j,j] is stored (in CSC format), i.e.

    A.nzval[p[j]] == A[j,j]

If the diagonal entry is structurally missing, p[j] = 0.

Why we need this:
- In FEAST we repeatedly form shifted matrices (zI - A) for many complex shifts z.
- We want to update ONLY the diagonal in-place, keeping the same sparse pattern.
- So we need O(n) access to diagonal locations in the CSC storage arrays.
"""
struct DiagPtr{Ti<:Integer}
    p::Vector{Ti} # p[j] = index into nzval of A[j,j], or 0 if missing
end

"""
    diagptr(A::SparseMatrixCSC{T,Ti}) -> DiagPtr{Ti}

Scan each CSC column j of A, locate the diagonal entry (row == j),
and store its nzval index in p[j]. If missing, p[j]=0.

Cost: O(nnz) worst-case, but done once per A.
"""
function diagptr(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    n=size(A,1)
    p=zeros(Ti,n)
    @inbounds for j in 1:n
        for t in A.colptr[j]:(A.colptr[j+1]-1)
            if A.rowval[t]==j
                p[j]=Ti(t)
                break
            end
        end
    end
    return DiagPtr{Ti}(p)
end

"""
    ensure_diagonal(A::SparseMatrixCSC{Float64,Int}) -> SparseMatrixCSC{Float64,Int}

Guarantee that every diagonal entry A[j,j] exists structurally in the sparse
pattern (even if its numerical value is 0).

Why:
- If a diagonal entry is missing from the sparse structure, we cannot shift it
  without changing sparsity (which would break LU reuse and be expensive).
- Adding `spdiagm(0 => zeros(n))` inserts missing diagonal structural zeros.

Note:
- This changes the sparsity structure (adds diagonal zeros), but only once.
"""
function ensure_diagonal(A::SparseMatrixCSC{Float64,Int})
    n=size(A,1)
    dp=diagptr(A)
    if any(dp.p.==0)
        A=A+spdiagm(0=>zeros(n))
    end
    return A
end

"""
    shift_diag!(Az, A0, dp, z)

Given:
- `A0` : real sparse matrix (Float64) with an explicit diagonal
- `Az` : complex sparse matrix with IDENTICAL sparsity pattern as A0
- `dp` : diagonal pointer map for A0 (and Az, since sparsity matches)
- `z`  : complex shift

Overwrite ONLY the diagonal entries of Az to represent:

    Az := A0 - z I

i.e. Az[j,j] = A0[j,j] - z, while leaving off-diagonal entries unchanged.

This is used when assembling (zI - A) efficiently:
- We start from Az = A (complex copy)
- We shift its diagonal to (A - zI)
- Then we negate all values to obtain (zI - A)
"""
@inline function shift_diag!(Az::SparseMatrixCSC{ComplexF64,Ti},A0::SparseMatrixCSC{Float64,Ti},dp::DiagPtr{Ti},z::ComplexF64) where {Ti<:Integer}
    @inbounds for j in 1:length(dp.p)
        t=dp.p[j]
        t==0 && throw(ArgumentError("missing diagonal entry at ($j,$j); call ensure_diagonal(A)"))
        Az.nzval[t]=ComplexF64(A0.nzval[t])-z
    end
    return nothing
end

#################################################################################
##############################  QUADRATURES   ###################################
#################################################################################

"""
    ellipse_nodes_weights(emin, emax; nodes=64, eta=0.6) -> (z, w)

Construct contour nodes `z[k]` and quadrature weights `w[k]` that approximate
the spectral projector integral:

    P = (1/(2π i)) ∮_Γ (zI - A)^{-1} dz
      ≈ Σ_{k=1..nodes} w[k] * (z[k] I - A)^{-1}.

Contour Γ: an ellipse enclosing the real interval [emin, emax], lifted off the
real axis to avoid near-singular shifted solves:
    z(θ) = c + a cosθ + i b sinθ
    c = (emin+emax)/2
    a = (emax-emin)/2
    b = eta*a   (vertical semi-axis)

Trapezoidal rule with θ_k = (k-1)Δθ, Δθ = 2π/nodes:

    z_k = z(θ_k)
    dz  = z'(θ_k) Δθ
    w_k = (1/(2π i)) dz

The factor (1/(2π i)) is built into w_k so you can write directly:
    Y = Σ w_k (z_k I - A)^{-1} Q.
"""
function ellipse_nodes_weights(emin::Float64,emax::Float64;nodes::Int=64,eta::Float64=0.6)
    nodes>0 || throw(ArgumentError("nodes must be positive"))
    c=0.5*(emin+emax)
    a=0.5*(emax-emin)
    a>0 || throw(ArgumentError("need emin<emax"))
    b=eta*a
    Δθ=2*pi/nodes
    z=Vector{ComplexF64}(undef,nodes)
    w=Vector{ComplexF64}(undef,nodes)
    @inbounds for k in 1:nodes
        θ=(k-0.5)*Δθ  
        ct=cos(θ); st=sin(θ)
        z[k]=c+a*ct+im*b*st
        dz=(-a*st+im*b*ct)*Δθ
        w[k]=dz/(2*pi*im)
    end
    return z,w
end

"""
    ellipse_nodes_weights_half(emin, emax; nodes=64, eta=0.6) -> (z_half, w_half)

Construct contour nodes and weights for the half-ellipse that encloses [emin, emax] in the upper half-plane. Used in the real symmetric case to exploit conjugate symmetry and reduce the number of shifted solves by half.
"""
function ellipse_nodes_weights_half(emin::Float64,emax::Float64;nodes::Int=64,eta::Float64=0.6)
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    nh=nodes÷2
    return z[1:nh],w[1:nh] # midpoint makes imag(z)>0 for first half
end

#################################################################################
##############################  ZOLOTAREV HELPERS   #############################
#################################################################################

"""
    ellipk(m;tol=1e-14)

Compute the complete elliptic integral of the first kind

    K(m) = ∫₀^{π/2} dθ / √(1 - m sin²θ)

Arguments
---------
- `m`   : elliptic parameter (`0 ≤ m < 1`)
- `tol` : tolerance for quadgk

Returns
-------
The value `K(m)`.
"""
function ellipk(m::Float64;tol::Float64=1e-14) 
    0.0<=m<1.0 || throw(ArgumentError("need 0 <= m < 1"))
    f(θ)=inv(sqrt(1-m*sin(θ)^2))
    val,_=quadgk(f,0.0,π/2;rtol=tol)
    return val
end
"""
    ellipf(φ,m;tol=1e-14)

Compute the incomplete elliptic integral of the first kind

    F(φ|m) = ∫₀^φ dθ / √(1 - m sin²θ)

Arguments
---------
- `φ`   : amplitude
- `m`   : elliptic parameter (`0 ≤ m < 1`)
- `tol` : tolerance for quadgk

Returns
-------
The value `F(φ|m)`.
"""
function ellipf(φ::Float64,m::Float64;tol::Float64=1e-14)
    0.0<=m<1.0 || throw(ArgumentError("need 0 <= m < 1"))
    0.0<=φ<=π/2 || throw(ArgumentError("need 0 <= φ <= π/2"))
    f(θ)=inv(sqrt(1-m*sin(θ)^2))
    val,_=quadgk(f,0.0,φ;rtol=tol)
    return val
end
"""
    jacobi_sn(u,m;tol=1e-14,maxit=20)

Compute the Jacobi elliptic function

    sn(u | m)

by solving

    F(φ | m) = u

for `φ` using bisection and returning `sin(φ)`.
Reference: Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, S. Güttel, https://arxiv.org/abs/1407.8078


Arguments
---------
- `u`     : argument
- `m`     : elliptic parameter (`0 ≤ m < 1`)
- `tol`  : tolerance used in elliptic integral evaluation
- `φtol`  : tolerance for φ convergence
- `maxit` : maximum Newton iterations

Returns
-------
`sn(u | m)`.
"""
function jacobi_sn(u::Float64,m::Float64;tol::Float64=1e-12,φtol::Float64=1e-13,maxit::Int=200)
    0.0<=m<1.0 || throw(ArgumentError("need 0 <= m < 1"))
    K=ellipk(m;tol=tol)
    0.0<=u<=K || throw(ArgumentError("need 0 <= u <= K(m); got u=$u, K=$K"))
    u==0.0 && return 0.0
    abs(u-K)<=10*eps(Float64)*max(1.0,K) && return 1.0
    lo,hi=0.0,π/2
    for _ in 1:maxit
        mid=0.5*(lo + hi)
        fmid=ellipf(mid,m;tol=tol)-u
        if abs(fmid)<=10*tol || (hi-lo)<=φtol
            return sin(mid)
        end
        if fmid>0.0
            hi=mid
        else
            lo=mid
        end
    end
    return sin(0.5*(lo+hi))
end

"""
    zphi(x,codd,ceven)

Evaluate the rational function

    φ(x) = x ∏(x² + c_even) / ∏(x² + c_odd)

appearing in the canonical Zolotarev filter construction.
Reference: Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, S. Güttel, https://arxiv.org/abs/1407.8078

Arguments
---------
- `x`      : evaluation point
- `codd`   : vector of odd-index coefficients
- `ceven`  : vector of even-index coefficients

Returns
-------
The value `φ(x)`.
"""
@inline zphi(x,codd,ceven)=x*(isempty(ceven) ? one(x) : prod(x^2+c for c in ceven))/prod(x^2+c for c in codd)

@inline function zlogder_y(y,codd,ceven)
    x2=exp(2*y)
    s=one(y)
    @inbounds for c in ceven
        s+=2*x2/(x2+c)
    end
    @inbounds for c in codd
        s-=2*x2/(x2+c)
    end
    return s
end

function _bisect(f,a::Float64,b::Float64;tol::Float64=sqrt(eps(Float64)),maxit::Int=256)
    fa=f(a);fb=f(b)
    fa==0 && return a
    fb==0 && return b
    fa*fb<0 || throw(ArgumentError("root not bracketed"))
    lo,hi,flo,fhi=a,b,fa,fb
    for _ in 1:maxit
        mid=(lo+hi)/2
        fm=f(mid)
        (abs(fm)<=sqrt(eps(Float64)) || (hi-lo)<=tol*max(1.0,abs(mid))) && return mid
        if flo*fm<0
            hi,fhi=mid,fm
        else
            lo,flo=mid,fm
        end
    end
    return (lo+hi)/2
end
"""
    zolotarev_D(codd,ceven,R;ngrid=20000,tol=1e-14)

Compute the normalization constant `D` for the Zolotarev rational filter.

The constant `D` is defined by the equioscillation condition

    D = 2 / (φ_min + φ_max)

where `φ(x)` is evaluated over `[1,R]`.
Reference: Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, S. Güttel, https://arxiv.org/abs/1407.8078

Arguments
---------
- `codd`, `ceven` : coefficient vectors defining `φ(x)`
- `R`             : Zolotarev interval parameter
- `ngrid`         : grid used to locate derivative sign changes
- `tol`           : tolerance for root finding

Returns
-------
The normalization constant `D`.
"""
function zolotarev_D(codd,ceven,R::Float64;ngrid::Int=20000,tol::Float64=1e-14)
    yL=0.0;yR=log(R)
    ys=collect(range(yL,yR;length=ngrid))
    f(y)=zlogder_y(y,codd,ceven)
    cand=[1.0,R]
    fp=f(ys[1])
    @inbounds for k in 1:length(ys)-1
        a=ys[k];b=ys[k+1]
        fa=fp;fb=f(b)
        if fa==0
            push!(cand,exp(a))
        elseif fa*fb<0
            yr=_bisect(f,a,b;tol=tol)
            push!(cand,exp(yr))
        end
        fp=fb
    end
    fp==0 && push!(cand,R)
    ϕmin=typemax(Float64);ϕmax=-typemax(Float64)
    @inbounds for x in cand
        ϕ=zphi(x,codd,ceven)
        ϕ<ϕmin && (ϕmin=ϕ)
        ϕ>ϕmax && (ϕmax=ϕ)
    end
    return 2/(ϕmin+ϕmax)
end

"""
    zolotarev_half(m,G;tol=1e-14,ngrid=20000)

Construct the canonical half-plane Zolotarev rational filter on `[-G,G]`,
with `0<G<1`.

The returned rational filter has the form

    r(x)=α₀ + Σ_j ω_j/(x-p_j) + Σ_j conj(ω_j)/(x-conj(p_j))

with poles `p_j` in the upper half-plane. It is intended to approximate
the spectral projector onto the passband `[-G,G]`.
Reference: Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, S. Güttel, https://arxiv.org/abs/1407.8078


Arguments
---------
- `m`     : number of poles in the upper half-plane
- `G`     : canonical gap parameter, must satisfy `0<G<1`
- `tol`   : tolerance used in elliptic/Jacobi evaluations
- `ngrid` : grid size used in precise normalization constant determination

Returns
-------
Named tuple

    (poles,weights,alpha0,G,R,cvals,D)

where

- `poles`   : upper-half-plane poles
- `weights` : corresponding residues
- `alpha0`  : constant term
- `G`       : canonical passband parameter
- `R`       : associated Zolotarev interval parameter
- `cvals`   : auxiliary coefficients
- `D`       : normalization constant
"""
function zolotarev_half(m::Int,G::Float64;tol::Float64=1e-14,ngrid::Int=20000)
    m>=1 || throw(ArgumentError("m>=1 required"))
    0<G<1 || throw(ArgumentError("need 0<G<1"))
    R=((1+G)/(1-G))^2
    mpar=1-inv(R^2)
    K=ellipk(mpar;tol=tol)
    cvals=Vector{Float64}(undef,2*m-1)
    @inbounds for j in 1:(2*m-1)
        sn=jacobi_sn(j*K/(2*m),mpar;tol=tol)
        cvals[j]=sn^2/(1-sn^2)
    end
    codd=cvals[1:2:end]
    ceven=cvals[2:2:end]
    D=zolotarev_D(codd,ceven,R;ngrid=ngrid,tol=max(eps(Float64),tol^2))
    sqrtR=sqrt(R)
    poles=Vector{ComplexF64}(undef,m)
    weights=Vector{ComplexF64}(undef,m)
    @inbounds for j in 1:m
        cj=codd[j]
        α=complex(0.0,sqrt(cj))
        z=(α-sqrtR)/(α+sqrtR)
        poles[j]=z
        num=isempty(ceven) ? 1.0 : prod(c-cj for c in ceven)
        den=1.0
        for l in 1:m
            l==j && continue
            den*=codd[l]-cj
        end
        resx=(D/2)*(num/den)
        tp=2*sqrtR/(1-z)^2
        weights[j]=-(resx/tp)/2
    end
    alpha0=complex((D*zphi(-sqrtR,codd,ceven)+1.0)/2,0.0)
    return (poles=poles,weights=weights,alpha0=alpha0,G=G,R=R,cvals=cvals,D=D)
end

"""
    map_zolotarev(rf,emin,emax)

Map a canonical Zolotarev filter defined on `[-G,G]` to a physical
spectral interval `[emin,emax]`.
Reference: Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, S. Güttel, https://arxiv.org/abs/1407.8078

Arguments
---------
- `rf`   : canonical filter returned by `zolotarev_half`
- `emin` : lower bound of target interval
- `emax` : upper bound of target interval

Returns
-------
Named tuple with poles and residues mapped to the new interval.
"""
function map_zolotarev(rf::NamedTuple,emin::Float64,emax::Float64)
    T=typeof(real(rf.G))
    a=T(emin);b=T(emax);a<b || throw(ArgumentError("need emin<emax"))
    c=(a+b)/2;d=(b-a)/2
    s=d/rf.G
    return (poles=c.+s.*rf.poles,weights=s.*rf.weights,alpha0=rf.alpha0,center=c,halfwidth=d,scale=s)
end

"""
    zolotarev_nodes_weights(emin,emax;nodes=8,G=0.9,tol=1e-14,ngrid=20000)

Return poles and weights for a Zolotarev rational filter mapped to
the spectral interval `[emin,emax]`.

Arguments
---------
- `emin,emax` : spectral interval
- `nodes`     : number of poles in the upper half-plane
- `G`         : canonical gap parameter
- `tol`       : elliptic/Jacobi tolerance
- `ngrid`     : grid size used in normalization computation

Returns
-------
where

- `z`      : complex poles (including conjugates)
- `w`      : residues
- `alpha0` : constant term of the rational filter
"""
function zolotarev_nodes_weights(emin::Float64,emax::Float64;nodes::Int=8,G::Float64=0.9,tol::Float64=1e-14,ngrid::Int=20000)
    rf=zolotarev_half(nodes,G;tol=tol,ngrid=ngrid)
    mp=map_zolotarev(rf,emin,emax)
    zh=mp.poles;wh=mp.weights
    z=Vector{Complex{typeof(real(zh[1]))}}(undef,2length(zh))
    w=similar(z)
    n=length(zh)
    @inbounds for j in 1:n
        z[j]=zh[j]
        w[j]=wh[j]
        z[n+j]=conj(zh[j])
        w[n+j]=conj(wh[j])
    end
    return z,w,mp.alpha0
end

"""
    zolotarev_nodes_weights_half(emin,emax;nodes=8,G=0.9,T=Float64,tol=1e-14,ngrid=20000)

Return only the upper-half-plane poles and residues of the Zolotarev
filter mapped to `[emin,emax]`. This is for real symmetric matrices where the filter is conjugate symmetric, so we can save half the shifted solves.

Returns
-------
Notes
-----
Intended for real-symmetric FEAST implementations using the identity

    Y = real(alpha0)*Q + 2*Re Σ w_j (z_j I - A)^(-1) Q

which halves the number of shifted linear solves.
"""
function zolotarev_nodes_weights_half(emin::Float64,emax::Float64;nodes::Int=8,G::Float64=0.9,tol::Float64=1e-14,ngrid::Int=20000)
    rf=zolotarev_half(nodes,G;tol=tol,ngrid=ngrid)
    mp=map_zolotarev(rf,emin,emax)
    return mp.poles,mp.weights,mp.alpha0
end

"""
    ZolotarevParams

Container storing precomputed parameters of the Zolotarev rational filter.

The Zolotarev construction depends only on the number of poles and the
bandwidth parameter `G`. Computing the poles and residues is relatively
expensive (due to elliptic integrals and root finding), so these quantities
are computed once and reused across FEAST sweeps.

# Fields
- `G::Float64`
  Zolotarev bandwidth parameter controlling the steepness of the filter.

- `polesh::Vector{ComplexF64}`
  Poles of the canonical Zolotarev filter in the upper half-plane.

- `weightsh::Vector{ComplexF64}`
  Corresponding residues (weights) of the rational approximation.

- `alpha0::ComplexF64`
  Constant term of the rational filter.

The stored poles correspond only to the upper half-plane. For non-Hermitian
problems the lower half-plane poles are generated by conjugation when needed.
"""
struct ZolotarevParams
    G::Float64
    polesh::Vector{ComplexF64}
    weightsh::Vector{ComplexF64}
    alpha0::ComplexF64
end

"""
    ZolotarevParams(nodes,G;tol=1e-14,ngrid=20000)

Construct a `ZolotarevParams` object by computing the Zolotarev
rational approximation parameters.

# Arguments
- `nodes::Int`
  Number of poles in the upper half-plane (half-filter size).

- `G::Float64`
  Bandwidth parameter of the Zolotarev filter (`0 < G < 1`).

# Keyword Arguments
- `tol`
  Numerical tolerance used in the elliptic integral and root-finding steps.

- `ngrid`
  Grid resolution used when locating extrema in the normalization step.

# Returns
A `ZolotarevParams` object containing poles, weights, and the constant term
of the rational filter.
"""
function ZolotarevParams(nodes::Int,G::Float64;tol::Float64=1e-14,ngrid::Int=20000)
    rf=zolotarev_half(nodes,G;tol=tol,ngrid=ngrid)
    return ZolotarevParams(G,rf.poles,rf.weights,rf.alpha0)
end
"""
    zolotarev_nodes_weights_half(zp, emin, emax)

Map the canonical Zolotarev filter stored in `zp` to the interval `[emin, emax]`
for the half-contour formulation used for real symmetric problems.

# Arguments
- `zp::ZolotarevParams`
  Precomputed Zolotarev filter parameters.

- `emin::Float64`
  Lower bound of the target eigenvalue interval.

- `emax::Float64`
  Upper bound of the target eigenvalue interval.

# Returns
`(z, w, alpha0)` where

- `z` are the shifted poles
- `w` are the scaled residues
- `alpha0` is the constant term

Only the upper half-plane poles are returned. This variant is used when
the matrix spectrum is real and symmetry allows evaluation on half
of the complex contour.
"""
@inline function zolotarev_nodes_weights_half(zp::ZolotarevParams,emin::Float64,emax::Float64)
    c=(emin+emax)/2
    s=(emax-emin)/(2*zp.G)
    return c.+s.*zp.polesh,s.*zp.weightsh,zp.alpha0
end
"""
    zolotarev_nodes_weights(zp, emin, emax)

Return the full Zolotarev filter poles and weights mapped to the interval
`[emin, emax]`.

This version is used for **general (non-Hermitian) problems** where both
halves of the complex contour must be evaluated.

# Arguments
- `zp::ZolotarevParams`
  Precomputed Zolotarev filter parameters.

- `emin::Float64`
  Lower bound of the target eigenvalue interval.

- `emax::Float64`
  Upper bound of the target eigenvalue interval.

# Returns
`(z, w, alpha0)` where

- `z` contains poles in both half-planes
- `w` are the corresponding residues
- `alpha0` is the constant term

The lower half-plane poles are generated by conjugating the stored
upper half-plane poles.
"""
function zolotarev_nodes_weights(zp::ZolotarevParams,emin::Float64,emax::Float64)
    zh,wh,alpha0=zolotarev_nodes_weights_half(zp,emin,emax)
    n=length(zh)
    z=Vector{ComplexF64}(undef,2*n)
    w=Vector{ComplexF64}(undef,2*n)
    @inbounds for j in 1:n
        z[j]=zh[j]
        w[j]=wh[j]
        z[n+j]=conj(zh[j])
        w[n+j]=conj(wh[j])
    end
    return z,w,alpha0
end

####################################################################
####################################################################

"""
    residuals!(res, A, X, λ, AX)

Compute absolute residual norms for approximate eigenpairs (λ_j, x_j):

    res[j] = ||A*x_j - λ_j*x_j||_2

Inputs:
- A  : sparse real matrix
- X  : n×m matrix of (complex) approximate eigenvectors (columns)
- λ  : length-m vector of (complex) approximate eigenvalues
- AX : workspace n×m to store A*X

Notes:
- For symmetric real A and a good FEAST subspace, λ should be ~ real.
- Absolute residuals are fine for scaled tests; for production, a relative
  residual is often more meaningful:
      ||A x - λ x|| / (||A x|| + |λ| ||x||).
"""
function residuals!(res::Vector{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{T},λ::Vector{T},AX::Matrix{T}) where T<:Union{Float64,ComplexF64}
    mul!(AX,A,X)
    n,m=size(X)
    @inbounds for j in 1:m
        lj=λ[j]
        s=0.0
        for i in 1:n
            v=AX[i,j]-lj*X[i,j]
            s+=real(v*conj(v))
        end
        res[j]=sqrt(s)
    end
    return nothing
end

"""
    thin_orth!(Qc, Qtmp, Z)

Replace Qc by an orthonormal basis of its column space:

    Qc ← orth(Qc)

Parameters:
- Qc: n×m0 matrix to be orthonormalized in-place (columns replaced by orthonormal basis)
- Qtmp, Z: n×m0 workspace matrices (can be reused across calls)


and ensure the result has size n×m0.

Implementation:
- Compute QR factorization: Qc = Q R.
- Materialize Q as a dense matrix and keep its first m0 columns.

Why we do this:
- FEAST is a subspace method. It requires Rayleigh–Ritz in an ORTHONORMAL basis.
- After projector application, columns become correlated / ill-conditioned.
- Orthonormalizing stabilizes the subspace iteration and ensures that the small
  projected matrix Aq = Q'*A*Q is (numerically) Hermitian for symmetric A.
"""
function thin_orth!(Qc::Matrix{Float64},Qtmp::Matrix{Float64},τ::Vector{Float64})
    n,m=size(Qc) # dims of Qc
    k=min(n,m) # smallest dimension (m0)
    LAPACK.geqrf!(Qc,τ) # QR factorization of Qc with reflectors τ
    copyto!(@view(Qtmp[:,1:k]),@view(Qc[:,1:k])) # get the m0 columns of Qc (inplace QR) and copy to temp
    LAPACK.orgqr!(@view(Qtmp[:,1:k]),τ,k) # get Q (orthogonal) matrix from Qtmp above
    copyto!(Qc,@view(Qtmp[:,1:k])) # copy to Qc the mo columns of the matrix Q (in Qc=Q*R)
    return nothing
end
# there is no LAPACK.ungqr! exposed in LinearAlgebra.jl
function thin_orth!(Qc::Matrix{ComplexF64},Qtmp::Matrix{ComplexF64},Z::Matrix{ComplexF64})
    F=qr!(Qc)
    mul!(Qtmp,F.Q,Z)
    copyto!(Qc,Qtmp)
    return nothing
end
# workspaces for not allocating thin_orth! each iter in max_iter loop in general_feast_sparse_interval
function make_thin_orth_workspace(::Type{ComplexF64},n::Int,m0::Int)
    Qtmp=Matrix{ComplexF64}(undef,n,m0)
    Z=zeros(ComplexF64,n,m0)
    @inbounds for j in 1:m0
        Z[j,j]=1.0+0.0im
    end
    return Qtmp,Z
end
function make_thin_orth_workspace(::Type{Float64},n::Int,m0::Int)
    Qtmp=Matrix{Float64}(undef,n,m0)
    τ=Vector{Float64}(undef,min(n,m0))
    return Qtmp,τ
end

#= OLD GENERALIZED 
Since we already have this, we can just use multiple dispatch and restriuct the type T: function thin_orth!(Qc::Matrix{T},Qtmp::Matrix{T},Z::Matrix{T}) where T<:Union{Float64,ComplexF64}
    F=qr!(Qc)
    mul!(Qtmp,F.Q,Z)
    copyto!(Qc,Qtmp)
    return nothing
end

# workspaces for not allocating thin_orth! each iter in max_iter loop in general_feast_sparse_interval
function make_thin_orth_workspace(T::Type,n::Int,m0::Int)
    Qtmp=Matrix{T}(undef,n,m0)
    Z=zeros(T,n,m0)
    @inbounds for j in 1:m0 # initialize Z to identity for thin_orth! (Qtmp is overwritten by Q*Z)
        Z[j,j]=1
    end
    return Qtmp,Z
end
=#

###########################################################################
####################### MATRIX - FREE FUNCTIONS ###########################
###########################################################################

"""
    MatrixFreeRealSymOp{F}<:AbstractMatrix{Float64}

Matrix-free wrapper for a real symmetric operator, contains a user-provided `matvec!` function that computes `y <- A*x` for a real symmetric operator `A` and `n` is the dimension of the operator.
"""
struct MatrixFreeRealSymOp{F}<:AbstractMatrix{Float64}
    n::Int
    matvec!::F
end
# helper for size of the operator as a matrix
Base.size(A::MatrixFreeRealSymOp)=(A.n,A.n)
"""
    LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})

Matrix-free matrix-vector product for a real symmetric operator `A` represented by a `MatrixFreeRealSymOp`. Computes `y <- A*x` using the user-provided `matvec!` function.
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})
    A.matvec!(y,x)
    return y
end
#TODO need to implement a block_mul! for block GMRES
"""
    apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})

Block matrix-free application of a real symmetric operator `A` to a block of vectors `X`, storing the result in `Y`. Computes `Y[:,j] <- A*X[:,j]` for each column `j` of `X`. 
# NOTE: Currently it does GMRES column by column, but it should use block GMRES.
"""
@inline function apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    n,m=size(X)
    @inbounds for j in 1:m
        mul!(view(Y,:,j),A,view(X,:,j))
    end
    return Y
end
"""
# Complex shifted operator S = zI - A, still matrix-free.
#
# Since A is real, for complex x = xr + i xi:
#     A*x = A*xr + i A*xi
#
# We therefore reuse four real work vectors:
#     xr, xi, yr, yi
# and assemble
#     y = z*x - (A*x)
"""
mutable struct ShiftedMatrixFreeOp{F}<:AbstractMatrix{ComplexF64}
    n::Int
    z::ComplexF64
    matvec!::F
    xr::Vector{Float64}
    xi::Vector{Float64}
    yr::Vector{Float64}
    yi::Vector{Float64}
end
"""
    ShiftedMatrixFreeOp(n::Int,z::ComplexF64,matvec!::F) where {F}

Constructor for `ShiftedMatrixFreeOp`, which creates a matrix-free operator representing the shifted operator `S = zI - A` for a real symmetric operator `A` defined by the user-provided `matvec!` function. It initializes the necessary work vectors for the matrix-vector product.
"""
function ShiftedMatrixFreeOp(n::Int,z::ComplexF64,matvec!::F) where {F}
    xr=zeros(Float64,n)
    xi=zeros(Float64,n)
    yr=zeros(Float64,n)
    yi=zeros(Float64,n)
    return ShiftedMatrixFreeOp{F}(n,z,matvec!,xr,xi,yr,yi)
end
# helper for size of the operator as a matrix
Base.size(S::ShiftedMatrixFreeOp)=(S.n,S.n)
"""
    LinearAlgebra.mul!(y::AbstractVector{ComplexF64},S::ShiftedMatrixFreeOp,x::AbstractVector{ComplexF64})

Matrix-free matrix-vector product for the shifted operator `S = zI - A` represented by a `ShiftedMatrixFreeOp`. Computes `y <- S*x` using the user-provided `matvec!` function for `A` and the shift `z`.
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},S::ShiftedMatrixFreeOp,x::AbstractVector{ComplexF64})
    n=S.n
    xr=S.xr;xi=S.xi
    yr=S.yr;yi=S.yi
    @inbounds for i in 1:n
        xr[i]=real(x[i]);xi[i]=imag(x[i])
    end
    S.matvec!(yr,xr) # yr <- A*real(x)
    S.matvec!(yi,xi) # yi <- A*imag(x)
    z=S.z
    @inbounds for i in 1:n
        y[i]=z*x[i]-ComplexF64(yr[i],yi[i])
    end
    return y
end
"""
    LinearAlgebra.mul!(Y::AbstractMatrix{ComplexF64},S::ShiftedMatrixFreeOp,X::AbstractMatrix{ComplexF64})

Block matrix-free application of the shifted operator `S = zI - A` to a block of vectors `X`, storing the result in `Y`. Computes `Y[:,j] <- S*X[:,j]` for each column `j` of `X` using the user-provided `matvec!` function for `A` and the shift `z`.
"""
@inline function LinearAlgebra.mul!(Y::AbstractMatrix{ComplexF64},S::ShiftedMatrixFreeOp,X::AbstractMatrix{ComplexF64})
    n,m=size(X)
    @inbounds for j in 1:m
        mul!(view(Y,:,j),S,view(X,:,j))
    end
    return Y
end
"""
    residuals!(res::Vector{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},λ::Vector{Float64},AX::Matrix{Float64})

Compute the residual norms `res[j] = ||A*X[:,j] - λ[j]*X[:,j]||` for each column `j` of `X`, where `A` is a real symmetric operator represented by a `MatrixFreeRealSymOp`. The matrix-vector product `A*X` is computed using the user-provided `matvec!` function, and the results are stored in `AX`.
"""
function residuals!(res::Vector{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},λ::Vector{Float64},AX::Matrix{Float64})
    apply_A_block!(AX,A,X)
    n,m=size(X)
    @inbounds for j in 1:m
        lj=λ[j]
        s=0.0
        for i in 1:n
            v=AX[i,j]-lj*X[i,j]
            s+=v*v
        end
        res[j]=sqrt(s)
    end
    return nothing
end
#TODO This is still very slow...
"""
    solve_shifted_block_matrixfree!(X::Matrix{ComplexF64},Sop::ShiftedMatrixFreeOp,B::Matrix{ComplexF64};tol=1e-12,maxiter=5000)

Solve the shifted linear systems defined by `Sop` for each column of `B`, storing the results in `X`. This function uses an iterative linear solver (e.g., GMRES) to solve `Sop * x = b` for each column `b` of `B`, where `Sop` is a `ShiftedMatrixFreeOp` representing the shifted operator `S = zI - A`. The user can specify the linear solver algorithm and convergence parameters.
"""
function solve_shifted_block_matrixfree!(X::Matrix{ComplexF64},Sop::ShiftedMatrixFreeOp,B::Matrix{ComplexF64};tol=1e-12,maxiter=5000,ls_alg::Symbol=:gmres)
    # OLD WAY TO DO IT COLUMN BY COLUMN
    #=
    n,m=size(B)
    @inbounds for j in 1:m
        bj=view(B,:,j);xj=view(X,:,j)
        prob=LinearSolve.LinearProblem(Sop,bj;u0=xj)
        sol=isnothing(ls_alg) ?
        LinearSolve.solve(prob; abstol=tol, reltol=tol, maxiters=maxiter) : LinearSolve.solve(prob, ls_alg; abstol=tol, reltol=tol, maxiters=maxiter)
        copyto!(xj,sol.u)
    end
    =#
    # BLOCK IS THE WAY TO GO
    if ls_alg==:gmres
        Xsol,stats=block_gmres(Sop,B;atol=tol,rtol=tol,itmax=maxiter)
    else
        @warn "Unsupported linear solver algorithm specified: $ls_alg. Falling back to GMRES."
        Xsol,stats=block_gmres(Sop,B;atol=tol,rtol=tol,itmax=maxiter)
    end
    copyto!(X,Xsol)
    return X
end

"""
    apply_A_vec!(y,A::SparseMatrixCSC{Float64,Int},x)

Compute the sparse matrix–vector product `y = A*x` in place.

This is the basic single-vector operator application used throughout the
POLFED implementation. It assumes `A` is real symmetric, but the routine
itself only performs the generic sparse multiply.

Arguments
- `y::AbstractVector{Float64}`: output vector, overwritten with `A*x`
- `A::SparseMatrixCSC{Float64,Int}`: sparse matrix operator
- `x::AbstractVector{Float64}`: input vector

Returns
- `y`, overwritten in place.
"""
@inline apply_A_vec!(y::AbstractVector{Float64},A::SparseMatrixCSC{Float64,Int},x::AbstractVector{Float64})=mul!(y,A,x)

"""
    apply_A_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})

Apply the matrix-free real symmetric operator `A` to a single vector `x`
and store the result in `y`:

    y = A*x

This is the single-vector operator interface used by POLFED in the
matrix-free setting. Internally it dispatches to the user-provided
`matvec!` stored in `A`.

Arguments
- `y::AbstractVector{Float64}`: output vector, overwritten with `A*x`
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `x::AbstractVector{Float64}`: input vector

Returns
- `y`, overwritten in place.
"""
@inline apply_A_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})=mul!(y,A,x)

"""
    apply_A_block!(Y,A::SparseMatrixCSC{Float64,Int},X)

Compute the sparse matrix–block product `Y = A*X` in place.

This is the block analogue of `apply_A_vec!` and is used when applying the
operator to several vectors at once, used during residual evaluation
or local Rayleigh–Ritz refinement.

Arguments
- `Y::AbstractMatrix{Float64}`: output block, overwritten with `A*X`
- `A::SparseMatrixCSC{Float64,Int}`: sparse matrix operator
- `X::AbstractMatrix{Float64}`: input block whose columns are vectors

Returns
- `Y`, overwritten in place.
"""
@inline apply_A_block!(Y::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},X::AbstractMatrix{Float64})=mul!(Y,A,X)

"""
    scaled_apply_vec!(y,A,x,c,d,ax)

Apply the scaled operator
    y = ((A-cI)/d) * x
in place.

This helper is used inside the Chebyshev/Clenshaw recurrence, where the
original operator is first mapped from its spectral interval `[smin,smax]`
to `[-1,1]` via
    Â = (A-cI)/d,
with `c = (smin+smax)/2` and `d = (smax-smin)/2`.

Arguments
- `y::AbstractVector{Float64}`: output vector
- `A`: linear operator compatible with `apply_A_vec!`
- `x::AbstractVector{Float64}`: input vector
- `c::Float64`: spectral center shift
- `d::Float64`: spectral half-width scale
- `ax::AbstractVector{Float64}`: workspace used to store `A*x`

Returns
- `y`, overwritten in place.
"""
function scaled_apply_vec!(y::AbstractVector{Float64},A::SparseMatrixCSC{Float64,Int},x::AbstractVector{Float64},c::Float64,d::Float64,ax::AbstractVector{Float64})
    apply_A_vec!(ax,A,x)
    α=1/d;β=-c/d
    @inbounds @simd for i in eachindex(x)
        y[i]=α*ax[i]+β*x[i]
    end
end

"""
    scaled_apply_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64},c::Float64,d::Float64,ax::AbstractVector{Float64})

Apply the scaled matrix-free operator

    y = ((A-cI)/d) * x

in place.

This helper is used inside the Chebyshev/Clenshaw recurrence for POLFED.
The original operator is mapped from the spectral interval `[smin,smax]`
to `[-1,1]` via

    Â = (A-cI)/d,
    c = (smin+smax)/2,
    d = (smax-smin)/2.

Arguments
- `y::AbstractVector{Float64}`: output vector
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `x::AbstractVector{Float64}`: input vector
- `c::Float64`: spectral center shift
- `d::Float64`: spectral half-width scale
- `ax::AbstractVector{Float64}`: workspace used to store `A*x`

Returns
- `y`, overwritten in place.
"""
function scaled_apply_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64},c::Float64,d::Float64,ax::AbstractVector{Float64})
    apply_A_vec!(ax,A,x)
    α=1/d;β=-c/d
    @inbounds @simd for i in eachindex(x)
        y[i]=α*ax[i]+β*x[i]
    end
end

###########################################################
##################### ALGORITHMS ##########################
###########################################################

"""
    general_feast_sparse_interval(Ain; emin, emax, m0=200, nodes=64, eta=0.6, maxiter=8, tol=1e-10, res_gate=1e-6, debug=false, is_hermitian=true) -> (λsel::Vector{Float64}, Xsel::Matrix{ComplexF64}, ressel::Vector{Float64})

Compute eigenpairs of a real symmetric sparse matrix A with eigenvalues in the
target interval [emin, emax].
---------------------------------------
Algorithm (mathematical steps)
---------------------------------------

Let Γ be the ellipse enclosing [emin, emax] (lifted off the real axis).
Define the spectral projector onto eigenvectors inside Γ:

    P = (1/(2π i)) ∮_Γ (zI - A)^{-1} dz.

Approximate P by trapezoidal quadrature:

    P ≈ Σ_{k=1..nodes} w_k (z_k I - A)^{-1}.

Subspace iteration (rational filtering):

1) Initialize Q ∈ ℂ^{n×m0} random.
2) For it = 1..maxiter:
    a) Orthonormalize:      Q ← orth(Q).
    b) Apply projector:     Y = P Q ≈ Σ w_k (z_k I - A)^{-1} Q.
    c) Orthonormalize:      Q ← orth(Y).
    d) Rayleigh–Ritz:
           Aq = Q* A Q  (m0×m0)
           Solve Aq u = λ u (small dense problem)
           Lift x = Q u.
    e) Compute residuals r_j = ||A x_j - λ_j x_j||.
    f) Count eigenpairs inside interval (and optionally below res_gate), stop if max residual < tol.
    g) Finally, return the subset with λ in [emin, emax] and residual below a gate.

---------------------------------------
Parameters
---------------------------------------
- m0:
    Size of the search subspace. Must exceed the number of eigenvalues inside
    [emin, emax] (or Γ), plus padding.
    If m0 is too small you will miss eigenvalues.
- nodes:
    Number of quadrature nodes on Γ. Larger nodes => better approximation of P,
    but more factorizations and solves.
- eta:
    Imaginary lift of ellipse (b = eta*a). Too small => (zI-A) close to singular
    near real spectrum; too large => filter becomes less selective.
- tol:
    Convergence criterion based on residuals of accepted eigenpairs.
- res_gate:
    A reporting/selection threshold used for counting inside.
    For debugging, you may set this large or ignore it; for production,
    this prevents counting Ritz junk.
- is_hermitian:
    If true, assume Aq is Hermitian and use Hermitian eigensolver.
    This is correct when Q is orthonormal and A is symmetric.
- debug:
    If true, print iteration info (number of inside eigenvalues, max residual).
- show_progress:
    If true, display progress bars for factorization and iteration loops.
- backend:
    :lu (default) uses direct LU factorization for shifted solves.
    :ls uses an iterative linear solver (e.g. GMRES) for shifted solves.
- ls_alg:
    If backend=:ls, you can specify the iterative solver algorithm (e.g. MKLPardisoFactorization()) from LinearSolve.jl; if not provided, the default is used
- two_hit:
    If true (default), require two consecutive iterations with maxres<tol to declare convergence, for robustness against transient small residuals.

Returns:
- λsel: selected eigenvalues (Float64, sorted)
- Xsel: corresponding eigenvectors (columns) in ℂ^{n×m}
- ressel: residual norms for those selected eigenpairs
"""
function general_feast_sparse_interval(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small; use 32/64+ for robustness"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    A=ensure_diagonal(Ain)
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    R=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0);inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)

    Ffact=backend===:lu ? Vector{Any}(undef,nodes) : nothing
    lins=backend===:ls ? Vector{Any}(undef,nodes) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing

    # This part prepares the P = Σ w_k (z_k I - A)^{-1} operator by factorizing (z_k I - A) for each node k. This is not 
    # the most expensive part, but can be significant for large nodes.
    p_fact=show_progress ? Progress(nodes;desc="FEAST: factorizations") : nothing
    @inbounds for k in 1:nodes
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,z[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t] # (zI-A)
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    vY=vec(Y)
    # the reall bottleneck, as it requires maxiter*nodes solves; the factorization above is a one-time cost while solve! 
    # is really expensive and dominates the runtime for large problems.
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nodes;desc="FEAST: subspace iteration ($(it))") : nothing
        fill!(Y,0.0+0.0im)
        @inbounds for k in 1:nodes
            if backend===:lu
                ldiv!(Tm,Ffact[k],Qc) # solve (zI-A) Tm = Qc
                LinearAlgebra.BLAS.axpy!(w[k],vec(Tm),vY) # Y += w[k] * Tm (accumulate into Y)
                show_progress && next!(p_it)
            else
                lins[k].b=Qc
                sol=LinearSolve.solve!(lins[k]) # solve (zI-A) Tm = Qc using iterative solver
                LinearAlgebra.BLAS.axpy!(w[k],vec(sol.u),vY) # Y += w[k] * Tm (accumulate into Y)
                show_progress && next!(p_it)
            end
        end
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z) # relatively cheap since nodes loop dominates
        mul!(R,A,Qc) # BLAS gemm: O(n m0^2)
        mul!(Aq,adjoint(Qc),R) # BLAS gemm: O(n m0^2)
        if is_hermitian
            E=eigen!(Hermitian(Aq)) # O(m0^3) but m0 is small!!!
            λ.=ComplexF64.(E.values)
            Xq.=E.vectors
        else
            E=eigen(Aq) # O(m0^3) but m0 is small!!!
            λ.=E.values
            Xq.=E.vectors
        end
        mul!(X,Qc,Xq) # O(n m0^2)
        residuals!(res,A,X,λ,AX) # O(n m0^2) but dominated by solves
        # each solve is O(n^2) for direct LU, or depends on the iterative solver and convergence (check LinearSolve.jl docs)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("inside=%d  maxres=%.6e\n",count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    general_feast_sparse_interval_distributed(Ain; emin, emax, m0=200, nodes=64, eta=0.6,
                                      maxiter=8, tol=1e-10, res_gate=1e-6,
                                      debug=false, is_hermitian=true,
                                      backend=:lu, ls_alg=nothing, blockrhs=0, two_hit=true, show_progress=false)

Low-memory FEAST implementation for computing eigenpairs of a large sparse real matrix

    A x = λ x

with eigenvalues in the interval `[emin, emax]`.

Algorithmically this is identical to `general_feast_sparse_interval`:

    P = (1/(2π i)) ∮_Γ (zI - A)⁻¹ dz
      ≈ Σ_k w_k (z_k I - A)⁻¹

and the FEAST iteration is

    Q ← orth(Q)
    Y = Σ_k w_k (z_k I - A)⁻¹ Q
    Q ← orth(Y)
    Aq = Qᴴ A Q
    solve Aq u = λ u
    x = Q u

Returns:
    λsel  – eigenvalues in `[emin, emax]`
    Xsel  – corresponding eigenvectors
    ressel – residual norms
"""
function general_feast_sparse_interval_distributed(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=ensure_diagonal(Ain)
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    # Sparse shifted operator template: reuse (colptr,rowval); only nzval changes with z_k.
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    
    # unavoidable O(n*m0) memory
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    R=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)

    vY=vec(Y)
    # FEAST: spectral projector P = (1/2πi)∮(zI-A)^{-1}dz ≈ Σ_k w_k (z_k I - A)^{-1} 
    prev_ok=false
    for it in 1:maxiter
        fill!(Y,0.0+0.0im)
        # Y = Σ_k w_k (z_k I - A)^{-1} Qc
        progress_k=show_progress ? Progress(nodes;desc="FEAST: node iteration, i=($(it)) from maxiter=$(maxiter)") : nothing
        @inbounds for k in 1:nodes
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,z[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t] # (zI-A)
            end 
            wk=w[k]
            if backend===:lu
                F=lu(Az)
                if blockrhs==m0
                    ldiv!(Tm,F,Qc)
                    LinearAlgebra.BLAS.axpy!(wk,vec(Tm),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        ldiv!(Ublk,F,Bblk)
                        @views Y[:,j0:j0+bs-1].+=wk.*Ublk[:,1:bs]
                        j0+=bs
                    end
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                if blockrhs==m0
                    prob=LinearSolve.LinearProblem(Ak,Qc;u0=Tm)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    LinearAlgebra.BLAS.axpy!(wk,vec(sol.u),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                        lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                        sol=LinearSolve.solve!(lin)
                        @views Y[:,j0:j0+bs-1].+=wk.*sol.u[:,1:bs]
                        j0+=bs
                    end
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        # Next subspace
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z) # cost is O(n m0^2)
        # Rayleigh–Ritz: Aq = Qᴴ A Q (m0×m0), eigen solve is O(m0^3) and tiny vs sparse solves.
        mul!(R,A,Qc) # BLAS gemm: O(n m0^2)
        mul!(Aq,adjoint(Qc),R) # BLAS gemm: O(n m0^2)
        if is_hermitian
            E=eigen!(Hermitian(Aq)) # O(m0^3) but m0 is small!!!
            λ.=ComplexF64.(E.values) 
            Xq.=E.vectors
        else
            E=eigen(Aq) # O(m0^3) but m0 is small!!!
            λ.=E.values
            Xq.=E.vectors
        end
        # Lift eigenvectors + residual filter
        mul!(X,Qc,Xq) # O(n m0^2)
        residuals!(res,A,X,λ,AX) # O(n m0^2) 
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half(Ain; emin, emax, m0=200, nodes=64, eta=0.6,
                                   maxiter=8, tol=1e-10, res_gate=1e-6,
                                   debug=false, show_progress=false,
                                   backend=:lu, ls_alg=nothing, two_hit=true)

FEAST eigensolver for a real symmetric sparse matrix

    A x = λ x

that computes eigenpairs with eigenvalues in the interval `[emin, emax]`.

This version uses a half-contour quadrature to accelerate the spectral
projector evaluation. For real matrices and real FEAST subspaces `Q`,
contributions from conjugate contour nodes satisfy

    w (zI - A)⁻¹ Q + w̄ (z̄I - A)⁻¹ Q = 2 Re[w (zI - A)⁻¹ Q].

Therefore the projector can be evaluated using only the nodes in the
upper half-plane, reducing the number of shifted linear solves by
approximately a factor of two.

The Rayleigh–Ritz step is performed in a real subspace and the projected
matrix is solved using a symmetric eigensolver.

# Parameters
- `m0` : dimension of the FEAST search subspace (must exceed the number of eigenvalues in the interval).
- `nodes` : number of quadrature nodes on the full contour (only `nodes/2` solves are performed).
- `eta` : imaginary lift of the ellipse contour (`b = eta*a`).
- `maxiter` : maximum FEAST iterations.
- `tol` : convergence tolerance based on residual norms.
- `res_gate` : residual threshold used when selecting eigenpairs inside the interval.
- `backend` : solver for shifted systems (`:lu` or `:ls` via LinearSolve.jl).
- `ls_alg` : optional LinearSolve algorithm.
- `two_hit` : if true, require two consecutive iterations with maxres<tol to declare convergence, for robustness against transient small residuals.

# Returns
- `λsel::Vector{Float64}` : eigenvalues in `[emin, emax]` (sorted).
- `Xsel::Matrix{Float64}` : corresponding eigenvectors.
- `ressel::Vector{Float64}` : residual norms.
"""
function sym_feast_sparse_interval_half(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)

    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))

    A=ensure_diagonal(Ain)
    zh,wh=ellipse_nodes_weights_half(emin,emax;nodes=nodes,eta=eta)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    # Complex solve workspaces
    Bc=Matrix{ComplexF64}(undef,n,m0) 
    Tm=Matrix{ComplexF64}(undef,n,m0) 
    vTm=vec(Tm)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    Ffact=backend===:lu ? Vector{Any}(undef,nh) : nothing
    lins=backend===:ls ? Vector{Any}(undef,nh) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing
    p_fact=show_progress ? Progress(nh;desc="FEAST(sym,half): factorizations") : nothing
    @inbounds for k in 1:nh
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,zh[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t] # (zI-A)
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nh;desc="FEAST(sym,half): iter $(it)") : nothing
        fill!(Y,0.0)
        @views for j in 1:m0
            @inbounds for i in 1:n
                Bc[i,j]=Q[i,j]
            end
        end
        @inbounds for k in 1:nh
            if backend===:lu
                ldiv!(Tm,Ffact[k],Bc)
            else
                lins[k].b=Bc
                sol=LinearSolve.solve!(lins[k])
                copyto!(Tm,sol.u)
            end
            wk=wh[k]
            @inbounds for idx in eachindex(vTm)
                Y[idx]+=2.0*real(wk*vTm[idx])   # half-node accumulation
            end
            show_progress && next!(p_it)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(R,A,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half_distributed(Ain; emin, emax, m0=200, nodes=64, eta=0.6, maxiter=8, tol=1e-10, res_gate=1e-6, debug=false, backend=:lu, ls_alg=nothing, blockrhs=0, show_progress=false, two_hit=true)

Low-memory FEAST implementation for computing eigenpairs of a real
symmetric sparse matrix

    A x = λ x

with eigenvalues in `[emin, emax]`.

This routine uses the same half-contour spectral projector
as `sym_feast_sparse_interval_half`, exploiting conjugate symmetry of
the contour quadrature to evaluate

    P Q ≈ 2 Σ Re[w_k (z_k I - A)⁻¹ Q]

using only the contour nodes in the upper half-plane.

Unlike the pre-factorized version, this implementation **does not store
factorizations for all contour nodes**. Each shifted system `(zI-A)` is
assembled and solved on the fly, reducing memory usage for large
problems.

Right-hand sides may be processed in blocks controlled by `blockrhs`.

# Parameters
- `m0` : FEAST search subspace dimension.
- `nodes` : number of quadrature nodes on the full contour.
- `eta` : ellipse lift parameter.
- `blockrhs` : number of right-hand sides solved simultaneously.
- `backend` : linear solver (`:lu` or `:ls`).
- `ls_alg` : optional LinearSolve algorithm.
- `two_hit` : if true, require two consecutive iterations with maxres<tol to declare convergence.

# Returns
- `λsel::Vector{Float64}` : eigenvalues in `[emin, emax]`.
- `Xsel::Matrix{Float64}` : corresponding eigenvectors.
- `ressel::Vector{Float64}` : residual norms.
"""
function sym_feast_sparse_interval_half_distributed(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)

    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)

    A=ensure_diagonal(Ain)
    zh,wh=ellipse_nodes_weights_half(emin,emax;nodes=nodes,eta=eta)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    # block RHS controls memory/solve shape
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    # complex block workspaces (only these are complex)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)
    prev_ok=false
    for it in 1:maxiter
        fill!(Y,0.0)
        progress_k=show_progress ? Progress(nh;desc="FEAST(sym,half,dist): iter $(it)") : nothing
        @inbounds for k in 1:nh
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,zh[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t] # (zI-A)
            end
            wk=wh[k]
            if backend===:lu
                F=lu(Az)
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @views Bblk[:,1:bs].=Q[:,j0:j0+bs-1]
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    ldiv!(Ublk,F,Bblk)
                    # accumulate: Y[:,j0:j0+bs-1] += 2*real(wk*Ublk[:,1:bs])
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*Ublk[i,jj])
                        end
                    end
                    j0+=bs
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Bblk[i,jj]=Q[i,col]
                        end
                    end
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*sol.u[i,jj])
                        end
                    end
                    j0+=bs
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(R,A,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    gershgorin_bounds(A) -> (smin,smax)

Cheap spectral bounds for a real symmetric sparse matrix using Gershgorin discs. Safe but often loose.
"""
function gershgorin_bounds(A::SparseMatrixCSC{Float64,Int})
    n=size(A,1);size(A,2)==n||throw(ArgumentError("A must be square"))
    d=zeros(Float64,n)
    r=zeros(Float64,n)
    @inbounds for j in 1:n
        for t in A.colptr[j]:(A.colptr[j+1]-1)
            i=A.rowval[t]
            v=A.nzval[t]
            if i==j
                d[i]=v
            else
                r[i]+=abs(v)
            end
        end
    end
    return minimum(d.-r),maximum(d.+r)
end

"""
    jackson_factor(k,m)

Jackson damping factor for Chebyshev truncation degree m.
"""
@inline function jackson_factor(k::Int,m::Int)
    n=m
    return ((n-k+1)*cos(pi*k/(n+2))+sin(pi*k/(n+2))/tan(pi/(n+2)))/(n+2)
end

#################################################
############## CHEBYSHEV WINDOW #################
#################################################

"""
    cheb_window_coeffs(a,b,m; jackson=true)

Chebyshev coefficients for the indicator/window function 1_[a,b](x) on [-1,1]:

    f(x) ≈ Σ_{k=0}^m c[k+1] T_k(x)

with optional Jackson damping.
"""
function cheb_window_coeffs(a::Float64,b::Float64,m::Int;jackson::Bool=true)
    -1.0<=a<b<=1.0 || throw(ArgumentError("need -1 <= a < b <= 1"))
    m>=0 || throw(ArgumentError("need m >= 0"))
    α=acos(b)
    β=acos(a)
    c=zeros(Float64,m+1)
    c[1]=(β-α)/pi
    @inbounds for k in 1:m
        c[k+1]=(2/pi)*(sin(k*β)-sin(k*α))/k
    end
    if jackson
        @inbounds for k in 0:m
            c[k+1]*=jackson_factor(k,m)
        end
    end
    return c
end

#################################################
########### CHEBYSHEV SMOOTH WINDOW #############
#################################################

function cheb_smooth_window_coeffs(a::Float64,b::Float64,m::Int;δ::Float64=0.02,jackson::Bool=false,nq::Int=max(4*m,1024))
    -1.0<=a<b<=1.0||throw(ArgumentError("need -1 <= a < b <= 1"))
    m>=0||throw(ArgumentError("need m >= 0"))
    δ>0||throw(ArgumentError("need δ > 0"))
    nq>=2||throw(ArgumentError("need nq >= 2"))
    c=zeros(Float64,m+1)
    Δθ=π/nq
    @inbounds for j in 0:(nq-1)
        θ=(j+0.5)*Δθ
        x=cos(θ)
        fx=0.5*(tanh((x-a)/δ)-tanh((x-b)/δ))
        c[1]+=fx
        for k in 1:m
            c[k+1]+=fx*cos(k*θ)
        end
    end
    c[1]*=1/nq
    @inbounds for k in 1:m
        c[k+1]*=2/nq
    end
    if jackson
        @inbounds for k in 0:m
            c[k+1]*=jackson_factor(k,m)
        end
    end
    return c
end

#################################################
############## CHEBYSHEV DELTA ##################
#################################################

"""
    cheb_delta_coeffs(σ,m;jackson=true)

Chebyshev coefficients for a delta-like filter centered at `σ∈[-1,1]`:

    p_m(x) ≈ Σ_{k=0}^m c[k+1] T_k(x)

with optional Jackson damping.
"""
function cheb_delta_coeffs(σ::Float64,m::Int;jackson::Bool=true)
    -1.0<=σ<=1.0 || throw(ArgumentError("need -1 <= σ <= 1"))
    m>=0 || throw(ArgumentError("need m >= 0"))
    θ=acos(σ)
    c=zeros(Float64,m+1)
    c[1]=1/pi
    @inbounds for k in 1:m
        c[k+1]=(2/pi)*cos(k*θ)
    end
    if jackson
        @inbounds for k in 0:m
            c[k+1]*=jackson_factor(k,m)
        end
    end
    return c
end

#################################################

"""
    scaled_mul!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})

Compute Y = ((A - cI)/d) * X using AX as workspace for A*X.
"""
function scaled_mul!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})
    mul!(AX,A,X)
    n,m=size(X)
    invd=1/d
    @inbounds for j in 1:m
        for i in 1:n
            Y[i,j]=(AX[i,j]-c*X[i,j])*invd
        end
    end
    return nothing
end

"""
    scaled_mul!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})

Compute Y = ((A - cI)/d) * X for a matrix-free real symmetric operator A, using AX as workspace for A*X.
"""
function scaled_mul!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})
    apply_A_block!(AX,A,X)
    n,m=size(X)
    invd=1/d
    @inbounds for j in 1:m, i in 1:n
        Y[i,j]=(AX[i,j]-c*X[i,j])*invd
    end
    return nothing
end

"""
    apply_cheb_filter!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})

Apply the Chebyshev polynomial filter of `A` with coefficients `coeffs` to the block `Q`.

The matrix is first rescaled to
    Â = (A-cI)/d
where
    c = (smin+smax)/2, d = (smax-smin)/2

so that σ(Â) ⊂ [-1,1].

The output is
    Y = Σ_{k=0}^m coeffs[k+1] T_k(Â) Q.
"""
function apply_cheb_filter!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64};multithreading::Bool=false)
    m=length(coeffs)-1
    n,blk=size(Q) # n is matrix size, blk is block size (number of vectors in Q)
    c=0.5*(smin+smax) # center of spectral interval
    d=0.5*(smax-smin) # half-width of spectral interval
    d>0 || throw(ArgumentError("need smin < smax"))
    # U0 = T0(Â)Q = Q
    copyto!(U0,Q) # workspace for T_k(Â) * Q
    fill!(Y,0.0) # workspace for the output
    c0=coeffs[1] # contribution of T_0(Â) * Q = Q
    @inbounds for j in 1:blk, i in 1:n
        Y[i,j]=c0*U0[i,j] # Y <- c_0 * U_0 = c_0 * T_0(Â) * Q = c_0 * Q
    end
    m==0 && return nothing # if the poly has degree 0, then p_0(x) = c_0m is a constant and we are done
    # the first chebyshev polynomial is T_1(x) = x (so T_1(Â) = Â), so U1 = T1(Â) * Q = Â * Q
    scaled_mul!(U1,A,Q,c,d,AX) # U1 = T1(Â) * Q = Â * Q
    c1=coeffs[2] # contribution of U1 = T_1(Â) * Q = Â * Q
    LinearAlgebra.BLAS.axpy!(c1,vec(U1),vec(Y)) # now we add the first (linear) term (c_1) to the output Y as Y <- Y + c_1 * U_1 = c_0 * Q + c_1 * Â * Q
    m==1 && return nothing # if order of polynomial is 1, then we are done
    # now we use the three-term recurrence for Chebyshev polynomials to compute U_k = T_k(Â) * Q for k=2,...,m:
    # U_k = 2 * Â * U_{k-1} - U_{k-2}, from T_{k+1}(x) = 2x * T_k(x)-T_{k-1}(x).
    # at the start of each iteration U0 hold T_{k-1}(Â) * Q and U1 holds T_k(Â) * Q, so we can compute U_{k+1} = 2Â * U_k - U_{k-1} = 2Â * U1 - U0
    invd=1/d
    @inbounds for k in 1:(m-1) # sequential 3-term reccurence
        # U2 = 2ÂU1 - U0
        mul!(AX,A,U1) # AX = A * U1 (not Â)
        for j in 1:blk
            for i in 1:n
            U2[i,j]=2*((AX[i,j]-c*U1[i,j])*invd)-U0[i,j] # U2 = 2Â * U1 - U0 = 2((A - cI) / d) * U1 - U0 = 2(A * U1 - c * U1)/d - U0; because A in the previous line is not scaled Â, we need to do the scaling here
        
            end
        end
        ck1=coeffs[k+2] # contribution of U_{k+1} = T_{k+1}(Â) * Q (shifted by 1 index because coeffs[1] is for T_0 ...)
        LinearAlgebra.BLAS.axpy!(ck1,vec(U2),vec(Y)) # add the contribution of U_{k+1} = T_{k+1}(Â) * Q to the output Y
        U0,U1,U2=U1,U2,U0 # reasing workspaces for the next iteration: now U0 holds T_k(Â) * Q and U1 holds T_{k+1}(Â) * Q for the next iteration
    end
    return nothing
end

"""
    apply_cheb_filter!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})

Apply the Chebyshev polynomial filter of `A` with coefficients `coeffs` to the block `Q`, where now `A` is a linear map.

"""
function apply_cheb_filter!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})
    m=length(coeffs)-1
    n,blk=size(Q) # blk is the number of vector columns
    c=0.5*(smin+smax) # center of the window
    d=0.5*(smax-smin) # half width of the window
    copyto!(U0,Q) #U0 is the workspace before accumulating to Y (both Q and U0/1/2 are workspaces - there are 3 for U due to three term recurrence relation for the chebyshev polys)
    fill!(Y,0.0) # initialize the output
    c0=coeffs[1] # first coeff (constant) cheb poly
    @inbounds for j in 1:blk, i in 1:n
        Y[i,j]=c0*U0[i,j] # add the first contribution to Y from the constant term cheb poly
    end
    m==0 && return nothing # if constant, return (not the case mostly if not wrong)
    scaled_mul!(U1,A,Q,c,d,AX) # U1 = T1(Â) * Q = Â * Q (this rescales the window to [-1,1] and computes the 1st order chebyshev poly)
    LinearAlgebra.BLAS.axpy!(coeffs[2],vec(U1),vec(Y)) # add the 2nd order cheb poly with its weight/coefficient into the result accumulator Y
    m==1 && return nothing # if only first order return
    invd=1/d
    @inbounds for k in 1:(m-1) # sequential 3-term reccurence
        apply_A_block!(AX,A,U1) # apply matrix A (here it is a linear operator) to U1 and store into temp AX
        for j in 1:blk
            for i in 1:n
            U2[i,j]=2*((AX[i,j]-c*U1[i,j])*invd)-U0[i,j] # U2 = 2Â * U1 - U0 = 2((A - cI) / d) * U1 - U0 = 2(A * U1 - c * U1)/d - U0; because A in the previous line is not scaled Â, we need to do the scaling here
            end
        end
        LinearAlgebra.BLAS.axpy!(coeffs[k+2],vec(U2),vec(Y)) # accumulate the new k-th term into the output
        U0,U1,U2=U1,U2,U0 # reshift the indexes: T0,T1,T2 -> T3, next iter T1,T2,T3 -> T4 etc.
    end
    return nothing
end

"""
    chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int};σ::Float64=0.0,emin::Float64,emax::Float64,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

Compute eigenpairs of a real symmetric sparse matrix with eigenvalues in [emin,emax]
using Chebyshev polynomial filtered subspace iteration.

Parameters
- `σ`:
    In case of the window_type == :delta, this is the center of the delta function.
- `m0`:
    Search subspace dimension. Must exceed the number of eigenvalues in the interval.
- `degree`:
    Degree of the Chebyshev window polynomial. Larger degree => sharper spectral filter.
- `jackson`:
    If true, use Jackson damping to reduce Gibbs oscillations.
- `two_hit`:
    If true, require 2 consecutive iterations with maxres<tol (except immediate break at it=1).
- `window_type::Symbol`
    If :delta, approximates a "wide" delta and if :step gives the step function in the given wanted energy interval. Also can use :smooth to give a smoothed indicator function on the interval for better convergence.

Returns
- `λsel`: selected eigenvalues in [emin,emax], sorted
- `Xsel`: corresponding eigenvectors
- `ressel`: residual norms
"""
function chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int},emin::Float64,emax::Float64;smin::Float64,smax::Float64,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

    issymmetric(Ain) || throw(ArgumentError("Chebyshev filtered solver requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    degree>=0 || throw(ArgumentError("degree must be nonnegative"))
    n=size(Ain,1)
    size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=Ain
    smin_,smax_=gershgorin_bounds(A) # cheap spectral bounds
    (emin>=smin_ && emax<=smax_) || @warn "target interval not contained in supplied spectral bounds; filter quality may degrade"

    # target interval mapped to [-1,1]
    c=0.5*(smin_+smax_) # center of spectral interval
    d=0.5*(smax_-smin_) # half-width of spectral interval
    a=(emin-c)/d # left endpoint of target interval mapped to [-1,1]
    b=(emax-c)/d # right endpoint of target interval mapped to [-1,1]
    -1.0<=a<b<=1.0 || throw(ArgumentError("mapped interval [$a,$b] not inside [-1,1]; provide better spectral bounds"))
    σ̂=(σ-c)/d
     # give the window coefficients for the mapped interval
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree;jackson=jackson)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknow window type, possible are :step and :delta")
    end
    Random.seed!(seed)
    Q=randn(n,m0) # initialize a random block of m0 vectors; will be replaced by the filtered subspace
    Y=Matrix{Float64}(undef,n,m0) # workspace for the filtered block Y = p(Â)Q
    U0=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    U1=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    U2=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    AXf=Matrix{Float64}(undef,n,m0) # workspace for A*Q in the Chebyshev recurrence
    # RR workspaces
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0) # workspaces for thin orthogonalization of Q
    R=Matrix{Float64}(undef,n,m0) # workspace for A*Q in the Rayleigh-Ritz step
    Aq=Matrix{Float64}(undef,m0,m0) # workspace for the Rayleigh quotient matrix Q'*A*Q
    Xq=Matrix{Float64}(undef,m0,m0) # workspace for Ritz vectors in the Rayleigh-Ritz step
    λ=Vector{Float64}(undef,m0) # workspace for Ritz values in the Rayleigh-Ritz step
    X=Matrix{Float64}(undef,n,m0) # workspace for Ritz vectors in the original space
    AX=Matrix{Float64}(undef,n,m0) # workspace for A*X in the residual computation
    res=Vector{Float64}(undef,m0) # workspace for residual norms
    inside=falses(m0) # workspace for tracking which Ritz pairs are inside the target interval and have residuals below res_gate
    
    Qlock=Matrix{Float64}(undef,n,0)
    λlock=Float64[]
    reslock=Float64[]
    lockmask=falses(m0)
    
    # ALGORITHM
    # For it in 1:maxiter
    #     1. Q = orthonormalize(Q)
    #     2. Y = p(Â)Q - apply the Chebyshev filter to the block Q; Y is the filtered block
    #     3. Q = Y - replace Q by the filtered block; will be orthogonalized in the next iteration
    #     4. Q = orthonormalize(Q) - orthogonalize the filtered block Q
    #     5. R = A*Q - compute A*Q for the Rayleigh-Ritz step
    #     6. Aq = Q'*A*Q - compute the Rayleigh quotient matrix
    #     7. E = eigen(Symmetric(Aq)) - Ritz values/vectors of Aq; E.values are sorted in ascending order
    #     8. X = Q*E.vectors - Ritz vectors in the original space
    #     9. res = residuals(A,X,λ,AX) - compute residual norms for the Ritz pairs
    #     10. track which Ritz pairs are inside the target interval and have residuals below res_gate
    #     11. maxres = maximum residual norm among the Ritz pairs inside the target interval
    #     12. if maxres < tol for 2 consecutive iterations (it two_hit, except immediate break at it=1), then break
    # end
    prev_ok=false # flag to allow termination after 2 consecutive iterations with maxres<tol (except immediate break at it=1)
    for it in 1:maxiter
        apply_cheb_filter!(Y,A,Q,coeffs,smin_,smax_,U0,U1,U2,AXf) # Y = p(Â)Q - filtered block
        Q.=Y # replace Q by the filtered block; will be orthogonalized in the next iteration
        thin_orth!(Q,Qtmp,Z) # orthogonalize the filtered block Q
        mul!(R,A,Q) # compute A*Q for the Rayleigh-Ritz step
        mul!(Aq,transpose(Q),R) # Aq = Q'*A*Q - Rayleigh quotient matrix
        E=eigen!(Symmetric(Aq)) # Ritz values/vectors of Aq; E.values are sorted in ascending order
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq) # Ritz vectors in the original space
        residuals!(res,A,X,λ,AX) # compute residual norms for the Ritz pairs
        
        fill!(inside,false)
        @inbounds for j in 1:m0 
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate) # track which Ritz pairs are inside the target interval and have residuals below res_gate
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf # maximum residual norm among the Ritz pairs inside the target interval
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    chebyshev_poly_real_symm_sparse(matvec!::F;σ::Float64=0.0,emin::Float64,emax::Float64,smin::Float64,smax::Float64,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

For details on parameters see the `chebyshev_poly_real_symm_sparse` dispatch with the concrete matrix `Ain`. The input is the callnack with signature `f(y,x) := y = A*x` and `n` is the matrix dimension of the linear operator.
"""
function chebyshev_poly_real_symm_sparse(matvec!::F,n::Int;σ::Float64=0.0,emin::Float64,emax::Float64,smin::Float64,smax::Float64,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step) where {F}
    Ain=MatrixFreeRealSymOp(n,matvec!)
    n=size(Ain,1)
    m0=min(m0,n)
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    a=(emin-c)/d
    b=(emax-c)/d
    σ̂=(σ-c)/d
    -1.0<=a<b<=1.0 || throw(ArgumentError("mapped interval [$a,$b] not inside [-1,1]"))
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree;jackson=jackson)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknow window type, possible are :step and :delta")
    end
    Random.seed!(seed)
    Q=randn(n,m0)
    Y=Matrix{Float64}(undef,n,m0)
    U0=Matrix{Float64}(undef,n,m0)
    U1=Matrix{Float64}(undef,n,m0)
    U2=Matrix{Float64}(undef,n,m0)
    AXf=Matrix{Float64}(undef,n,m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    prev_ok=false
    for it in 1:maxiter
        apply_cheb_filter!(Y,Ain,Q,coeffs,smin,smax,U0,U1,U2,AXf)
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        apply_A_block!(R,Ain,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,Ain,X,λ,AX)
        fill!(inside, false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx]
    p=sortperm(λsel)
    idx=idx[p]
    λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

#################################################
############## ZOLOTAREV FUNCTIONS ##############
#################################################

"""
    general_feast_sparse_interval_zolotarev(Ain;emin, emax,m0=200,maxiter=8,tol=1e-10,res_gate=1e-6,debug=false,is_hermitian=true,show_progress=false,backend=:lu,ls_alg=nothing,two_hit=true)

Compute eigenpairs of a sparse matrix `A` with eigenvalues in `[emin, emax]`
using FEAST subspace iteration with a Zolotarev rational filter.

----------------------------------------------------------------------
Algorithm
----------------------------------------------------------------------

The spectral projector onto eigenvalues inside `[emin,emax]` is

    P = (1/(2π i)) ∮_Γ (zI - A)⁻¹ dz.

Instead of contour quadrature (ellipse rule), this routine uses a rational
filter

    P ≈ α₀ I + Σ w_k (z_k I - A)⁻¹

where `(z_k, w_k)` are the poles and residues of the Zolotarev approximation.

FEAST iteration:

    Y = α₀ Q + Σ w_k (z_k I - A)⁻¹ Q
    Q ← orth(Y)
    Aq = Qᴴ A Q
    solve Aq u = λ u
    x = Q u
    compute residuals

until convergence.

----------------------------------------------------------------------
Parameters
----------------------------------------------------------------------

`Ain`
    Input sparse matrix (real or complex).

`zp`
    Zolotarev approximation parameters. See `zolotarev_nodes_weights` for details.

`emin, emax`
    Target eigenvalue interval.

`m0`
    Dimension of the FEAST search subspace.

    Must exceed the number of eigenvalues in `[emin,emax]`.

`maxiter`
    Maximum FEAST iterations.

`tol`
    Convergence tolerance based on residual norms.

`res_gate`
    Residual threshold used when selecting eigenpairs inside the interval.

`debug`
    If `true`, prints iteration diagnostics.

`show_progress`
    If `true`, displays progress bars.

`is_hermitian`
    If `true`, uses Hermitian eigensolver for the projected problem.

`backend`
    Linear solver for shifted systems.

    Options:

        :lu   — direct LU factorization
        :ls   — LinearSolve.jl iterative/direct solver

`ls_alg`
    Optional LinearSolve algorithm.

    Examples:

        LinearSolve.KLUFactorization()
        LinearSolve.UMFPACKFactorization()

`two_hit`
    If `true`, require two consecutive iterations with `maxres < tol`
    before declaring convergence.

----------------------------------------------------------------------
Returns
----------------------------------------------------------------------

`λsel::Vector{Float64}`
    Eigenvalues in `[emin,emax]` (sorted).

`Xsel::Matrix{ComplexF64}`
    Corresponding eigenvectors.

`ressel::Vector{Float64}`
    Residual norms.
"""
function general_feast_sparse_interval_zolotarev(Ain::SparseMatrixCSC{Float64,Int},zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    A=ensure_diagonal(Ain)
    z,w,alpha0=zolotarev_nodes_weights(zp,emin,emax)
    np=length(z)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    Rm=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)
    Ffact=backend===:lu ? Vector{Any}(undef,np) : nothing
    lins=backend===:ls ? Vector{Any}(undef,np) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing
    p_fact=show_progress ? Progress(np;desc="FEAST(zolo): factorizations") : nothing
    @inbounds for k in 1:np
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,z[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t]
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    vY=vec(Y)
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(np;desc="FEAST(zolo): iter $(it)") : nothing
        #thin_orth!(Qc,Qtmp,Z)
        copyto!(Y,Qc)
        rmul!(Y,alpha0)
        @inbounds for k in 1:np
            if backend===:lu
                ldiv!(Tm,Ffact[k],Qc)
                LinearAlgebra.BLAS.axpy!(w[k],vec(Tm),vY)
            else
                lins[k].b=Qc
                sol=LinearSolve.solve!(lins[k])
                LinearAlgebra.BLAS.axpy!(w[k],vec(sol.u),vY)
            end
            show_progress && next!(p_it)
        end
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z)
        mul!(Rm,A,Qc)
        mul!(Aq,adjoint(Qc),Rm)
        if is_hermitian
            E=eigen!(Hermitian(Aq))
            λ.=ComplexF64.(E.values)
            Xq.=E.vectors
        else
            E=eigen(Aq)
            λ.=E.values
            Xq.=E.vectors
        end
        mul!(X,Qc,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("inside=%d maxres=%.6e\n",count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    general_feast_sparse_interval_zolotarev_distributed(Ain; ...)

Low-memory FEAST eigensolver using a Zolotarev rational filter. (See `general_feast_sparse_interval_zolotarev` for details.)

This version avoids storing factorizations for every pole. Instead, each
shifted system `(z_k I - A)` is assembled and solved on the fly.

----------------------------------------------------------------------
Additional parameter
----------------------------------------------------------------------

`blockrhs`

Number of right-hand sides solved simultaneously.

Examples:

    blockrhs = m0      # solve full block
    blockrhs = 32      # process vectors in blocks

----------------------------------------------------------------------
Returns
----------------------------------------------------------------------

Same as `general_feast_sparse_interval_zolotarev`.
"""
function general_feast_sparse_interval_zolotarev_distributed(Ain::SparseMatrixCSC{Float64,Int},zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=ensure_diagonal(Ain)
    z,w,alpha0=zolotarev_nodes_weights(zp,emin,emax)
    np=length(z)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    Rm=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)
    vY=vec(Y)
    prev_ok=false
    for it in 1:maxiter
        #thin_orth!(Qc,Qtmp,Z)
        copyto!(Y,Qc)
        rmul!(Y,alpha0)
        progress_k=show_progress ? Progress(np;desc="FEAST(zolo,dist): iter $(it)") : nothing
        @inbounds for k in 1:np
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,z[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t]
            end
            wk=w[k]
            if backend===:lu
                F=lu(Az)
                if blockrhs==m0
                    ldiv!(Tm,F,Qc)
                    LinearAlgebra.BLAS.axpy!(wk,vec(Tm),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        ldiv!(Ublk,F,Bblk)
                        @views Y[:,j0:j0+bs-1].+=wk.*Ublk[:,1:bs]
                        j0+=bs
                    end
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                if blockrhs==m0
                    prob=LinearSolve.LinearProblem(Ak,Qc;u0=Tm)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    LinearAlgebra.BLAS.axpy!(wk,vec(sol.u),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                        lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                        sol=LinearSolve.solve!(lin)
                        @views Y[:,j0:j0+bs-1].+=wk.*sol.u[:,1:bs]
                        j0+=bs
                    end
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z)
        mul!(Rm,A,Qc)
        mul!(Aq,adjoint(Qc),Rm)
        if is_hermitian
            E=eigen!(Hermitian(Aq))
            λ.=ComplexF64.(E.values)
            Xq.=E.vectors
        else
            E=eigen(Aq)
            λ.=E.values
            Xq.=E.vectors
        end
        mul!(X,Qc,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half_zolotarev(Ain; ...)

FEAST eigensolver for real symmetric sparse matrices using a
half-contour Zolotarev rational filter. (See `general_feast_sparse_interval_zolotarev` for details.)

----------------------------------------------------------------------
Half-contour idea
----------------------------------------------------------------------

For real symmetric matrices:

    w (zI - A)⁻¹ + w̄ (z̄I - A)⁻¹ = 2 Re[w (zI - A)⁻¹]

Therefore the rational filter can be evaluated using only poles in the
upper half-plane, reducing the number of linear solves by ~2×.

The projector becomes

    Y = α₀ Q + 2 Re Σ w_k (z_k I - A)⁻¹ Q

----------------------------------------------------------------------
Parameters
----------------------------------------------------------------------

Same as `general_feast_sparse_interval_zolotarev`.
`nodes` still denotes the **total number of poles of the full filter.
Only `nodes/2` solves are performed.

----------------------------------------------------------------------
Returns
----------------------------------------------------------------------

`λsel::Vector{Float64}`
    Eigenvalues in `[emin,emax]`.

`Xsel::Matrix{Float64}`
    Eigenvectors.

`ressel::Vector{Float64}`
    Residual norms.
"""
function sym_feast_sparse_interval_half_zolotarev(Ain::SparseMatrixCSC{Float64,Int},zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)
    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    A=ensure_diagonal(Ain)
    zh,wh,alpha0=zolotarev_nodes_weights_half(zp,emin,emax)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    Bc=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    vTm=vec(Tm)
    Rm=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)

    Ffact=backend===:lu ? Vector{Any}(undef,nh) : nothing
    lins=backend===:ls ? Vector{Any}(undef,nh) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing

    p_fact=show_progress ? Progress(nh;desc="FEAST(sym,zolo): factorizations") : nothing
    @inbounds for k in 1:nh
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,zh[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t]
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nh;desc="FEAST(sym,zolo): iter $(it)") : nothing
        #thin_orth!(Q,Qtmp,Z)
        αr=real(alpha0)
        @inbounds for j in 1:m0, i in 1:n
            Y[i,j]=αr*Q[i,j]
            Bc[i,j]=Q[i,j]
        end
        @inbounds for k in 1:nh
            if backend===:lu
                ldiv!(Tm,Ffact[k],Bc)
            else
                lins[k].b=Bc
                sol=LinearSolve.solve!(lins[k])
                copyto!(Tm,sol.u)
            end
            wk=wh[k]
            @inbounds for idx in eachindex(vTm)
                Y[idx]+=2.0*real(wk*vTm[idx])
            end
            show_progress && next!(p_it)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(Rm,A,Q)
        mul!(Aq,transpose(Q),Rm)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half_zolotarev_distributed(Ain; ...)

Low-memory symmetric FEAST solver using a **half-contour Zolotarev filter**.

This variant recomputes shifted solves at every iteration instead of storing
all factorizations. (See `sym_feast_sparse_interval_half_zolotarev` for details.)

----------------------------------------------------------------------
Returns
----------------------------------------------------------------------

Same as `sym_feast_sparse_interval_half_zolotarev`.
"""
function sym_feast_sparse_interval_half_zolotarev_distributed(Ain::SparseMatrixCSC{Float64,Int},zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)
    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=ensure_diagonal(Ain)
    zh,wh,alpha0=zolotarev_nodes_weights_half(zp,emin,emax)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    Rm=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)
    prev_ok=false
    αr=real(alpha0)
    for it in 1:maxiter
        #thin_orth!(Q,Qtmp,Z)
        @inbounds for j in 1:m0, i in 1:n
            Y[i,j]=αr*Q[i,j]
        end
        progress_k=show_progress ? Progress(nh;desc="FEAST(sym,zolo,dist): iter $(it)") : nothing
        @inbounds for k in 1:nh
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,zh[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t]
            end
            wk=wh[k]
            if backend===:lu
                F=lu(Az)
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @views Bblk[:,1:bs].=Q[:,j0:j0+bs-1]
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    ldiv!(Ublk,F,Bblk)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*Ublk[i,jj])
                        end
                    end
                    j0+=bs
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Bblk[i,jj]=Q[i,col]
                        end
                    end
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*sol.u[i,jj])
                        end
                    end
                    j0+=bs
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(Rm,A,Q)
        mul!(Aq,transpose(Q),Rm)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half_zolotarev(matvec!::F,n::Int,zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,show_progress::Bool=false,two_hit::Bool=true,maxiter_solve=5000) where {F}

FEAST eigensolver for real symmetric operators using a half-contour Zolotarev rational filter, in a matrix-free setting. The operator is defined by a user-provided `matvec!` function that computes `y <- A*x` for a real symmetric operator `A`. (See `sym_feast_sparse_interval_half_zolotarev` with materialized matrix `Ain` for details.)

# ADDITIONAL PARAMETERS
- `matvec!::F`: A user-provided function that computes the matrix-vector product `y <- A*x` for a real symmetric operator `A`. The function should have the signature `matvec!(y::Vector{Float64}, x::Vector{Float64})` and should overwrite `y` with the result of `A*x`.
- `n::Int`: The dimension of the operator `A`.
- `maxiter_solve::Int`: Maximum number of iterations to be used for the block GMRES for solving `(z * Id - A)^{-1} X = Q` at each iteration where `X` and `Q` are matrices.

# EXAMPLE

# ------------------------------
# 1D Dirichlet Laplacian
# A = tridiagonal(-1,2,-1)
# y_i = 2x_i - x_{i-1} - x_{i+1}
# ------------------------------
function lap1d_matvec!(y,x)
    n=length(x)
    @inbounds begin
        y[1]=2*x[1]-x[2]
        for i in 2:n-1
            y[i]=2*x[i]-x[i-1]-x[i+1]
        end
        y[n]=2*x[n]-x[n-1]
    end
end
"""
function sym_feast_sparse_interval_half_zolotarev(matvec!::F,n::Int,zp::ZolotarevParams;emin::Float64,emax::Float64,m0::Int=200,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,show_progress::Bool=false,two_hit::Bool=true,maxiter_solve=5000,ls_alg::Symbol=:gmres) where {F}
    A=MatrixFreeRealSymOp(n,matvec!)
    zh,wh,alpha0=zolotarev_nodes_weights_half(zp,emin,emax)
    nh=length(zh)
    Sops=[ShiftedMatrixFreeOp(n,zh[k],matvec!) for k in 1:nh]
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    Bc=Matrix{ComplexF64}(undef,n,m0)
    Tm=zeros(ComplexF64,n,m0) 
    vTm=vec(Tm)
    Rm=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nh;desc="iter $(it)") : nothing
        αr=real(alpha0)
        @inbounds for j in 1:m0, i in 1:n
            Y[i,j]=αr*Q[i,j]
            Bc[i,j]=Q[i,j]+0.0im
        end
        @inbounds for k in 1:nh
            solve_shifted_block_matrixfree!(Tm,Sops[k],Bc,tol=tol,maxiter=maxiter_solve,ls_alg=ls_alg)
            wk=wh[k]
            @inbounds for idx in eachindex(vTm)
                Y[idx]+=2.0*real(wk*vTm[idx])
            end
            show_progress && next!(p_it)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        apply_A_block!(Rm,A,Q)
        mul!(Aq,transpose(Q),Rm)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)

        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx]
    p=sortperm(λsel)
    idx=idx[p]
    λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

##############################################
################### POLFED ###################
##############################################

"""
    clenshaw_cheb_apply_vec!(y,A,x,coeffs,smin,smax,b1,b2,tmp,ax)

Apply a Chebyshev polynomial of the scaled operator to a single vector using
the Clenshaw recurrence.

Given coefficients `coeffs = [c₀,c₁,...,c_K]`, this computes
    y = p(Â)x,
where
    p(t) = Σ_{k=0}^K c_k T_k(t),
and
    Â = (A-cI)/d,
with `c=(smin+smax)/2` and `d=(smax-smin)/2`.

This routine is the core filtered-operator application used in POLFED.

Arguments
- `y::AbstractVector{Float64}`: output vector
- `A`: linear operator compatible with `apply_A_vec!`
- `x::AbstractVector{Float64}`: input vector
- `coeffs::AbstractVector{Float64}`: Chebyshev expansion coefficients
- `smin::Float64`, `smax::Float64`: spectral bounds used for scaling
- `b1`, `b2`, `tmp`, `ax`: work vectors of the same length as `x`

Returns
- `y`, overwritten in place.
"""
function clenshaw_cheb_apply_vec!(y::AbstractVector{Float64},A::SparseMatrixCSC{Float64,Int},x::AbstractVector{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,b1::AbstractVector{Float64},b2::AbstractVector{Float64},tmp::AbstractVector{Float64},ax::AbstractVector{Float64})
    K=length(coeffs)-1;c=0.5*(smin+smax);d=0.5*(smax-smin)
    fill!(b1,0.0);fill!(b2,0.0)
    @inbounds for k=(K+1):-1:2
    scaled_apply_vec!(tmp,A,b1,c,d,ax);ck=coeffs[k]
    @simd for i in eachindex(x)
        tmp[i]=ck*x[i]+2*tmp[i]-b2[i]
    end
    copyto!(b2,b1)
    copyto!(b1,tmp)
    end
    scaled_apply_vec!(tmp,A,b1,c,d,ax);c0=coeffs[1]
    @inbounds @simd for i in eachindex(x)
        y[i]=c0*x[i]+tmp[i]-b2[i]
    end
end

"""
    clenshaw_cheb_apply_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,b1::AbstractVector{Float64},b2::AbstractVector{Float64},tmp::AbstractVector{Float64},ax::AbstractVector{Float64})

Apply a Chebyshev polynomial of the scaled matrix-free operator to a
single vector using the Clenshaw recurrence.

Given coefficients `coeffs = [c₀,c₁,...,c_K]`, this computes

    y = p(Â)x,

where

    p(t) = Σ_{k=0}^K c_k T_k(t),
    Â = (A-cI)/d,

with `c=(smin+smax)/2` and `d=(smax-smin)/2`.

This is the core single-vector filtered operator application used by
POLFED in the matrix-free setting.

Arguments
- `y::AbstractVector{Float64}`: output vector
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `x::AbstractVector{Float64}`: input vector
- `coeffs::AbstractVector{Float64}`: Chebyshev expansion coefficients
- `smin::Float64`, `smax::Float64`: spectral bounds used for scaling
- `b1`, `b2`, `tmp`, `ax`: work vectors of the same length as `x`

Returns
- `y`, overwritten in place.
"""
function clenshaw_cheb_apply_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,b1::AbstractVector{Float64},b2::AbstractVector{Float64},tmp::AbstractVector{Float64},ax::AbstractVector{Float64})
    K=length(coeffs)-1
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    fill!(b1,0.0)
    fill!(b2,0.0)
    @inbounds for k=(K+1):-1:2
        scaled_apply_vec!(tmp,A,b1,c,d,ax)
        ck=coeffs[k]
        @simd for i in eachindex(x)
            tmp[i]=ck*x[i]+2*tmp[i]-b2[i]
        end
        copyto!(b2,b1)
        copyto!(b1,tmp)
    end
    scaled_apply_vec!(tmp,A,b1,c,d,ax)
    c0=coeffs[1]
    @inbounds @simd for i in eachindex(x)
        y[i]=c0*x[i]+tmp[i]-b2[i]
    end
end

"""
    POLFEDWorkspace

Workspace container for block polynomial application in POLFED.

Fields
- `b1`, `b2`: block Clenshaw recurrence states
- `tmp`: temporary block used during recurrence updates
- `ax`: workspace for block operator application

All matrices have size `(n,b)`, where `n` is the problem dimension and `b`
is the block size.
"""
struct POLFEDWorkspace
    b1::Matrix{Float64}
    b2::Matrix{Float64}
    tmp::Matrix{Float64}
    ax::Matrix{Float64}
end

POLFEDWorkspace(n,b)=POLFEDWorkspace(zeros(n,b),zeros(n,b),zeros(n,b),zeros(n,b))

"""
    apply_poly_block!(Y,A,Q,coeffs,smin,smax,work;threaded=true)

Apply the Chebyshev polynomial filter blockwise:
    Y[:,j] = p(Â)Q[:,j],   j = 1,...,b.

Each column is treated independently via `clenshaw_cheb_apply_vec!`. When
`threaded=true`, columns are distributed across Julia threads.

Arguments
- `Y::Matrix{Float64}`: output block
- `A`: linear operator compatible with `apply_A_vec!`
- `Q::Matrix{Float64}`: input block
- `coeffs::Vector{Float64}`: Chebyshev coefficients
- `smin::Float64`, `smax::Float64`: spectral bounds used for scaling
- `work::POLFEDWorkspace`: block work arrays
- `threaded::Bool=true`: whether to thread over block columns

Returns
- `Y`, overwritten in place.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::POLFEDWorkspace;threaded::Bool=true)
    n,b=size(Q)
    if threaded && Threads.nthreads()>1
        Threads.@threads for j in 1:b
            clenshaw_cheb_apply_vec!(view(Y,:,j),A,view(Q,:,j),coeffs,smin,smax,view(work.b1,:,j),view(work.b2,:,j),view(work.tmp,:,j),view(work.ax,:,j))
        end
    else
        @inbounds for j in 1:b
            clenshaw_cheb_apply_vec!(view(Y,:,j),A,view(Q,:,j),coeffs,smin,smax,view(work.b1,:,j),view(work.b2,:,j),view(work.tmp,:,j),view(work.ax,:,j))
        end
    end
end

"""
    apply_poly_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::POLFEDWorkspace;threaded::Bool=true)

Apply the Chebyshev polynomial filter blockwise for a matrix-free operator:

    Y[:,j] = p(Â) Q[:,j],   j = 1,...,b.

Each column is processed independently using
`clenshaw_cheb_apply_vec!`. When `threaded=true`, the block columns
are distributed across Julia threads.

Arguments
- `Y::AbstractMatrix{Float64}`: output block
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `Q::AbstractMatrix{Float64}`: input block
- `coeffs::AbstractVector{Float64}`: Chebyshev coefficients
- `smin::Float64`, `smax::Float64`: spectral bounds used for scaling
- `work::POLFEDWorkspace`: preallocated block workspaces
- `threaded::Bool=true`: whether to thread over block columns

Returns
- `Y`, overwritten in place.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::POLFEDWorkspace;threaded::Bool=true)
    n,b=size(Q)
    if threaded && Threads.nthreads()>1
        Threads.@threads for j in 1:b
            clenshaw_cheb_apply_vec!(view(Y,:,j),A,view(Q,:,j),coeffs,smin,smax,view(work.b1,:,j),view(work.b2,:,j),view(work.tmp,:,j),view(work.ax,:,j))
        end
    else
        @inbounds for j in 1:b
            clenshaw_cheb_apply_vec!(view(Y,:,j),A,view(Q,:,j),coeffs,smin,smax,view(work.b1,:,j),view(work.b2,:,j),view(work.tmp,:,j),view(work.ax,:,j))
        end
    end
end

"""
    orth_block_polfed!(Q;rtol=1e-12)

Orthonormalize the columns of `Q` and drop numerically rank-deficient
directions.

A thin QR factorization is computed, and only columns corresponding to
diagonal entries of `R` larger than `rtol*maximum(abs.(diag(R)))` are kept.

Arguments
- `Q::AbstractMatrix{Float64}`: input block
- `rtol::Float64=1e-12`: relative rank threshold

Returns
- `Matrix{Float64}` with orthonormal columns spanning the numerically
  independent subspace of the original columns.
"""
function orth_block_polfed!(Q::AbstractMatrix{Float64};rtol=1e-12)
    #=
    F=qr(Q)
    R=Matrix(F.R)
    k=min(size(R,1),size(R,2))
    k==0 && return Matrix{Float64}(undef,size(Q,1),0)
    dr=abs.(diag(R))
    r0=maximum(dr)
    r0==0 && return Matrix{Float64}(undef,size(Q,1),0)
    r=count(>(rtol*r0),dr)
    r==0 && return Matrix{Float64}(undef,size(Q,1),0)
    return Matrix(F.Q[:,1:r])
    =#
    
    n,s=size(Q)
    F=qr!(Q)
    R=F.R
    dmax=0.0
    @inbounds for i in 1:s
        v=abs(R[i,i])
        v>dmax && (dmax=v)
    end
    r=s
    if dmax>0
        thresh=rtol*dmax
        r=0
        @inbounds for i in 1:s
            abs(R[i,i])>thresh && (r+=1)
        end
    end
    Q1=Matrix(F.Q[:,1:max(r,1)])
    if r<s
        Qfill=randn(n,s-r)
        Qtmp=hcat(Q1,Qfill)
        return Matrix(qr!(Qtmp).Q[:,1:s])
    else
        return Q1
    end
    
end

"""
    reorth!(W,Qhist)

Reorthogonalize the block `W` against a history of previously computed basis
blocks.

For each `Q` in `Qhist`, this performs the projection
    W <- W - Q(Q'W),
so that the updated `W` is orthogonal to the span of all prior blocks.

Arguments
- `W::AbstractMatrix{Float64}`: block to be projected
- `Qhist::AbstractVector{<:AbstractMatrix{Float64}}`: previously stored
  orthonormal basis blocks

Returns
- `W`, modified in place.
"""
function reorth!(W::AbstractMatrix{Float64},Qhist::AbstractVector{<:AbstractMatrix{Float64}})
    for Q in Qhist
        W.-=Q*(transpose(Q)*W)
    end
end

"""
    assemble_T(Ablocks,Bblocks)

Assemble the symmetric reduced block tridiagonal matrix from diagonal and
off-diagonal block coefficients.

If `Ablocks[j]` is the `j`-th diagonal block and `Bblocks[j]` couples block
`j` to block `j+1`, then this constructs the full reduced matrix `T`.

Arguments
- `Ablocks::Vector{Matrix{Float64}}`: diagonal block matrices
- `Bblocks::Vector{Matrix{Float64}}`: upper off-diagonal coupling blocks

Returns
- `Symmetric{Float64,Matrix{Float64}}`: dense reduced block tridiagonal matrix.
"""
function assemble_T(Ablocks,Bblocks)
    nb=length(Ablocks)
    dims=[size(A,1) for A in Ablocks]
    N=sum(dims);T=zeros(Float64,N,N)
    o=1
    for j in 1:nb
        r=dims[j]
        I=o:o+r-1
        T[I,I].=Ablocks[j]
        if j<nb
            B=Bblocks[j]
            J=(o+r):(o+r+size(B,2)-1)
            T[I,J].=B
            T[J,I].=transpose(B)
        end
        o+=r
    end
    return Symmetric(T)
end

"""
    stack_LEGACY(Qhist)

Horizontally concatenate all basis blocks stored in `Qhist`.

This is used to build the full basis matrix corresponding to the block
Lanczos history.

Arguments
- `Qhist::Vector{Matrix{Float64}}`: basis blocks

Returns
- `Matrix{Float64}` equal to `hcat(Qhist...)`.
"""
function stack_LEGACY(Qhist)
    hcat(Qhist...)
end

function build_X(Qhist::Vector{Matrix{Float64}},U::Matrix{Float64})
    n=size(Qhist[1],1)
    m=size(U,2)
    X=zeros(n,m)
    o=1
    for Q in Qhist
        r=size(Q,2)
        X.+= Q*view(U,o:o+r-1,:)
        o+=r
    end
    return X
end

"""
    rayleigh!(λ,res,AX,A,X)

Compute Rayleigh quotients and residual norms for a block of candidate vectors.

For each column `x_j = X[:,j]`, this computes
    λ_j = x_j' A x_j
and
    res_j = ||A x_j - λ_j x_j||₂.

Arguments
- `λ::Vector{Float64}`: output Rayleigh quotients
- `res::Vector{Float64}`: output residual norms
- `AX::Matrix{Float64}`: workspace, overwritten with `A*X`
- `A`: linear operator compatible with `apply_A_block!`
- `X::Matrix{Float64}`: candidate vectors stored columnwise

Returns
- `λ` and `res`, overwritten in place.
"""
function rayleigh!(λ::Vector{Float64},res::Vector{Float64},AX::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},X::AbstractMatrix{Float64})
    apply_A_block!(AX,A,X)
    k=size(X,2)
    @inbounds for j in 1:k
        x=view(X,:,j)
        ax=view(AX,:,j)
        λj=dot(x,ax)
        λ[j]=λj
        s=0.0
        @simd for i in eachindex(x)
            r=ax[i]-λj*x[i];s+=r*r
        end
        res[j]=sqrt(s)
    end
end

"""
    rayleigh!(λ::Vector{Float64},res::Vector{Float64},AX::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})

Compute Rayleigh quotients and residual norms for a block of candidate
vectors with respect to a matrix-free real symmetric operator.

For each column `x_j = X[:,j]`, this computes

    λ_j = x_j' A x_j
    res_j = ||A x_j - λ_j x_j||₂.

The block product `A*X` is evaluated through `apply_A_block!`.

Arguments
- `λ::Vector{Float64}`: output Rayleigh quotients
- `res::Vector{Float64}`: output residual norms
- `AX::AbstractMatrix{Float64}`: workspace overwritten with `A*X`
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `X::AbstractMatrix{Float64}`: candidate vectors stored columnwise

Returns
- `λ` and `res`, overwritten in place.
"""
function rayleigh!(λ::Vector{Float64},res::Vector{Float64},AX::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    apply_A_block!(AX,A,X)
    k=size(X,2)
    @inbounds for j in 1:k
        x=view(X,:,j)
        ax=view(AX,:,j)
        λj=dot(x,ax)
        λ[j]=λj
        s=0.0
        @simd for i in eachindex(x)
            r=ax[i]-λj*x[i]
            s+=r*r
        end
        res[j]=sqrt(s)
    end
end

"""
    refine_original_rr(A,X)

Perform a local Rayleigh–Ritz refinement in the subspace spanned by the
columns of `X` for the original operator `A`.

The input block is first orthonormalized. Then the projected dense matrix
    H = Q' A Q
is diagonalized, the Ritz vectors are backprojected to the original space,
and final Rayleigh quotients / residuals are computed.

Arguments
- `A`: linear operator compatible with `apply_A_block!`
- `X::Matrix{Float64}`: candidate subspace basis

Returns
- `λ::Vector{Float64}`: refined Ritz values sorted increasingly
- `Y::Matrix{Float64}`: refined Ritz vectors as columns
- `res::Vector{Float64}`: corresponding residual norms
"""
function refine_original_rr(A,X)
    Q=orth_block_polfed!(copy(X))
    k=size(Q,2)
    k==0 && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
    AQ=zeros(size(Q))
    apply_A_block!(AQ,A,Q)
    H=Symmetric(0.5*(transpose(Q)*AQ+transpose(AQ)*Q))
    E=eigen(H)
    Y=Q*E.vectors
    AY=zeros(size(Y))
    λ=zeros(k);res=zeros(k)
    rayleigh!(λ,res,AY,A,Y)
    ps=sortperm(λ)
    λ[ps],Y[:,ps],res[ps]
end

"""
    refine_original_rr(A::MatrixFreeRealSymOp,X::Matrix{Float64})

Perform a local Rayleigh–Ritz refinement in the subspace spanned by the
columns of `X` for a matrix-free real symmetric operator `A`.

The input block is first orthonormalized. Then the projected dense matrix

    H = Q' A Q

is formed and diagonalized. The Ritz vectors are backprojected to the
original space, and final Rayleigh quotients / residuals are computed.

Arguments
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `X::Matrix{Float64}`: candidate subspace basis

Returns
- `λ::Vector{Float64}`: refined Ritz values sorted increasingly
- `Y::Matrix{Float64}`: refined Ritz vectors as columns
- `res::Vector{Float64}`: corresponding residual norms
"""
function refine_original_rr(A::MatrixFreeRealSymOp,X::Matrix{Float64})
    Q=orth_block_polfed!(copy(X))
    k=size(Q,2)
    k==0 && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
    AQ=zeros(size(Q))
    apply_A_block!(AQ,A,Q)
    H=Symmetric(0.5*(transpose(Q)*AQ+transpose(AQ)*Q))
    E=eigen(H)
    Y=Q*E.vectors
    AY=zeros(size(Y))
    λ=zeros(k)
    res=zeros(k)
    rayleigh!(λ,res,AY,A,Y)
    ps=sortperm(λ)
    return λ[ps],Y[:,ps],res[ps]
end

"""
    extract_pairs(Qhist,T,A;nev,emin,emax,overextract=8,refine=true)

Extract approximate eigenpairs in the original problem from the reduced
block-Lanczos problem.

Arguments
- `Qhist::Vector{Matrix{Float64}}`: history of basis blocks
- `T::Symmetric{Float64,<:AbstractMatrix}`: reduced transformed matrix
- `A`: original operator
- `nev::Int`: number of desired eigenpairs
- `emin::Float64`, `emax::Float64`: target spectral window in the original problem
- `overextract::Int=8`: multiplier controlling how many transformed Ritz
  vectors are examined before interval filtering
- `refine::Bool=true`: whether to perform local original-space RR refinement

Returns
- `λ::Vector{Float64}`: selected Ritz values
- `X::Matrix{Float64}`: selected Ritz vectors
- `res::Vector{Float64}`: corresponding residual norms
"""
function extract_pairs(Qhist,T,A;nev,emin,emax,overextract=8,refine=true)
    E=eigen(T)
    m=min(length(E.values),max(overextract*nev,nev))
    p=sortperm(E.values;rev=true)[1:m]
    U=E.vectors[:,p]
    X=build_X(Qhist,U)
    k=size(X,2)
    λ=zeros(k); res=zeros(k); AX=zeros(size(X))
    rayleigh!(λ,res,AX,A,X)
    inside=findall(i->emin<=λ[i]<=emax,eachindex(λ))
    isempty(inside) && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
    Xc=X[:,inside]
    if refine
        λc,Xc,resc=refine_original_rr(A,Xc)
        inside2=findall(i->emin<=λc[i]<=emax,eachindex(λc))
        isempty(inside2) && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
        λc=λc[inside2]; Xc=Xc[:,inside2]; resc=resc[inside2]
        ps=sortperm(resc)
        length(ps)>nev && (ps=ps[1:nev])
        λf=λc[ps]; Xf=Xc[:,ps]; rf=resc[ps]
        pe=sortperm(λf)
        return λf[pe],Xf[:,pe],rf[pe]

    else
        λc=λ[inside]
        resc=res[inside]
        ps=sortperm(resc)
        length(ps)>nev && (ps=ps[1:nev])
        λf=λc[ps]
        Xf=Xc[:,ps]
        rf=resc[ps]
        pe=sortperm(λf)
        return λf[pe],Xf[:,pe],rf[pe]
    end
end

"""
    extract_pairs(Qhist,T,A::MatrixFreeRealSymOp;nev,emin,emax,overextract=8,refine=true)

Extract approximate eigenpairs of the original matrix-free problem from
the reduced block-Lanczos problem.

Arguments
- `Qhist::Vector{Matrix{Float64}}`: history of basis blocks
- `T::Symmetric{Float64,<:AbstractMatrix}`: reduced transformed matrix
- `A::MatrixFreeRealSymOp`: original matrix-free operator
- `nev::Int`: number of desired eigenpairs
- `emin::Float64`, `emax::Float64`: target spectral window
- `overextract::Int=8`: multiplier controlling how many transformed Ritz
  vectors are examined before interval filtering
- `refine::Bool=true`: whether to perform local original-space RR refinement

Returns
- `λ::Vector{Float64}`: selected Ritz values
- `X::Matrix{Float64}`: selected Ritz vectors
- `res::Vector{Float64}`: corresponding residual norms
"""
function extract_pairs(Qhist,T,A::MatrixFreeRealSymOp;nev,emin,emax,overextract=8,refine=true)
    E=eigen(T)
    m=min(length(E.values),max(overextract*nev,nev))
    p=sortperm(E.values;rev=true)[1:m]
    U=E.vectors[:,p]
    X=build_X(Qhist,U)
    k=size(X,2)
    λ=zeros(k); res=zeros(k); AX=zeros(size(X))
    rayleigh!(λ,res,AX,A,X)
    inside=findall(i->emin<=λ[i]<=emax,eachindex(λ))
    isempty(inside) && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
    Xc=X[:,inside]
    if refine
        λc,Xc,resc=refine_original_rr(A,Xc)
        inside2=findall(i->emin<=λc[i]<=emax,eachindex(λc))
        isempty(inside2) && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
        λc=λc[inside2]
        Xc=Xc[:,inside2]
        resc=resc[inside2]
        ps=sortperm(resc)
        length(ps)>nev && (ps=ps[1:nev])
        λf=λc[ps]
        Xf=Xc[:,ps]
        rf=resc[ps]
        pe=sortperm(λf)
        return λf[pe],Xf[:,pe],rf[pe]
    else
        λc=λ[inside]
        resc=res[inside]
        ps=sortperm(resc)
        length(ps)>nev && (ps=ps[1:nev])
        λf=λc[ps]
        Xf=Xc[:,ps]
        rf=resc[ps]
        pe=sortperm(λf)
        return λf[pe],Xf[:,pe],rf[pe]
    end
end

function make_restart_block(X::Matrix{Float64},n::Int,s::Int;pad=8,seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    nkeep=min(size(X,2),max(1,s-pad))
    Q0=Matrix{Float64}(undef,n,s)
    Q0[:,1:nkeep].=X[:,1:nkeep]
    if nkeep<s
        Q0[:,nkeep+1:s].=randn(n,s-nkeep)
    end
    orth_block_polfed!(Q0)
end

"""
    polfed(A,nev,degree;s=8,smin=nothing,smax=nothing,sigma=0.0,
           emin=-Inf,emax=Inf,tol=1e-10,maxblocks=nothing,
           threaded=true,overextract=8,refine=true)

POLFED block Lanczos method on a polynomially transformed operator centered at `sigma` (if window_type=:delta, otherwise an indicator interval given by `smin/smax`), and extract eigenpairs of the original operator in the target interval `[emin,emax]`.

Arguments
- `A`: original sparse symmetric operator
- `nev::Int`: number of desired eigenpairs
- `degree::Int`: polynomial degree of the delta filter
- `s::Int=8`: block size of the Lanczos expansion
- `smin`, `smax`: optional global spectral bounds; if omitted they are
  estimated with Gershgorin bounds
- `sigma::Float64=0.0`: target energy around which the filter is centered
- `emin::Float64=-Inf`, `emax::Float64=Inf`: target interval in the original spectrum
- `tol::Float64=1e-10`: stopping tolerance on the maximum selected residual
- `maxblocks=nothing`: maximum number of block Lanczos steps; defaults to
  `ceil(Int,2.8*nev/s)`
- `threaded::Bool=true`: whether to thread the block polynomial apply
- `overextract::Int=8`: transformed-space overextraction factor
- `refine::Bool=true`: whether to perform local original-space refinement
- `window_type::Symbol=:step`: Window for chebyshev polynomial, :delta focuses on a narrow window while :step uses the indicator function in a given energy window and :smooth is the smoothed version of :step for usually faster convergence.

Returns
- `bestλ::Vector{Float64}`: best interval eigenvalue approximations found
- `bestX::Matrix{Float64}`: corresponding eigenvector approximations
- `bestres::Vector{Float64}`: corresponding residual norms
"""
#=
function polfed(A::SparseMatrixCSC{Float64,Int},nev::Int,degree::Int;s=8,smax=nothing,smin=nothing,sigma=0.0,emin=-Inf,emax=Inf,tol=1e-10,maxblocks=nothing,threaded=true,overextract=8,refine=true,window_type::Symbol=:step,jackson::Bool=true)
    n=size(A,1)
    if smin===nothing || smax===nothing
        smin,smax=gershgorin_bounds(A)
    end
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    σ̂=(sigma-c)/d
    #coeffs=cheb_delta_coeffs(σ̂,degree)
    a=(emin-c)/d
    b=(emax-c)/d
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknow window type, possible are :step and :delta")
    end
    maxblocks===nothing && (maxblocks=ceil(Int,2.8*nev/s))
    Random.seed!(1)
    Q=orth_block_polfed!(randn(n,s))
    work=POLFEDWorkspace(n,s)
    Qhist=Matrix{Float64}[]
    Ablocks=Matrix{Float64}[]
    Bblocks=Matrix{Float64}[]
    Qprev=zeros(n,0)
    Bprev=zeros(0,0)
    bestλ=Float64[]
    bestX=Matrix{Float64}(undef,n,0)
    bestres=Float64[]
    for it in 1:maxblocks
        push!(Qhist,copy(Q))
        r=size(Q,2)
        W=zeros(n,r)
        apply_poly_block!(W,A,Q,coeffs,smin,smax,work;threaded=threaded)
        if it>1
            W.-=Qprev*transpose(Bprev)
        end
        Aj=transpose(Q)*W
        push!(Ablocks,Matrix(Aj))
        W.-=Q*Aj
        reorth!(W,Qhist)
        Wraw=copy(W)
        Qnext=orth_block_polfed!(W)
        if it<maxblocks && size(Qnext,2)>0
            Bj=transpose(Qnext)*Wraw
            push!(Bblocks,transpose(Matrix(Bj)))
        end
        T=assemble_T(Ablocks,Bblocks)
        λ,X,res=extract_pairs(Qhist,T,A;nev=nev,emin=emin,emax=emax,overextract=overextract,refine=refine)
        if !isempty(res)
            if isempty(bestres) || length(λ)>length(bestλ) || (length(λ)==length(bestλ) && maximum(res)<maximum(bestres))
                bestλ=copy(λ);bestX=copy(X);bestres=copy(res)
            end
            mr=maximum(res)
            rs=sort(res)
            println("it=",it,"  inside=",length(λ),"  r90=",rs[clamp(round(Int,0.9*length(rs)),1,length(rs))],"  r95=",rs[clamp(round(Int,0.95*length(rs)),1,length(rs))],"  maxres=",rs[end])
            mr<tol && length(λ)>=nev && return λ,X,res
        else
            println("it=",it,"  inside=0  maxres=Inf")
        end
        size(Qnext,2)==0 && break
        Qprev=Q
        Q=Qnext
        Bprev=Bblocks[end]
    end
    return bestλ,bestX,bestres
end
=#
function polfed(A::SparseMatrixCSC{Float64,Int},nev::Int,degree::Int;s=8,smin=nothing,smax=nothing,sigma=0.0,emin=-Inf,emax=Inf,tol=1e-10,threaded=true,overextract=8,refine=true,window_type::Symbol=:step,jackson::Bool=true,mcycle::Int=6,maxcycles::Int=30)
    n=size(A,1)
    if smin===nothing || smax===nothing
        smin,smax=gershgorin_bounds(A)
    end
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    σ̂=(sigma-c)/d
    a=(emin-c)/d
    b=(emax-c)/d
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree;jackson=jackson)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        error("Unknown window type: $window_type")
    end
    Random.seed!(1)
    Q=orth_block_polfed!(randn(n,s))
    work=POLFEDWorkspace(n,s)
    bestλ=Float64[]
    bestX=Matrix{Float64}(undef,n,0)
    bestres=Float64[]
    for cyc in 1:maxcycles
        Qhist=Matrix{Float64}[]
        Ablocks=Matrix{Float64}[]
        Bblocks=Matrix{Float64}[]
        Qprev=zeros(n,0)
        Bprev=zeros(0,0)
        for it in 1:mcycle
            push!(Qhist,copy(Q))
            r=size(Q,2)
            W=zeros(n,r)
            apply_poly_block!(W,A,Q,coeffs,smin,smax,work;threaded=threaded)
            if it>1
                W.-=Qprev*transpose(Bprev)
            end
            Aj=transpose(Q)*W
            push!(Ablocks,Matrix(Aj))
            W.-=Q*Aj
            reorth!(W,Qhist)
            Wraw=copy(W)
            Qnext=orth_block_polfed!(W)
            if it<mcycle && size(Qnext,2)>0
                Bj=transpose(Qnext)*Wraw
                push!(Bblocks,transpose(Matrix(Bj)))
            end
            size(Qnext,2)==0 && break
            Qprev=Q
            Q=Qnext
            !isempty(Bblocks) && (Bprev=Bblocks[end])
        end
        T=assemble_T(Ablocks,Bblocks)
        λ,X,res=extract_pairs(Qhist,T,A;nev=nev,emin=emin,emax=emax,overextract=overextract,refine=refine)
        if !isempty(res)
            if isempty(bestres) || length(λ)>length(bestλ) || (length(λ)==length(bestλ) && maximum(res)<maximum(bestres))
                bestλ=copy(λ);bestX=copy(X);bestres=copy(res)
            end
            rs=sort(res)
            println("cyc=",cyc,"  inside=",length(λ),"  r90=",rs[clamp(round(Int,0.9*length(rs)),1,length(rs))],"  r95=",rs[clamp(round(Int,0.95*length(rs)),1,length(rs))],"  maxres=",rs[end])
            length(λ)>=nev && rs[min(nev,length(rs))]<tol && return λ,X,res
            Q=make_restart_block(X,n,s)
        else
            println("cyc=",cyc,"  inside=0  maxres=Inf")
            Q=orth_block_polfed!(randn(n,s))
        end
    end

    return bestλ,bestX,bestres
end

"""
    polfed(matvec!::F,n::Int,nev::Int,degree::Int;
           s=8,smin::Float64,smax::Float64,sigma::Float64=0.0,
           emin::Float64=-Inf,emax::Float64=Inf,tol::Float64=1e-10,
           maxblocks=nothing,threaded::Bool=true,overextract::Int=8,
           refine::Bool=true,seed::Int=1) where {F}

Compute eigenpairs of a real symmetric operator in a target interval using
a matrix-free POLFED-style block Lanczos method with a Chebyshev step/delta/smooth filter.

Arguments
- `matvec!::F`: user-provided matrix-vector product `y <- A*x`
- `n::Int`: dimension of the operator
- `nev::Int`: number of desired eigenpairs
- `degree::Int`: polynomial degree of the delta filter
- `s::Int=8`: block size of the Lanczos expansion
- `smin::Float64`, `smax::Float64`: global spectral bounds used for scaling
- `sigma::Float64=0.0`: target energy around which the filter is centered
- `emin::Float64=-Inf`, `emax::Float64=Inf`: target interval in the original spectrum
- `tol::Float64=1e-10`: stopping tolerance on the maximum selected residual
- `maxblocks=nothing`: maximum number of block Lanczos steps; defaults to
  `ceil(Int,2.8*nev/s)`
- `threaded::Bool=true`: whether to thread the block polynomial apply
- `overextract::Int=8`: transformed-space overextraction factor
- `refine::Bool=true`: whether to perform local original-space refinement
- `seed::Int=1`: random seed used for the initial block
- `window_type::Symbol=:step`: Window for chebyshev polynomial, :delta focuses on a narrow window while :step uses the indicator function in a given energy window and :smooth is the smoothed version of :step for usually faster convergence.

Returns
- `bestλ::Vector{Float64}`: best interval eigenvalue approximations found
- `bestX::Matrix{Float64}`: corresponding eigenvector approximations
- `bestres::Vector{Float64}`: corresponding residual norms
"""
function polfed(matvec!::F,n::Int,nev::Int,degree::Int;s=8,smin=nothing,smax=nothing,sigma::Float64=0.0,emin::Float64=-Inf,emax::Float64=Inf,tol::Float64=1e-10,maxblocks=nothing,threaded::Bool=true,overextract::Int=8,refine::Bool=true,seed::Int=1,window_type::Symbol=:step,jackson::Bool=true) where {F}
    A=MatrixFreeRealSymOp(n,matvec!)
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    σ̂=(sigma-c)/d
    if smin===nothing || smax===nothing
        error("Matrix-free mode requires spectral bounds smin and smax")
    end
    a=(emin-c)/d
    b=(emax-c)/d
    #coeffs=cheb_delta_coeffs(σ̂,degree)
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknow window type, possible are :step and :delta")
    end
    maxblocks===nothing && (maxblocks=ceil(Int,2.8*nev/s))
    Random.seed!(seed)
    Q=orth_block_polfed!(randn(n,s))
    work=POLFEDWorkspace(n,s)
    Qhist=Matrix{Float64}[]
    Ablocks=Matrix{Float64}[]
    Bblocks=Matrix{Float64}[]
    Qprev=zeros(n,0)
    Bprev=zeros(0,0)
    bestλ=Float64[]
    bestX=Matrix{Float64}(undef,n,0)
    bestres=Float64[]
    for it in 1:maxblocks
        push!(Qhist,copy(Q))
        r=size(Q,2)
        W=zeros(n,r)
        apply_poly_block!(W,A,Q,coeffs,smin,smax,work;threaded=threaded)
        if it>1
            W.-=Qprev*transpose(Bprev)
        end
        Aj=transpose(Q)*W
        push!(Ablocks,Matrix(Aj))
        W.-=Q*Aj
        reorth!(W,Qhist)
        Wraw=copy(W)
        Qnext=orth_block_polfed!(W)
        if it<maxblocks && size(Qnext,2)>0
            Bj=transpose(Qnext)*Wraw
            push!(Bblocks,transpose(Matrix(Bj)))
        end
        T=assemble_T(Ablocks,Bblocks)
        λ,X,res=extract_pairs(Qhist,T,A;nev=nev,emin=emin,emax=emax,overextract=overextract,refine=refine)
        if !isempty(res)
            if isempty(bestres) || length(λ)>length(bestλ) || (length(λ)==length(bestλ) && maximum(res)<maximum(bestres))
                bestλ=copy(λ);bestX=copy(X);bestres=copy(res)
            end
            rs=sort(res)
            println("it=",it,"  inside=",length(λ),"  r90=",rs[clamp(round(Int,0.9*length(rs)),1,length(rs))],"  r95=",rs[clamp(round(Int,0.95*length(rs)),1,length(rs))],"  maxres=",rs[end])
            maximum(res)<tol && length(λ)>=nev && return λ,X,res
        else
            println("it=",it,"  inside=0  maxres=Inf")
        end
        size(Qnext,2)==0 && break
        Qprev=Q
        Q=Qnext
        Bprev=Bblocks[end]
    end
    return bestλ,bestX,bestres
end

"""
    polfed(A::MatrixFreeRealSymOp,nev::Int,degree::Int;
           s=8,smin=nothing,smax=nothing,sigma::Float64=0.0,
           emin::Float64=-Inf,emax::Float64=Inf,tol::Float64=1e-10,
           maxblocks=nothing,threaded::Bool=true,overextract::Int=8,
           refine::Bool=true,seed::Int=1)

Convenience wrapper for calling the matrix-free POLFED solver directly on
a `MatrixFreeRealSymOp`.

This simply forwards to

    polfed(A.matvec!,A.n,nev,degree; ...)

with the same keyword arguments.

Arguments
- `A::MatrixFreeRealSymOp`: matrix-free real symmetric operator
- `nev::Int`: number of desired eigenpairs
- `degree::Int`: polynomial degree of the delta filter

Keyword arguments
- same as for `polfed(matvec!,n,nev,degree; ...)`

Returns
- `bestλ::Vector{Float64}`: best interval eigenvalue approximations found
- `bestX::Matrix{Float64}`: corresponding eigenvector approximations
- `bestres::Vector{Float64}`: corresponding residual norms
"""
function polfed(A::MatrixFreeRealSymOp,nev::Int,degree::Int;s=8,smin=nothing,smax=nothing,sigma::Float64=0.0,emin::Float64=-Inf,emax::Float64=Inf,tol::Float64=1e-10,maxblocks=nothing,threaded::Bool=true,overextract::Int=8,refine::Bool=true,seed::Int=1,window_type::Symbol=:step,jackson::Bool=true)
    return polfed(A.matvec!,A.n,nev,degree;s=s,smin=smin,smax=smax,sigma=sigma,emin=emin,emax=emax,tol=tol,maxblocks=maxblocks,threaded=threaded,overextract=overextract,refine=refine,seed=seed,window_type=window_type,jackson=jackson)
end


















if abspath(PROGRAM_FILE)==@__FILE__

    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing

    # SOLVER PARAMETERS; NODES/POLES - CRITICAL TO GET SMALLEST POSSIBLE WITH LOWEST ITERATION COUNT
    nodes_general=32 # this is the total nodes, sym_ automatically uses half the nodes for the half-contour version
    nodes_chebyshev=96 # Chebyshev filter (actually quite a high degree is needed)
    nodes_zolotarev=16 # Zolotarev rational filter, requiring typically lower node count than ellipse or Chebyshev filters for similar accuracy
    degrees_chebyshev=[8,16,32,48,64,96,128] # degrees of the chebyshev polynomial approximating the delta function filter
    counts_chebyshev=reverse([100,150,200,250,500]) # how many eigenvalues we want in the delta window, more eigenvealues (m0), less degree we need
    do_polfed=true 3 # polfed algorithm from "Polynomially Filtered Exact Diagonalization Approach to Many-Body Localization" by Sierant, P., Lewenstein, M., Zakrzewski, J.

    # TOLERANCE FOR RESIDUAL OF EIGENVECTOR
    tol=1e-8 # test tolerance (max |Δλ| and max residual)

    # MATRIX SIZES SPARSE/DENSE
    ns=[5000] # sparse matrix size lap1d to test
    n=200 # size of the dense matrix (just to test that the code works on dense matrices as well)

    # OPTIONS 
    do_general=false # test for general sparse matrices (not necessarily symmetric)
    do_half=false # test half-contour version for symmetric matrices (2x faster but only for symmetric problems)
    do_chebyshev=false # test Chebyshev filtered subspace iteration for symmetric matrices in a window (step vs delta)
    do_zolotarev=false # test Zolotarev rational filter for full 
    do_zolotarev_half=false # test Zolotarev rational filter for half (real symmetric matrices)

    # ADDITIONAL OPTIONS
    do_distributed=false # if we want for every option above to test the "distributed" version that doesn't store factorizations but recomputes them at every iteration (this is more memory efficient but more expensive computationally, so it's not the default for testing)
    show_progress=true # to see progress bars for each benchmark. Use false to avoid visual clutter and influence on performance

    # ============================================================
    # TESTS
    # ============================================================

    # 1D Laplacian tridiag(-1,2,-1)
    function lap1d(n::Int)
        d0=fill(2.0,n)
        d1=fill(-1.0,n-1)
        return spdiagm(-1=>d1,0=>d0,1=>d1)
    end
    function lap1d_matvec!(y,x)
            n=length(x)
            @inbounds begin
                y[1]=2*x[1]-x[2]
                for i in 2:n-1
                    y[i]=2*x[i]-x[i-1]-x[i+1]
                end
                y[n]=2*x[n]-x[n-1]
            end
        end
    lap1d_eigs_exact(n::Int)=[2-2*cos(k*pi/(n+1)) for k in 1:n]
    function max_abs_err(a::Vector{Float64},b::Vector{Float64})
        length(a)==length(b) || return Inf
        maximum(abs.(a.-b))
    end

    function test_feast_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        any(max_abs_err(λ,λ_ref)>1e-8) && @warn "max |Δλ| = $(max_abs_err(λ,λ_ref)) exceeds tolerance"
        #isempty(res) || @warn "max residual reported = $(maximum(res))"
        return λ,λ_ref,res
    end

    function test_feast_on_lap1d_zolotarev(zp::ZolotarevParams;n=1000,emin=0.2,emax=0.8,m0=400,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval_zolotarev(A,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            debug=debug,backend=backend,ls_alg=ls_alg,
            show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        any(max_abs_err(λ,λ_ref)>1e-8) && @warn "ZOLO: max |Δλ| = $(max_abs_err(λ,λ_ref)) exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "HALF: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    # BETTER TEST SINCE WE ARE APPROXIMATING THE DELTA WINDOW POTENTIALLY AND NOT A FIXED INTERVAL for window_type=:delta
    ################################################################################################################
    function nearest_k_to_sigma(vals::Vector{Float64},σ::Float64,k::Int)
        p=sortperm(abs.(vals .- σ))
        out=vals[p[1:min(k,length(vals))]]
        sort!(out)
        return out
    end
    function local_match_error(λ::Vector{Float64},λexact::Vector{Float64},σ::Float64)
        k=min(length(λ),length(λexact))
        k==0 && return Inf
        a=nearest_k_to_sigma(λ,σ,k)
        b=nearest_k_to_sigma(λexact,σ,k)
        return maximum(abs.(a.-b))
    end
    function match_stats_sorted(ref::Vector{Float64},got::Vector{Float64};tol::Float64=1e-8)
        nref=length(ref)
        ngot=length(got)
        ref_hit=falses(nref)
        got_hit=falses(ngot)
        i=1
        j=1
        while i<=nref && j<=ngot
            d=got[j]-ref[i]
            if abs(d)<=tol
                ref_hit[i]=true
                got_hit[j]=true
                i+=1
                j+=1
            elseif d<0
                j+=1
            else
                i+=1
            end
        end
        missed_ref=findall(!,ref_hit)
        extra_got=findall(!,got_hit)
        return (nref=nref,ngot=ngot,nmatched_ref=count(ref_hit),nmatched_got=count(got_hit),nmissed=length(missed_ref),nextra=length(extra_got),missed_idx=missed_ref,extra_idx=extra_got,ref_hit=ref_hit,got_hit=got_hit)
    end
    function classify_against_full_spectrum(λgot::Vector{Float64},λall::Vector{Float64};tol::Float64=1e-8)
        ng=length(λgot)
        na=length(λall)
        got_legit=falses(ng)
        i=1
        j=1
        while i<=ng && j<=na
            d=λgot[i]-λall[j]
            if abs(d)<=tol
                got_legit[i]=true
                i+=1
                j+=1
            elseif d<0
                i+=1
            else
                j+=1
            end
        end
        legit_idx=findall(got_legit)
        spurious_idx=findall(!,got_legit)
        return (nlegit=length(legit_idx),nspurious=length(spurious_idx),legit_idx=legit_idx,spurious_idx=spurious_idx,got_legit=got_legit)
    end
    function classify_local_extras(λref::Vector{Float64},λgot::Vector{Float64},λall::Vector{Float64};tol::Float64=1e-8)
        local_stats=match_stats_sorted(λref,λgot;tol=tol)
        full_stats=classify_against_full_spectrum(λgot,λall;tol=tol)
        legit_extra_idx=Int[]
        spurious_extra_idx=Int[]
        local_extra_mask=falses(length(λgot))
        local_extra_mask[local_stats.extra_idx].=true
        for j in local_stats.extra_idx
            if full_stats.got_legit[j]
                push!(legit_extra_idx,j)
            else
                push!(spurious_extra_idx,j)
            end
        end
        return (nmatched_target=local_stats.nmatched_ref,nmissed_target=local_stats.nmissed,nextra=local_stats.nextra,nlegit_extra=length(legit_extra_idx),nspurious_extra=length(spurious_extra_idx),missed_target_idx=local_stats.missed_idx,extra_idx=local_stats.extra_idx,legit_extra_idx=legit_extra_idx,spurious_extra_idx=spurious_extra_idx)
    end
    function lap1d_interval_by_count(n::Int,nev_target::Int;center_frac::Float64=0.5,pad::Int=2)
        λ=lap1d_eigs_exact(n)
        ic=clamp(round(Int,center_frac*n),1,n)
        h=nev_target÷2
        i1=max(1,ic-h)
        i2=min(n,i1+nev_target-1)
        i1=max(1,i2-nev_target+1)
        emin=λ[max(1,i1-pad)]
        emax=λ[min(n,i2+pad)]
        σ=0.5*(λ[i1]+λ[i2])
        λref=λ[i1:i2]
        return emin,emax,σ,collect(λref),(i1,i2)
    end
    function test_cheb_delta_on_lap1d_count(matvec!;n=5000,nev_target=50,center_frac=0.5,m0_pad=1.5,degree=64,tol=1e-8,maxiter=30,res_gate=1e-6,show_progress=false,window_type=:step,matrix_free=false)
        smin=0.0
        smax=4.0
        A=lap1d(n)
        λall=lap1d_eigs_exact(n)
        emin,emax,σ,λref,(i1,i2)=lap1d_interval_by_count(n,nev_target;center_frac=center_frac)
        m0=min(n,ceil(Int,m0_pad*nev_target))

        println("CHEBΔ count-test: n=",n," nev_target=",nev_target,
                " idx=[",i1,",",i2,"] interval=[",emin,", ",emax,"] σ=",σ,
                " m0=",m0," degree=",degree)
        flush(stdout)
        if matrix_free
            λ,X,res=chebyshev_poly_real_symm_sparse(
            lap1d_matvec!,n;
            σ=σ, emin=emin, emax=emax,
            smin=smin, smax=smax,
            m0=m0, degree=degree,
            maxiter=maxiter, tol=tol, res_gate=res_gate,
            jackson=true, window_type=window_type)
        else
            λ,X,res=chebyshev_poly_real_symm_sparse(
            A,emin,emax;
            σ=σ,
            smin=smin, smax=smax,
            m0=m0, degree=degree,
            maxiter=maxiter, tol=tol, res_gate=res_gate,
            jackson=true, window_type=window_type)
        end
        λ=sort(λ)
        λref=sort(λref)
        λall=sort(λall)
        cls=classify_local_extras(λref,λ,λall;tol=1e-8)
        println("matched target = ",cls.nmatched_target," / ",length(λref))
        println("missed target  = ",cls.nmissed_target)
        println("extra total    = ",cls.nextra)
        println("legit extra    = ",cls.nlegit_extra)
        println("spurious extra = ",cls.nspurious_extra)
        Δloc=local_match_error(λ,λref,σ)
        rmax=isempty(res) ? Inf : maximum(res)
        meta=(emin=emin,emax=emax,σ=σ,m0=m0,i1=i1,i2=i2,Δloc=Δloc,rmax=rmax,nmatched_target=cls.nmatched_target,nmissed_target=cls.nmissed_target,nextra=cls.nextra,nlegit_extra=cls.nlegit_extra,nspurious_extra=cls.nspurious_extra)
        return λ,λref,res,meta
    end
    
    ################################################################################################################

    function test_sym_half_on_lap1d_zolotarev(zp::ZolotarevParams;n=1000,emin=0.2,emax=0.8,m0=400,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half_zolotarev(A,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            backend=backend,ls_alg=ls_alg,debug=debug,
            show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO HALF: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "ZOLO HALF: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,debug=false,backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A) # force it to be sparse to test the sparse FEAST implementation, even though it's a dense matrix; this is just for testing correctness and timing of the sparse version on small matrices.
        λ,X,res=general_feast_sparse_interval(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,maxiter=maxiter,tol=tol,res_gate=res_gate,debug=debug,is_hermitian=true,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        any(maxΔ>1e-8) && @warn "max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small_zolotarev(zp::ZolotarevParams;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,tol=tol,maxiter=10,res_gate=1e-8,debug=false,backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=general_feast_sparse_interval_zolotarev(As,zp;
            emin=emin,emax=emax,
            m0=m0,
            maxiter=maxiter,tol=tol,res_gate=res_gate,
            debug=debug,is_hermitian=true,
            backend=backend,ls_alg=ls_alg,
            show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "ZOLO(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=sym_feast_sparse_interval_half(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "HALF(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small_zolotarev(zp::ZolotarevParams;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=sym_feast_sparse_interval_half_zolotarev(As,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            backend=backend,ls_alg=ls_alg,
            debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO HALF(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "ZOLO HALF(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end
    
    # for 1d Laplacian do a sweep over matrix sizes and intervals, with timing. Try increasing matrix size for any anomalies. Careful to manually increase m0 and nodes as n grows to ensure good convergence.
    function bench_sweep(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4] # interval sizes, can increase if needed to more or larger intervals.
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    function bench_sweep_half(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    # ============================================================
    # DISTRIBUTED VERSION (no node preallocations)
    # ============================================================

    function test_feast_on_lap1d_distributed(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval_distributed(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,is_hermitian=true,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "DISTRIBUTED: expected $(length(λ_ref)) eigs but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "DISTRIBUTED: max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_lap1d_zolotarev_distributed(zp::ZolotarevParams;n=1000,emin=0.2,emax=0.8,m0=400,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval_zolotarev_distributed(A,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            debug=debug,is_hermitian=true,
            backend=backend,ls_alg=ls_alg,
            blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO DIST: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "ZOLO DIST: max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_lap1d_distributed(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half_distributed(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF/DIST: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "HALF/DIST: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_lap1d_zolotarev_distributed(zp::ZolotarevParams;n=1000,emin=0.2,emax=0.8,m0=400,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half_zolotarev_distributed(A,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            backend=backend,ls_alg=ls_alg,
            blockrhs=blockrhs,debug=debug,
            show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO HALF/DIST: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "ZOLO HALF/DIST: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_cheb_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,degree=nodes_chebyshev,tol=tol,maxiter=8,res_gate=1e-6,debug=false,show_progress::Bool=false,window_type=:step)
        
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=chebyshev_poly_real_symm_sparse(A;emin=emin,emax=emax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,debug=debug,show_progress=show_progress,window_type=window_type)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "CHEB: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "CHEB: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small_distributed(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n;A[i,i]+=diagshift;end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=general_feast_sparse_interval_distributed(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,is_hermitian=true,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "DISTRIBUTED: expected $(length(λ_ref)) eigs but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "DISTRIBUTED: max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small_zolotarev_distributed(zp::ZolotarevParams;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=general_feast_sparse_interval_zolotarev_distributed(As,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            debug=false,is_hermitian=true,
            backend=backend,ls_alg=ls_alg,
            blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO DIST(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "ZOLO DIST(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small_distributed(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res = sym_feast_sparse_interval_half_distributed(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF/DIST(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "HALF/DIST(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small_zolotarev_distributed(zp::ZolotarevParams;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=sym_feast_sparse_interval_half_zolotarev_distributed(As,zp;
            emin=emin,emax=emax,
            m0=m0,
            tol=tol,maxiter=maxiter,res_gate=res_gate,
            backend=backend,ls_alg=ls_alg,
            blockrhs=blockrhs,debug=debug,
            show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "ZOLO HALF/DIST(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "ZOLO HALF/DIST(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function bench_sweep_distributed(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general;eta=0.8;tol=tol;maxiter=10;res_gate=1e-6;emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0;emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d_distributed(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    function bench_sweep_zolotarev(zp::ZolotarevParams;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_zolotarev
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d_zolotarev(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    # BETTER BENCHMARK FOR APPROXIMATING DELTA FUNCTION
    function bench_sweep_cheb_counts(;n=5000,counts=[20,50,100,150],degrees=[16,32,48,64,96,128],center_frac=0.5,maxiter=30,tol=1e-8,res_gate=1e-6,m0_pad=1.5,window_type=:step,matrix_free::Bool=false)

        println("==========================================================================")
        println("CHEB DELTA local-count benchmark")
        println("==========================================================================")
        @printf("%8s %8s %8s %8s %8s %8s %8s %12s\n",
                "nev","deg","m0","got","match","lextra","spur","localerr")
        println("--------------------------------------------------------------------------")

        for nev in counts
            for deg in degrees
                t=@elapsed begin
                    λ,λref,res,meta=test_cheb_delta_on_lap1d_count(lap1d_matvec!,
                    n=n,nev_target=nev,center_frac=center_frac,
                    m0_pad=m0_pad,degree=deg,
                    tol=tol,maxiter=maxiter,res_gate=res_gate,
                    show_progress=false,window_type=window_type,matrix_free=matrix_free)
                    global _λ=λ
                    global _λref=λref
                    global _res=res
                    global _meta=meta
                end
                @printf("%8d %8d %8d %8d %8d %8d %8d %12.3e\n",
                        nev,deg,_meta.m0,length(_λ),
                        _meta.nmatched_target,
                        _meta.nlegit_extra,
                        _meta.nspurious_extra,
                        _meta.Δloc)
                println("time[s] = ", @sprintf("%.4f", t))
                flush(stdout)
            end
            println("--------------------------------------------------------------------------")
        end
    end

    function bench_sweep_half_distributed(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=32,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d_distributed(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    function bench_sweep_half_zolotarev(zp::ZolotarevParams;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d_zolotarev(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    function bench_sweep_zolotarev_distributed(zp::ZolotarevParams;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_zolotarev
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d_zolotarev_distributed(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    function bench_sweep_half_zolotarev_distributed(zp::ZolotarevParams;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=32,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_zolotarev
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d_zolotarev_distributed(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    function bench_sweep_cheb(;ns=[5_000,10_000,20_000],degree=nodes_chebyshev,tol=tol,maxiter=20,show_progress::Bool=false,window_type=:step)
        Δs=[0.1,0.2,0.4]
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_cheb_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,degree=degree,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,show_progress=show_progress,window_type=window_type)
        end
    end

    function lap1d_target_window(n::Int,nev::Int)
        λexact=lap1d_eigs_exact(n)
        i1=div(n-nev,2)+1
        i2=i1+nev-1
        emin=λexact[i1]
        emax=λexact[i2]
        sigma=0.5*(emin+emax)
        return λexact,i1,i2,emin,emax,sigma
    end

    function test_polfed_lap1d(;n::Int=50000,nev::Int=1000,degree::Int=128,smin=0.0,smax=4.0,s::Int=8,tol::Float64=1e-8,threaded::Bool=true,maxblocks=50,matrix_free::Bool=true,window_type=:delta)
        A=lap1d(n)
        λexact,i1,i2,emin,emax,sigma=lap1d_target_window(n,nev)
        println("================================================")
        println("POLFED Lap1D benchmark")
        println("================================================")
        println("n            = ",n)
        println("target nev   = ",nev)
        println("degree       = ",degree)
        println("block size   = ",s)
        println("target interval:")
        println("  emin = ",emin)
        println("  emax = ",emax)
        println("  sigma = ",sigma)
        println("  index range ≈ [",i1, ", ",i2, "]")
        println()
        t0=time()
        if matrix_free
            A=MatrixFreeRealSymOp(n,lap1d_matvec!)
            λ,X,res=polfed(A,nev,degree,sigma=sigma,s=s,emin=emin,emax=emax,tol=tol,threaded=threaded,maxblocks=maxblocks,window_type=window_type,smin=smin,smax=smax)
        else
            λ,X,res=polfed(A,nev,degree,sigma=sigma,s=s,emin=emin,emax=emax,tol=tol,threaded=threaded,#=maxblocks=maxblocks,=#window_type=window_type)
        end
        ttot=time()-t0
        inside=findall(x->emin<=x<=emax,λ)
        p=sortperm(λ[inside])
        λin=λ[inside][p]
        rin=res[inside][p]
        λtarget=λexact[i1:i2]
        ngot=length(λin)
        ncmp=min(length(λtarget),ngot)

        maxerr=ncmp>0 ? maximum(abs.(λin[1:ncmp].-λtarget[1:ncmp])) : Inf
        maxres=ngot>0 ? maximum(rin) : Inf
        minres=ngot>0 ? minimum(rin) : Inf

        println("================================================")
        println("RESULTS")
        println("================================================")
        println("returned total         = ",length(λ))
        println("returned inside window = ",ngot)
        println("runtime [s]            = ",ttot)
        println("max residual inside    = ",maxres)
        println("min residual inside    = ",minres)
        println("max eig error inside   = ",maxerr)
        println()

        if ngot>0
            println("first few inside-window eigenvalues:")
            println(λin[1:min(10,ngot)])
        end
        return λ,X,res,λin,rin,λtarget
    end





    @info "Benchmarking FEAST implementations for SPARSE and DENSE symmetric matrices. The backends are either LU (:lu) or solvers from LinearSolve.jl (:ls). The algorithms for :lu are default=nothing, while for :ls we have either UMPACK or KLU. For SPARSE matrices the best option is :ls with KLUFactorization (at least for small to medium ns), while for DENSE matrices hte choice between default Julia's LU or UMFPACKFactorization give basically the same runtimes."

    if do_general

    println()
    println("--------------------------------------------------")
    println("DEFAULT (general_feast_sparse_interval)")
    println("--------------------------------------------------")
    # sweep over matrix sizes and intervals, with timing for the 1d Laplacian test. 5000 is default, try n=20000.
    
    s=time()
    println("SPARSE")
    println("LU (default)")
    bench_sweep(ns=ns,backend=:lu) # slow for small ns, for large ns could be faster ?
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    bench_sweep(ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization()) # slow for small ns
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization()) # fast for small ns
    println("Done (3/3)")
    e=time()
    # Analysis:
    println("--------------------------------------------------")
    println("Timing summary for 1d Laplacian test (SPARSE):")
    println("LU backend: ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend: ",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend: ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------") # try increasing matrix size for any anomalies. Careful to manually increase m0 and nodes as n grows to ensure good convergence.

    #= #TODO
    if HAVE_PARDISO[]
        @warn "Not yet correctly implemented, ignore for now!"
        try
            local s=time()
            bench_sweep(ns=ns,backend=:ls,ls_alg=LinearSolve.MKLPardisoFactorize())
            local en=time()
            println("MKL Pardiso backend: ",@sprintf("%.3f seconds",en-s))
        catch e
            @warn "MKLPardiso failed: $e"
        end
    end
    =#

    println()
    n=1000
    s=time()
    println("DENSE")
    println("LU (default)")
    test_feast_on_dense_small(n=n,backend=:lu,show_progress=show_progress) # fast
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    test_feast_on_dense_small(n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress) # # fast (maybe slightly faster than default LU)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    test_feast_on_dense_small(n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress) # slow
    println("Done (3/3)")
    e=time()
    # Analysis:
    println("Timing summary for dense small matrix test (DENSE):")
    println("LU backend: ", @sprintf("%.3f seconds", s1-s))
    println("LinearSolve UMFPACK backend: ", @sprintf("%.3f seconds", s2-s1))
    println("LinearSolve KLU backend: ", @sprintf("%.3f seconds", e-s2))
    println("--------------------------------------------------")

        if do_distributed

        println()
        println("--------------------------------------------------")
        println("DISTRIBUTED VERSION TESTS (general_feast_sparse_interval_distributed)")
        println("--------------------------------------------------")
        
        # timing sweep: compare LU vs LinearSolve(KLU/UMFPACK) for distributed version
        
        s=time()
        println("SPARSE / DISTRIBUTED")
        println("LU (blockrhs=m0)")
        bench_sweep_distributed(ns=ns,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        println("LU (blockrhs=32)")
        bench_sweep_distributed(ns=ns,backend=:lu,blockrhs=32,show_progress=show_progress)
        s2=time()
        println("LinearSolve UMFPACK (blockrhs=32)")
        bench_sweep_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),blockrhs=32,show_progress=show_progress)
        s3=time()
        println("LinearSolve KLU (blockrhs=32)")
        bench_sweep_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("Timing summary (SPARSE / DISTRIBUTED):")
        println("LU blockrhs=m0:      ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=32:      ",@sprintf("%.3f seconds",s2-s1))
        println("LS UMFPACK blockrhs=32: ",@sprintf("%.3f seconds",s3-s2))
        println("LS KLU blockrhs=32:     ",@sprintf("%.3f seconds",e-s3))
        println("--------------------------------------------------")

        # dense small matrix smoke test for distributed version
        println()
        println("DENSE / DISTRIBUTED")
        
        s=time()
        test_feast_on_dense_small_distributed(n=n,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        test_feast_on_dense_small_distributed(n=n,backend=:lu,blockrhs=16,show_progress=show_progress)
        s2=time()
        test_feast_on_dense_small_distributed(n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),blockrhs=16,show_progress=show_progress)
        s3=time()
        test_feast_on_dense_small_distributed(n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=16,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("Timing summary (DENSE / DISTRIBUTED):")
        println("LU blockrhs=0:      ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=16:     ",@sprintf("%.3f seconds",s2-s1))
        println("LS UMFPACK blockrhs=16: ",@sprintf("%.3f seconds",s3-s2))
        println("LS KLU blockrhs=16:     ",@sprintf("%.3f seconds",e-s3))
        println("--------------------------------------------------")

        end

    end

    if do_half

    println()
    println("--------------------------------------------------")
    println("HALF-CONTOUR (sym_feast_sparse_interval_half)")
    println("--------------------------------------------------")

    s=time()
    println("SPARSE / HALF")
    println("LU (default)")
    bench_sweep_half(ns=ns,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    bench_sweep_half(ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep_half(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()
    println("--------------------------------------------------")
    println("Timing summary (SPARSE / HALF):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / HALF")
    println("LU (default)")
    test_sym_half_on_dense_small(n=n,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    test_sym_half_on_dense_small(n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    test_sym_half_on_dense_small(n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()
    println("--------------------------------------------------")
    println("Timing summary (DENSE / HALF):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

        if do_distributed

        println()
        println("--------------------------------------------------")
        println("HALF-CONTOUR DISTRIBUTED (sym_feast_sparse_interval_half_distributed)")
        println("--------------------------------------------------")

        
        s=time()
        println("SPARSE / HALF / DISTRIBUTED")
        println("LU (blockrhs=m0)")
        bench_sweep_half_distributed(ns=ns,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        println("LU (blockrhs=32)")
        bench_sweep_half_distributed(ns=ns,backend=:lu,blockrhs=32,show_progress=show_progress)
        s2=time()
        println("LinearSolve UMFPACK (blockrhs=32)")
        bench_sweep_half_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),blockrhs=32,show_progress=show_progress)
        s3=time()
        println("LinearSolve KLU (blockrhs=32)")
        bench_sweep_half_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("Timing summary (SPARSE / HALF / DISTRIBUTED):")
        println("LU blockrhs=m0:             ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=32:             ",@sprintf("%.3f seconds",s2-s1))
        println("LS UMFPACK blockrhs=32:     ",@sprintf("%.3f seconds",s3-s2))
        println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s3))
        println("--------------------------------------------------")

        println()
        println("DENSE / HALF / DISTRIBUTED")
        
        s=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:lu,blockrhs=16,show_progress=show_progress)
        s2=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),blockrhs=16,show_progress=show_progress)
        s3=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=16,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("Timing summary (DENSE / HALF / DISTRIBUTED):")
        println("LU blockrhs=0:              ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=16:             ",@sprintf("%.3f seconds",s2-s1))
        println("LS UMFPACK blockrhs=16:     ",@sprintf("%.3f seconds",s3-s2))
        println("LS KLU blockrhs=16:         ",@sprintf("%.3f seconds",e-s3))
        println("--------------------------------------------------")

        end

    end

    if do_zolotarev

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV FILTER FEAST")
    println("--------------------------------------------------")

    
    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    s=time()
    println("SPARSE / ZOLOTAREV")
    println("LU (default)")
    bench_sweep_zolotarev(zp,ns=ns,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    bench_sweep_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (SPARSE / ZOLOTAREV):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / ZOLOTAREV")
    println("LU (default)")
    test_feast_on_dense_small_zolotarev(zp,n=n,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    test_feast_on_dense_small_zolotarev(zp,n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    test_feast_on_dense_small_zolotarev(zp,n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (DENSE / ZOLOTAREV):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    end

    if do_zolotarev_half

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV HALF-FILTER")
    println("--------------------------------------------------")

    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    s=time()
    println("SPARSE / ZOLOTAREV HALF")
    println("LU (default)")
    bench_sweep_half_zolotarev(zp,ns=ns,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    bench_sweep_half_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep_half_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (SPARSE / ZOLOTAREV HALF):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / ZOLOTAREV HALF")
    println("LU (default)")
    test_sym_half_on_dense_small_zolotarev(zp,n=n,backend=:lu,show_progress=show_progress)
    println("Done (1/3)")
    s1=time()
    println("LinearSolve UMFPACK")
    test_sym_half_on_dense_small_zolotarev(zp,n=n,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    println("Done (2/3)")
    s2=time()
    println("LinearSolve KLU")
    test_sym_half_on_dense_small_zolotarev(zp,n=n,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    println("Done (3/3)")
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (DENSE / ZOLOTAREV HALF):")
    println("LU backend:                 ",@sprintf("%.3f seconds",s1-s))
    println("LinearSolve UMFPACK backend:",@sprintf("%.3f seconds",s2-s1))
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    end

    if do_distributed && do_zolotarev

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV DISTRIBUTED")
    println("--------------------------------------------------")

    s=time()
    println("SPARSE / ZOLOTAREV / DISTRIBUTED")
    println("LU (blockrhs=m0)")
    bench_sweep_zolotarev_distributed(zp,ns=ns,backend=:lu,blockrhs=0,show_progress=show_progress)
    s1=time()
    println("LU (blockrhs=32)")
    bench_sweep_zolotarev_distributed(zp,ns=ns,backend=:lu,blockrhs=32,show_progress=show_progress)
    s2=time()
    println("LinearSolve UMFPACK (blockrhs=32)")
    bench_sweep_zolotarev_distributed(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),blockrhs=32,show_progress=show_progress)
    s3=time()
    println("LinearSolve KLU (blockrhs=32)")
    bench_sweep_zolotarev_distributed(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (SPARSE / ZOLOTAREV / DISTRIBUTED):")
    println("LU blockrhs=m0:             ",@sprintf("%.3f seconds",s1-s))
    println("LU blockrhs=32:             ",@sprintf("%.3f seconds",s2-s1))
    println("LS UMFPACK blockrhs=32:     ",@sprintf("%.3f seconds",s3-s2))
    println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s3))
    println("--------------------------------------------------")

    end

    if do_distributed && do_zolotarev_half

    println()
    println("SPARSE / ZOLOTAREV HALF / DISTRIBUTED")
    s=time()
    println("LU (blockrhs=m0)")
    bench_sweep_half_zolotarev_distributed(ns=ns,backend=:lu,blockrhs=0,show_progress=show_progress)
    s1=time()
    println("LU (blockrhs=32)")
    bench_sweep_half_zolotarev_distributed(ns=ns,backend=:lu,blockrhs=32,show_progress=show_progress)
    s2=time()
    println("LinearSolve UMFPACK (blockrhs=32)")
    bench_sweep_half_zolotarev_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.UMFPACKFactorization(),show_progress=show_progress)
    s3=time()
    println("LinearSolve KLU (blockrhs=32)")
    bench_sweep_half_zolotarev_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (SPARSE / ZOLOTAREV HALF / DISTRIBUTED):")
    println("LU blockrhs=m0:             ",@sprintf("%.3f seconds",s1-s))
    println("LU blockrhs=32:             ",@sprintf("%.3f seconds",s2-s1))
    println("LS UMFPACK blockrhs=32:     ",@sprintf("%.3f seconds",s3-s2))
    println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s3))
    println("--------------------------------------------------")

    end

    if do_chebyshev
    
    # matrix free
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=ns[1],counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:delta,matrix_free=true)

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA WINDOW MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=ns[1],counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:step,matrix_free=true)
    
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB DELTA")
    
    bench_sweep_cheb_counts(n=ns[1],counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:delta,matrix_free=false)

    println("--------------------------------------------------")
    println("CHEBYSHEV WINDOW FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB WINDOW")
    bench_sweep_cheb_counts(n=ns[1],counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:step,matrix_free=false)

    end

    if do_polfed

    println("POLFED CONCRETE MATRIX")
   
    L=15
    n=Int(2^L)
    nev=200
    s=max(200,3*nev)
    degree=256
    λ,X,res,λin,rin,λtarget=test_polfed_lap1d(n=n,nev=nev,degree=degree,s=s,tol=1e-6,threaded=true,maxblocks=200,matrix_free=false,window_type=:smooth)

    println()
    println("POLFED MATRIX FREE")

    L=15
    n=Int(2^L)
    nev=200
    s=max(200,3*nev)
    degree=128
    n=Int(2^L)
    λ,X,res,λin,rin,λtarget=test_polfed_lap1d(n=n,nev=nev,smin=0.0,smax=4.0,degree=degree,s=s,tol=1e-6,threaded=true,maxblocks=200,matrix_free=true,window_type=:smooth)

    end

end
