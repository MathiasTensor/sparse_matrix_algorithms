# -----------------------------------------------------------------------------
#  Sparse FEAST (projector iteration) for the symmetric EVP
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
#
# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# EDITING AND SUPERVISION: Orel 10/3/2026
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
# - CHEBYSHEV FILTER
# chebyshev_poly_real_symm_sparse(...) <-> REAL SYMMETRIC SPARSE MATRIX
#
# - ZOLOTAREV RATIONAL FILTER ! Very fast, should be the GOTO
# ! REQUIRES THE CONSTRUCTION OF ZolotarevParams (constructor below) TO PRECOMPUTE THE POLES !
# general_feast_sparse_interval_zolotarev(...) <-> GENERAL SPARSE MATRIX
# sym_feast_sparse_interval_half_zolotarev(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR
# general_feast_sparse_interval_zolotarev_distributed(...) <-> GENERAL SPARSE MATRIX, DISTRIBUTED
# sym_feast_sparse_interval_half_zolotarev_distributed(...) <-> REAL SYMMETRIC SPARSE MATRIX, HALF CONTOUR, DISTRIBUTED
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# EXAMPLES (see tests below)
# -----------------------------------------------------------------------------



using LinearAlgebra,SparseArrays,Random,Printf,BenchmarkTools,LinearSolve,ProgressMeter,QuadGK

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
- Our `shift_diag!` overwrites Az.nzval at the diagonal locations.
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

Performance note:
- `Matrix(F.Q)` allocates an n×n matrix in principle, but Julia’s QRCompactWY
  materialization returns a dense n×n; for correctness tests this is ok.
- For production, you’d want a lower-allocation thin-QR routine (e.g. store
  reflectors and apply them to an n×m0 identity block).
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
        #thin_orth!(Qc,Qtmp,Z) # relatively cheap since nodes loop dominates
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
        #thin_orth!(Qc,Qtmp,Z)
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
        #thin_orth!(Q,Qtmp,Z)
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
        #thin_orth!(Q,Qtmp,Z)
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

"""
    scaled_mul!(Y,A,X,c,d,AX)

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
    apply_cheb_filter!(Y,A,Q,coeffs,smin,smax,U0,U1,U2,AX)

Apply the Chebyshev polynomial filter with coefficients `coeffs` to the block `Q`.

The matrix is first rescaled to
    Â = (A-cI)/d
where
    c = (smin+smax)/2, d = (smax-smin)/2

so that σ(Â) ⊂ [-1,1].

The output is
    Y = Σ_{k=0}^m coeffs[k+1] T_k(Â) Q.
"""
function apply_cheb_filter!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})
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
    @inbounds for k in 1:(m-1)
        # U2 = 2ÂU1 - U0
        mul!(AX,A,U1) # AX = A * U1 (not Â)
        for j in 1:blk, i in 1:n
            U2[i,j]=2*((AX[i,j]-c*U1[i,j])*invd)-U0[i,j] # U2 = 2Â * U1 - U0 = 2((A - cI) / d) * U1 - U0 = 2(A * U1 - c * U1)/d - U0; because A in the previous line is not scaled Â, we need to do the scaling here
        end
        ck1=coeffs[k+2] # contribution of U_{k+1} = T_{k+1}(Â) * Q (shifted by 1 index because coeffs[1] is for T_0 ...)
        LinearAlgebra.BLAS.axpy!(ck1, vec(U2), vec(Y)) # add the contribution of U_{k+1} = T_{k+1}(Â) * Q to the output Y
        U0,U1,U2=U1,U2,U0 # reasing workspaces for the next iteration: now U0 holds T_k(Â) * Q and U1 holds T_{k+1}(Â) * Q for the next iteration
    end
    return nothing
end

"""
    chebyshev_poly_real_symm_sparse(Ain; emin, emax, m0=200, degree=80, smin=nothing, smax=nothing, maxiter=8, tol=1e-10, res_gate=1e-6, jackson=true, debug=false, two_hit=true, seed=1) -> (λsel::Vector{Float64}, Xsel::Matrix{Float64}, ressel::Vector{Float64})

Compute eigenpairs of a real symmetric sparse matrix with eigenvalues in [emin,emax]
using Chebyshev polynomial filtered subspace iteration.

Parameters
- `m0`:
    Search subspace dimension. Must exceed the number of eigenvalues in the interval.
- `degree`:
    Degree of the Chebyshev window polynomial. Larger degree => sharper spectral filter.
- `jackson`:
    If true, use Jackson damping to reduce Gibbs oscillations.
- `two_hit`:
    If true, require 2 consecutive iterations with maxres<tol (except immediate break at it=1).

Returns
- `λsel`: selected eigenvalues in [emin,emax], sorted
- `Xsel`: corresponding eigenvectors
- `ressel`: residual norms
"""
function chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1)

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

    coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson) # give the window coefficients for the mapped interval
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
        #thin_orth!(Q,Qtmp,Z) # orthogonalize the block before applying the filter to improve numerical stability
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

    Q ← orth(Q)
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

FEAST eigensolver for **real symmetric sparse matrices** using a
**half-contour Zolotarev rational filter**. (See `general_feast_sparse_interval_zolotarev` for details.)

----------------------------------------------------------------------
Half-contour idea
----------------------------------------------------------------------

For real symmetric matrices:

    w (zI - A)⁻¹ + w̄ (z̄I - A)⁻¹ = 2 Re[w (zI - A)⁻¹]

Therefore the rational filter can be evaluated using only poles in the
**upper half-plane**, reducing the number of linear solves by ~2×.

The projector becomes

    Y = α₀ Q + 2 Re Σ w_k (z_k I - A)⁻¹ Q

----------------------------------------------------------------------
Parameters
----------------------------------------------------------------------

Same as `general_feast_sparse_interval_zolotarev`.
`nodes` still denotes the **total number of poles of the full filter**.
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






















if abspath(PROGRAM_FILE)==@__FILE__

    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing

    # SOLVER PARAMETERS; NODES/POLES - CRITICAL TO GET SMALLEST POSSIBLE WITH LOWEST ITERATION COUNT
    nodes_general=32 # this is the total nodes, sym_ automatically uses half the nodes for the half-contour version
    nodes_chebyshev=96 # Chebyshev filter (actually quite a high degree is needed)
    nodes_zolotarev=16 # Zolotarev rational filter, requiring typically lower node count than ellipse or Chebyshev filters for similar accuracy

    # TOLERANCE FOR RESIDUAL OF EIGENVECTOR
    tol=1e-8 # test tolerance (max |Δλ| and max residual)

    # MATRIX SIZES SPARSE/DENSE
    ns=[5000] # sparse matrix size lap1d to test
    n=200 # size of the dense matrix (just to test that the code works on dense matrices as well)

    # OPTIONS 
    do_general=false # test for general sparse matrices (not necessarily symmetric)
    do_half=true # test half-contour version for symmetric matrices (2x faster but only for symmetric problems)
    do_chebyshev=false # test Chebyshev filtered subspace iteration for symmetric matrices
    do_zolotarev=true # test Zolotarev rational filter for full 
    do_zolotarev_half=true # test Zolotarev rational filter for half (real symmetric matrices)

    # ADDITIONAL OPTIONS
    do_distributed=false # if we want for every option above to test the "distributed" version that doesn't store factorizations but recomputes them at every iteration (this is more memory efficient but more expensive computationally, so it's not the default for testing)
    show_progress=false # to see progress bars for each benchmark. Use false to avoid visual clutter and influence on performance

    # ============================================================
    # TESTS
    # ============================================================

    # 1D Laplacian tridiag(-1,2,-1)
    function lap1d(n::Int)
        d0=fill(2.0,n)
        d1=fill(-1.0,n-1)
        return spdiagm(-1=>d1,0=>d0,1=>d1)
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

    function test_cheb_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,degree=nodes_chebyshev,tol=tol,maxiter=8,res_gate=1e-6,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=chebyshev_poly_real_symm_sparse(A;emin=emin,emax=emax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,debug=debug,show_progress=show_progress)
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

    function test_cheb_on_dense_small(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,degree=nodes_chebyshev,tol=tol,maxiter=20,res_gate=1e-8,debug=false,show_progress::Bool=false)
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
        λ,X,res=chebyshev_poly_real_symm_sparse(As;emin=emin,emax=emax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "CHEB(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "CHEB(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
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

    function bench_sweep_cheb(;ns=[5_000,10_000,20_000],degree=nodes_chebyshev,tol=tol,maxiter=20,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            m0=ceil(Int,1.3*expected)
            test_cheb_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,degree=degree,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,show_progress=show_progress)
        end
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

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV POLYNOMIAL FILTER")
    println("--------------------------------------------------")

    s=time()
    println("SPARSE / CHEB")
    bench_sweep_cheb(ns=ns,degree=nodes_chebyshev,maxiter=30,tol=1e-8,show_progress=show_progress) # surprisingly the larger matrices converge in fewer iterations
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (SPARSE / CHEB):")
    println("degree=", nodes_chebyshev, ": ", @sprintf("%.3f seconds", e-s))
    println("--------------------------------------------------")

    # DO NOT USE CHEBYSHEV FILTER FOR DENSE MATRICES, IT PROBABLY WILL NOT CONVERGE.

    end

end