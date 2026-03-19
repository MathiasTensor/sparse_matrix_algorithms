# ==============================================================================
# Zolotarev-Accelerated FEAST Sparse / Matrix-Free Eigenvalue Solver
# ==============================================================================
#
# This file implements FEAST-type eigensolvers based on Zolotarev rational
# filters for computing interior eigenpairs of large sparse or matrix-free
# operators.
#
# ------------------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------------------
#
# We solve the eigenvalue problem
#
#     A x = λ x
#
# for eigenvalues λ inside a target interval [emin, emax].
#
# Instead of approximating the spectral projector with contour quadrature on an
# ellipse, this code uses a Zolotarev rational approximation to the ideal step
# function / projector. The resulting FEAST filter has the form
#
#     r(x) = α₀ + Σ_j ω_j / (x - p_j) + Σ_j conj(ω_j) / (x - conj(p_j)),
#
# where p_j are poles in the upper half-plane and ω_j are the corresponding
# residues. After mapping from the canonical interval [-G,G] to the physical
# interval [emin, emax], the filter is applied to the current FEAST subspace.
#
# ------------------------------------------------------------------------------
# Mathematical Idea
# ------------------------------------------------------------------------------
#
# The spectral projector onto eigenvalues in [emin, emax] is approximated by a
# near-optimal rational filter of Zolotarev type:
#
#     P ≈ r(A)
#
# with
#
#     r(A)Q = α₀ Q + Σ_j ω_j (p_j I - A)^{-1} Q
#                        + Σ_j conj(ω_j) (conj(p_j) I - A)^{-1} Q.
#
# For real symmetric problems, conjugate symmetry allows the reduced form
#
#     r(A)Q = α₀ Q + 2 Re Σ_j ω_j (p_j I - A)^{-1} Q,
#
# so only upper-half-plane poles must be evaluated.
#
# ------------------------------------------------------------------------------
# Implemented Components
# ------------------------------------------------------------------------------
#
# 1. Zolotarev construction
#    - complete / incomplete elliptic integrals
#    - Jacobi elliptic function sn(u|m)
#    - canonical Zolotarev rational filter on [-G,G]
#    - mapping of poles and residues to [emin, emax]
#
# 2. Parameter containers
#    - ZolotarevParams
#    - precompute poles / weights once and reuse across sweeps
#
# 3. Sparse FEAST solvers
#    - general_feast_sparse_interval_zolotarev
#    - general_feast_sparse_interval_zolotarev_distributed
#    - sym_feast_sparse_interval_half_zolotarev
#    - sym_feast_sparse_interval_half_zolotarev_distributed
#
# 4. Matrix-free FEAST solver (EXPERIMENTAL)
#    - sym_feast_sparse_interval_half_zolotarev(matvec!, ...)
#    - uses block GMRES on shifted matrix-free operators
#
# 5. Shared utilities
#    - sparse diagonal shifting with fixed CSC sparsity
#    - thin QR orthogonalization
#    - residual evaluation
#    - matrix-free real symmetric operator wrappers
#
# ------------------------------------------------------------------------------
# FEAST Iteration with Zolotarev Filter
# ------------------------------------------------------------------------------
#
# Given a search subspace Q ∈ ℂ^{n×m0}, each FEAST iteration performs:
#
#   1. Rational filtering:
#
#          Y = α₀ Q + Σ_j ω_j (p_j I - A)^{-1} Q + ...
#
#      or, in the real-symmetric half-plane variant,
#
#          Y = α₀ Q + 2 Re Σ_j ω_j (p_j I - A)^{-1} Q
#
#   2. Subspace orthogonalization:
#
#          Q ← orth(Y)
#
#   3. Rayleigh–Ritz projection:
#
#          Aq = Qᴴ A Q
#
#   4. Dense projected eigensolve:
#
#          Aq u = λ u
#
#   5. Lift Ritz vectors:
#
#          X = Q u
#
#   6. Residual filtering / convergence check
#
# ------------------------------------------------------------------------------
# Implemented Variants
# ------------------------------------------------------------------------------
#
# A. General sparse Zolotarev FEAST
#    - full pole set
#    - complex arithmetic
#    - applicable to general real sparse matrices
#
# B. Symmetric half-plane Zolotarev FEAST
#    - real symmetric sparse matrices only
#    - uses upper-half-plane poles only
#    - ~2× fewer shifted solves
#    - real Rayleigh–Ritz stage
#
# C. Distributed / low-memory variants
#    - do not store all shifted factorizations
#    - refactorize or rebuild shifted solves on the fly
#    - slower but more memory-efficient
#
# D. Matrix-free symmetric half-plane variant (EXPERIMENTAL)
#    - user supplies y ← A*x
#    - shifted systems solved iteratively (block GMRES)
#    - useful when A is too large to materialize explicitly
#
# ------------------------------------------------------------------------------
# Key Parameters
# ------------------------------------------------------------------------------
#
# - G        : canonical Zolotarev gap / steepness parameter - used here is 0.9
# - nodes    : number of poles in the upper half-plane
# - m0       : FEAST search subspace dimension
# - tol      : convergence tolerance
# - res_gate : residual threshold for accepted eigenpairs
# - backend  : shifted linear solver backend (:lu or :ls)
# - blockrhs : number of RHS solved simultaneously in low-memory variants

# ------------------------------------------------------------------------------
# Reference
# ------------------------------------------------------------------------------
#
#   Zolotarev Quadrature Rules and Load Balancing for the FEAST Eigensolver, Güttel S., Tak Peter Tang P., Viaud G., https://arxiv.org/abs/1407.8078
#
# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# ==============================================================================


using LinearAlgebra
using SparseArrays
using Random
using Printf
using BenchmarkTools
using LinearSolve
using ProgressMeter
using Statistics
using Krylov
using QuadGK

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

"""
    MatrixFreeRealSymOp{F,G}

Matrix-free real symmetric operator. Represents a real symmetric linear map A on ℝⁿ without explicitly storing a full
matrix. If `matblock!` is supplied, block applications can be more efficient
than repeated columnwise `matvec!` calls.

Fields
------
- `n::Int`: operator dimension.
- `matvec!::F`: in-place action `y ← A*x`.
- `matblock!::Union{Nothing,G}`: optional in-place block action `Y ← A*X`.
"""
struct MatrixFreeRealSymOp{F,G,H}
    n::Int
    matvec!::F
    matblock!::Union{Nothing,G}
end
"""
    MatrixFreeRealSymOp(n,matvec!)

Construct a matrix-free operator with only vector action.
"""
MatrixFreeRealSymOp(n::Int,matvec!::F) where {F}=MatrixFreeRealSymOp{F,Nothing,Nothing}(n,matvec!,nothing)
"""
    MatrixFreeRealSymOp(n,matvec!,matblock!)

Construct a matrix-free operator with both vector and block actions.
"""
MatrixFreeRealSymOp(n::Int,matvec!::F,matblock!::G) where {F,G}=MatrixFreeRealSymOp{F,G,Nothing}(n,matvec!,matblock!)

# helper for size of the operator as a matrix
Base.size(A::MatrixFreeRealSymOp)=(A.n,A.n)
Base.size(A::MatrixFreeRealSymOp,d::Int)=(d==1) || (d==2) ? A.n : 1
"""
    LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})

Matrix-free matrix-vector product for a real symmetric operator `A` represented by a `MatrixFreeRealSymOp`. Computes `y <- A*x` using the user-provided `matvec!` function.
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})
    A.matvec!(y,x)
    return y
end
"""
    apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})

Block matrix-free application of a real symmetric operator `A` to a block of vectors `X`, storing the result in `Y`. Computes `Y[:,j] <- A*X[:,j]` for each column `j` of `X`. 
"""
@inline function apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    if !isnothing(A.matblock!)
        A.matblock!(Y,X)
        return Y
    else
        n,m=size(X)
        @inbounds for j in 1:m
            mul!(view(Y,:,j),A,view(X,:,j))
        end
        return Y
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
    solve_shifted_block_matrixfree!(X::Matrix{ComplexF64},Sop::ShiftedMatrixFreeOp,B::Matrix{ComplexF64};tol=1e-12,maxiter=5000)

Solve the shifted linear systems defined by `Sop` for each column of `B`, storing the results in `X`. This function uses an iterative linear solver (e.g., GMRES) to solve `Sop * x = b` for each column `b` of `B`, where `Sop` is a `ShiftedMatrixFreeOp` representing the shifted operator `S = zI - A`. The user can specify the linear solver algorithm and convergence parameters.
"""
function solve_shifted_block_matrixfree!(X::Matrix{ComplexF64},Sop::ShiftedMatrixFreeOp,B::Matrix{ComplexF64};tol=1e-12,maxiter=5000,ls_alg::Symbol=:gmres)
    #TODO This is still very slow...
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

















if abspath(PROGRAM_FILE)==@__FILE__

    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing
    nodes_zolotarev=16 # Zolotarev rational filter, requiring typically lower node count than ellipse or Chebyshev filters for similar accuracy
    do_general=true
    do_half=true
    do_distributed=true
    tol=1e-8
    show_progress=true # progress bars for all tests, for cleaner output set to false

    ns=[20_000] # matrix sizes to test (sparse)
    n=1000

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
            @info "ZOLO: n=$n, emin=$emin, emax=$emax, expected eigenvalues in interval = $(ceil(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d_zolotarev(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
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
            @info "ZOLO HALF: n=$n, emin=$emin, emax=$emax, expected eigenvalues in interval = $(ceil(Int,expected))"
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
            @info "ZOLO DIST: n=$n, emin=$emin, emax=$emax, expected eigenvalues in interval = $(ceil(Int,expected))"
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
            @info "ZOLO HALF/DIST: n=$n, emin=$emin, emax=$emax, expected eigenvalues in interval = $(ceil(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d_zolotarev_distributed(zp,n=n,emin=emin,emax=emax,m0=m0,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    if do_half

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV HALF-FILTER")
    println("--------------------------------------------------")

    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    s=time()
    println("LinearSolve KLU")
    bench_sweep_half_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / ZOLOTAREV HALF")
    println("LU (default)")
    test_sym_half_on_dense_small_zolotarev(zp,n=n,backend=:lu,show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("LU backend:                 ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    end

    if do_general

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV FILTER FEAST")
    println("--------------------------------------------------")

    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    s=time()
    println("SPARSE / ZOLOTAREV")
    println("LinearSolve KLU")
    bench_sweep_zolotarev(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / ZOLOTAREV")
    println("LU (default)")
    test_feast_on_dense_small_zolotarev(zp,n=n,backend=:lu,show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("Timing summary (DENSE / ZOLOTAREV):")
    println("LU backend:                 ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    end

    if do_distributed && do_general

    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    println()
    println("--------------------------------------------------")
    println("ZOLOTAREV DISTRIBUTED")
    println("--------------------------------------------------")

    s=time()
    println("SPARSE / ZOLOTAREV / DISTRIBUTED")
    println("LinearSolve KLU (blockrhs=32)")
    bench_sweep_zolotarev_distributed(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    end

    if do_distributed && do_half

    Gzolo=0.9
    @time "Zolotarev params construction" zp=ZolotarevParams(nodes_zolotarev÷2,Gzolo;tol=1e-14,ngrid=20000)

    println()
    println("SPARSE / ZOLOTAREV HALF / DISTRIBUTED")
    s=time()
    println("LinearSolve KLU (blockrhs=32)")
    bench_sweep_half_zolotarev_distributed(zp,ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    e=time()

    println("--------------------------------------------------")
    println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

    end

end
