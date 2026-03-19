# ==============================================================================
# Chebyshev Filtered Subspace Iteration
# ==============================================================================
#
# This file implements polynomial-filtered subspace iteration for computing
# interior eigenpairs of large real symmetric sparse or matrix-free operators.
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
# Instead of using shifted linear solves (as in FEAST / rational filtering),
# this method constructs a polynomial filter p(A) that amplifies spectral
# components inside the desired interval while damping components outside it.
#
# The filtered subspace iteration step is
#
#     Y = p(A) Q
#
# followed by orthogonalization and Rayleigh–Ritz extraction.
#
# ------------------------------------------------------------------------------
# Mathematical Idea
# ------------------------------------------------------------------------------
#
# The matrix A is first rescaled to
#
#     Â = (A - cI) / d
#
# where
#
#     c = (smin + smax)/2,
#     d = (smax - smin)/2,
#
# so that the spectrum of Â lies approximately inside [-1, 1].
#
# A polynomial filter is then built in the Chebyshev basis:
#
#     p(x) = Σ_{k=0}^m c_k T_k(x),
#
# where T_k(x) is the Chebyshev polynomial of degree k.
#
# Three filter types are supported:
#
#   1. step   : approximates the indicator of [emin, emax]
#   2. delta  : approximates a localized peak near σ
#   3. smooth : approximates a smoothed interval window
#
# Jackson damping may be applied to reduce Gibbs oscillations and improve
# numerical stability.
#
# ------------------------------------------------------------------------------
# Implemented Variants
# ------------------------------------------------------------------------------
#
# 1. Sparse matrix version
#    - chebyshev_poly_real_symm_sparse(A, emin, emax; ...)
#    - uses sparse matrix–block products
#
# 2. Matrix-free version
#    - chebyshev_poly_real_symm_sparse(matvec!, n, emin, emax; ...)
#    - uses user-supplied matrix-vector or block-matrix-vector products
#
# ------------------------------------------------------------------------------
# Algorithm (per iteration)
# ------------------------------------------------------------------------------
#
# Given a search subspace Q ∈ ℝ^{n×m0}:
#
#   1. Apply polynomial filter:
#
#          Y = p(Â) Q
#
#      using the three-term Chebyshev recurrence
#
#          T₀(Â)Q = Q
#          T₁(Â)Q = ÂQ
#          T_{k+1}(Â)Q = 2Â T_k(Â)Q - T_{k-1}(Â)Q
#
#   2. Orthonormalize:
#
#          Q ← orth(Y)
#
#   3. Rayleigh–Ritz projection:
#
#          Aq = Qᵀ A Q
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
# Key Parameters
# ------------------------------------------------------------------------------
#
# - m0          : subspace size (must exceed number of wanted eigenvalues)
# - degree      : polynomial degree of the filter
# - window_type : :step, :delta, or :smooth
# - jackson     : whether to use Jackson damping
# - smin, smax  : spectral bounds for rescaling
# - σ           : center for delta-like filters
#
# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# ==============================================================================
#TODO Lock converged vectors efficiently without degrading the quality of the others. How to make a reasonable locking tolerance where the other ones wont degrade due to locking a part of the space?

using LinearAlgebra
using SparseArrays
using Random
using Printf
using BenchmarkTools
using ProgressMeter
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
        Threads.@threads for j in 1:m
            mul!(view(Y,:,j),A,view(X,:,j))
        end
        return Y
    end
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

function make_thin_orth_workspace(::Type{Float64},n::Int,m0::Int)
    Qtmp=Matrix{Float64}(undef,n,m0)
    τ=Vector{Float64}(undef,min(n,m0))
    return Qtmp,τ
end

"""
    orth_against!(W,Q)

Project block `W` orthogonally against the column space of `Q`:

    W ← W - Q(Q^T W).

This is the standard orthogonal projector used in block orthogonalization.
"""
function orth_against!(W::AbstractMatrix{Float64},Q::AbstractMatrix{Float64})
    size(Q,2)==0 && return W
    W.-=Q*(transpose(Q)*W)
    return W
end

"""
    orth_block!(Q;rtol=1e-12)

Return an orthonormal basis for the columns of `Q`, truncating nearly dependent
directions using the diagonal of the QR R-factor.

Criterion
---------
If `R=QR` and `r0=max|R_ii|`, keep only columns with

    |R_ii| > rtol*r0.

This acts as a numerical rank-revealing QR truncation.
"""
function orth_block!(Q::AbstractMatrix{Float64};rtol::Float64=1e-12)
    F=qr!(Q)
    R=Matrix(F.R)
    k=min(size(R,1),size(R,2))
    k==0 && return Matrix{Float64}(undef,size(Q,1),0)
    dr=abs.(diag(R[1:k,1:k]))
    r0=maximum(dr)
    r0==0 && return Matrix{Float64}(undef,size(Q,1),0)
    r=count(>(rtol*r0),dr)
    r==0 && return Matrix{Float64}(undef,size(Q,1),0)
    return Matrix(F.Q[:,1:r])
end

# helper to orthogonalize a new block against a locked subspace so as to potentially reduce the workload after each iteration
function append_locked!(Xlock::Matrix{Float64},Xnew::Matrix{Float64};rtol::Float64=1e-12)
    size(Xnew,2)==0 && return Xlock
    if size(Xlock,2)>0
        Xnew.-=Xlock*(transpose(Xlock)*Xnew)
    end
    Qnew=orth_block!(copy(Xnew);rtol=rtol)
    size(Qnew,2)==0 && return Xlock
    return size(Xlock,2)==0 ? Qnew : hcat(Xlock,Qnew)
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
    jackson_kernel(K)

Return Jackson damping coefficients `g_0,...,g_K`. Jackson damping suppresses Gibbs oscillations in truncated Chebyshev series,
trading sharpness for smoother, more stable polynomial filters.
Ref: A CHEBYSHEV–JACKSON SERIES BASED BLOCK SS–RR ALGORITHM FOR COMPUTING PARTIAL EIGENPAIRS OF REAL SYMMETRIC MATRICES, ZHONGXIAO J. AND TIANHANG L. https://arxiv.org/pdf/2508.20456
"""
function jackson_kernel(K::Int)
    K<=0 && return [1.0]
    g=zeros(Float64,K+1)
    α=pi/(K+2)
    sα=sin(α)
    @inbounds for k in 0:K
        g[k+1]=sin((k+1)*α)/((K+2)*sα)+(1-(k+1)/(K+2))*cos(k*α)
    end
    return g
end

"""
    cheb_delta_coeffs(σ̂,K;jackson=true)

Return Chebyshev coefficients for a delta-like peak centered at `σ̂∈[-1,1]`.

Mathematical form
-----------------
Approximates a sharply peaked kernel in the Chebyshev basis

    p(x) ≈ 1/π + 2/π Σ_{k≥1} cos(k arccos(σ̂)) T_k(x),

optionally Jackson-damped.
"""
function cheb_delta_coeffs(σ̂::Float64,K::Int;jackson::Bool=true)
    c=zeros(Float64,K+1)
    θ=acos(clamp(σ̂,-1.0,1.0))
    c[1]=1/pi
    @inbounds for k in 1:K
        c[k+1]=2*cos(k*θ)/pi
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

"""
    cheb_window_coeffs(a,b,K;jackson=true)

Return Chebyshev coefficients for the indicator of the interval `[a,b]⊂[-1,1]`.

Mathematical form
-----------------
For the characteristic function χ_[a,b], the Chebyshev coefficients are

    c_0 = (β-α)/π,
    c_k = 2(sin(kβ)-sin(kα))/(kπ),  k≥1,

with `α=acos(b)`, `β=acos(a)`, assuming `a≤b`.
"""
function cheb_window_coeffs(a::Float64,b::Float64,K::Int;jackson::Bool=true)
    aa=max(-1.0,min(1.0,a))
    bb=max(-1.0,min(1.0,b))
    aa>bb && ((aa,bb)=(bb,aa))
    α=acos(bb)
    β=acos(aa)
    c=zeros(Float64,K+1)
    c[1]=(β-α)/pi
    @inbounds for k in 1:K
        c[k+1]=2*(sin(k*β)-sin(k*α))/(k*pi)
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

"""
    cheb_smooth_window_coeffs(a,b,K;jackson=true,nsamp=4096,δ=0.02)

Return Chebyshev coefficients of a smoothed window on `[a,b]`, using tanh ramps.

Mathematical form
-----------------
The window is

    f(x)=½(1+tanh((x-a)/δ)) * ½(1-tanh((x-b)/δ)),

and coefficients are computed by numerical quadrature in the θ-variable
(`x=cos θ`). This often produces a more stable filter than the discontinuous
indicator.
"""
function cheb_smooth_window_coeffs(a::Float64,b::Float64,K::Int;jackson::Bool=true,nsamp::Int=4096,δ::Float64=0.02)
    aa=max(-1.0,min(1.0,a))
    bb=max(-1.0,min(1.0,b))
    aa>bb && ((aa,bb)=(bb,aa))
    c=zeros(Float64,K+1)
    w=pi/(nsamp-1)
    @inbounds for k in 0:K
        s=0.0
        for i in 1:nsamp
            θ=(i-1)*w
            xi=cos(θ)
            fi=0.5*(1+tanh((xi-aa)/δ))*0.5*(1-tanh((xi-bb)/δ))
            wt=(i==1 || i==nsamp) ? 0.5 : 1.0
            s+=wt*fi*cos(k*θ)
        end
        c[k+1]=(k==0 ? 1/pi : 2/pi)*w*s
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

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

struct Chebyshev_workspace
    b1::Matrix{Float64}
    b2::Matrix{Float64}
    tmp::Matrix{Float64}
    ax::Matrix{Float64}
end
"""
    Chebyshev_workspace(n,b)

Allocate a workspace for block size `b` in dimension `n`.
"""
Chebyshev_workspace(n::Int,b::Int)=Chebyshev_workspace(zeros(n,b),zeros(n,b),zeros(n,b),zeros(n,b))
"""
    apply_poly_block!(Y,A,Q,coeffs,smin,smax,work)

Apply the Chebyshev polynomial filter to a block `Q` for an explicit sparse A.

Output
------
Computes approximately

    Y = p(Â)Q,   Â=(A-cI)/d.

Method
------
Uses a block Clenshaw recurrence for stability:

    b_{K+1}=b_{K+2}=0,
    b_k = α_k Q + 2Â b_{k+1} - b_{k+2},
    p(Â)Q = α_0 Q + Â b_1 - b_2.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::Chebyshev_workspace)
    # Apply Y = p(Â)Q by block Clenshaw recurrence, where Â=(A-cI)/d.
    # Recurrence:
    #   B_{K+1}=B_{K+2}=0
    #   B_k = α_k Q + 2 Â B_{k+1} - B_{k+2},   k=K,...,1
    #   Y   = α₀ Q + Â B₁ - B₂
    #   Dominant cost: repeated block applications of A.
    K=length(coeffs)-1
    c=0.5*(smin+smax)
    invd=2.0/(smax-smin)
    α=2.0*invd
    β=-2.0*c*invd
    B1=work.b1
    B2=work.b2
    TMP=work.tmp
    AX=work.ax
    n,b=size(Q)
    fill!(B1,0.0)
    fill!(B2,0.0)
    @inbounds for k=(K+1):-1:2
        mul!(AX,A,B1)
        ck=coeffs[k]
        for j in 1:b
            @simd for i in 1:n
                TMP[i,j]=ck*Q[i,j]+α*AX[i,j]+β*B1[i,j]-B2[i,j]
            end
        end
        B1,B2,TMP=TMP,B1,B2
    end
    mul!(AX,A,B1)
    c0=coeffs[1]
    @inbounds for j in 1:b
        @simd for i in 1:n
            Y[i,j]=c0*Q[i,j]+invd*AX[i,j]-c*invd*B1[i,j]-B2[i,j]
        end
    end
    return Y
end

"""
    apply_poly_block!(Y,Aop,Q,coeffs,smin,smax,work)

Apply the Chebyshev polynomial filter to a block `Q` for a matrix-free operator. This is the generic version where the operator is accessed only through `apply_A_block!`.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::Chebyshev_workspace)
    # Apply Y = p(Â)Q by block Clenshaw recurrence, where Â=(A-cI)/d.
    # Recurrence:
    #   B_{K+1}=B_{K+2}=0
    #   B_k = α_k Q + 2 Â B_{k+1} - B_{k+2},   k=K,...,1
    #   Y   = α₀ Q + Â B₁ - B₂
    #   Dominant cost: repeated block applications of A.
    K=length(coeffs)-1
    c=0.5*(smin+smax)
    invd=2.0/(smax-smin)
    α=2.0*invd
    β=-2.0*c*invd
    B1=work.b1
    B2=work.b2
    TMP=work.tmp
    AX=work.ax
    n,b=size(Q)
    fill!(B1,0.0)
    fill!(B2,0.0)
    @inbounds for k=(K+1):-1:2
        apply_A_block!(AX,A,B1)
        ck=coeffs[k]
        for j in 1:b
            @simd for i in 1:n
                TMP[i,j]=ck*Q[i,j]+α*AX[i,j]+β*B1[i,j]-B2[i,j]
            end
        end
        B1,B2,TMP=TMP,B1,B2
    end
    apply_A_block!(AX,A,B1)
    c0=coeffs[1]
    @inbounds for j in 1:b
        @simd for i in 1:n
            Y[i,j]=c0*Q[i,j]+invd*AX[i,j]-c*invd*B1[i,j]-B2[i,j]
        end
    end
    return Y
end

"""
    chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int},emin::Float64,emax::Float64;smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

Compute eigenpairs of a real symmetric sparse matrix with eigenvalues in [emin,emax]
using Chebyshev polynomial filtered subspace iteration.

Inputs
- `Ain`: real symmetric sparse matrix (in CSC format)
- `emin`, `emax`: target interval for eigenvalues (must satisfy emin<emax)

Parameters
- `smin`, `smax`: spectral bounds for the matrix (must satisfy smin<smax and contain [emin,emax]); if not provided, will be estimated with Gershgorin bounds (cheap but often loose, which can degrade filter quality)
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
function chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int},emin::Float64,emax::Float64;smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

    issymmetric(Ain) || throw(ArgumentError("Chebyshev filtered solver requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    degree>=0 || throw(ArgumentError("degree must be nonnegative"))
    n=size(Ain,1)
    size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=Ain
    if smin===nothing || smax===nothing
        smin_,smax_=gershgorin_bounds(A) # cheap spectral bounds
    else
        smin_=smin
        smax_=smax
    end
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
        @error("Unknown window type, possible are :step, :delta, and :smooth")
    end
    Random.seed!(seed)
    Q=randn(n,m0) # initialize a random block of m0 vectors; will be replaced by the filtered subspace
    Y=Matrix{Float64}(undef,n,m0) # workspace for the filtered block Y = p(Â)Q
    work=Chebyshev_workspace(n,m0)
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
        apply_poly_block!(Y,A,Q,coeffs,smin_,smax_,work) # Y = p(Â)Q - filtered block
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
    chebyshev_poly_real_symm_sparse(matvec!::F,n::Int,emin::Float64,emax::Float64;matblock::Union{G,Nothing}=nothing,smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step) where {F,G}

For details on parameters see the `chebyshev_poly_real_symm_sparse` dispatch with the concrete matrix `Ain`. The input is the callnack with signature `f(y,x) := y = A*x` and `n` is the matrix dimension of the linear operator.

Additional Inputs
- `matvec!::F`: a user-provided function that computes the matrix-vector product `y = A*x` for the linear operator `A` whose eigenpairs we want to compute; the function should have the signature `f(y,x) := y = A*x` where `y` and `x` are vectors of length `n`.

Additional Parameters
- `matblock::Union{G,Nothing}`: optional additional argument to be passed to the `matvec!` function, for example to allow for a block matrix-vector product; if not needed, can be left as `nothing` (default).

"""
function chebyshev_poly_real_symm_sparse(matvec!::F,n::Int,emin::Float64,emax::Float64;matblock::Union{G,Nothing}=nothing,smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step) where {F,G}
    Ain=MatrixFreeRealSymOp(n,matvec!,matblock)
    n=size(Ain,1)
    m0=min(m0,n)
    if smin===nothing || smax===nothing
        smin_,smax_=gershgorin_bounds(Ain)
    else
        smin_=smin
        smax_=smax
    end
    c=0.5*(smin_+smax_)
    d=0.5*(smax_-smin_)
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
        @error("Unknown window type, possible are :step, :delta, and :smooth")
    end
    Random.seed!(seed)
    Q=randn(n,m0)
    Y=Matrix{Float64}(undef,n,m0)
    work=Chebyshev_workspace(n,m0)
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
        apply_poly_block!(Y,Ain,Q,coeffs,smin_,smax_,work)
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
        debug && @printf("it=%d inside=%d maxres=%.3e r90=%.3e\n",it,count(inside),maxres,isempty(res[inside]) ? Inf : quantile(res[inside],0.9))
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

    n=10_000 # size of the matrix
    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing
    degrees_chebyshev=[48,64,128,256,512] # degrees of the chebyshev polynomial approximating the delta function filter
    counts_chebyshev=reverse([100,150,200,250,500,1000]) # how many eigenvalues we want in the delta window, more eigenvealues (m0), less degree we need
    do_matrix_free=true
    do_matrix_concrete=false

    maxiter=10
    show_progress=true
    m0_pad=2.0 # multiplicative padding to the search subspace
    tol=1e-8

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
    function lap1d_block_matvec!(Y::AbstractMatrix{Float64},X::AbstractMatrix{Float64})
        n,b=size(X)
        Threads.@threads for j in 1:b
            @inbounds begin
                Y[1,j]=2.0*X[1,j]-X[2,j]
                for i in 2:n-1
                    Y[i,j]=-X[i-1,j]+2.0*X[i,j]-X[i+1,j]
                end
                Y[n,j]=-X[n-1,j]+2.0*X[n,j]
            end
        end
        return Y
    end
    lap1d_eigs_exact(n::Int)=[2-2*cos(k*pi/(n+1)) for k in 1:n]

    function nearest_k_to_sigma(vals::Vector{Float64},σ::Float64,k::Int)
        p=sortperm(abs.(vals.-σ))
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
    function test_cheb_delta_on_lap1d_count(matvec!,matblock!;n=5000,nev_target=50,center_frac=0.5,degree=64,tol=1e-8,res_gate=1e-6,window_type=:step,matrix_free=false)
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
            λ,X,res=chebyshev_poly_real_symm_sparse(lap1d_matvec!,n,emin,emax;matblock=lap1d_block_matvec!,σ=σ,smin=smin,smax=smax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,window_type=window_type,debug=show_progress,two_hit=false)
        else
            λ,X,res=chebyshev_poly_real_symm_sparse(A,emin,emax;σ=σ,smin=smin,smax=smax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,window_type=window_type,debug=show_progress,two_hit=false)
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

    function bench_sweep_cheb_counts(;n=5000,counts=[20,50,100,150],degrees=[16,32,48,64,96,128],center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:step,matrix_free::Bool=false)

        println("==========================================================================")
        println("CHEB DELTA local-count benchmark")
        println("==========================================================================")
        @printf("%8s %8s %8s %8s %8s %8s %8s %12s\n",
                "nev","deg","m0","got","match","lextra","spur","localerr")
        println("--------------------------------------------------------------------------")

        for nev in counts
            for deg in degrees
                t=@elapsed begin
                    λ,λref,res,meta=test_cheb_delta_on_lap1d_count(lap1d_matvec!,lap1d_block_matvec!,n=n,nev_target=nev,center_frac=center_frac,degree=deg,tol=tol,res_gate=res_gate,window_type=window_type,matrix_free=matrix_free)
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
                nev==_meta.nmatched_target && break # if we got all the target eigenvalues, no need to keep increasing degree for this count
            end
            println("--------------------------------------------------------------------------")
        end
    end

    if do_matrix_free
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:delta,matrix_free=true)

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV STEP WINDOW MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:step,matrix_free=true)

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV SMOOTH WINDOW MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:smooth,matrix_free=true)

    end

    if do_matrix_concrete
    
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB DELTA")
    
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:delta,matrix_free=false)

    println("--------------------------------------------------")
    println("CHEBYSHEV STEP FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB STEP")
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:step,matrix_free=false)

    println("--------------------------------------------------")
    println("CHEBYSHEV SMOOTH FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB SMOOTH")
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,tol=1e-8,res_gate=1e-6,window_type=:smooth,matrix_free=false)

    end

end

#########################
#### RESULTS SUMMARY ####
#########################

#=
--------------------------------------------------
CHEBYSHEV DELTA FILTER MATRIX FREE
--------------------------------------------------
==========================================================================
CHEB DELTA local-count benchmark
==========================================================================
     nev      deg       m0      got    match   lextra     spur     localerr
--------------------------------------------------------------------------
CHEBΔ count-test: n=50000 nev_target=1000 idx=[24500,25499] interval=[1.9368657406389307, 2.0628830617279608] σ=1.9998744006869607 m0=2000 degree=128
it=1 inside=0 maxres=Inf r90=Inf
it=2 inside=0 maxres=Inf r90=Inf
it=3 inside=245 maxres=9.983e-07 r90=9.181e-07
it=4 inside=982 maxres=9.882e-07 r90=1.720e-07
it=5 inside=1003 maxres=4.247e-07 r90=3.969e-09
it=6 inside=1003 maxres=2.088e-08 r90=4.642e-11
it=7 inside=1003 maxres=5.394e-10 r90=1.053e-12
matched target = 1000 / 1000
missed target  = 0
extra total    = 3
legit extra    = 3
spurious extra = 0
    1000      128     2000     1003     1000        3        0    3.775e-15
time[s] = 117.0751
=#
